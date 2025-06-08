import os
import time
import sys
import argparse
import traceback
import subprocess
from pathlib import Path
from PIL import Image

import torch
import gc
import numpy as np
import librosa
from tqdm import tqdm

from mmgp import offload, profile_type
from wan.modules.attention import get_attention_modes, get_supported_attention_modes
from wan.utils.utils import cache_video

# --- Bắt đầu phần code được giữ lại và tái cấu trúc từ wgp.py ---

# Các biến toàn cục và cấu hình được đơn giản hóa
bfloat16_supported = torch.cuda.get_device_capability()[0] >= 8

# Danh sách các mô hình cần thiết để get_model_filename hoạt động
hunyuan_choices = [
    "ckpts/hunyuan_video_avatar_720_bf16.safetensors",
    "ckpts/hunyuan_video_avatar_720_quanto_bf16_int8.safetensors",
]
transformer_choices = hunyuan_choices

model_signatures = {
    "hunyuan_avatar": "hunyuan_video_avatar"
}


def get_model_family(model_filename):
    if "hunyuan" in model_filename:
        return "hunyuan"
    raise Exception(f"Unknown model family for model'{model_filename}'")


def get_transformer_dtype(model_family, transformer_dtype_policy):
    if len(transformer_dtype_policy) == 0:
        return torch.bfloat16 if bfloat16_supported else torch.float16
    elif transformer_dtype_policy == "fp16":
        return torch.float16
    else:
        return torch.bfloat16


def get_model_filename(model_type, quantization="int8", dtype_policy=""):
    signature = model_signatures[model_type]
    choices = [name for name in transformer_choices if signature in name]
    if len(quantization) == 0:
        quantization = "bf16"

    model_family = get_model_family(choices[0])
    dtype = get_transformer_dtype(model_family, dtype_policy)

    sub_choices = [name for name in choices if quantization in name]
    if len(sub_choices) > 0:
        dtype_str = "fp16" if dtype == torch.float16 else "bf16"
        new_sub_choices = [name for name in sub_choices if dtype_str in name]
        raw_filename = new_sub_choices[0] if len(new_sub_choices) > 0 else sub_choices[0]
    else:
        raw_filename = choices[0]

    return raw_filename


def get_hunyuan_text_encoder_filename(text_encoder_quantization):
    if text_encoder_quantization == "int8":
        return "ckpts/llava-llama-3-8b/llava-llama-3-8b-v1_1_vlm_quanto_int8.safetensors"
    else:
        return "ckpts/llava-llama-3-8b/llava-llama-3-8b-v1_1_vlm_fp16.safetensors"


def load_hunyuan_model(model_filename, text_encoder_quantization, dtype, VAE_dtype, mixed_precision_transformer):
    from hyvideo.hunyuan import HunyuanVideoSampler
    from hyvideo.modules.models import get_linear_split_map

    print(f"Loading '{model_filename[-1]}' model...")
    hunyuan_model = HunyuanVideoSampler.from_pretrained(
        model_filepath=model_filename,
        text_encoder_filepath=get_hunyuan_text_encoder_filename(text_encoder_quantization),
        dtype=dtype,
        VAE_dtype=VAE_dtype,
        mixed_precision_transformer=mixed_precision_transformer
    )

    pipe = {"transformer": hunyuan_model.model, "text_encoder": hunyuan_model.text_encoder,
            "text_encoder_2": hunyuan_model.text_encoder_2, "vae": hunyuan_model.vae}
    if hunyuan_model.wav2vec is not None:
        pipe["wav2vec"] = hunyuan_model.wav2vec

    split_linear_modules_map = get_linear_split_map()
    hunyuan_model.model.split_linear_modules_map = split_linear_modules_map
    offload.split_linear_modules(hunyuan_model.model, split_linear_modules_map)

    return hunyuan_model, pipe


def initialize_model(args):
    """Tải và cấu hình mô hình Hunyuan Avatar."""
    model_type = "hunyuan_avatar"
    transformer_filename = get_model_filename(model_type, args.quantization, args.dtype)
    text_encoder_quantization = args.quantization

    print("--- Model Configuration ---")
    print(f"Transformer: {transformer_filename}")
    print(f"Quantization: {args.quantization}")
    print(f"Data Type: {args.dtype}")
    print("---------------------------")

    if not Path(transformer_filename).exists():
        print(f"ERROR: Model file not found at {transformer_filename}")
        sys.exit(1)

    model_filelist = [transformer_filename]
    transformer_dtype = get_transformer_dtype("hunyuan", args.dtype)
    VAE_dtype = torch.float16  # Giữ mặc định VAE fp16 cho hiệu năng

    # Tải mô hình
    wan_model, pipe = load_hunyuan_model(
        model_filelist,
        text_encoder_quantization=text_encoder_quantization,
        dtype=transformer_dtype,
        VAE_dtype=VAE_dtype,
        mixed_precision_transformer=False
    )
    wan_model._model_file_name = transformer_filename

    # Cấu hình offload
    profile = profile_type.LowRAM_LowVRAM  # Profile 4, an toàn cho nhiều hệ thống
    kwargs = {"budgets": {"*": 3000}}  # Cấp phát VRAM động

    offload.profile(pipe, profile_no=profile, convertWeightsFloatTo=transformer_dtype, **kwargs)

    return wan_model


def run_inference(wan_model, args):
    """Chạy suy luận và tạo video."""
    torch.set_grad_enabled(False)

    # 1. Chuẩn bị đầu vào
    try:
        image_start = Image.open(args.image).convert('RGB')
    except FileNotFoundError:
        print(f"ERROR: Image file not found at {args.image}")
        sys.exit(1)

    if not Path(args.audio).exists():
        print(f"ERROR: Audio file not found at {args.audio}")
        sys.exit(1)

    prompt = args.prompt
    image_refs = [image_start]

    width, height = 624, 832  # Độ phân giải 480p, tỉ lệ 3:4

    # 2. Cài đặt các tham số suy luận
    seed = args.seed if args.seed != -1 else np.random.randint(0, 999999999)
    num_inference_steps = args.steps
    guidance_scale = 7.5
    flow_shift = 5.0
    fps = 25  # Mặc định cho Hunyuan Avatar

    # Tính toán độ dài video dựa trên audio
    audio_duration = librosa.get_duration(path=args.audio)
    video_length = min(int(fps * audio_duration // 4) * 4 + 5, 401)  # Giới hạn max 401 frames
    print(f"Audio duration: {audio_duration:.2f}s, calculated video length: {video_length} frames.")

    # 3. Tạo callback để theo dõi tiến trình
    progress_bar = tqdm(total=num_inference_steps, desc="Denoising")

    def callback(step_idx, latent, force_refresh, **kwargs):
        progress_bar.update(1)
        if wan_model._interrupt:
            raise InterruptedError("Generation aborted.")

    # 4. Chạy mô hình
    start_time = time.time()
    try:
        samples = wan_model.generate(
            input_prompt=prompt,
            image_refs=image_refs,
            frame_num=video_length,
            height=height,
            width=width,
            sampling_steps=num_inference_steps,
            guide_scale=guidance_scale,
            shift=flow_shift,
            seed=seed,
            callback=callback,
            audio_guide=args.audio,
            fps=fps,
            model_filename=wan_model._model_file_name,
        )
    except Exception as e:
        print("\nAn error occurred during video generation:")
        traceback.print_exc()
        return
    finally:
        progress_bar.close()

    if samples is None:
        print("Generation was interrupted or failed.")
        return

    samples = samples["x"].to("cpu")

    # 5. Lưu kết quả
    os.makedirs(args.output_dir, exist_ok=True)
    base_filename = f"avatar_seed{seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    temp_video_path = os.path.join(args.output_dir, f"{base_filename}_temp.mp4")
    final_video_path = os.path.join(args.output_dir, f"{base_filename}.mp4")

    print(f"\nSaving temporary video to {temp_video_path}...")
    cache_video(tensor=samples[None], save_file=temp_video_path, fps=fps, nrow=1, normalize=True, value_range=(-1, 1))

    print(f"Combining video and audio into {final_video_path} using ffmpeg...")
    ffmpeg_command = [
        "ffmpeg", "-y",
        "-i", temp_video_path,
        "-i", args.audio,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        "-loglevel", "warning",
        "-nostats",
        final_video_path,
    ]
    try:
        subprocess.run(ffmpeg_command, check=True)
        os.remove(temp_video_path)
        end_time = time.time()
        print("\n--- Generation Complete! ---")
        print(f"Video saved to: {final_video_path}")
        print(f"Total time: {end_time - start_time:.2f} seconds.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nERROR: ffmpeg command failed. Please ensure ffmpeg is installed and in your system's PATH.")
        print(f"Your video (without audio) is available at: {temp_video_path}")


def main():
    parser = argparse.ArgumentParser(description="Command-line interface for Hunyuan Video Avatar generation.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--audio", type=str, required=True, help="Path to the input audio file.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the video.")

    parser.add_argument("--output_dir", type=str, default="outputs_cli", help="Directory to save the output video.")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--seed", type=int, default=-1, help="Seed for generation. -1 for random.")

    parser.add_argument("--quantization", type=str, default="int8", choices=["int8", "bf16"],
                        help="Model quantization type.")
    parser.add_argument("--dtype", type=str, default="", choices=["", "fp16", "bf16"],
                        help="Transformer data type. Default is auto-detect.")

    args = parser.parse_args()

    # Tải mô hình
    wan_model = initialize_model(args)

    # Chạy suy luận
    run_inference(wan_model, args)

    # Dọn dẹp
    if offload.last_offload_obj is not None:
        offload.last_offload_obj.release()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()