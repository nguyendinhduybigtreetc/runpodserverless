################################
#          handler.py          #
################################
"""
Handler RunPod Serverless:
- Nhận input JSON chứa image_url, audio_url, text
- Tải file về /tmp
- Gọi run_avatar_cli.py -> trả stdout (hoặc URL video nếu script tự upload)
"""

import os
import sys
import json
import subprocess
import requests
from pathlib import Path

TMP_DIR = "/tmp"
SCRIPT = Path("python /runpod-volume/Wan2GP/run_avatar_cli.py")  # đường dẫn cố định

subprocess.run(
    ["python", "/runpod-volume/Wan2GP/download_model_cli.py"],
    check=True
)

def _download(url: str, fname: str) -> str:
    """Tải url về /tmp rồi trả lại path."""
    dst = Path(TMP_DIR) / fname
    with requests.get(url, timeout=60, stream=True) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
    return str(dst)


def handler(event):
    try:
        inp = event.get("input", {})
        text = inp.get("text", "")

        # 1. Tải file từ URL (nếu có)
        img_path = _download(inp["image_url"], "input.png") if "image_url" in inp else None
        aud_path = _download(inp["audio_url"], "input.wav") if "audio_url" in inp else None

        # 2. Gọi CLI (điều chỉnh arg tên tham số tuỳ file run_avatar_cli.py của bạn)
        cmd = [sys.executable, str(SCRIPT), "--prompt", text]
        if img_path:
            cmd += [" --image ", img_path]
        if aud_path:
            cmd += [" --audio ", aud_path]
        print(cmd)
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )

        # 3. Thử parse stdout thành JSON, không được thì trả raw string
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError:
            result = completed.stdout.strip()

        return {"result": result}

    except subprocess.CalledProcessError as e:
        return {
            "error": "run_avatar_cli.py exited with non-zero code",
            "code":  e.returncode,
            "stderr": e.stderr.strip(),
        }
    except Exception as e:
        return {"error": str(e)}
