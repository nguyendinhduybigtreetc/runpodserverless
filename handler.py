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

# TMP_DIR = "/tmp"
SCRIPT = Path("python run_avatar_cli.py")  # đường dẫn cố định

subprocess.run(
    ["python", "download_model_cli.py"],
    check=True
)

def _download(url: str, filename: str) -> str:
    """
    Tải thẳng về filename (ngay thư mục hiện hành) rồi trả lại tên file.
    """
    with open(filename, "wb") as f:
        f.write(requests.get(url, timeout=30).content)
    return filename          # chỉ cần relative path


def handler(event):
    try:
        inp = event.get("input", {})
        prompt = inp.get("text", "")

        # 1. Tải file từ URL (nếu có)
        img_path = _download(inp["image_url"], "input.png") if "image_url" in inp else None
        aud_path = _download(inp["audio_url"], "input.wav") if "audio_url" in inp else None


        cmd = [
            "python",  # Python interpreter hiện tại
            "run_avatar_cli.py",  # File CLI cần chạy
            "--image", "input.png",
            "--audio", "input.wav",
            "--prompt", prompt,
        ]
        print(cmd)
        result = subprocess.run(
            cmd,
            capture_output=True,  # lấy cả stdout và stderr
            text=True,  # trả về str thay vì bytes
            check=True  # tự động raise CalledProcessError nếu lỗi
        )
        # return result.stdout

        # 3. Thử parse stdout thành JSON, không được thì trả raw string
        try:
            result = json.loads(result.stdout)
        except json.JSONDecodeError:
            result = result.stdout.strip()

        return {"result": result}

    except subprocess.CalledProcessError as e:
        return {
            "error": "run_avatar_cli.py exited with non-zero code",
            "code":  e.returncode,
            "stderr": e.stderr.strip(),
        }
    except Exception as e:
        return {"error": str(e)}
