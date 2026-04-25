from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from importlib.metadata import version  # python >= 3.8
from pathlib import Path

from packaging.version import parse

import_name = {"py-cpuinfo": "cpuinfo", "protobuf": "google.protobuf"}


def is_installed(
    package: str,
    min_version: str | None = None,
    max_version: str | None = None,
):
    name = import_name.get(package, package)
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return False

    if spec is None:
        return False

    if not min_version and not max_version:
        return True

    if not min_version:
        min_version = "0.0.0"
    if not max_version:
        max_version = "99999999.99999999.99999999"

    try:
        pkg_version = version(package)
        return parse(min_version) <= parse(pkg_version) <= parse(max_version)
    except Exception:
        return False


def run_pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", *args], check=True)


def download_models():
    """Download required YOLO models for ADetailer"""
    try:
        from huggingface_hub import hf_hub_download
        import requests
    except ImportError:
        print("[-] ADetailer: Installing dependencies for model download...")
        run_pip("huggingface_hub", "requests")
        from huggingface_hub import hf_hub_download
        import requests
    
    # Get the extension directory
    ext_dir = Path(__file__).parent
    models_dir = ext_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # YOLOv8 models from Hugging Face
    yolov8_models = [
        ("Bingsu/adetailer", "face_yolov8s.pt"),
        ("Bingsu/adetailer", "hand_yolov8n.pt"),
        ("Bingsu/adetailer", "person_yolov8n-seg.pt"),
        ("Bingsu/adetailer", "person_yolov8s-seg.pt"),
        ("Bingsu/yolo-world-mirror", "yolov8x-worldv2.pt"),
    ]
    
    print("[-] ADetailer: Checking for YOLOv8 models...")
    for repo_id, filename in yolov8_models:
        model_path = models_dir / filename
        if model_path.exists():
            continue
        
        try:
            print(f"[-] ADetailer: Downloading {filename}...")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(models_dir),
                local_dir_use_symlinks=False
            )
            print(f"[-] ADetailer: Downloaded {filename}")
        except Exception as e:
            print(f"[-] ADetailer: Failed to download {filename}: {e}")
    
    print("[-] ADetailer: YOLOv8 model check complete")


def download_yolov11_models():
    """Download YOLOv11 models for enhanced face detection"""
    try:
        import requests
    except ImportError:
        print("[-] ADetailer: Installing requests for YOLOv11 download...")
        run_pip("requests")
        import requests
    
    ext_dir = Path(__file__).parent
    models_dir = ext_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # YOLOv11 face detection model from GitHub
    yolov11_models = [
        {
            "name": "face_yolo11n.pt",
            "url": "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11n-face.pt",
            "size": "~7MB"
        },
        {
            "name": "face_yolo11s.pt",
            "url": "https://huggingface.co/deepghs/yolo-face/resolve/main/yolov11s-face/model.pt",
            "size": "~25MB"
        }
    ]
    
    print("[-] ADetailer: Checking for YOLOv11 models...")
    
    for model_info in yolov11_models:
        model_path = models_dir / model_info["name"]
        if model_path.exists():
            print(f"[-] ADetailer: {model_info['name']} already exists")
            continue
        
        try:
            print(f"[-] ADetailer: Downloading {model_info['name']} ({model_info['size']}) from GitHub...")
            response = requests.get(model_info["url"], stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\r[-] ADetailer: Progress: {progress:.1f}%", end="")
            
            print(f"\n[-] ADetailer: Downloaded {model_info['name']}")
        except Exception as e:
            print(f"\n[-] ADetailer: Failed to download {model_info['name']}: {e}")
            print(f"[-] ADetailer: You can manually download from: {model_info['url']}")
    
    print("[-] ADetailer: YOLOv11 model check complete")


def download_insightface():
    """Download InsightFace wheel for Python 3.13 compatibility"""
    
    # Check if InsightFace is already installed
    if is_installed("insightface"):
        print("[-] ADetailer: InsightFace already installed")
        return

    # On non-Windows platforms (Linux/macOS), rely on standard PyPI installation
    if sys.platform != "win32":
        print(f"[-] ADetailer: Installing InsightFace via standard pip for {sys.platform}...")
        run_pip("insightface")
        return

    try:
        import requests
    except ImportError:
        print("[-] ADetailer: Installing requests for InsightFace download...")
        run_pip("requests")
        import requests
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"[-] ADetailer: Python version: {python_version}")
    
    # Determine wheel file based on Python version
    if python_version == "3.13":
        wheel_file = "insightface-0.7.3-cp313-cp313-win_amd64.whl"
    elif python_version == "3.12":
        wheel_file = "insightface-0.7.3-cp312-cp312-win_amd64.whl"
    elif python_version == "3.11":
        wheel_file = "insightface-0.7.3-cp311-cp311-win_amd64.whl"
    else:
        print(f"[-] ADetailer: Python {python_version} not supported for InsightFace wheel")
        return
    
    print(f"[-] ADetailer: Downloading InsightFace wheel for Python {python_version}...")
    
    # Download wheel from Hugging Face
    wheel_url = f"https://huggingface.co/ussoewwin/Insightface_for_windows/resolve/main/{wheel_file}"
    
    try:
        response = requests.get(wheel_url, stream=True)
        response.raise_for_status()
        
        # Install wheel directly
        print(f"[-] ADetailer: Installing {wheel_file}...")
        run_pip(f"{wheel_url}")
        
        print("[-] ADetailer: InsightFace installation completed")
        
    except Exception as e:
        print(f"[-] ADetailer: Failed to download InsightFace: {e}")
        print("[-] ADetailer: You can manually install InsightFace from: https://huggingface.co/ussoewwin/Insightface_for_windows")


def install():
    deps = [
        # requirements
        ("ultralytics", "8.3.75", None),  # YOLOv11 support requires 8.3.0+
        ("rich", "13.0.0", None),
        ("huggingface_hub", None, None),
        ("requests", None, None),  # for YOLOv11 model download
        # InsightFace dependencies
        ("onnxruntime", "1.16.0", None),  # Required by InsightFace
        ("ml_dtypes", "0.4.0", None),  # Fix for InsightFace compatibility
        ("onnx", "1.15.0", None),  # ONNX for InsightFace
    ]

    pkgs = []
    for pkg, low, high in deps:
        if not is_installed(pkg, low, high):
            if low and high:
                cmd = f"{pkg}>={low},<={high}"
            elif low:
                cmd = f"{pkg}>={low}"
            elif high:
                cmd = f"{pkg}<={high}"
            else:
                cmd = pkg
            pkgs.append(cmd)

    if pkgs:
        run_pip(*pkgs)
    
    # Download YOLOv8 models after installing dependencies
    download_models()
    
    # Download YOLOv11 models for enhanced face detection
    download_yolov11_models()
    
    # Download InsightFace for Python 3.13 compatibility
    download_insightface()


try:
    import launch

    skip_install = launch.args.skip_install
except Exception:
    skip_install = False

if not skip_install:
    install()
