# Stable Diffusion web UI

A web interface for Stable Diffusion, implemented using the Gradio library.

## Python Version Support

**This repository supports Python 3.12 only.**

Other Python versions are not supported. Please ensure you are using Python 3.12 before proceeding with installation.

**Note:** Not all extensions may be compatible with Python 3.12. Some extensions may require additional modifications or may not work correctly.

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Windows  | Fully supported | Prebuilt wheels for FA2 / SciPy / NumPy / Insightface |
| Linux    | Supported | Flash-Attention 2 is built from source (requires CUDA toolkit + `nvcc`, ~30 min) |
| macOS    | Supported (limited) | Flash-Attention 2 is skipped (CUDA required; MPS backend cannot use FA2) |

All platform-specific handling is performed automatically by `modules/launch_utils.py` at startup. Windows install flow is byte-identical to the pre-1.03 behaviour; Linux / macOS branches are additive.

## Default Package Versions

The following packages are installed automatically during initial setup:

- **PyTorch**: 2.10.0+cu130 (CUDA 13.0)
- **Flash-Attention 2**:
  - Windows: `2.8.3+cu130torch2.10.0` (prebuilt wheel)
  - Linux: `flash-attn==2.8.3` (source build)
  - macOS: skipped
- **transformers**: 5.4.0+
- **protobuf**: 7.34.1
- **scipy**: 1.16.1
- **numpy**: 1.26.4

## Installation

### Common Prerequisites (all platforms)

1. **Python 3.12**: Ensure Python 3.12 is installed.

2. **Create and activate a virtual environment**:
   - Windows:
     ```cmd
     python -m venv venv
     venv\Scripts\activate
     ```
   - Linux / macOS:
     ```bash
     python3.12 -m venv venv
     source venv/bin/activate
     ```

3. **Upgrade pip**:
   ```bash
   python -m pip install --upgrade pip
   ```

4. **Install PyTorch 2.10.0+cu130**:
   ```bash
   pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu130
   ```

### Windows-specific extras

Triton / ONNX / Insightface are shipped as Windows prebuilt wheels:
```cmd
pip install triton-windows
python.exe -m pip install importlib_metadata onnx polygraphy
pip install coloredlogs flatbuffers packaging protobuf sympy
pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ onnxruntime-gpu
pip install https://huggingface.co/ussoewwin/Insightface_for_windows/resolve/main/insightface-0.7.3-cp312-cp312-win_amd64.whl
```

### Linux prerequisites

- CUDA toolkit 13.0 with `nvcc` on `PATH` (required to build Flash-Attention 2 from source).
- Standard build toolchain (`gcc`, `g++`, `make`, Python headers).
- First startup will spend ~30 minutes building FA2. To use an alternate prebuilt wheel, set:
  ```bash
  export FLASH_ATTN_PACKAGE=<url-or-wheel-path>
  ```

### macOS notes

- FA2 is skipped automatically (the MPS backend is not CUDA-compatible).
- Other Python 3.12 adjustments are shared with Linux.

### Launching

- Windows:
  ```cmd
  webui-user.bat
  ```
- Linux / macOS:
  ```bash
  ./webui.sh
  ```

## How Linux / macOS support works

`modules/launch_utils.py` branches by `platform.system()` at startup:

- **Flash-Attention 2**: Windows wheel / Linux `--no-build-isolation` source build / macOS skip.
- **SciPy**: Windows HuggingFace prebuilt wheel / Linux + macOS PyPI `scipy==1.16.1`.
- **NumPy**: local Windows `whl/numpy-*.whl` is used when present (Windows only); otherwise NumPy is installed from PyPI at the pinned version.
- **clip.py `pkg_resources` auto-fix**: targets `venv/Lib/...` on Windows and `venv/lib/pythonX.Y/...` on Linux / macOS (major / minor resolved dynamically).

See [`md/LINUX_MAC_PY312_STARTUP_FIX.md`](md/LINUX_MAC_PY312_STARTUP_FIX.md) for the full fix design and [`md/PYTHON312_COMPATIBILITY.md`](md/PYTHON312_COMPATIBILITY.md) for the overall Python 3.12 compatibility notes.

## Changelog

See [`md/CHANGELOG.md`](md/CHANGELOG.md) for the full change history.

## Documentation

For detailed features, installation instructions, and usage documentation, please refer to the official upstream repository:

https://github.com/AUTOMATIC1111/stable-diffusion-webui
