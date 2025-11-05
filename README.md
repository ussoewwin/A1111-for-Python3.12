# Stable Diffusion web UI

A web interface for Stable Diffusion, implemented using Gradio library.

## Python Version Support

**This repository supports Python 3.12 only.**

Other Python versions are not supported. Please ensure you are using Python 3.12 before proceeding with installation.

**Note:** Not all extensions may be compatible with Python 3.12. Some extensions may require additional modifications or may not work correctly.

## Default Package Versions

The following packages are automatically installed during initial setup:

- **PyTorch**: 2.8.0+cu129 (CUDA 12.9)
- **xformers**: 0.0.32.post2
- **Flash-Attention-2**: 2.8.2+cu129torch2.8.0

## Installation

**Note: The following installation instructions are for Windows only. Linux and macOS are not officially supported and require advanced manual modifications to work properly.**

**Why Windows only?** This repository is designed specifically for Windows. Many packages use Windows-specific binary wheel files (`.whl`) that are hardcoded in the installation process, including:
- NumPy: Local `whl/numpy-*.whl` files (Windows `win_amd64` builds)
- SciPy: Windows-specific wheel from HuggingFace
- Flash-Attention-2: Windows-specific CUDA build
- Other dependencies: Windows-specific requirements file (`requirements_versions_py312_windows.txt`)

To use on Linux or macOS, you would need to:
- Build or source Linux/macOS-compatible wheel files for all packages
- Modify the installation scripts to use platform-appropriate paths and binaries
- Create or adapt requirements files for your target platform

### Prerequisites

1. **Python 3.12**: Ensure you have Python 3.12 installed on your system.

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   ```bash
   venv\Scripts\activate
   ```

4. **Upgrade pip**:
   ```bash
   python -m pip install --upgrade pip
   ```

5. **Install PyTorch 2.8.0+cu129**:
   ```bash
   pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129 --force-reinstall --no-deps
   ```

6. **Install xformers 0.0.32.post2**:
   ```bash
   pip install xformers==0.0.32.post2
   ```

7. **Install numpy 1.26.4**:
   ```bash
   pip install numpy==1.26.4
   ```

8. **Launch the web UI**:
   ```bash
   webui-user.bat
   ```

## Documentation

For detailed features, installation instructions, and usage documentation, please refer to the official repository:

https://github.com/AUTOMATIC1111/stable-diffusion-webui
