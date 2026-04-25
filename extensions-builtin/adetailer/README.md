# ADetailer (Python 3.13 Compatible)

ADetailer is an extension for the stable diffusion webui that does automatic masking and inpainting. It is similar to the Detection Detailer.

## 🎉 Features of This Modified Version

This version is a modified ADetailer designed to **work with Python 3.13**:

- **Python 3.13 Compatible**: Fully compatible with Python 3.13+
- **YOLOv8 & YOLOv11 Models**: Uses proven YOLO models with enhanced YOLOv11 face detection
- **InsightFace Integration**: High-precision face detection using InsightFace for enhanced accuracy
- **Hybrid Detection**: YOLO + InsightFace combination for maximum face detection coverage
- **Automatic Model Download**: All required models are automatically downloaded on first run
- **Enhanced Face Detection**: YOLOv11 + InsightFace provides superior face detection accuracy
- **MediaPipe Alternative**: InsightFace replaces MediaPipe for Python 3.13 compatibility

### Modifications

- **InsightFace Integration**: Added InsightFace for high-precision face detection
- **Hybrid Detection System**: YOLO detects faces first, InsightFace finds missed faces
- **Enhanced Accuracy**: Combines YOLO speed with InsightFace precision
- `install.py`: Automatic model download from Hugging Face and GitHub
- `common.py`: Added InsightFace model support and hybrid detection
- `pyproject.toml`: Updated dependencies for Python 3.13 compatibility
- Added YOLOv11 face detection model for enhanced accuracy

Original project: [Bing-su/adetailer](https://github.com/Bing-su/adetailer)

## Install

### Quick Installation

**Note**: While the repository name is `ADetailer_without_mediapipe`, the extension works under the same name as the original ADetailer. If you already have the original ADetailer installed, please uninstall it before installing this version.

1. Open "Extensions" tab in WebUI
2. Open "Install from URL" tab
3. Enter `https://github.com/ussoewwin/ADetailer_without_mediapipe.git` to "URL for extension's git repository"
4. Press "Install" button
5. Go to "Installed" tab, click "Check for updates", and then click "Apply and restart UI"
6. **All required models will be downloaded automatically on first run** (approximately 100-200MB)
7. Restart WebUI completely

That's it! No additional dependencies or manual model downloads required.

## Options

### Model & Prompts

| Option | Description | Default |
|--------|-------------|---------|
| **ADetailer model** | Determine what to detect | `None` = disable |
| **ADetailer model classes** | Comma separated class names to detect (YOLO World models only) | If blank, use default values<br/>default = [COCO 80 classes](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) |
| **ADetailer prompt, negative prompt** | Prompts and negative prompts to apply | If left blank, uses the same as input |
| **Skip img2img** | Skip img2img (changes step count to 1) | img2img only |

### Detection

| Option | Description | Default |
|--------|-------------|---------|
| **Detection model confidence threshold** | Only objects with confidence above this threshold are used | - |
| **Mask min/max ratio** | Only use masks whose area is between these ratios | - |
| **Mask only the top k largest** | Only use the k objects with largest bbox area | 0 to disable |

If you want to exclude objects in the background, try setting the min ratio to around `0.01`.

### Mask Preprocessing

| Option | Description | Default |
|--------|-------------|---------|
| **Mask x, y offset** | Moves the mask horizontally and vertically | - |
| **Mask erosion (-) / dilation (+)** | Enlarge or reduce the detected mask | [opencv example](https://docs.opencv.org/4.7.0/db/df6/tutorial_erosion_dilatation.html) |
| **Mask merge mode** | `None`: Inpaint each mask<br/>`Merge`: Merge all masks and inpaint<br/>`Merge and Invert`: Merge all masks and Invert, then inpaint | - |

Applied in this order: x, y offset → erosion/dilation → merge/invert.

#### Inpainting

Each option corresponds to a corresponding option on the inpaint tab. Therefore, please refer to the inpaint tab for usage details on how to use each option.

## ControlNet Inpainting

You can use the ControlNet extension if you have ControlNet installed and ControlNet models.

Support `inpaint, scribble, lineart, openpose, tile, depth` controlnet models. Once you choose a model, the preprocessor is set automatically. It works separately from the model set by the Controlnet extension.

If you select `Passthrough`, the controlnet settings you set outside of ADetailer will be used.

## Advanced Options

API request example: [wiki/REST-API](https://github.com/Bing-su/adetailer/wiki/REST-API)

`[SEP], [SKIP], [PROMPT]` tokens: [wiki/Advanced](https://github.com/Bing-su/adetailer/wiki/Advanced)

## Models

### YOLOv8 Models (Stable)

| Model | Target | mAP 50 | mAP 50-95 | Size |
|-------|--------|--------|-----------|------|
| **face_yolov8s.pt** | 2D / realistic face | 0.713 | 0.404 | 11.2MB |
| **hand_yolov8n.pt** | 2D / realistic hand | 0.767 | 0.505 | 6.4MB |
| **person_yolov8n-seg.pt** | 2D / realistic person | 0.782 (bbox)<br/>0.761 (mask) | 0.555 (bbox)<br/>0.460 (mask) | 6.7MB |
| **person_yolov8s-seg.pt** | 2D / realistic person | 0.824 (bbox)<br/>0.809 (mask) | 0.605 (bbox)<br/>0.508 (mask) | 11.8MB |

### YOLOv11 Models (Enhanced) ✨

| Model | Target | Expected Improvement | Size |
|-------|--------|---------------------|------|
| **face_yolo11n.pt** | 2D / realistic face | ~9% better accuracy than YOLOv8n | ~7MB |
| **face_yolo11s.pt** | 2D / realistic face | Higher accuracy than YOLOv11n | ~25MB |

### InsightFace Models (High Precision) 🎯

| Model | Target | Accuracy | Speed | Size |
|-------|--------|----------|-------|------|
| **insightface_buffalo_l** | 2D / realistic face | Highest | Medium | ~100MB |
| **insightface_buffalo_m** | 2D / realistic face | High | Fast | ~50MB |
| **insightface_buffalo_s** | 2D / realistic face | Good | Fastest | ~25MB |

**All models are automatically downloaded on first run.** InsightFace models provide superior face detection accuracy compared to YOLO models, especially for small or angled faces.

**Note**: InsightFace requires manual installation for Python 3.13 compatibility. The install script will automatically download the appropriate wheel file from [ussoewwin/Insightface_for_windows](https://huggingface.co/ussoewwin/Insightface_for_windows).

**Model Sources:**
- YOLOv8 models: [Bingsu/adetailer on Hugging Face](https://huggingface.co/Bingsu/adetailer)
- YOLOv11n face model: [akanametov/yolo-face on GitHub](https://github.com/akanametov/yolo-face)
- YOLOv11s face model: [deepghs/yolo-face on Hugging Face](https://huggingface.co/deepghs/yolo-face)

**Documentation:**
- YOLOv8: https://docs.ultralytics.com/models/yolov8/
- YOLOv11: https://docs.ultralytics.com/models/yolo11/
- YOLO World: https://docs.ultralytics.com/models/yolo-world/

### Additional Model

Put your [ultralytics](https://github.com/ultralytics/ultralytics) yolo model in `models/adetailer`. The model name should end with `.pt`.

It must be a bbox detection or segment model and use all label.

## How it works

ADetailer works in three simple steps.

1. Create an image.
2. Detect object with a detection model and create a mask image.
3. Inpaint using the image from 1 and the mask from 2.

## Development

ADetailer is developed and tested using the stable-diffusion 1.5 model, for the latest version of [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) repository only.

## License

ADetailer is a derivative work that uses two AGPL-licensed works (stable-diffusion-webui, ultralytics) and is therefore distributed under the AGPL license.

## See Also

- https://github.com/ototadana/sd-face-editor
- https://github.com/continue-revolution/sd-webui-segment-anything
- https://github.com/portu-sim/sd-webui-bmab

---

**Note**: All commits are GPG signed for verification.

## Latest Update (v1.03)

### InsightFace Hybrid Detection System

This release introduces a powerful hybrid detection system that significantly improves face detection accuracy, especially for SDXL and Pony models.

#### Key Features

- **Hybrid Detection Architecture**: Combines YOLO (fast, general detection) with InsightFace (high-precision face detection)
- **Complementary Detection**: YOLO detects faces first, then InsightFace finds missed faces
- **Intelligent Deduplication**: Automatically removes duplicate detections using 30% overlap threshold
- **SDXL/Pony Optimized**: Lower confidence thresholds specifically tuned for anime/stylized faces
- **Python 3.13 Full Support**: Custom InsightFace wheels for Python 3.11, 3.12, and 3.13

#### How It Works

1. **Primary Detection (YOLO)**: Fast initial face detection with YOLOv11
   - Confidence: 25.5% (85% of user setting for anime styles)
   - Optimized for speed and general face detection

2. **Secondary Detection (InsightFace)**: High-precision complementary detection
   - Confidence: 15-20% (lower threshold to catch difficult faces)
   - Only adds faces that YOLO missed (no duplicates)
   - Particularly effective for Pony and stylized anime faces

3. **Result Merging**: Combines both detection results
   - Removes overlapping detections (>30% overlap)
   - Logs detection counts: "YOLO: X, InsightFace: Y, Combined: Z"

#### Why This Matters

- **Better SDXL Detection**: SDXL and Pony models produce unique face styles that single detectors often miss
- **No False Negatives**: Two-stage detection catches faces that YOLO alone would miss
- **Minimal False Positives**: Overlap checking prevents duplicate processing
- **Automatic Fallback**: If InsightFace is unavailable, gracefully falls back to YOLO-only detection

#### Installation Notes

- **Automatic Setup**: `install.py` automatically downloads appropriate InsightFace wheels
- **Python 3.13**: Uses custom wheel from `huggingface.co/ussoewwin/Insightface_for_windows`
- **Python 3.11/3.12**: Also uses custom wheels for better compatibility
- **Dependencies**: `onnxruntime>=1.16.0`, `ml_dtypes>=0.4.0`, `onnx>=1.15.0` auto-installed

#### Technical Details

- **Detection Models**:
  - YOLO: `face_yolo11n.pt` (default), `face_yolo11s.pt`, `face_yolov8s.pt`
  - InsightFace: `buffalo_l` (high accuracy), `buffalo_m` (balanced), `buffalo_s` (fast)

- **Confidence Tuning**:
  - YOLO: `max(0.25, user_confidence * 0.85)` - Slightly lower for anime styles
  - InsightFace: `max(0.2, user_confidence - 0.15)` - Much lower to catch edge cases

- **Performance Impact**: InsightFace adds ~200-500ms per image (negligible compared to inpainting time)

#### Previous Updates

- **YOLOv11 Integration**: Replaced YOLOv8n with YOLOv11n/s models
- **Default Model**: `face_yolo11n.pt` set as default (smaller, faster)
- **Model Options**: Both 11n and 11s available for speed/accuracy tradeoff
