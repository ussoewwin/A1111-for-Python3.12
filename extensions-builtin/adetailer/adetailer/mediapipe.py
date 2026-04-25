"""
MediaPipe replacement using InsightFace for Python 3.13 compatibility.
Provides the same interface as original MediaPipe implementation.
"""

from __future__ import annotations

from functools import partial

import cv2
import numpy as np
from PIL import Image, ImageDraw

from adetailer import PredictOutput
from adetailer.common import create_bbox_from_mask, create_mask_from_bbox

# Import InsightFace detector
try:
    from .insightface_detector import get_insightface_detector, InsightFaceDetector
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("[-] ADetailer: InsightFace not available")


def mediapipe_predict(
    model_type: str, image: Image.Image, confidence: float = 0.3
) -> PredictOutput:
    """
    MediaPipe-compatible interface using InsightFace.
    
    Parameters
    ----------
    model_type : str
        Model type (mapped to InsightFace models)
    image : Image.Image
        Input image
    confidence : float
        Detection confidence threshold
        
    Returns
    -------
    PredictOutput
        Detection results
    """
    if not INSIGHTFACE_AVAILABLE:
        print("[-] ADetailer: InsightFace not available, returning empty results")
        return PredictOutput()
    
    # Map MediaPipe model types to InsightFace models
    mapping = {
        "mediapipe_face_short": partial(insightface_face_detection, "buffalo_s"),  # Fast
        "mediapipe_face_full": partial(insightface_face_detection, "buffalo_l"),  # Accurate
        "mediapipe_face_mesh": partial(insightface_face_detection, "buffalo_l"),  # Accurate
        "mediapipe_face_mesh_eyes_only": partial(insightface_face_detection, "buffalo_l"),  # Accurate
    }
    
    if model_type in mapping:
        func = mapping[model_type]
        try:
            return func(image, confidence)
        except Exception as e:
            print(f"[-] ADetailer: InsightFace detection failed: {e}")
            return PredictOutput()
    
    msg = f"[-] ADetailer: Invalid model type: {model_type}, Available: {list(mapping.keys())!r}"
    raise RuntimeError(msg)


def insightface_face_detection(
    model_name: str, image: Image.Image, confidence: float = 0.3
) -> PredictOutput[float]:
    """
    Face detection using InsightFace (MediaPipe replacement).
    
    Parameters
    ----------
    model_name : str
        InsightFace model name (buffalo_l, buffalo_m, buffalo_s)
    image : Image.Image
        Input image
    confidence : float
        Detection confidence threshold
        
    Returns
    -------
    PredictOutput[float]
        Detection results with bounding boxes and confidence scores
    """
    if not INSIGHTFACE_AVAILABLE:
        return PredictOutput()
    
    try:
        # Get InsightFace detector
        detector = get_insightface_detector()
        if detector is None:
            # Create new detector with specific model
            detector = InsightFaceDetector(model_name)
        
        # Detect faces
        faces = detector.detect_faces(image, confidence)
        
        if not faces:
            return PredictOutput()
        
        # Convert to MediaPipe-compatible format
        bboxes = []
        confidences = []
        masks = []
        
        for face in faces:
            bbox = face['bbox']
            conf = face['confidence']
            
            # Convert bbox format [x1, y1, x2, y2] to [x1, y1, x2, y2]
            bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            confidences.append(conf)
            
            # Create mask from bbox
            mask = Image.new('L', image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle(bbox, fill=255)
            masks.append(mask)
        
        # Create preview image
        preview_array = np.array(image)
        preview = Image.fromarray(preview_array)
        
        return PredictOutput(
            bboxes=bboxes,
            masks=masks,
            confidences=confidences,
            preview=preview
        )
        
    except Exception as e:
        print(f"[-] ADetailer: InsightFace face detection failed: {e}")
        return PredictOutput()


def insightface_face_mesh(
    image: Image.Image, confidence: float = 0.3
) -> PredictOutput[int]:
    """
    Face mesh detection using InsightFace (MediaPipe replacement).
    For now, uses regular face detection as InsightFace doesn't have mesh.
    """
    return insightface_face_detection("buffalo_l", image, confidence)


def insightface_face_mesh_eyes_only(
    image: Image.Image, confidence: float = 0.3
) -> PredictOutput[int]:
    """
    Eye-only detection using InsightFace (MediaPipe replacement).
    For now, uses regular face detection as InsightFace doesn't have eye-specific detection.
    """
    return insightface_face_detection("buffalo_l", image, confidence)


def draw_preview(
    preview: Image.Image, bboxes: list[list[int]], masks: list[Image.Image]
) -> Image.Image:
    """Draw preview with bounding boxes and masks."""
    red = Image.new("RGB", preview.size, "red")
    for mask in masks:
        masked = Image.composite(red, preview, mask)
        preview = Image.blend(preview, masked, 0.25)

    draw = ImageDraw.Draw(preview)
    for bbox in bboxes:
        draw.rectangle(bbox, outline="red", width=2)

    return preview
