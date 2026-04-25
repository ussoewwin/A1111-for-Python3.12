from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from adetailer import PredictOutput
from adetailer.common import create_mask_from_bbox, insightface_predict, INSIGHTFACE_AVAILABLE

if TYPE_CHECKING:
    import torch
    from ultralytics import YOLO, YOLOWorld


def ultralytics_predict(
    model_path: str | Path,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
    classes: str = "",
) -> PredictOutput[float]:
    from ultralytics import YOLO

    model = YOLO(model_path)
    apply_classes(model, model_path, classes)
    pred = model(image, conf=confidence, device=device)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if bboxes.size == 0:
        return PredictOutput()
    bboxes = bboxes.tolist()

    if pred[0].masks is None:
        masks = create_mask_from_bbox(bboxes, image.size)
    else:
        masks = mask_to_pil(pred[0].masks.data, image.size)

    confidences = pred[0].boxes.conf.cpu().numpy().tolist()

    preview = pred[0].plot()
    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    preview = Image.fromarray(preview)

    return PredictOutput(
        bboxes=bboxes, masks=masks, confidences=confidences, preview=preview
    )


def apply_classes(model: YOLO | YOLOWorld, model_path: str | Path, classes: str):
    if not classes or "-world" not in Path(model_path).stem:
        return
    parsed = [c.strip() for c in classes.split(",") if c.strip()]
    if parsed:
        model.set_classes(parsed)


def mask_to_pil(masks: torch.Tensor, shape: tuple[int, int]) -> list[Image.Image]:
    """
    Parameters
    ----------
    masks: torch.Tensor, dtype=torch.float32, shape=(N, H, W).
        The device can be CUDA, but `to_pil_image` takes care of that.

    shape: tuple[int, int]
        (W, H) of the original image
    """
    n = masks.shape[0]
    return [to_pil_image(masks[i], mode="L").resize(shape) for i in range(n)]


def hybrid_face_predict(
    model_path: str | Path,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
    classes: str = "",
    use_insightface: bool = True,
    insightface_confidence: float = 0.5,
) -> PredictOutput[float]:
    """
    Hybrid face detection using YOLO + InsightFace for enhanced accuracy.
    YOLO detects faces first, then InsightFace finds missed faces.
    This replaces the original MediaPipe + YOLO combination.
    
    Parameters
    ----------
    model_path : str | Path
        YOLO model path
    image : Image.Image
        Input image
    confidence : float
        YOLO confidence threshold
    device : str
        Device for YOLO
    classes : str
        YOLO classes filter
    use_insightface : bool
        Whether to use InsightFace for missed faces
    insightface_confidence : float
        InsightFace confidence threshold
        
    Returns
    -------
    PredictOutput[float]
        Combined detection results
    """
    # First, run YOLO detection (main detector)
    yolo_result = ultralytics_predict(
        model_path=model_path,
        image=image,
        confidence=confidence,
        device=device,
        classes=classes,
    )
    
    # Log YOLO results
    yolo_count = len(yolo_result.bboxes)
    print(f"[-] ADetailer: YOLO detection - {yolo_count} faces found (confidence: {confidence:.2f})")
    
    if not use_insightface or not INSIGHTFACE_AVAILABLE:
        if not use_insightface:
            print(f"[-] ADetailer: InsightFace disabled, using YOLO only")
        else:
            print(f"[-] ADetailer: InsightFace not available, using YOLO only")
        return yolo_result
    
    # Use InsightFace to find additional faces (complementary detection)
    try:
        # Use InsightFace to find missed faces
        print(f"[-] ADetailer: Running InsightFace detection (confidence: {insightface_confidence:.2f})")
        insightface_result = insightface_predict(
            image=image,
            model_name="insightface_buffalo_l",  # Use high-accuracy model
            confidence=insightface_confidence,
        )
        print(f"[-] ADetailer: InsightFace detection - {len(insightface_result.bboxes)} faces found")
        
        # Combine results (YOLO + InsightFace)
        combined_bboxes = yolo_result.bboxes.copy()
        combined_masks = yolo_result.masks.copy()
        combined_confidences = yolo_result.confidences.copy() if hasattr(yolo_result, 'confidences') else []
        
        # Add InsightFace results that don't overlap significantly with YOLO results
        for i, insight_bbox in enumerate(insightface_result.bboxes):
            is_duplicate = False
            for yolo_bbox in yolo_result.bboxes:
                if _bbox_overlap(insight_bbox, yolo_bbox) > 0.3:  # 30% overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                combined_bboxes.append(insight_bbox)
                if i < len(insightface_result.masks):
                    combined_masks.append(insightface_result.masks[i])
                if i < len(insightface_result.confidences):
                    combined_confidences.append(insightface_result.confidences[i])
        
        # Log detailed detection results
        yolo_count = len(yolo_result.bboxes)
        insightface_count = len(insightface_result.bboxes)
        combined_count = len(combined_bboxes)
        added_count = combined_count - yolo_count
        
        print(f"[-] ADetailer: Hybrid detection - YOLO: {yolo_count}, InsightFace: {insightface_count}, Combined: {combined_count}")
        if added_count > 0:
            print(f"[-] ADetailer: InsightFace added {added_count} missed faces")
        elif insightface_count > 0:
            print(f"[-] ADetailer: InsightFace found {insightface_count} faces but all overlapped with YOLO")
        else:
            print(f"[-] ADetailer: No faces detected by either detector")
        
        return PredictOutput(
            bboxes=combined_bboxes,
            masks=combined_masks,
            confidences=combined_confidences,
            preview=yolo_result.preview,
        )
        
    except Exception as e:
        print(f"[-] ADetailer: InsightFace hybrid detection failed: {e}")
        return yolo_result


def _bbox_overlap(bbox1: list[float], bbox2: list[float]) -> float:
    """
    Calculate overlap ratio between two bounding boxes.
    
    Parameters
    ----------
    bbox1, bbox2 : list[float]
        Bounding boxes in format [x1, y1, x2, y2]
        
    Returns
    -------
    float
        Overlap ratio (0.0-1.0)
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
    x1_2, y1_2, x2_2, y2_2 = bbox2[:4]
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0
