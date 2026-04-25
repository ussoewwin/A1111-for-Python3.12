"""InsightFace detector for complementary face detection."""
from __future__ import annotations

import logging
import os
from contextlib import redirect_stdout
from io import StringIO
import numpy as np
from PIL import Image

# Suppress ONNX Runtime and InsightFace verbose logging
os.environ['ORT_LOGGING_LEVEL'] = '3'  # ERROR level
logging.getLogger('onnxruntime').setLevel(logging.ERROR)
logging.getLogger('onnxruntime.providers').setLevel(logging.ERROR)

try:
    import insightface
    # Suppress InsightFace verbose output
    logging.getLogger('insightface').setLevel(logging.ERROR)
    
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except (ImportError, AttributeError):
    INSIGHTFACE_AVAILABLE = False


class InsightFaceDetector:
    """InsightFace detector wrapper for face detection."""
    
    def __init__(self, model_name: str = "buffalo_l", det_size: tuple[int, int] = (1024, 1024)):
        """
        Initialize InsightFace detector.
        
        Parameters
        ----------
        model_name : str
            Model name (buffalo_l, buffalo_m, buffalo_s)
        det_size : tuple[int, int]
            Detection size (width, height) - higher = more accurate but slower
        """
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace is not available")
        
        # Suppress verbose logging during initialization
        with redirect_stdout(StringIO()):
            self.app = FaceAnalysis(name=model_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.det_size = det_size
            self.app.prepare(ctx_id=0, det_size=det_size)
    
    def detect_faces(self, image: Image.Image, confidence: float = 0.3) -> list[dict]:
        """
        Detect faces in the image.
        
        Parameters
        ----------
        image : Image.Image
            Input image
        confidence : float
            Confidence threshold (0.0-1.0)
        
        Returns
        -------
        list[dict]
            List of detected faces with 'bbox' and 'confidence' keys
        """
        # Convert PIL image to numpy array (RGB)
        img_array = np.array(image)
        
        # InsightFace expects BGR format
        if img_array.shape[2] == 3:
            img_array = img_array[:, :, ::-1]
        
        # Detect faces
        faces = self.app.get(img_array)
        
        # Filter by confidence and convert to standard format
        results = []
        for face in faces:
            det_score = face.det_score
            if det_score >= confidence:
                bbox = face.bbox.astype(int).tolist()  # [x1, y1, x2, y2]
                results.append({
                    'bbox': bbox,
                    'confidence': float(det_score)
                })
        
        return results


def get_insightface_detector(model_name: str = "buffalo_l", det_size: tuple[int, int] = (1024, 1024)) -> InsightFaceDetector | None:
    """
    Get or create InsightFace detector instance.
    
    Parameters
    ----------
    model_name : str
        Model name (buffalo_l, buffalo_m, buffalo_s)
    det_size : tuple[int, int]
        Detection size (width, height) - higher = more accurate but slower
    
    Returns
    -------
    InsightFaceDetector | None
        Detector instance or None if not available
    """
    if not INSIGHTFACE_AVAILABLE:
        return None
    
    try:
        return InsightFaceDetector(model_name, det_size)
    except Exception as e:
        print(f"[-] InsightFace: Failed to initialize detector: {e}")
        return None

