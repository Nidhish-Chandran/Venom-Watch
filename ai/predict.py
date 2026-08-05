"""
============================================
Snake Detection — Core Prediction Logic
============================================
This is the MAIN ENTRY POINT of the AI module.

The function detect_snake() is the ONLY function
exposed to external modules.

CONTRACT:
    detect_snake(image_path) -> dict

    Returns:
        {
            "snake": bool,
            "confidence": float,        # 0 to 100
            "bbox": [x1, y1, x2, y2],   # or None
            "message": str,
            "annotated_image_path": str  # or None
        }

FUTURE MODULES MUST CALL THIS FUNCTION.
They must NOT access model_loader or utils directly.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np

from .model_loader import get_model_loader
from .utils import (
    validate_image,
    load_image,
    draw_bounding_box,
    save_annotated_image,
    format_detection_result,
    mock_detection,
)

logger = logging.getLogger('ai.predict')

# ============================================
# CONFIGURATION (read from environment or defaults)
# ============================================
_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get('CONFIDENCE_THRESHOLD', '0.5')
)
_DEVICE: str = os.environ.get('DEVICE', 'cpu')
_MODEL_PATH: str = os.environ.get('MODEL_PATH', 'weights/snake_yolov11.pt')


# ============================================
# MODEL INITIALIZATION (runs on import)
# ============================================
def _initialize_model() -> None:
    """
    Initialize the YOLO model on module import.
    This ensures the model is loaded ONCE when the app starts.
    """
    try:
        loader = get_model_loader()
        loader.load_model(model_path=_MODEL_PATH, device=_DEVICE)
        logger.info("AI model initialization complete.")
    except Exception as e:
        logger.error(f"Model initialization failed: {e}", exc_info=True)
        logger.warning("Continuing in MOCK mode.")


# Initialize on import (module-level)
_initialize_model()


# ============================================
# MAIN DETECTION FUNCTION
# ============================================
def detect_snake(image_path: str) -> Dict[str, Any]:
    """
    Detect whether a snake is present in the given image.

    This is the PUBLIC API of the AI module.
    Future modules (Species ID, Venomous Classification, etc.)
    must call ONLY this function.

    Args:
        image_path (str): Absolute or relative path to the image file.
                          Supported formats: .jpg, .jpeg, .png

    Returns:
        dict: Standardized detection result:
            {
                "snake": bool,
                "confidence": float,           # 0 to 100
                "bbox": [x1, y1, x2, y2],      # pixel coordinates, or None
                "message": str,                # "Snake Detected" or "No Snake Detected"
                "annotated_image_path": str    # path to annotated image, or None
            }

    Example:
        >>> from ai import detect_snake
        >>> result = detect_snake("photo.jpg")
        >>> result["snake"]
        True
        >>> result["confidence"]
        95.72
        >>> result["bbox"]
        [120, 80, 450, 390]
    """
    logger.info(f"Detection requested for: {image_path}")

    # ------------------------------------------
    # Step 1: Validate input image
    # ------------------------------------------
    is_valid, validation_msg = validate_image(image_path)
    if not is_valid:
        logger.warning(f"Image validation failed: {validation_msg}")
        return format_detection_result(
            snake_detected=False,
            confidence=0.0,
            bbox=None,
            annotated_image_path=None,
        )

    # ------------------------------------------
    # Step 2: Get the model (or use mock)
    # ------------------------------------------
    loader = get_model_loader()
    model = loader.get_model()

    if model is None:
        logger.warning("No model loaded. Using mock detection.")
        return mock_detection(image_path)

    # ------------------------------------------
    # Step 3: Load image
    # ------------------------------------------
    image = load_image(image_path)
    if image is None:
        logger.error("Failed to load image for inference.")
        return format_detection_result(
            snake_detected=False,
            confidence=0.0,
            bbox=None,
            annotated_image_path=None,
        )

    # ------------------------------------------
    # Step 4: Run YOLO inference
    # ------------------------------------------
    try:
        logger.debug(f"Running inference with threshold={_CONFIDENCE_THRESHOLD}")
        results = model(
            image,
            conf=_CONFIDENCE_THRESHOLD,
            device=_DEVICE,
            verbose=False,
        )

        if not results or len(results) == 0:
            logger.info("No results returned from model.")
            return format_detection_result(
                snake_detected=False,
                confidence=99.0,
                bbox=None,
                annotated_image_path=None,
            )

        result = results[0]  # First image's results

        # ------------------------------------------
        # Step 5: Extract detections
        # ------------------------------------------
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            logger.info("No detections above threshold.")
            return format_detection_result(
                snake_detected=False,
                confidence=99.0,
                bbox=None,
                annotated_image_path=None,
            )

        # Get all detection confidences
        confidences = boxes.conf.cpu().numpy()  # Shape: (N,)
        class_ids = boxes.cls.cpu().numpy()     # Shape: (N,)
        xyxy = boxes.xyxy.cpu().numpy()         # Shape: (N, 4)

        # Find the highest-confidence detection
        best_idx = int(np.argmax(confidences))
        best_conf = float(confidences[best_idx])
        best_bbox = xyxy[best_idx].astype(int).tolist()  # [x1, y1, x2, y2]

        logger.info(
            f"Snake detected! confidence={best_conf:.4f}, bbox={best_bbox}"
        )

        # ------------------------------------------
        # Step 6: Annotate image with bounding box
        # ------------------------------------------
        annotated_image = draw_bounding_box(
            image=image,
            bbox=best_bbox,
            confidence=best_conf * 100,
            label="Snake",
        )

        annotated_path = save_annotated_image(annotated_image)

        # ------------------------------------------
        # Step 7: Return standardized result
        # ------------------------------------------
        return format_detection_result(
            snake_detected=True,
            confidence=best_conf * 100,  # Convert to percentage
            bbox=best_bbox,
            annotated_image_path=annotated_path,
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return format_detection_result(
            snake_detected=False,
            confidence=0.0,
            bbox=None,
            annotated_image_path=None,
        )


# ============================================
# OPTIONAL: Batch detection utility
# ============================================
def detect_snake_batch(image_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Detect snakes in multiple images.

    Args:
        image_paths: List of image file paths.

    Returns:
        List of detection result dictionaries (one per image).
    """
    results = []
    for path in image_paths:
        try:
            result = detect_snake(path)
            results.append(result)
        except Exception as e:
            logger.error(f"Batch detection failed for '{path}': {e}")
            results.append(
                format_detection_result(
                    snake_detected=False,
                    confidence=0.0,
                    bbox=None,
                    annotated_image_path=None,
                )
            )
    return results