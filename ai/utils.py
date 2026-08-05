"""
============================================
AI Utilities
============================================
Helper functions for image validation, annotation, and file operations.
No Django dependency — pure Python + OpenCV + Pillow.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger('ai.utils')


# ============================================
# IMAGE VALIDATION
# ============================================
def validate_image(image_path: str) -> Tuple[bool, str]:
    """
    Validate that a file is a readable, non-corrupted image.

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of (is_valid, message).
    """
    path = Path(image_path)

    # Check existence
    if not path.exists():
        return False, f"File does not exist: {image_path}"

    # Check file size (not empty)
    if path.stat().st_size == 0:
        return False, "File is empty (0 bytes)."

    # Check extension
    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    if path.suffix.lower() not in allowed_extensions:
        return False, (
            f"Unsupported file format: '{path.suffix}'. "
            f"Allowed formats: {', '.join(allowed_extensions)}"
        )

    # Try opening with Pillow (catches corruption)
    try:
        with Image.open(path) as img:
            img.verify()  # Verify without loading pixel data
    except UnidentifiedImageError:
        return False, "File is not a valid image (corrupted or wrong format)."
    except Exception as e:
        return False, f"Image validation failed: {str(e)}"

    # Double-check with OpenCV (catches edge cases Pillow misses)
    try:
        img_cv = cv2.imread(str(path))
        if img_cv is None:
            return False, "OpenCV could not read the image (possibly corrupted)."
    except Exception as e:
        return False, f"OpenCV read error: {str(e)}"

    logger.debug(f"Image validated successfully: {image_path}")
    return True, "Image is valid."


# ============================================
# IMAGE LOADING
# ============================================
def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image as a NumPy array (BGR format for OpenCV).

    Args:
        image_path: Path to the image file.

    Returns:
        NumPy array (H, W, 3) in BGR format, or None on failure.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"OpenCV returned None for: {image_path}")
            return None
        logger.debug(f"Image loaded: shape={img.shape}, dtype={img.dtype}")
        return img
    except Exception as e:
        logger.error(f"Failed to load image '{image_path}': {e}", exc_info=True)
        return None


# ============================================
# BOUNDING BOX ANNOTATION
# ============================================
def draw_bounding_box(
    image: np.ndarray,
    bbox: List[int],
    confidence: float,
    label: str = "Snake",
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 3
) -> np.ndarray:
    """
    Draw a bounding box with label on an image.

    Args:
        image: Input image (NumPy array, BGR).
        bbox: Bounding box coordinates [x1, y1, x2, y2].
        confidence: Confidence score (0-100).
        label: Label text to display.
        color: Box color in BGR.
        thickness: Box line thickness.

    Returns:
        Annotated image (NumPy array).
    """
    annotated = image.copy()
    x1, y1, x2, y2 = bbox

    # Draw rectangle
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

    # Prepare label text
    text = f"{label} {confidence:.2f}%"

    # Calculate text size for background rectangle
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)

    # Draw label background
    cv2.rectangle(
        annotated,
        (x1, y1 - text_h - baseline - 10),
        (x1 + text_w + 10, y1),
        color,
        -1  # Filled rectangle
    )

    # Draw label text
    cv2.putText(
        annotated,
        text,
        (x1 + 5, y1 - baseline - 5),
        font,
        font_scale,
        (255, 255, 255),  # White text
        text_thickness,
        cv2.LINE_AA
    )

    return annotated


# ============================================
# SAVE ANNOTATED IMAGE
# ============================================
def save_annotated_image(
    image: np.ndarray,
    output_dir: str = 'media/results',
    prefix: str = 'result'
) -> Optional[str]:
    """
    Save an annotated image to disk with a unique filename.

    Args:
        image: Image to save (NumPy array, BGR).
        output_dir: Directory to save into.
        prefix: Filename prefix.

    Returns:
        Relative path to saved image, or None on failure.
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"{prefix}_{uuid.uuid4().hex[:12]}.jpg"
        full_path = output_path / filename

        success = cv2.imwrite(str(full_path), image)
        if not success:
            logger.error(f"cv2.imwrite returned False for: {full_path}")
            return None

        logger.info(f"Annotated image saved: {full_path}")
        return str(full_path)

    except Exception as e:
        logger.error(f"Failed to save annotated image: {e}", exc_info=True)
        return None


# ============================================
# FORMAT DETECTION RESULT
# ============================================
def format_detection_result(
    snake_detected: bool,
    confidence: float,
    bbox: Optional[List[int]] = None,
    annotated_image_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build a standardized result dictionary.

    This is the contract that all future modules will rely on.

    Returns:
        {
            "snake": bool,
            "confidence": float,       # 0-100
            "bbox": [x1, y1, x2, y2] or None,
            "message": str,
            "annotated_image_path": str or None
        }
    """
    message = "Snake Detected" if snake_detected else "No Snake Detected"

    return {
        "snake": snake_detected,
        "confidence": round(float(confidence), 2),
        "bbox": bbox,
        "message": message,
        "annotated_image_path": annotated_image_path,
    }


# ============================================
# MOCK DETECTION (for development without model)
# ============================================
def mock_detection(image_path: str) -> Dict[str, Any]:
    """
    Generate a mock detection result when no model is loaded.
    Useful during development/testing.

    Args:
        image_path: Path to the image (used to generate consistent results).

    Returns:
        Standard detection result dictionary.
    """
    logger.warning("Using MOCK detection (no real model loaded).")

    # Pseudo-random but deterministic based on filename
    hash_val = sum(ord(c) for c in Path(image_path).name) % 10

    if hash_val > 4:
        # Mock snake detection
        img = load_image(image_path)
        if img is not None:
            h, w = img.shape[:2]
            bbox = [
                int(w * 0.2),
                int(h * 0.3),
                int(w * 0.8),
                int(h * 0.7),
            ]
            annotated = draw_bounding_box(img, bbox, 92.50, "Snake (MOCK)")
            annotated_path = save_annotated_image(annotated)
        else:
            bbox = [100, 100, 400, 400]
            annotated_path = None

        return format_detection_result(
            snake_detected=True,
            confidence=92.50,
            bbox=bbox,
            annotated_image_path=annotated_path,
        )
    else:
        return format_detection_result(
            snake_detected=False,
            confidence=98.50,
            bbox=None,
            annotated_image_path=None,
        )