"""
============================================
Model Loader — Singleton Pattern
============================================
Loads the YOLO model ONLY ONCE during application startup.
Never reloads the model for each request.

WHY SINGLETON?
- Loading a YOLO model takes 2-5 seconds and significant RAM.
- Reloading per request would make the app extremely slow.
- Singleton ensures one model instance lives in memory for all requests.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger('ai.model_loader')

# ============================================
# Try importing Ultralytics (YOLOv11 / YOLOv8)
# ============================================
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    logger.info("✅ Ultralytics package imported successfully.")
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None
    logger.warning("⚠️ Ultralytics not installed. Using mock mode.")


class ModelLoader:
    """
    Thread-safe Singleton that loads and caches the YOLO model.

    Usage:
        loader = ModelLoader()
        model = loader.get_model()
    """

    _instance: Optional['ModelLoader'] = None
    _lock: threading.Lock = threading.Lock()
    _model: Optional[Any] = None
    _loaded: bool = False

    def __new__(cls) -> 'ModelLoader':
        """Ensure only one instance exists (Singleton)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    logger.info("ModelLoader singleton instance created.")
        return cls._instance

    def load_model(
        self,
        model_path: str = 'weights/snake_yolov11.pt',
        device: str = 'cpu'
    ) -> Any:
        """
        Load the YOLO model from the given path.

        Args:
            model_path: Path to the .pt weights file.
            device: 'cpu' or 'cuda' for GPU inference.

        Returns:
            Loaded YOLO model instance.

        Raises:
            FileNotFoundError: If weights file does not exist.
            RuntimeError: If model loading fails.
        """
        if self._loaded and self._model is not None:
            logger.debug("Model already loaded. Returning cached instance.")
            return self._model

        with self._lock:
            if self._loaded and self._model is not None:
                return self._model

            # ------------------------------------------
            # Check if Ultralytics is available
            # ------------------------------------------
            if not ULTRALYTICS_AVAILABLE:
                logger.warning(
                    "Ultralytics not available. Operating in MOCK mode. "
                    "Install with: pip install ultralytics"
                )
                self._model = None
                self._loaded = True
                return None

            # ------------------------------------------
            # Resolve model path
            # ------------------------------------------
            # Try multiple possible locations
            possible_paths = [
                Path(model_path),
                Path(__file__).parent.parent / model_path,
                Path(__file__).parent.parent / 'weights' / 'snake_yolov11.pt',
                Path(__file__).parent.parent / 'weights' / 'snake_yolov8.pt',
            ]

            resolved_path: Optional[Path] = None
            for p in possible_paths:
                if p.exists():
                    resolved_path = p
                    break

            if resolved_path is None:
                logger.error(
                    f"Model weights not found. Searched: {[str(p) for p in possible_paths]}"
                )
                logger.warning("Running in MOCK mode (no weights found).")
                self._model = None
                self._loaded = True
                return None

            # ------------------------------------------
            # Load the model
            # ------------------------------------------
            try:
                logger.info(f"Loading YOLO model from: {resolved_path}")
                self._model = YOLO(str(resolved_path))

                # Warm up the model with a dummy inference
                # This pre-allocates memory and speeds up first real request
                logger.info("Warming up model (dummy inference)...")
                import numpy as np
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                self._model(dummy, verbose=False, device=device)

                self._loaded = True
                logger.info("✅ YOLO model loaded and warmed up successfully.")
                return self._model

            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
                self._model = None
                self._loaded = True
                raise RuntimeError(f"Model loading failed: {e}") from e

    def get_model(self) -> Optional[Any]:
        """Return the cached model (or None if in mock mode)."""
        if not self._loaded:
            raise RuntimeError("Model not loaded yet. Call load_model() first.")
        return self._model

    def is_loaded(self) -> bool:
        """Check if the model has been loaded."""
        return self._loaded and self._model is not None


# ============================================
# Module-level convenience instance
# ============================================
_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get the global ModelLoader singleton instance."""
    global _loader
    if _loader is None:
        _loader = ModelLoader()
    return _loader