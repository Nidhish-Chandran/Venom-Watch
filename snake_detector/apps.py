"""App configuration for snake_detector."""
import os
import sys
import logging
from django.apps import AppConfig
from django.conf import settings

logger = logging.getLogger('snake_detector.apps')

class SnakeDetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'snake_detector'
    verbose_name = 'Snake Detection Module'

    def ready(self) -> None:
        """Called when Django starts."""
        try:
            # Force add the project root to Python path
            root_dir = str(settings.BASE_DIR)
            if root_dir not in sys.path:
                sys.path.append(root_dir)
            
            import ai  # noqa: F401
            logger.info("AI module initialized successfully.")
        except Exception as e:
            logger.warning(f"AI module failed to initialize on startup: {e}")
            logger.warning("App will run in MOCK mode.")