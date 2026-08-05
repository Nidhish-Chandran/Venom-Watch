"""
============================================
Venom Watch — AI Module
============================================
Independent Snake Detection AI Package.
"""

from .predict import detect_snake

__all__ = ['detect_snake']
__version__ = '1.0.0'















# """
# ============================================
# Venom Watch — AI Module
# ============================================
# Independent Snake Detection AI Package.

# USAGE (for future teammates):
#     from ai import detect_snake

#     result = detect_snake("path/to/image.jpg")
#     print(result)
#     # {
#     #     "snake": True,
#     #     "confidence": 95.72,
#     #     "bbox": [120, 80, 450, 390],
#     #     "message": "Snake Detected",
#     #     "annotated_image_path": "media/results/img_xxx.jpg"
#     # }

# This module does NOT depend on Django.
# It can be used in any Python project, CLI, notebook, or microservice.
# """

# from .predict import detect_snake

# __all__ = ['detect_snake']
# __version__ = '1.0.0'
# __author__ = 'Venom Watch Team'


