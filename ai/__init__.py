default_app_config = "ai.apps.AiConfig"
try:
    from .predict import detect_snake
except Exception:
    detect_snake = None
