"""
URL routes for the snake_detector app.
"""
from django.urls import path
from .views import index, DetectAPIView

app_name = 'snake_detector'

urlpatterns = [
    path('', index, name='index'),
    path('api/detect/', DetectAPIView.as_view(), name='api_detect'),
]