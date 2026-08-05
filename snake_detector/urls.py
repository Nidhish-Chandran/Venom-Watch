"""
URL routes for the snake_detector app.
"""
from django.urls import path
from .views import index, DetectAPIView

app_name = 'snake_detector'

urlpatterns = [
    # Web page
    path('', index, name='index'),

    # REST API endpoint
    path('api/detect/', DetectAPIView.as_view(), name='api_detect'),
]