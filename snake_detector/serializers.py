"""
DRF Serializers for the snake detection API.
"""
from rest_framework import serializers


class SnakeDetectionRequestSerializer(serializers.Serializer):
    """Serializer for validating the detection API request."""
    image = serializers.ImageField(required=True)


class SnakeDetectionResponseSerializer(serializers.Serializer):
    """Serializer for the standardized detection response."""

    snake = serializers.BooleanField()
    confidence = serializers.FloatField()
    bbox = serializers.ListField(
        child=serializers.IntegerField(),
        allow_null=True,
        required=False,
    )
    message = serializers.CharField()
    annotated_image_url = serializers.CharField(allow_null=True, required=False)
    uploaded_image_url = serializers.CharField(allow_null=True, required=False)
    record_id = serializers.CharField(allow_null=True, required=False)