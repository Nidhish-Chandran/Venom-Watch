"""
Database models for storing detection records.
This allows future modules (dashboard, analytics) to query detection history.
"""
import uuid
from pathlib import Path

from django.db import models
from django.conf import settings


class DetectionRecord(models.Model):
    """
    Stores every snake detection request and its result.
    Future modules (government dashboard, hotspot detection)
    can query this table.
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        db_index=True,
    )

    # Original uploaded image
    uploaded_image = models.ImageField(
        upload_to='uploads/%Y/%m/%d/',
        max_length=500,
    )

    # Annotated result image (with bounding box)
    annotated_image = models.CharField(
        max_length=500,
        blank=True,
        null=True,
        help_text="Path to annotated image with bounding box",
    )

    # Detection result
    snake_detected = models.BooleanField(default=False)
    confidence_score = models.FloatField(default=0.0)
    bbox_coordinates = models.JSONField(
        blank=True,
        null=True,
        help_text="Bounding box [x1, y1, x2, y2]",
    )

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Client info (useful for analytics)
    ip_address = models.GenericIPAddressField(blank=True, null=True)
    user_agent = models.CharField(max_length=500, blank=True, null=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Detection Record'
        verbose_name_plural = 'Detection Records'
        indexes = [
            models.Index(fields=['snake_detected']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self) -> str:
        status = "Snake" if self.snake_detected else "No Snake"
        return f"[{self.created_at:%Y-%m-%d %H:%M}] {status} ({self.confidence_score:.1f}%)"


class ModelInfo(models.Model):
    """
    Stores metadata about the currently loaded AI model.
    Useful for the future dashboard to display model version.
    """

    MODEL_STATUS_CHOICES = [
        ('loaded', 'Loaded'),
        ('mock', 'Mock Mode'),
        ('failed', 'Failed to Load'),
    ]

    model_name = models.CharField(max_length=200)
    model_path = models.CharField(max_length=500)
    model_status = models.CharField(
        max_length=20,
        choices=MODEL_STATUS_CHOICES,
        default='mock',
    )
    confidence_threshold = models.FloatField(default=0.5)
    loaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Model Info'
        verbose_name_plural = 'Model Info'

    def __str__(self) -> str:
        return f"{self.model_name} ({self.model_status})"