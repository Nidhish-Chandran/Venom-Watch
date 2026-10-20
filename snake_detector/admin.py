"""Django admin registration for detection records."""
from django.contrib import admin
from .models import DetectionRecord, ModelInfo


@admin.register(DetectionRecord)
class DetectionRecordAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'snake_detected',
        'confidence_score',
        'created_at',
        'ip_address',
    )
    list_filter = ('snake_detected', 'created_at')
    search_fields = ('id', 'ip_address')
    readonly_fields = (
        'id', 'uploaded_image', 'annotated_image', 'snake_detected',
        'confidence_score', 'bbox_coordinates', 'created_at', 'updated_at',
        'ip_address', 'user_agent',
    )
    ordering = ('-created_at',)
    date_hierarchy = 'created_at'


@admin.register(ModelInfo)
class ModelInfoAdmin(admin.ModelAdmin):
    list_display = ('model_name', 'model_status', 'confidence_threshold', 'loaded_at')
    readonly_fields = ('loaded_at',)