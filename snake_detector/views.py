"""
============================================
Views — Web Page + REST API
============================================
"""
import logging
import os
import sys
import uuid
from pathlib import Path

from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse, HttpRequest, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser

from .forms import SnakeImageUploadForm
from .models import DetectionRecord

# Add root directory to Python path so we can find the 'ai' folder
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Import the AI module
from ai import detect_snake

logger = logging.getLogger('snake_detector.views')


# ============================================
# WEB PAGE VIEW
# ============================================
def index(request: HttpRequest) -> HttpResponse:
    """
    Render the main upload page.
    """
    return render(request, 'snake_detector/index.html', {
        'title': 'Venom Watch — Snake Detection',
        'max_upload_mb': settings.MAX_UPLOAD_SIZE // (1024 * 1024),
    })


# ============================================
# REST API VIEW
# ============================================
@method_decorator(csrf_exempt, name='dispatch')
class DetectAPIView(APIView):
    """
    POST /api/detect/
    """
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request: HttpRequest) -> Response:
        logger.info("=== Detection API request received ===")

        # Step 1: Validate form
        form = SnakeImageUploadForm(request.POST, request.FILES)

        if not form.is_valid():
            errors = []
            for field, error_list in form.errors.items():
                for err in error_list:
                    errors.append(str(err))
            error_msg = '; '.join(errors) if errors else 'Invalid request.'
            logger.warning(f"Form validation failed: {error_msg}")
            return Response(
                {'error': error_msg},
                status=status.HTTP_400_BAD_REQUEST,
            )

        image_file = form.cleaned_data['image']

        # Step 2: Save uploaded image to media/uploads/
        upload_dir = Path(settings.MEDIA_ROOT) / 'uploads'
        upload_dir.mkdir(parents=True, exist_ok=True)

        ext = Path(image_file.name).suffix.lower()
        filename = f"upload_{uuid.uuid4().hex[:12]}{ext}"
        upload_path = upload_dir / filename

        try:
            with open(upload_path, 'wb+') as destination:
                for chunk in image_file.chunks():
                    destination.write(chunk)
            logger.info(f"Image saved to: {upload_path}")
        except Exception as e:
            logger.error(f"Failed to save uploaded image: {e}", exc_info=True)
            return Response(
                {'error': 'Failed to save uploaded image. Please try again.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # Step 3: Call the AI module
        try:
            result = detect_snake(str(upload_path))
            logger.info(f"Detection result: snake={result['snake']}, confidence={result['confidence']}")

        except Exception as e:
            logger.error(f"AI detection failed: {e}", exc_info=True)
            if upload_path.exists():
                upload_path.unlink()

            return Response(
                {
                    'snake': False,
                    'confidence': 0.0,
                    'bbox': None,
                    'message': 'Detection failed due to an internal error.',
                    'error': str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # Step 4: Build URLs for frontend
        uploaded_image_url = None
        annotated_image_url = None

        if upload_path.exists():
            relative_path = upload_path.relative_to(settings.MEDIA_ROOT)
            uploaded_image_url = f"{settings.MEDIA_URL}{relative_path}"

        if result.get('annotated_image_path'):
            annotated_full = Path(result['annotated_image_path'])
            if annotated_full.exists():
                try:
                    relative = annotated_full.relative_to(settings.MEDIA_ROOT)
                    annotated_image_url = f"{settings.MEDIA_URL}{relative}"
                except ValueError:
                    annotated_image_url = f"/{result['annotated_image_path']}"

        # Step 5: Save detection record to database
        record = None
        try:
            record = DetectionRecord.objects.create(
                uploaded_image=str(relative_path) if upload_path.exists() else str(upload_path),
                annotated_image=result.get('annotated_image_path'),
                snake_detected=result['snake'],
                confidence_score=result['confidence'],
                bbox_coordinates=result.get('bbox'),
                ip_address=self._get_client_ip(request),
                user_agent=request.META.get('HTTP_USER_AGENT', '')[:500],
            )
            logger.info(f"Detection record saved: {record.id}")
        except Exception as e:
            logger.error(f"Failed to save detection record: {e}", exc_info=True)

        # Step 6: Return JSON response
        response_data = {
            'snake': result['snake'],
            'confidence': result['confidence'],
            'bbox': result.get('bbox'),
            'message': result['message'],
            'annotated_image_url': annotated_image_url,
            'uploaded_image_url': uploaded_image_url,
            'record_id': str(record.id) if record else None,
        }

        logger.info("=== Detection API request complete ===")
        return Response(response_data, status=status.HTTP_200_OK)

    def _get_client_ip(self, request: HttpRequest) -> str:
        """Extract client IP address from request."""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0].strip()
        return request.META.get('REMOTE_ADDR', '')