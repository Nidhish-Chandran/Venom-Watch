"""
Image upload form with validation.
Validates file type and file size before saving.
"""
import logging

from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError

logger = logging.getLogger('snake_detector.forms')


class SnakeImageUploadForm(forms.Form):
    """
    Form for handling snake image uploads.
    Validates:
    1. File is provided
    2. File extension is .jpg, .jpeg, or .png
    3. File size does not exceed MAX_UPLOAD_SIZE
    """

    image = forms.ImageField(
        label='Snake Image',
        required=True,
        error_messages={
            'required': 'Please select an image file.',
            'invalid': 'Please upload a valid image file.',
            'missing': 'No file was uploaded.',
            'empty': 'The uploaded file is empty.',
        },
    )

    def clean_image(self) -> object:
        """
        Custom validation for the uploaded image.
        """
        image = self.cleaned_data.get('image')

        if not image:
            raise ValidationError('No image file was provided.')

        # ------------------------------------------
        # Validate file extension
        # ------------------------------------------
        allowed_extensions = settings.ALLOWED_IMAGE_EXTENSIONS
        ext = ''
        if hasattr(image, 'name') and image.name:
            ext = '.' + image.name.rsplit('.', 1)[-1].lower() if '.' in image.name else ''

        if ext not in allowed_extensions:
            logger.warning(f"Rejected file with extension: {ext}")
            raise ValidationError(
                f'Unsupported file format "{ext}". '
                f'Allowed formats: {", ".join(allowed_extensions)}'
            )

        # ------------------------------------------
        # Validate file size
        # ------------------------------------------
        max_size = settings.MAX_UPLOAD_SIZE
        if hasattr(image, 'size') and image.size > max_size:
            size_mb = image.size / (1024 * 1024)
            max_mb = max_size / (1024 * 1024)
            logger.warning(f"Rejected file too large: {size_mb:.1f}MB")
            raise ValidationError(
                f'File size ({size_mb:.1f} MB) exceeds the maximum allowed '
                f'size of {max_mb:.0f} MB.'
            )

        logger.info(
            f"Image accepted: name={image.name}, size={getattr(image, 'size', 'unknown')} bytes"
        )
        return image