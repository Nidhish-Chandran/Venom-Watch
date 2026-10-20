import os
import io
import tempfile
from unittest.mock import patch, MagicMock
from PIL import Image

from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from accounts.models import UserProfile
from ai.services import SnakebiteScreeningService, WoundScreeningService


class SnakebiteScreeningServiceTestCase(TestCase):
    """
    Unit and integration tests for the EfficientNetB0-based Snakebite Wound Screening pipeline.
    Validates model artifact integrity, preprocessing, decision thresholds, uncertainty handling,
    and mandatory clinical disclaimers.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Create a genuine temporary JPEG image for inference testing
        cls.test_image_dir = tempfile.mkdtemp()
        cls.test_image_path = os.path.join(cls.test_image_dir, 'sample_wound.jpg')
        img = Image.new('RGB', (300, 300), color=(180, 50, 50))
        img.save(cls.test_image_path, 'JPEG')

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_image_path):
            os.remove(cls.test_image_path)
        if os.path.exists(cls.test_image_dir):
            os.rmdir(cls.test_image_dir)
        super().tearDownClass()

    def test_model_artifact_exists_and_loads(self):
        """Verify the physical EfficientNetB0 .keras artifact exists and loads as a valid Keras model."""
        self.assertTrue(
            os.path.exists(SnakebiteScreeningService.MODEL_PATH),
            f"Model artifact not found at {SnakebiteScreeningService.MODEL_PATH}"
        )
        model = SnakebiteScreeningService.get_model()
        self.assertIsNotNone(model)
        # Verify expected input shape (None, 224, 224, 3) and output shape (None, 1)
        self.assertEqual(model.input_shape, (None, 224, 224, 3))
        self.assertEqual(model.output_shape, (None, 1))

    def test_missing_image_file_handling(self):
        """Missing or non-existent files must return a safe fallback with mandatory disclaimer."""
        result = SnakebiteScreeningService.analyze_wound_image('non_existent_path_xyz.jpg')
        self.assertFalse(result['is_snake_bite'])
        self.assertEqual(result['result_label'], 'Not Snake Bite')
        self.assertEqual(result['status_title'], 'Image Not Found')
        self.assertIsNone(result['confidence'])
        self.assertIn("medical care immediately", result['disclaimer'])
        self.assertFalse(result['uncertain'])

    def test_end_to_end_real_image_forward_pass(self):
        """Run actual forward pass with a real image through the loaded EfficientNetB0 model."""
        result = SnakebiteScreeningService.analyze_wound_image(self.test_image_path)
        self.assertIn('is_snake_bite', result)
        self.assertIsInstance(result['is_snake_bite'], bool)
        self.assertIn(result['result_label'], ['Snake Bite', 'Not Snake Bite'])
        self.assertIsInstance(result['raw_output'], float)
        self.assertGreaterEqual(result['raw_output'], 0.0)
        self.assertLessEqual(result['raw_output'], 1.0)
        self.assertIsInstance(result['confidence'], float)
        self.assertIn("AI-assisted screening tool", result['disclaimer'])
        self.assertEqual(result['method_label'], 'AI-Assisted Screening (EfficientNetB0)')

    @patch('ai.services.SnakebiteScreeningService.get_model')
    def test_high_confidence_positive_screening(self, mock_get_model):
        """Raw output >= 0.60 must classify as Possible Snakebite Pattern Detected."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [[0.85]]
        mock_get_model.return_value = mock_model

        result = SnakebiteScreeningService.analyze_wound_image(self.test_image_path)
        self.assertTrue(result['is_snake_bite'])
        self.assertEqual(result['result_label'], 'Snake Bite')
        self.assertEqual(result['status_title'], 'Possible Snakebite Pattern Detected')
        self.assertEqual(result['confidence'], 85.0)
        self.assertFalse(result['uncertain'])
        self.assertIn("Immobilize the limb", result['recommendation'])

    @patch('ai.services.SnakebiteScreeningService.get_model')
    def test_high_confidence_negative_screening(self, mock_get_model):
        """Raw output < 0.40 must classify as No Snakebite Pattern Detected."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [[0.10]]
        mock_get_model.return_value = mock_model

        result = SnakebiteScreeningService.analyze_wound_image(self.test_image_path)
        self.assertFalse(result['is_snake_bite'])
        self.assertEqual(result['result_label'], 'Not Snake Bite')
        self.assertEqual(result['status_title'], 'No Snakebite Pattern Detected')
        self.assertEqual(result['confidence'], 90.0)  # (1 - 0.10) * 100
        self.assertFalse(result['uncertain'])
        self.assertIn("does not rule out a snakebite", result['recommendation'])

    @patch('ai.services.SnakebiteScreeningService.get_model')
    def test_uncertainty_boundary_interval(self, mock_get_model):
        """Raw output within [0.40, 0.60] must set uncertain=True and override status title."""
        mock_model = MagicMock()
        # Test lower boundary of uncertainty
        mock_model.predict.return_value = [[0.45]]
        mock_get_model.return_value = mock_model

        result = SnakebiteScreeningService.analyze_wound_image(self.test_image_path)
        self.assertTrue(result['uncertain'])
        self.assertEqual(result['status_title'], 'Unable to Confidently Classify')
        self.assertIn("could not confidently classify", result['recommendation'])

        # Test upper boundary of uncertainty
        mock_model.predict.return_value = [[0.55]]
        result_upper = SnakebiteScreeningService.analyze_wound_image(self.test_image_path)
        self.assertTrue(result_upper['uncertain'])
        self.assertEqual(result_upper['status_title'], 'Unable to Confidently Classify')

    def test_deprecated_service_delegates_to_efficientnetb0(self):
        """WoundScreeningService should cleanly delegate to SnakebiteScreeningService."""
        result = WoundScreeningService.analyze_wound_image('non_existent.jpg')
        self.assertEqual(result['status_title'], 'Image Not Found')
        self.assertEqual(result['disclaimer'], SnakebiteScreeningService.MANDATORY_DISCLAIMER)


class WoundCheckerViewTestCase(TestCase):
    """
    Tests for the citizen-facing wound screening view (wound_checker_view).
    Verifies authentication gating, multipart image handling, and cache-control security headers.
    """

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user('testcitizen', 'citizen@example.com', 'Pass123!Secure')
        self.user.profile.role = UserProfile.ROLE_CITIZEN
        self.user.profile.account_status = UserProfile.ACCOUNT_ACTIVE
        self.user.profile.save()

    def test_unauthenticated_user_redirected_to_login(self):
        """Unauthenticated requests must be redirected to login."""
        response = self.client.get('/ai/wound-check/')
        self.assertEqual(response.status_code, 302)
        self.assertIn('/accounts/login/', response.url)

    def test_authenticated_user_can_access_wound_checker_page(self):
        """Authenticated citizen can access the wound screening upload page."""
        self.client.login(username='testcitizen', password='Pass123!Secure')
        response = self.client.get('/ai/wound-check/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Wound")
        self.assertContains(response, "screening")

    def test_authenticated_user_image_upload_and_cache_headers(self):
        """Uploading a valid wound image performs inference, renders disclaimer, and returns no-cache headers."""
        self.client.login(username='testcitizen', password='Pass123!Secure')

        file_obj = io.BytesIO()
        img = Image.new('RGB', (224, 224), color=(200, 100, 100))
        img.save(file_obj, 'JPEG')
        file_obj.seek(0)
        uploaded_file = SimpleUploadedFile("wound.jpg", file_obj.read(), content_type="image/jpeg")

        response = self.client.post('/ai/wound-check/', {'wound_image': uploaded_file})
        self.assertEqual(response.status_code, 200)

        # Verify no-cache security headers
        self.assertIn('no-store', response['Cache-Control'])
        self.assertIn('no-cache', response['Cache-Control'])
        self.assertEqual(response['Pragma'], 'no-cache')
        self.assertEqual(response['Expires'], '0')

        # Verify mandatory disclaimer is visible on page
        self.assertContains(response, "AI-assisted screening tool")
        self.assertContains(response, "NOT a medical diagnosis")
