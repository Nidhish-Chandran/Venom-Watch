from django.test import TestCase
from django.contrib.auth.models import User
from accounts.models import UserProfile
from reports.models import SightingReport
from hotspots.services import DBSCANHotspotService

class VenomWatchAccessControlTestCase(TestCase):
    def setUp(self):
        # Create Citizen User
        self.citizen = User.objects.create_user(username='john_citizen', password='password123')
        self.citizen.profile.role = UserProfile.ROLE_CITIZEN
        self.citizen.profile.save()

        # Create Second Citizen User
        self.other_citizen = User.objects.create_user(username='other_citizen', password='password123')
        self.other_citizen.profile.role = UserProfile.ROLE_CITIZEN
        self.other_citizen.profile.save()

        # Create Higher Authority User
        self.ranger = User.objects.create_user(username='ranger_officer', password='password123')
        self.ranger.profile.role = UserProfile.ROLE_AUTHORITY
        self.ranger.profile.save()

        # Create Citizen 1 Sighting Report
        self.citizen_report = SightingReport.objects.create(
            user=self.citizen,
            title="Citizen's Cobra Sighting",
            latitude=9.9312,
            longitude=76.2673,
            location_name='Kochi Sector 4',
            species_predicted='Indian Cobra (Naja naja)',
            venom_category=SightingReport.VENOM_HIGHLY,
            toxicity_level='CRITICAL',
            danger_score=95
        )

        # Create Other Citizen Sighting Report
        self.other_report = SightingReport.objects.create(
            user=self.other_citizen,
            title="Other Citizen's Viper Sighting",
            latitude=10.0420,
            longitude=76.3200,
            location_name='Kalamassery Industrial Area',
            species_predicted="Russell's Viper",
            venom_category=SightingReport.VENOM_HIGHLY,
            toxicity_level='CRITICAL',
            danger_score=98
        )

    def test_sighting_properties(self):
        self.assertTrue(self.citizen_report.is_venomous)
        self.assertTrue(self.citizen_report.is_active_48h)

    def test_dbscan_clustering(self):
        hotspot_data = DBSCANHotspotService.compute_hotspots(eps_km=1.0, min_samples=2, hours=48)
        self.assertEqual(hotspot_data['total_active_sightings'], 2)

    def test_citizen_access_control_denied(self):
        self.client.login(username='john_citizen', password='password123')

        # Citizen cannot access Authority Dashboard (403 Forbidden)
        response_dash = self.client.get('/')
        self.assertEqual(response_dash.status_code, 403)

        # Citizen cannot access Hotspot Radar (403 Forbidden)
        response_hotspots = self.client.get('/hotspots/')
        self.assertEqual(response_hotspots.status_code, 403)

        # Citizen cannot access Hotspots GeoJSON API (403 Forbidden)
        response_api = self.client.get('/hotspots/api/clusters/')
        self.assertEqual(response_api.status_code, 403)

        # Citizen cannot view other citizen's report detail (403 Forbidden)
        response_detail_other = self.client.get(f'/reports/{self.other_report.pk}/')
        self.assertEqual(response_detail_other.status_code, 403)

        # Citizen CAN view their own report detail (200 OK)
        response_detail_own = self.client.get(f'/reports/{self.citizen_report.pk}/')
        self.assertEqual(response_detail_own.status_code, 200)

    def test_citizen_sees_only_own_reports_in_list(self):
        self.client.login(username='john_citizen', password='password123')
        response = self.client.get('/reports/')
        self.assertEqual(response.status_code, 200)
        reports_in_context = response.context['reports']
        self.assertEqual(len(reports_in_context), 1)
        self.assertEqual(reports_in_context[0].pk, self.citizen_report.pk)

    def test_authority_access_control_allowed(self):
        self.client.login(username='ranger_officer', password='password123')

        # Higher Authority CAN access Dashboard (200 OK)
        response_dash = self.client.get('/')
        self.assertEqual(response_dash.status_code, 200)

        # Higher Authority CAN access Hotspot Radar (200 OK)
        response_hotspots = self.client.get('/hotspots/')
        self.assertEqual(response_hotspots.status_code, 200)

        # Higher Authority CAN view any report detail (200 OK)
        response_detail = self.client.get(f'/reports/{self.citizen_report.pk}/')
        self.assertEqual(response_detail.status_code, 200)

        # Higher Authority sees all reports in list view
        response_list = self.client.get('/reports/')
        self.assertEqual(len(response_list.context['reports']), 2)


class ReportingAndRiskZoneLifecycleTestCase(TestCase):
    def setUp(self):
        self.user1 = User.objects.create_user(username='citizen_one', password='password123')
        self.user1.profile.role = UserProfile.ROLE_CITIZEN
        self.user1.profile.save()

        self.user2 = User.objects.create_user(username='citizen_two', password='password123')
        self.user2.profile.role = UserProfile.ROLE_CITIZEN
        self.user2.profile.save()

    def test_report_numbering_and_reference_per_citizen(self):
        """
        Verifies:
        - The user's first report is numbered #1 for that citizen.
        - Subsequent reports increment sequentially for that citizen (#2).
        - Different citizens have independent submission counters starting at #1.
        - Each report has a stable, unique report_reference (e.g. VW-0001).
        """
        r1 = SightingReport.objects.create(
            user=self.user1,
            title="First sighting for user 1",
            latitude=9.9312,
            longitude=76.2673,
            location_name="Sector 1"
        )
        self.assertEqual(r1.citizen_submission_number, 1)
        self.assertEqual(r1.report_reference, f"VW-{r1.id:04d}")

        r2 = SightingReport.objects.create(
            user=self.user1,
            title="Second sighting for user 1",
            latitude=9.9320,
            longitude=76.2680,
            location_name="Sector 1 Park"
        )
        self.assertEqual(r2.citizen_submission_number, 2)
        self.assertEqual(r2.report_reference, f"VW-{r2.id:04d}")

        r3 = SightingReport.objects.create(
            user=self.user2,
            title="First sighting for user 2",
            latitude=10.0400,
            longitude=76.3200,
            location_name="Kalamassery"
        )
        self.assertEqual(r3.citizen_submission_number, 1)
        self.assertEqual(r3.report_reference, f"VW-{r3.id:04d}")

    def test_sighting_form_coordinate_validation(self):
        """Form validates coordinates boundaries and handles manual location input."""
        from reports.forms import SightingReportForm

        # Invalid latitude (> 90)
        form_invalid_lat = SightingReportForm(data={
            'title': 'Test Snake',
            'latitude': 95.0,
            'longitude': 76.2673,
            'location_name': 'Kochi'
        })
        self.assertFalse(form_invalid_lat.is_valid())
        self.assertIn('latitude', form_invalid_lat.errors)

        # Invalid longitude (> 180)
        form_invalid_lng = SightingReportForm(data={
            'title': 'Test Snake',
            'latitude': 9.9312,
            'longitude': 185.0,
            'location_name': 'Kochi'
        })
        self.assertFalse(form_invalid_lng.is_valid())
        self.assertIn('longitude', form_invalid_lng.errors)

        # Missing location name
        form_missing_loc = SightingReportForm(data={
            'title': 'Test Snake',
            'latitude': 9.9312,
            'longitude': 76.2673,
            'location_name': ''
        })
        self.assertFalse(form_missing_loc.is_valid())
        self.assertIn('location_name', form_missing_loc.errors)

    def test_risk_zone_lifecycle_and_distinct_incident_counting(self):
        """
        Tests the 4-state lifecycle and duplicate deduplication:
        - 0 active -> No active zones
        - 1-2 distinct active -> ORANGE
        - 3+ distinct active -> RED
        - Duplicates do not inflate active count
        - Resolved incident -> YELLOW (cooldown)
        - Cooldown expired -> GREEN
        - Green retention expired -> removed
        """
        from datetime import timedelta
        from django.utils import timezone

        # 1. Zero incidents -> empty zones
        res_zero = DBSCANHotspotService.compute_community_risk_zones()
        self.assertEqual(res_zero['total_active_incidents'], 0)
        self.assertEqual(len(res_zero['zones']), 0)

        # 2. 1 active incident -> ORANGE
        inc1 = SightingReport.objects.create(
            user=self.user1,
            title="Active Incident 1",
            latitude=9.9312,
            longitude=76.2673,
            location_name="Zone Central",
            status=SightingReport.STATUS_PENDING
        )
        res_one = DBSCANHotspotService.compute_community_risk_zones()
        self.assertEqual(res_one['total_active_incidents'], 1)
        self.assertEqual(len(res_one['zones']), 1)
        self.assertEqual(res_one['zones'][0]['state'], 'ORANGE')

        # 3. Add duplicate report for inc1 -> should NOT increase active count
        dup_report = SightingReport.objects.create(
            user=self.user2,
            title="Duplicate report for inc1",
            latitude=9.9313,
            longitude=76.2674,
            location_name="Zone Central",
            status=SightingReport.STATUS_PENDING,
            duplicate_flag=True,
            duplicate_of=inc1
        )
        res_dup = DBSCANHotspotService.compute_community_risk_zones()
        self.assertEqual(res_dup['total_active_incidents'], 1)
        self.assertEqual(res_dup['zones'][0]['state'], 'ORANGE')

        # 4. Add 2 more distinct incidents in same sector -> total 3 distinct active -> RED
        inc2 = SightingReport.objects.create(
            user=self.user1,
            title="Active Incident 2",
            latitude=9.9320,
            longitude=76.2680,
            location_name="Zone Central East",
            status=SightingReport.STATUS_VERIFIED
        )
        inc3 = SightingReport.objects.create(
            user=self.user2,
            title="Active Incident 3",
            latitude=9.9305,
            longitude=76.2665,
            location_name="Zone Central South",
            status=SightingReport.STATUS_PENDING
        )
        res_three = DBSCANHotspotService.compute_community_risk_zones()
        self.assertEqual(res_three['total_active_incidents'], 3)
        self.assertEqual(res_three['zones'][0]['state'], 'RED')

        # 5. Resolve all 3 incidents now -> zone enters YELLOW (cooldown)
        now = timezone.now()
        for inc in [inc1, inc2, inc3]:
            inc.status = SightingReport.STATUS_RESOLVED
            inc.save()
        res_resolved = DBSCANHotspotService.compute_community_risk_zones()
        self.assertEqual(res_resolved['total_active_incidents'], 0)
        self.assertEqual(res_resolved['zones'][0]['state'], 'YELLOW')

        # 6. Simulate cooldown expiry (resolved 26 hours ago, where cooldown = 24h)
        past_cooldown = now - timedelta(hours=26)
        SightingReport.objects.filter(pk__in=[inc1.pk, inc2.pk, inc3.pk]).update(
            updated_at=past_cooldown,
            created_at=past_cooldown
        )
        res_green = DBSCANHotspotService.compute_community_risk_zones()
        self.assertEqual(res_green['zones'][0]['state'], 'GREEN')

        # 7. Simulate retention expiry (> 24 + 48 = 72 hours ago)
        past_retention = now - timedelta(hours=75)
        SightingReport.objects.filter(pk__in=[inc1.pk, inc2.pk, inc3.pk]).update(
            updated_at=past_retention,
            created_at=past_retention
        )
        res_expired = DBSCANHotspotService.compute_community_risk_zones()
        self.assertEqual(len(res_expired['zones']), 0)
