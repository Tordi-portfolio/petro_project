"""Django tests. Run with:  python manage.py test wells"""
import tempfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from petrophysics.synthetic import make_synthetic_las

from .models import Analysis, Well

TMP_MEDIA = tempfile.mkdtemp()
PARAMS = {"sat_model": "indonesia", "a": 1, "m": 2, "n": 2, "surface_temp_f": 80, "bht_f": 210, "rmf": 0.35,
          "rw_temp_f": 75, "rmf_temp_f": 75, "vsh_method": "linear", "rho_sand": 2.65, "rho_lime": 2.71,
          "rho_dolo": 2.87, "rho_fluid": 1.0, "vsh_cut": 0.4, "phi_cut": 0.08, "sw_hc_max": 0.65,
          "sw_water_min": 0.85, "sw_pay_cut": 0.6, "sep_gas": 0.04, "min_bed_m": 0.9, "wash_thr_in": 1.5,
          "drho_thr": 0.1, "exclude_bad_hole": "on", "despike": "on"}


@override_settings(MEDIA_ROOT=TMP_MEDIA)
class WellFlowTests(TestCase):
    def upload(self, content=None, name="syn.las"):
        content = content if content is not None else make_synthetic_las()[0].encode()
        return self.client.post(reverse("well_list"), {"name": "", "las_file": SimpleUploadedFile(name, content)})

    def test_upload_valid_las(self):
        r = self.upload()
        well = Well.objects.get()
        self.assertRedirects(r, well.get_absolute_url())
        self.assertEqual(well.name, "SYN-1")
        self.assertIn("GR", well.curve_names)

    def test_upload_rejects_non_las(self):
        r = self.upload(b"this is not a las file")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(Well.objects.count(), 0)
        self.assertContains(r, "Not a readable LAS file")

    def test_run_analysis_and_report(self):
        self.upload()
        well = Well.objects.get()
        r = self.client.post(reverse("well_detail", args=[well.pk]), PARAMS)
        a = Analysis.objects.get()
        self.assertRedirects(r, a.get_absolute_url())
        self.assertTrue(a.suitable)
        page = self.client.get(a.get_absolute_url())
        self.assertContains(page, "Log quality analysis")
        self.assertContains(page, "Hydrocarbon saturation analysis")
        self.assertContains(page, "data:image/png;base64,")

    def test_downloads(self):
        self.upload()
        well = Well.objects.get()
        self.client.post(reverse("well_detail", args=[well.pk]), PARAMS)
        a = Analysis.objects.get()
        for fmt, ctype in (("html", "text/html"), ("md", "text/markdown"), ("csv", "text/csv")):
            r = self.client.get(reverse("analysis_download", args=[a.pk, fmt]))
            self.assertEqual(r.status_code, 200)
            self.assertIn(ctype, r["Content-Type"])
        self.assertEqual(self.client.get(reverse("analysis_download", args=[a.pk, "exe"])).status_code, 404)

    def test_missing_curve_shows_error_not_500(self):
        text = make_synthetic_las()[0].replace(" GR.GAPI", " XX.GAPI")
        self.upload(text.encode())
        well = Well.objects.get()
        r = self.client.post(reverse("well_detail", args=[well.pk]), PARAMS, follow=True)
        self.assertContains(r, "Analysis failed")
        self.assertEqual(Analysis.objects.count(), 0)

    def test_screen_all_ranks_wells(self):
        self.upload()
        r = self.client.post(reverse("screen_all"), follow=True)
        self.assertContains(r, "Select a suitable well")
        self.assertEqual(Analysis.objects.count(), 1)

    def test_bad_curve_map_rejected(self):
        self.upload()
        well = Well.objects.get()
        r = self.client.post(reverse("well_detail", args=[well.pk]), {**PARAMS, "curve_map_text": "RT LLD"})
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "CANONICAL=MNEMONIC")

    def test_delete_well(self):
        self.upload()
        well = Well.objects.get()
        self.client.post(reverse("well_delete", args=[well.pk]))
        self.assertEqual(Well.objects.count(), 0)
