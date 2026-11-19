"""Engine tests - need only numpy/pandas/matplotlib (no Django). Run:  python -m unittest discover -s petrophysics/tests -t ."""
import json
import unittest

import numpy as np

from petrophysics import fluids, lithology, plotting, report, saturation
from petrophysics.las import parse_las
from petrophysics.params import merge
from petrophysics.pipeline import evaluate
from petrophysics.synthetic import SYNTH_PARAMS, make_synthetic_las


class SyntheticWell(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text, cls.truth = make_synthetic_las()
        cls.ev = evaluate(cls.text, SYNTH_PARAMS)
        cls.s = cls.ev.summary
        cls.df = cls.ev.table

    def zone_mode(self, top, base):
        m = (self.df.DEPT > top) & (self.df.DEPT < base)
        return int(self.df.FLUID_CODE[m].mode()[0])

    def test_fluid_zones(self):
        self.assertEqual(self.zone_mode(2052, 2064), 3)   # gas
        self.assertEqual(self.zone_mode(2068, 2078), 2)   # oil
        self.assertEqual(self.zone_mode(2083, 2093), 1)   # water leg
        self.assertEqual(self.zone_mode(2153, 2183), 1)   # water sand
        self.assertEqual(self.zone_mode(2253, 2267), 2)   # shaly oil sand

    def test_contacts_within_one_metre(self):
        c = {x["contact"].split(" ")[0]: x["depth"] for x in self.s["fluids"]["contacts"]}
        self.assertAlmostEqual(c["GOC"], 2066.0, delta=1.0)
        self.assertAlmostEqual(c["OWC"], 2080.0, delta=1.0)

    def test_rw_estimate_reasonable(self):
        rw = self.s["saturation"]["rw_ref"]
        self.assertTrue(0.08 < rw < 0.16, rw)             # truth 0.12

    def test_porosity_and_sw_accuracy(self):
        d = self.df.DEPT.values
        t = self.truth
        for (a, b), sw_tol in (((2068, 2078), 0.10), ((2083, 2093), 0.15), ((2152, 2183), 0.15), ((2252, 2268), 0.12)):
            m = (d > a) & (d < b)
            tr = (t["depth"] > a) & (t["depth"] < b)
            self.assertAlmostEqual(self.df.PHIE[m].mean(), t["phie"][tr].mean(), delta=0.02)
            self.assertAlmostEqual(self.df.SW[m].mean(), t["sw"][tr].mean(), delta=sw_tol)

    def test_lithology(self):
        d = self.df.DEPT.values
        m = (d > 2152) & (d < 2183)
        self.assertEqual(self.df.LITH[m].mode()[0], "Sandstone")
        m = (d > 2216) & (d < 2218.5)
        self.assertEqual(self.df.LITH[m].mode()[0], "Limestone")
        m = (d > 2100) & (d < 2110)
        self.assertEqual(self.df.LITH[m].mode()[0], "Shale")
        self.assertGreater(self.s["lithology"]["pef_nd_agreement_pct"], 80)

    def test_qc_detects_planted_problems(self):
        qc = self.s["qc"]
        self.assertTrue(qc["environment"]["washout_intervals"])
        a, b = qc["environment"]["washout_intervals"][0]
        self.assertTrue(a < 2126 and b > 2120)
        self.assertGreaterEqual(qc["curves"]["GR"]["spikes"], 1)
        self.assertGreaterEqual(qc["curves"]["DT"]["flatline_samples"], 15)
        self.assertLess(qc["curves"]["DT"]["completeness_pct"], 100)

    def test_washout_not_called_reservoir(self):
        d = self.df.DEPT.values
        m = (d > 2121) & (d < 2125)
        self.assertTrue(set(self.df.FLUID_CODE[m]) <= {0, 5})

    def test_suitability(self):
        su = self.s["fluids"]["suitability"]
        self.assertTrue(su["suitable"])
        self.assertTrue(all(su["has"].values()))

    def test_summary_is_json_serialisable(self):
        json.dumps(self.s)

    def test_report_and_plots(self):
        secs = report.build_report(self.s)
        self.assertEqual(len(secs), 6)
        md = report.render_markdown(secs)
        for word in ("Log quality", "Lithology", "Fluid identification", "saturation", "Rw"):
            self.assertIn(word, md)
        self.assertTrue(plotting.log_plot(self.df, self.s["fluids"]["contacts"], "m", "t").startswith(b"\x89PNG"))
        self.assertTrue(plotting.crossplots(self.df, self.s).startswith(b"\x89PNG"))

    def test_html_is_escaped(self):
        secs = [{"id": "x", "title": "<script>", "blocks": [{"type": "text", "text": "<img src=x onerror=alert(1)>"}]}]
        out = report.render_html(secs)
        self.assertNotIn("<script>", out)
        self.assertNotIn("<img src=x", out)

    def test_user_rw_overrides_estimate(self):
        ev = evaluate(self.text, {**SYNTH_PARAMS, "rw": 0.12})
        self.assertEqual(ev.summary["saturation"]["rw_source"], "user supplied")
        self.assertAlmostEqual(ev.summary["saturation"]["rw_ref"], 0.12)

    def test_models_ordering_in_shaly_sand(self):
        # Archie ignores shale conductivity so it should give higher Sw than Indonesian in the shaly oil sand
        d = self.df.DEPT.values
        m = (d > 2253) & (d < 2267)
        self.assertGreater(self.df.SW_ARCHIE[m].mean(), self.df.SW_INDONESIA[m].mean())


class LasParser(unittest.TestCase):
    WRAPPED = """~V
 VERS. 2.0 : x
 WRAP. YES : wrapped
~W
 NULL. -999.25 : null
 WELL. W-1 : name
~C
 DEPT.FT : depth
 GR.GAPI : gr
 RT.OHMM : rt
 RHOB.G/C3 : rho
~A
 5000.0 50.0
 1.0 2.4
 5000.5 -999.25 2.0 2.5
 5001.0 60.0 3.0
 2.6
"""

    def test_wrapped_and_nulls(self):
        las = parse_las(self.WRAPPED)
        self.assertEqual(list(las.curves), ["DEPT", "GR", "RT", "RHOB"])
        self.assertEqual(len(las.depth), 3)
        self.assertTrue(np.isnan(las.curves["GR"][1]))
        self.assertEqual(las.depth[0], 5000.0)
        self.assertEqual(las.units["DEPT"], "FT")

    def test_rejects_garbage(self):
        with self.assertRaises(ValueError):
            parse_las("hello world")

    def test_bytes_input(self):
        las = parse_las(self.WRAPPED.encode())
        self.assertEqual(las.well["WELL"], "W-1")

    def test_duplicate_mnemonics_get_suffix(self):
        txt = "~V\n VERS. 2.0 : v\n~W\n NULL. -999.25 : n\n~C\n DEPT.M : d\n GR.GAPI : a\n GR.GAPI : b\n~A\n1 2 3\n2 4 5\n"
        las = parse_las(txt)
        self.assertIn("GR_2", las.curves)


class UnitsAndRobustness(unittest.TestCase):
    def _converted(self, **kw):
        """Same well, re-expressed in feet / percent / kg-m3 / conductivity, must give the same answer."""
        text, _ = make_synthetic_las()
        base = evaluate(text, SYNTH_PARAMS)
        las = parse_las(text)
        c = las.curves
        c["DEPT"] = c["DEPT"] / 0.3048
        las.units["DEPT"] = "FT"
        las.well_units["STRT"] = "FT"
        c["NPHI"] = c["NPHI"] * 100
        las.units["NPHI"] = "%"
        c["RHOB"] = c["RHOB"] * 1000
        las.units["RHOB"] = "KG/M3"
        c["CALI"] = c["CALI"] * 25.4
        las.units["CALI"] = "MM"
        c["BS"] = c["BS"] * 25.4
        las.units["BS"] = "MM"
        c["ILD"] = 1000.0 / c["ILD"]
        las.units["ILD"] = "MMHO/M"
        return base, evaluate(las, SYNTH_PARAMS)

    def test_unit_conversions_give_same_fluid_picture(self):
        base, conv = self._converted()
        self.assertEqual(conv.summary["depth_unit"], "ft")
        a = [(z["sand"], z["fluid"]) for z in base.summary["fluids"]["zones"]]
        b = [(z["sand"], z["fluid"]) for z in conv.summary["fluids"]["zones"]]
        self.assertEqual(a, b)
        for za, zb in zip(base.summary["fluids"]["zones"], conv.summary["fluids"]["zones"]):
            self.assertAlmostEqual(za["gross_m"], zb["gross_m"], delta=0.35)

    def test_missing_gr_raises_clear_error(self):
        text, _ = make_synthetic_las()
        las = parse_las(text)
        del las.curves["GR"]
        with self.assertRaisesRegex(ValueError, "GR"):
            evaluate(las, SYNTH_PARAMS)

    def test_missing_resistivity_raises(self):
        text, _ = make_synthetic_las()
        las = parse_las(text)
        for k in ("ILD", "MSFL"):
            del las.curves[k]
        with self.assertRaisesRegex(ValueError, "RT"):
            evaluate(las, SYNTH_PARAMS)

    def test_no_neutron_density_gives_undifferentiated_hc(self):
        text, _ = make_synthetic_las()
        las = parse_las(text)
        del las.curves["NPHI"]
        ev = evaluate(las, SYNTH_PARAMS)
        names = {z["fluid"] for z in ev.summary["fluids"]["zones"]}
        self.assertNotIn("Gas", names)
        self.assertNotIn("Oil", names)
        self.assertTrue(any("not separable" in n for n in names))

    def test_curve_map_override(self):
        text, _ = make_synthetic_las()
        las = parse_las(text)
        las.curves["MYRES"] = las.curves.pop("ILD")
        las.units["MYRES"] = "OHMM"
        ev = evaluate(las, {**SYNTH_PARAMS, "curve_map": {"RT": "MYRES"}})
        self.assertEqual(ev.summary["well"]["mapping"]["RT"], "MYRES")

    def test_bad_params_rejected(self):
        with self.assertRaises(ValueError):
            merge({"sat_model": "nonsense"})
        with self.assertRaises(ValueError):
            merge({"vsh_method": "nonsense"})

    def test_flat_gr_demands_manual_baselines(self):
        text, _ = make_synthetic_las()
        las = parse_las(text)
        las.curves["GR"] = np.full_like(las.curves["GR"], 60.0) + np.random.default_rng(1).normal(0, 1, len(las.depth))
        with self.assertRaisesRegex(ValueError, "baselines"):
            evaluate(las, SYNTH_PARAMS)


class Equations(unittest.TestCase):
    def test_archie_known_value(self):
        sw = saturation.archie_sw(np.array([20.0]), np.array([0.25]), 0.05, 1.0, 2.0, 2.0)
        self.assertAlmostEqual(float(sw[0]), (0.05 / (0.0625 * 20.0)) ** 0.5, places=6)

    def test_simandoux_reduces_to_archie_when_clean(self):
        rt, phi = np.array([15.0]), np.array([0.2])
        a = saturation.archie_sw(rt, phi, 0.08, 1, 2, 2)
        s = saturation.simandoux_sw(rt, phi, np.array([0.0]), 0.08, 2.0, 1, 2, 2)
        self.assertAlmostEqual(float(a[0]), float(s[0]), places=4)

    def test_indonesia_reduces_to_archie_when_clean(self):
        rt, phi = np.array([15.0]), np.array([0.2])
        a = saturation.archie_sw(rt, phi, 0.08, 1, 2, 2)
        s = saturation.indonesia_sw(rt, phi, np.array([0.0]), 0.08, 2.0, 1, 2, 2)
        self.assertAlmostEqual(float(a[0]), float(s[0]), places=4)

    def test_arps_temperature_conversion(self):
        # 1 ohm.m at 75 F -> at 175 F: (75+6.77)/(175+6.77)
        self.assertAlmostEqual(float(saturation.convert_res(1.0, 75, 175)), 81.77 / 181.77, places=4)

    def test_vsh_methods_bounded_and_ordered(self):
        gr = np.linspace(20, 120, 11)
        lin = lithology.vsh_from_gr(gr, 20, 120, "linear")
        for m in ("larionov_tertiary", "larionov_older", "clavier", "steiber"):
            v = lithology.vsh_from_gr(gr, 20, 120, m)
            self.assertTrue(np.all((v >= 0) & (v <= 1.0001)), m)
            self.assertTrue(np.all(v[1:-1] <= lin[1:-1] + 1e-9), m)     # all nonlinear methods are <= linear
        self.assertAlmostEqual(float(lin[0]), 0.0)
        self.assertAlmostEqual(float(lin[-1]), 1.0)


if __name__ == "__main__":
    unittest.main()
