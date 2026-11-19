from django import forms
from django.conf import settings

from petrophysics.las import parse_las
from petrophysics.params import CHOICES, DEFAULTS

from .models import Well


class WellUploadForm(forms.ModelForm):
    name = forms.CharField(max_length=200, required=False, help_text="Optional - defaults to the WELL name inside the LAS file.")

    class Meta:
        model = Well
        fields = ["name", "las_file"]

    def clean_las_file(self):
        f = self.cleaned_data["las_file"]
        limit = getattr(settings, "MAX_LAS_UPLOAD_BYTES", 100 * 1024 * 1024)
        if f.size > limit:
            raise forms.ValidationError(f"File is larger than {limit // (1024 * 1024)} MB.")
        raw = f.read()
        f.seek(0)
        try:
            self.parsed = parse_las(raw)
        except ValueError as exc:
            raise forms.ValidationError(f"Not a readable LAS file: {exc}")
        return f

    def save(self, commit=True):
        well = super().save(commit=False)
        las = self.parsed
        if not well.name:
            well.name = las.well_value("WELL", "WELL NAME", default=self.cleaned_data["las_file"].name)
        well.curve_names = list(las.curves)
        depth = las.depth
        well.top, well.base = float(depth.min()), float(depth.max())
        idx_unit = (las.units.get(las.index_name, "") or "").upper()
        well.depth_unit = "ft" if idx_unit.startswith("F") else "m"
        if commit:
            well.save()
        return well


def _f(label, help_text="", initial=None):
    return forms.FloatField(label=label, required=False, help_text=help_text, initial=initial)


class ParametersForm(forms.Form):
    """Interpretation parameters. Blank = use the default / automatic estimate."""

    # (group title, [field names]) - rendered in this order
    groups = [
        ("Saturation (Archie / shaly sand)", ["sat_model", "a", "m", "n", "rw", "rw_temp_f", "rmf", "rmf_temp_f", "rsh"]),
        ("Formation temperature", ["surface_temp_f", "bht_f"]),
        ("Shale volume", ["vsh_method", "gr_clean", "gr_shale"]),
        ("Matrix and fluid", ["rho_sand", "rho_lime", "rho_dolo", "rho_fluid"]),
        ("Cutoffs and classification", ["vsh_cut", "phi_cut", "sw_hc_max", "sw_water_min", "sw_pay_cut", "sep_gas", "min_bed_m"]),
        ("Quality control", ["wash_thr_in", "drho_thr", "exclude_bad_hole", "despike"]),
        ("Curve mapping (optional)", ["curve_map_text"]),
    ]

    sat_model = forms.ChoiceField(choices=[(c, c.capitalize()) for c in CHOICES["sat_model"]], initial=DEFAULTS["sat_model"],
                                  label="Saturation model")
    a = _f("Tortuosity factor a", initial=DEFAULTS["a"])
    m = _f("Cementation exponent m", initial=DEFAULTS["m"])
    n = _f("Saturation exponent n", initial=DEFAULTS["n"])
    rw = _f("Rw (ohm.m)", "Leave blank to estimate from the lowest clean wet-sand Rwa (assumes a water leg exists).")
    rw_temp_f = _f("Rw temperature (deg F)", initial=DEFAULTS["rw_temp_f"])
    rmf = _f("Rmf (ohm.m)", "Needed for Sxo from Rxo; blank = Sxo = Sw^0.2 fallback.")
    rmf_temp_f = _f("Rmf temperature (deg F)", initial=DEFAULTS["rmf_temp_f"])
    rsh = _f("Rsh (ohm.m)", "Blank = median Rt where Vsh > 0.8.")
    surface_temp_f = _f("Surface temperature (deg F)", initial=DEFAULTS["surface_temp_f"])
    bht_f = _f("Bottom-hole temperature (deg F)", "At the deepest logged depth. Blank = 1.6 F/100 ft gradient.")
    vsh_method = forms.ChoiceField(choices=[(c, c.replace("_", " ").capitalize()) for c in CHOICES["vsh_method"]],
                                   initial=DEFAULTS["vsh_method"], label="Vsh method")
    gr_clean = _f("GR clean sand (gAPI)", "Blank = 5th percentile of GR.")
    gr_shale = _f("GR shale (gAPI)", "Blank = 95th percentile of GR.")
    rho_sand = _f("Sandstone matrix density (g/cc)", initial=DEFAULTS["rho_sand"])
    rho_lime = _f("Limestone matrix density (g/cc)", initial=DEFAULTS["rho_lime"])
    rho_dolo = _f("Dolomite matrix density (g/cc)", initial=DEFAULTS["rho_dolo"])
    rho_fluid = _f("Fluid density (g/cc)", "Mud-filtrate density in the flushed zone.", initial=DEFAULTS["rho_fluid"])
    vsh_cut = _f("Vsh cutoff", initial=DEFAULTS["vsh_cut"])
    phi_cut = _f("PHIE cutoff", initial=DEFAULTS["phi_cut"])
    sw_hc_max = _f("Sw <= this: hydrocarbon", initial=DEFAULTS["sw_hc_max"])
    sw_water_min = _f("Sw >= this: water", initial=DEFAULTS["sw_water_min"])
    sw_pay_cut = _f("Net pay Sw cutoff", initial=DEFAULTS["sw_pay_cut"])
    sep_gas = _f("Gas crossover threshold (v/v)", initial=DEFAULTS["sep_gas"])
    min_bed_m = _f("Minimum bed thickness (m)", initial=DEFAULTS["min_bed_m"])
    wash_thr_in = _f("Washout threshold (in over bit)", initial=DEFAULTS["wash_thr_in"])
    drho_thr = _f("|DRHO| threshold (g/cc)", initial=DEFAULTS["drho_thr"])
    exclude_bad_hole = forms.BooleanField(required=False, initial=DEFAULTS["exclude_bad_hole"],
                                          label="Exclude bad-hole samples from fluid classification")
    despike = forms.BooleanField(required=False, initial=DEFAULTS["despike"], label="Remove isolated spikes")
    curve_map_text = forms.CharField(required=False, label="Curve mapping",
                                     help_text="Comma-separated CANONICAL=MNEMONIC, e.g. RT=LLD, RHOB=RHOZ, NPHI=TNPH. "
                                               "Canonical names: GR, CALI, BS, RHOB, NPHI, DT, PEF, DRHO, RT, RXO, RS, SP.")

    def clean_curve_map_text(self):
        text = self.cleaned_data.get("curve_map_text", "").strip()
        mapping = {}
        if text:
            for token in text.replace(";", ",").split(","):
                if not token.strip():
                    continue
                if "=" not in token:
                    raise forms.ValidationError(f"'{token.strip()}' is not in CANONICAL=MNEMONIC form.")
                k, v = (x.strip().upper() for x in token.split("=", 1))
                mapping[k] = v
        self._curve_map = mapping
        return text

    def grouped(self):
        return [(title, [self[name] for name in names]) for title, names in self.groups]

    def to_params(self) -> dict:
        data = {k: v for k, v in self.cleaned_data.items() if k != "curve_map_text"}
        data["curve_map"] = getattr(self, "_curve_map", {})
        return data
