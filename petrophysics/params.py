"""Interpretation parameters with defaults. Every value used in a run is echoed in the report."""
from __future__ import annotations

DEFAULTS = {
    # --- Archie / saturation ---
    "a": 1.0,
    "m": 2.0,
    "n": 2.0,
    "sat_model": "indonesia",        # archie | simandoux | indonesia
    "rw": None,                      # ohm.m at rw_temp_f; None -> estimated from Rwa (lowest clean wet sand)
    "rw_temp_f": 75.0,
    "rmf": None,                     # ohm.m at rmf_temp_f; None -> Sxo from Sxo = Sw**0.2 fallback
    "rmf_temp_f": 75.0,
    "rsh": None,                     # ohm.m; None -> median Rt where Vsh > 0.8
    # --- temperature ---
    "surface_temp_f": 80.0,
    "bht_f": None,                   # bottom-hole temp at max depth; None -> 1.6 F/100 ft gradient
    # --- shale volume ---
    "vsh_method": "linear",          # linear | larionov_tertiary | larionov_older | clavier | steiber
    "gr_clean": None,                # None -> 5th percentile of GR
    "gr_shale": None,                # None -> 95th percentile of GR
    # --- matrix / fluid ---
    "rho_sand": 2.65,
    "rho_lime": 2.71,
    "rho_dolo": 2.87,
    "rho_fluid": 1.0,                # mud-filtrate density in the flushed zone
    "dt_sand": 55.5,
    "dt_lime": 47.5,
    "dt_dolo": 43.5,
    # --- cutoffs ---
    "vsh_cut": 0.40,
    "phi_cut": 0.08,
    "sw_hc_max": 0.65,               # Sw <= this -> hydrocarbon-bearing
    "sw_water_min": 0.85,            # Sw >= this -> water-bearing; in between -> transition/uncertain
    "sw_pay_cut": 0.60,              # net pay Sw cutoff
    "sep_gas": 0.04,                 # density-neutron crossover (v/v) for gas
    "min_bed_m": 0.9,                # minimum bed / zone thickness (m)
    # --- QC / environment ---
    "wash_thr_in": 1.5,              # caliper - bit size (in) flagged as washout
    "drho_thr": 0.10,                # |DRHO| (g/cc) above which density is unreliable
    "exclude_bad_hole": True,
    "despike": True,
    "spike_window": 5,
    "flatline_samples": 15,
    "lith_smooth": 7,
    # --- optional curve overrides ---
    "curve_map": {},
}

CHOICES = {
    "sat_model": ["archie", "simandoux", "indonesia"],
    "vsh_method": ["linear", "larionov_tertiary", "larionov_older", "clavier", "steiber"],
}


def merge(user: dict | None) -> dict:
    p = dict(DEFAULTS)
    for k, v in (user or {}).items():
        if k in p and v not in (None, ""):
            p[k] = v
    for k in ("a", "m", "n", "rw", "rw_temp_f", "rmf", "rmf_temp_f", "rsh", "surface_temp_f", "bht_f",
              "gr_clean", "gr_shale", "rho_sand", "rho_lime", "rho_dolo", "rho_fluid", "dt_sand", "dt_lime",
              "dt_dolo", "vsh_cut", "phi_cut", "sw_hc_max", "sw_water_min", "sw_pay_cut", "sep_gas",
              "min_bed_m", "wash_thr_in", "drho_thr"):
        if p[k] is not None:
            p[k] = float(p[k])
    for k in ("spike_window", "flatline_samples", "lith_smooth"):
        p[k] = int(p[k])
    if p["sat_model"] not in CHOICES["sat_model"]:
        raise ValueError(f"sat_model must be one of {CHOICES['sat_model']}")
    if p["vsh_method"] not in CHOICES["vsh_method"]:
        raise ValueError(f"vsh_method must be one of {CHOICES['vsh_method']}")
    return p
