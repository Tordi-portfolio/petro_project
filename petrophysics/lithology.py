"""Shale volume and lithology determination.

Logs used
- GR           : shale volume (Vsh) and sand/shale discrimination
- PEF          : primary sand / dolomite / limestone discriminator (if present)
- RHOB + NPHI  : neutron-density separation (limestone-scale NPHI) as secondary / fallback discriminator,
                 plus coal and anhydrite recognition
"""
from __future__ import annotations

import numpy as np

from .util import mode_filter, nanmean, runs

CODES = {
    0: "Undefined", 1: "Shale", 2: "Shaly sandstone", 3: "Sandstone", 4: "Limestone",
    5: "Dolomite", 6: "Coal", 7: "Anhydrite", 8: "Argillaceous carbonate",
}
CLEAN_CUT = 0.25   # Vsh below this: clean lithology call
SHALE_CUT = 0.50   # Vsh above this: shale

# Neutron (limestone-scale) -> matrix-scale offsets, used for density-neutron crossover/porosity
NPHI_OFFSET = {"sand": 0.04, "lime": 0.0, "dolo": -0.02}


def vsh_from_gr(gr: np.ndarray, gr_clean: float, gr_shale: float, method: str = "linear") -> np.ndarray:
    igr = np.clip((gr - gr_clean) / (gr_shale - gr_clean), 0.0, 1.0)
    if method == "linear":
        v = igr
    elif method == "larionov_tertiary":
        v = 0.083 * (2 ** (3.7 * igr) - 1)
    elif method == "larionov_older":
        v = 0.33 * (2 ** (2 * igr) - 1)
    elif method == "clavier":
        v = 1.7 - np.sqrt(np.maximum(3.38 - (igr + 0.7) ** 2, 0))
    elif method == "steiber":
        v = igr / (3.0 - 2.0 * igr)
    else:
        raise ValueError(method)
    return np.clip(v, 0.0, 1.0)


def gr_baselines(gr: np.ndarray, p: dict):
    fin = gr[np.isfinite(gr)]
    clean = p["gr_clean"] if p.get("gr_clean") is not None else float(np.percentile(fin, 5))
    shale = p["gr_shale"] if p.get("gr_shale") is not None else float(np.percentile(fin, 95))
    src = "user" if p.get("gr_clean") is not None or p.get("gr_shale") is not None else "5th/95th percentile of GR"
    if shale - clean < 20:
        raise ValueError(f"GR clean ({clean:.0f}) and shale ({shale:.0f}) baselines are too close; "
                         "supply gr_clean / gr_shale manually.")
    return clean, shale, src


def nd_separation(nphi_ls, rhob, rho_fluid):
    """NPHI(LS) - DPHI(LS): ~ -0.075 clean sandstone, ~0 limestone, ~+0.11 dolomite (zero porosity)."""
    dphi_ls = (2.71 - rhob) / (2.71 - rho_fluid)
    return nphi_ls - dphi_ls


def classify(data: dict, vsh: np.ndarray, p: dict, dz_m: np.ndarray, bad_hole: np.ndarray):
    n = len(vsh)
    rhob, nphi, pef = data.get("RHOB"), data.get("NPHI"), data.get("PEF")
    have_nd = rhob is not None and nphi is not None
    sep = nd_separation(nphi, rhob, p["rho_fluid"]) if have_nd else np.full(n, np.nan)

    # clean-lithology call from the two independent indicators
    lith_pef = np.zeros(n, int)
    if pef is not None:
        lith_pef = np.where(~np.isfinite(pef), 0, np.where(pef < 2.5, 3, np.where(pef < 4.0, 5, 4)))
    lith_nd = np.zeros(n, int)
    sep_shale = 0.0
    if have_nd:
        # shale plots on the carbonate side of the ND separation, so correct for Vsh before thresholding
        ref = np.isfinite(vsh) & (vsh > 0.8) & np.isfinite(sep) & ~bad_hole
        sep_shale = float(np.median(sep[ref])) if ref.sum() > 10 else 0.0
        sep_c = sep - np.nan_to_num(vsh) * sep_shale
        lith_nd = np.where(~np.isfinite(sep_c), 0, np.where(sep_c < -0.035, 3, np.where(sep_c < 0.06, 4, 5)))
    clean_l = np.where(lith_pef > 0, lith_pef, lith_nd)

    codes = np.zeros(n, int)
    has_gr = np.isfinite(vsh)
    clean = has_gr & (vsh < CLEAN_CUT)
    mid = has_gr & (vsh >= CLEAN_CUT) & (vsh < SHALE_CUT)
    shale = has_gr & (vsh >= SHALE_CUT)
    codes[shale] = 1
    codes[clean] = clean_l[clean]
    mid_carb = mid & np.isin(clean_l, (4, 5))
    codes[mid & ~mid_carb] = 2
    codes[mid_carb] = 8

    if have_nd:
        pef_ok_coal = np.ones(n, bool) if pef is None else (~np.isfinite(pef) | (pef < 2.0))
        coal = np.isfinite(rhob) & (rhob < 1.9) & (nphi > 0.30) & has_gr & (vsh < SHALE_CUT) & pef_ok_coal & ~bad_hole
        anh_pef = np.ones(n, bool) if pef is None else (~np.isfinite(pef) | (pef > 4.0))
        anh = np.isfinite(rhob) & (rhob > 2.92) & (nphi < 0.03) & has_gr & (vsh < 0.3) & anh_pef
        codes[coal] = 6
        codes[anh] = 7

    codes = mode_filter(codes, p["lith_smooth"])

    # agreement between PEF and ND calls on clean intervals
    both = clean & (lith_pef > 0) & (lith_nd > 0)
    agree = float((lith_pef[both] == lith_nd[both]).mean() * 100) if both.sum() > 20 else None

    # statistics
    stats = []
    total = float(np.nansum(dz_m)) or 1.0
    for code, name in CODES.items():
        m = codes == code
        if not m.any():
            continue
        th = float(dz_m[m].sum())
        stats.append({
            "code": code, "lithology": name, "thickness_m": round(th, 1), "pct": round(100 * th / total, 1),
            "GR": nanmean(data["GR"][m]) if "GR" in data else None,
            "RHOB": nanmean(rhob[m]) if rhob is not None else None,
            "NPHI": nanmean(nphi[m]) if nphi is not None else None,
            "PEF": nanmean(pef[m]) if pef is not None else None,
        })
    return {
        "codes": codes, "sep": sep, "lith_pef": lith_pef, "lith_nd": lith_nd,
        "stats": stats, "pef_nd_agreement_pct": agree,
        "methods": {
            "vsh": p["vsh_method"], "have_pef": pef is not None, "have_nd": have_nd,
        },
    }


def intervals(codes, depth, dz_m, data, vsh, min_thickness_m):
    out = []
    start = 0
    n = len(codes)
    for i in range(1, n + 1):
        if i == n or codes[i] != codes[start]:
            th = float(dz_m[start:i].sum())
            if th >= min_thickness_m and codes[start] != 0:
                sl = slice(start, i)
                out.append({
                    "top": float(depth[start]), "base": float(depth[i - 1]), "thickness_m": round(th, 1),
                    "lithology": CODES[int(codes[start])],
                    "GR": nanmean(data["GR"][sl]) if "GR" in data else None,
                    "vsh": nanmean(vsh[sl]),
                    "RHOB": nanmean(data["RHOB"][sl]) if "RHOB" in data else None,
                    "NPHI": nanmean(data["NPHI"][sl]) if "NPHI" in data else None,
                    "PEF": nanmean(data["PEF"][sl]) if "PEF" in data else None,
                })
            start = i
    return out
