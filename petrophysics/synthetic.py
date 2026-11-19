"""Synthetic LAS generator for demos and tests.

The well has: Sand A with gas cap / oil leg / water leg (contacts at 2066 m and 2080 m), a water-bearing
Sand B, a tight limestone stringer, a shaly oil sand (Sand C), a washed-out shale interval, a GR spike,
a flat-lined DT segment and a null-filled DT top. Logs are generated with a forward model (Indonesian
Sw equation for resistivity) so this validates the *code path*, NOT field accuracy.
"""
from __future__ import annotations

import numpy as np

from . import saturation

RW_REF_F = 75.0
RW_REF = 0.12
RMF_REF = 0.35
RSH = 1.2
SURFACE_F, BHT_F, TD_M = 80.0, 210.0, 2300.0

LAYERS = [
    (2000, 2050, "shale", {}),
    (2050, 2066, "sand", dict(phi=0.27, vsh=0.06, fluid="gas", sw=0.25)),
    (2066, 2080, "sand", dict(phi=0.26, vsh=0.07, fluid="oil", sw=0.28)),
    (2080, 2095, "sand", dict(phi=0.25, vsh=0.08, fluid="water", sw=1.0)),
    (2095, 2150, "shale", {}),
    (2150, 2185, "sand", dict(phi=0.22, vsh=0.12, fluid="water", sw=1.0)),
    (2185, 2215, "shale", {}),
    (2215, 2219, "lime", dict(phi=0.03, vsh=0.03, fluid="water", sw=1.0)),
    (2219, 2250, "shale", {}),
    (2250, 2270, "sand", dict(phi=0.20, vsh=0.30, fluid="oil", sw=0.40)),
    (2270, 2300, "shale", {}),
]
FLUID_CODE = {"water": 1, "oil": 2, "gas": 3}
# flushed-zone state: (Sxo, fluid density, hydrogen index) of the filtrate/residual-HC mix
FLUSHED = {"water": (1.0, 1.0, 1.0), "oil": (0.75, 0.95, 1.0), "gas": (0.60, 0.68, 0.68)}
WASHOUT = (2120.0, 2126.0)


def _smooth(x, k=3):
    return np.convolve(np.pad(x, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


def make_synthetic_las(seed: int = 7, step: float = 0.1524):
    rng = np.random.default_rng(seed)
    depth = np.arange(2000.0, 2300.0, step)
    n = len(depth)
    vsh = np.ones(n); phi = np.zeros(n); sw = np.ones(n); sxo = np.ones(n)
    rho_ma = np.full(n, 2.65); off = np.full(n, -0.04); pef_ma = np.full(n, 1.85); dt_ma = np.full(n, 55.5)
    hi = np.ones(n); rho_fl = np.ones(n)
    truth_fluid = np.zeros(n, int)

    for top, base, kind, d in LAYERS:
        m = (depth >= top) & (depth < base)
        if kind == "shale":
            continue
        vsh[m] = d["vsh"] + rng.normal(0, 0.01, m.sum())
        phi[m] = d["phi"] + rng.normal(0, 0.006, m.sum())
        sw[m] = np.clip(d["sw"] + (rng.normal(0, 0.02, m.sum()) if d["sw"] < 1 else 0), 0.05, 1.0)
        sxo_v, rf, h = FLUSHED[d["fluid"]]
        sxo[m], rho_fl[m], hi[m] = sxo_v, rf, h
        if kind == "lime":
            rho_ma[m], off[m], pef_ma[m], dt_ma[m] = 2.71, 0.0, 5.08, 47.5
        truth_fluid[m] = FLUID_CODE[d["fluid"]] if d["phi"] > 0.08 else 0
    vsh = np.clip(vsh, 0, 1); phi = np.clip(phi, 0, 0.4)

    # ---- forward model ----
    rhob = (1 - vsh) * (rho_ma * (1 - phi) + rho_fl * phi) + vsh * 2.55
    nphi = (1 - vsh) * (phi * hi + off) + vsh * 0.35
    pef = (1 - vsh) * pef_ma + vsh * 3.4
    dt = (1 - vsh) * (dt_ma * (1 - phi) + 189.0 * phi) + vsh * 100.0
    gr = 25 + 85 * vsh
    phie = phi * (1 - vsh)

    p = {"surface_temp_f": SURFACE_F, "bht_f": BHT_F}
    tf, _ = saturation.formation_temp_f(depth, 1.0, {**p, "surface_temp_f": SURFACE_F}) if False else (
        SURFACE_F + (BHT_F - SURFACE_F) / TD_M * depth, "")
    rw_f = saturation.convert_res(RW_REF, RW_REF_F, tf)
    rmf_f = saturation.convert_res(RMF_REF, RW_REF_F, tf)

    def rt_forward(rw, s):
        inv = (np.power(vsh, 1 - vsh / 2) / np.sqrt(RSH) + np.power(phie, 1.0) / np.sqrt(rw)) * np.power(s, 1.0)
        return 1.0 / inv ** 2

    rt = rt_forward(rw_f, sw)
    rxo = rt_forward(rmf_f, sxo)

    cali = np.full(n, 8.5) + rng.normal(0, 0.04, n)
    cali = np.where(vsh > 0.9, cali + 0.2, cali - 0.1)
    bs = np.full(n, 8.5)
    drho = rng.normal(0.0, 0.015, n)

    # ---- environmental damage / data problems ----
    w = (depth >= WASHOUT[0]) & (depth < WASHOUT[1])
    cali[w] = 13.0 + rng.normal(0, 0.3, w.sum())
    rhob[w] = 2.15 + rng.normal(0, 0.03, w.sum())
    nphi[w] = 0.45 + rng.normal(0, 0.02, w.sum())
    drho[w] = 0.22 + rng.normal(0, 0.02, w.sum())
    pef[w] = 2.8

    # tool resolution + noise
    rhob = _smooth(rhob) + rng.normal(0, 0.008, n)
    nphi = _smooth(nphi) + rng.normal(0, 0.005, n)
    pef = _smooth(pef) + rng.normal(0, 0.08, n)
    dt = _smooth(dt) + rng.normal(0, 1.2, n)
    gr = _smooth(gr) + rng.normal(0, 2.0, n)
    rt = 10 ** (_smooth(np.log10(rt), 5) + rng.normal(0, 0.012, n))
    rxo = 10 ** (_smooth(np.log10(rxo), 3) + rng.normal(0, 0.012, n))

    gr[np.argmin(np.abs(depth - 2075.0))] += 90.0                           # single-sample GR spike
    f0 = np.argmin(np.abs(depth - 2100.0)); dt[f0:f0 + 22] = dt[f0]         # flat-lined sonic
    dt[:10] = np.nan                                                        # null-filled top

    curves = {"DEPT": depth, "GR": gr, "CALI": cali, "BS": bs, "RHOB": rhob, "NPHI": nphi, "PEF": pef,
              "DT": dt, "DRHO": drho, "ILD": rt, "MSFL": rxo}
    units = {"DEPT": "M", "GR": "GAPI", "CALI": "IN", "BS": "IN", "RHOB": "G/C3", "NPHI": "V/V", "PEF": "B/E",
             "DT": "US/F", "DRHO": "G/C3", "ILD": "OHMM", "MSFL": "OHMM"}
    desc = {"DEPT": "Depth", "GR": "Gamma ray", "CALI": "Caliper", "BS": "Bit size", "RHOB": "Bulk density",
            "NPHI": "Neutron porosity (limestone)", "PEF": "Photoelectric factor", "DT": "Sonic transit time",
            "DRHO": "Density correction", "ILD": "Deep induction", "MSFL": "Micro-spherical log"}

    lines = [
        "~VERSION INFORMATION", " VERS.   2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0", " WRAP.   NO  : ONE LINE PER DEPTH STEP",
        "~WELL INFORMATION",
        f" STRT.M  {depth[0]:.4f} : START DEPTH", f" STOP.M  {depth[-1]:.4f} : STOP DEPTH", f" STEP.M  {step:.4f} : STEP",
        " NULL.   -999.25 : NULL VALUE", " COMP.   SYNTHETIC OPERATOR : COMPANY", " WELL.   SYN-1 : WELL",
        " FLD.    DEMO FIELD : FIELD", " UWI.    SYN0001 : UNIQUE WELL ID",
        "~CURVE INFORMATION",
    ]
    for k in curves:
        lines.append(f" {k}.{units[k]}   : {desc[k]}")
    lines += ["~PARAMETER INFORMATION", " BHT.DEGF  210 : BOTTOM HOLE TEMP", " RMF.OHMM  0.35 : MUD FILTRATE RESISTIVITY @75F",
              "~ASCII"]
    for i in range(n):
        row = []
        for k in curves:
            v = curves[k][i]
            row.append("-999.25" if not np.isfinite(v) else f"{v:.4f}")
        lines.append(" ".join(row))
    truth = {"depth": depth, "fluid": truth_fluid, "sw": sw, "phie": phie, "vsh": vsh,
             "contacts": {"GOC": 2066.0, "OWC": 2080.0}, "rw_ref": RW_REF, "rmf_ref": RMF_REF}
    return "\n".join(lines) + "\n", truth


SYNTH_PARAMS = {"surface_temp_f": SURFACE_F, "bht_f": BHT_F, "rmf": RMF_REF}
