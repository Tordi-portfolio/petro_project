"""Formation temperature, Rw determination, and water/hydrocarbon saturation models.

Models (a, m, n as set in parameters):
  Archie     Sw = ((a Rw) / (phi^m Rt))^(1/n)
  Simandoux  1/Rt = phi^m Sw^n / (a Rw (1-Vsh)) + Vsh Sw / Rsh          (solved numerically)
  Indonesian 1/sqrt(Rt) = [Vsh^(1-Vsh/2)/sqrt(Rsh) + phi^(m/2)/sqrt(a Rw)] Sw^(n/2)
Sxo uses Archie with Rxo and Rmf when both exist, otherwise the Sxo = Sw^0.2 rule of thumb.
"""
from __future__ import annotations

import numpy as np

ARPS_C = 6.77  # Arps constant for temperatures in deg F


def formation_temp_f(depth, depth_unit_m: float, p: dict):
    d_m = depth * depth_unit_m
    td_m = float(np.nanmax(d_m))
    if p.get("bht_f") is not None and td_m > 0:
        grad = (p["bht_f"] - p["surface_temp_f"]) / td_m
        src = f"linear between surface {p['surface_temp_f']:.0f} F and BHT {p['bht_f']:.0f} F at {td_m:.0f} m"
    else:
        grad = 1.6 / (100 * 0.3048)
        src = f"default gradient 1.6 F/100 ft from surface {p['surface_temp_f']:.0f} F (no BHT given)"
    return p["surface_temp_f"] + grad * d_m, src


def convert_res(r_ref, t_ref_f, t_f):
    return r_ref * (t_ref_f + ARPS_C) / (t_f + ARPS_C)


def _erode(mask, k):
    """Keep only samples whose +/-k neighbours are all True (bed interiors, away from shoulder effects)."""
    if k <= 0:
        return mask
    w = 2 * k + 1
    cnt = np.convolve(mask.astype(int), np.ones(w, dtype=int), mode="same")
    return mask & (cnt == w)


def estimate_rw(rt, phie, vsh, codes, gas_x, bad_hole, tf, p, step_m=0.1524):
    """Rwa method: lowest apparent water resistivity in clean, porous, non-gas sand ~ Rw.
    Only bed-interior samples (>= ~0.5 m from the bed edges) are used, because tool-resolution
    shoulder effects bias Rwa low. Returned at the reference temperature rw_temp_f."""
    rwa_f = rt * np.power(np.maximum(phie, 1e-3), p["m"]) / p["a"]
    rwa_ref = rwa_f * (tf + ARPS_C) / (p["rw_temp_f"] + ARPS_C)     # formation temp -> reference temp
    base = np.isfinite(rwa_ref) & np.isfinite(vsh) & ~bad_hole & ~gas_x & np.isin(codes, (2, 3))
    k = max(2, int(round(0.5 / max(step_m, 1e-6))))
    for vcut, pcut in ((0.15, 0.12), (0.25, 0.08)):
        cand = _erode(base & (vsh < vcut) & (phie > pcut), k)
        if cand.sum() >= 20:
            vals = np.sort(rwa_ref[cand])
            k = max(5, int(0.1 * len(vals)))
            return float(np.median(vals[:k])), int(cand.sum()), f"Vsh<{vcut}, PHIE>{pcut}"
    raise ValueError("Cannot estimate Rw: no clean porous sand found. Supply Rw in the parameters.")


def archie_sw(rt, phi, rw, a, m, n):
    with np.errstate(divide="ignore", invalid="ignore"):
        sw = np.power((a * rw) / (np.power(phi, m) * rt), 1.0 / n)
    sw = np.where(phi > 0.01, sw, np.nan)
    return np.clip(sw, 0, 1)


def simandoux_sw(rt, phi, vsh, rw, rsh, a, m, n):
    v = np.clip(vsh, 0, 0.95)
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.power(phi, m) / (a * rw * (1 - v))
        B = v / rsh
        target = 1.0 / rt
        lo = np.zeros_like(rt, dtype=float)
        hi = np.ones_like(rt, dtype=float)
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            f = A * np.power(mid, n) + B * mid - target
            lo = np.where(f < 0, mid, lo)
            hi = np.where(f >= 0, mid, hi)
        sw = 0.5 * (lo + hi)
        # if even Sw = 1 gives f < 0 the solution is > 1 -> clip to 1
        f1 = A + B - target
        sw = np.where(f1 < 0, 1.0, sw)
    return np.where((phi > 0.01) & np.isfinite(rt), np.clip(sw, 0, 1), np.nan)


def indonesia_sw(rt, phi, vsh, rw, rsh, a, m, n):
    v = np.clip(vsh, 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.power(v, 1 - v / 2) / np.sqrt(rsh) + np.power(phi, m / 2) / np.sqrt(a * rw)
        sw = np.power((1 / np.sqrt(rt)) / denom, 2.0 / n)
    return np.where((phi > 0.01) & np.isfinite(rt), np.clip(sw, 0, 1), np.nan)


def compute(depth, m_per_unit, rt, rxo, vsh, phie, codes, gas_x, bad_hole, p, step_m=0.1524):
    tf, temp_src = formation_temp_f(depth, m_per_unit, p)
    info = {"temperature_source": temp_src}

    if p["rw"] is not None:
        rw_ref, rw_src = p["rw"], "user supplied"
    else:
        rw_ref, ncand, crit = estimate_rw(rt, phie, vsh, codes, gas_x, bad_hole, tf, p, step_m)
        rw_src = (f"estimated (Rwa method: median of lowest 10% of Rwa over {ncand} bed-interior clean porous sand samples, {crit}). "
                  "Assumes a water-bearing sand is present in the logged interval; otherwise Rw is overestimated "
                  "and hydrocarbon saturation underestimated.")
    rw_f = convert_res(rw_ref, p["rw_temp_f"], tf)
    info.update({"rw_ref": float(rw_ref), "rw_temp_f": p["rw_temp_f"], "rw_source": rw_src})

    if p["rsh"] is not None:
        rsh, rsh_src = p["rsh"], "user supplied"
    else:
        sel = np.isfinite(rt) & np.isfinite(vsh) & (vsh > 0.8) & ~bad_hole
        rsh = float(np.nanmedian(rt[sel])) if sel.sum() > 10 else 2.0
        rsh_src = "median Rt where Vsh > 0.8" if sel.sum() > 10 else "default 2.0 ohm.m (no shale found)"
    info.update({"rsh": float(rsh), "rsh_source": rsh_src})

    a, m, n = p["a"], p["m"], p["n"]
    sw_arch = archie_sw(rt, phie, rw_f, a, m, n)
    sw_sim = simandoux_sw(rt, phie, vsh, rw_f, rsh, a, m, n)
    sw_ind = indonesia_sw(rt, phie, vsh, rw_f, rsh, a, m, n)
    sw = {"archie": sw_arch, "simandoux": sw_sim, "indonesia": sw_ind}[p["sat_model"]]

    if rxo is not None and p["rmf"] is not None and np.isfinite(rxo).any():
        rmf_f = convert_res(p["rmf"], p["rmf_temp_f"], tf)
        sxo = archie_sw(rxo, phie, rmf_f, a, m, n)
        info["sxo_method"] = "Archie with Rxo and Rmf (corrected to formation temperature)"
    else:
        sxo = np.power(np.clip(sw, 0, 1), 0.2)
        info["sxo_method"] = "Sxo = Sw^0.2 rule of thumb (no Rxo and/or Rmf available)"
    sxo = np.where(np.isfinite(sw), np.clip(np.fmax(sxo, sw), 0, 1), np.nan)

    return {"sw": sw, "sw_archie": sw_arch, "sw_simandoux": sw_sim, "sw_indonesia": sw_ind, "sxo": sxo,
            "bvw": phie * sw, "rw_f": rw_f, "tf": tf, "info": info}
