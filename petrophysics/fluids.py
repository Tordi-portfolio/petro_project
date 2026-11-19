"""Fluid identification (gas / oil / water / transition), reservoir zoning, contact picks,
net-pay summary and well-suitability screening.

Evidence used
- Resistivity-derived Sw (via the selected saturation model): low Sw => hydrocarbon, high Sw => water
- Density-neutron crossover (NPHI below DPHI on a common matrix scale): gas vs oil among hydrocarbon zones
"""
from __future__ import annotations

import numpy as np

from .lithology import CODES as LITH
from .util import mode_filter, runs

FLUID = {
    0: "Non-reservoir", 1: "Water", 2: "Oil", 3: "Gas", 4: "Transition / uncertain",
    5: "Unreliable (bad hole)", 6: "Hydrocarbon (oil/gas not separable)",
}
HC_CODES = (2, 3, 6)


def _wavg(x, w, mask):
    m = mask & np.isfinite(x)
    if not m.any() or w[m].sum() == 0:
        return None
    return float(np.sum(x[m] * w[m]) / np.sum(w[m]))


def classify(vsh, phie, sw, gas_x, have_nd, lith, bad_hole, dz_m, p, step_m):
    n = len(vsh)
    valid = np.isfinite(vsh) & np.isfinite(phie) & np.isfinite(sw)
    res = valid & (vsh <= p["vsh_cut"]) & (phie >= p["phi_cut"]) & ~np.isin(lith, (6, 7))
    codes = np.zeros(n, int)
    codes[res & (sw >= p["sw_water_min"])] = 1
    codes[res & (sw < p["sw_water_min"]) & (sw > p["sw_hc_max"])] = 4
    hc = res & (sw <= p["sw_hc_max"])
    if have_nd:
        codes[hc & gas_x] = 3
        codes[hc & ~gas_x] = 2
    else:
        codes[hc] = 6
    if p["exclude_bad_hole"]:
        codes[bad_hole & np.isfinite(vsh) & (vsh <= p["vsh_cut"])] = 5
        codes[bad_hole & ~(np.isfinite(vsh) & (vsh <= p["vsh_cut"]))] = 0

    window = int(max(3, min(11, round(p["min_bed_m"] / max(step_m, 1e-6)))))
    codes = mode_filter(codes, window)

    # drop zones thinner than the minimum bed thickness
    for c in np.unique(codes):
        if c == 0:
            continue
        for s, e in runs(codes == c):
            if dz_m[s:e].sum() < p["min_bed_m"]:
                codes[s:e] = 0
    return codes, res


def build_zones(codes, depth, dz_m, data, vsh, phie, sw_all, sxo, bvw, rt, lith, p):
    sw = sw_all["sw"]
    pay = (np.isin(codes, HC_CODES) & (vsh <= p["vsh_cut"]) & (phie >= p["phi_cut"]) & (sw <= p["sw_pay_cut"]))

    # group non-zero code runs into sands (small zero-gaps allowed)
    runs_list = []
    cur = None
    for s, e in runs(codes != 0):
        if cur and dz_m[cur[1]:s].sum() <= p["min_bed_m"]:
            cur[1] = e
        else:
            if cur:
                runs_list.append(tuple(cur))
            cur = [s, e]
    if cur:
        runs_list.append(tuple(cur))

    zones, contacts = [], []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for si, (s0, s1) in enumerate(runs_list):
        sand_id = f"Sand {letters[si]}" if si < 26 else f"Sand {si + 1}"
        # sub-zones by code within the sand
        i = s0
        sub = []
        while i < s1:
            c = codes[i]
            j = i
            while j < s1 and codes[j] == c:
                j += 1
            if c != 0:
                sub.append((i, j, int(c)))
            i = j
        for (i0, i1, c) in sub:
            sl = slice(i0, i1)
            w = dz_m[sl]
            m_all = np.ones(i1 - i0, bool)
            lith_vals, lith_counts = np.unique(lith[sl], return_counts=True)
            zone = {
                "sand": sand_id, "fluid": FLUID[c], "fluid_code": c,
                "top": float(depth[i0]), "base": float(depth[i1 - 1]),
                "gross_m": round(float(w.sum()), 2),
                "net_pay_m": round(float(dz_m[sl][pay[sl]].sum()), 2),
                "lithology": LITH[int(lith_vals[lith_counts.argmax()])],
                "vsh": _wavg(vsh[sl], w, m_all), "phie": _wavg(phie[sl], w, m_all),
                "sw": _wavg(sw[sl], w, m_all), "sxo": _wavg(sxo[sl], w, m_all),
                "bvw": _wavg(bvw[sl], w, m_all), "rt": _wavg(rt[sl], w, m_all),
                "sw_archie": _wavg(sw_all["sw_archie"][sl], w, m_all),
                "sw_simandoux": _wavg(sw_all["sw_simandoux"][sl], w, m_all),
                "sw_indonesia": _wavg(sw_all["sw_indonesia"][sl], w, m_all),
                "gr": _wavg(data["GR"][sl], w, m_all) if "GR" in data else None,
            }
            zone["ng"] = round(zone["net_pay_m"] / zone["gross_m"], 2) if zone["gross_m"] else None
            if zone["phie"] is not None and zone["sw"] is not None and c in HC_CODES:
                zone["hcpv_m"] = round(float(np.sum(phie[sl] * (1 - sw[sl]) * w)), 3)   # phi*(1-Sw)*h  [m]
            else:
                zone["hcpv_m"] = None
            zones.append(zone)
        # contacts between adjacent sub-zones
        for (a0, a1, ca), (b0, b1, cb) in zip(sub[:-1], sub[1:]):
            depth_c = 0.5 * (depth[a1 - 1] + depth[b0])
            pair = (ca, cb)
            label = None
            if pair == (3, 2):
                label = "GOC (gas-oil contact)"
            elif pair == (2, 1):
                label = "OWC (oil-water contact)"
            elif pair == (3, 1):
                label = "GWC (gas-water contact)"
            elif pair == (6, 1):
                label = "HC-water contact"
            elif ca in HC_CODES and cb == 4:
                label = "Base of clean hydrocarbon column / top of transition"
            elif ca == 4 and cb == 1:
                label = "Base of transition (apparent water level)"
            elif ca == 1 and cb in HC_CODES:
                label = "Water above hydrocarbon - inverted order (check: separate sands, or erroneous Rw/lithology)"
            if label:
                contacts.append({"sand": sand_id, "contact": label, "depth": float(depth_c)})
    return zones, contacts


def suitability(zones, contacts, qc, p):
    min_t = p["min_bed_m"]
    thick = {c: sum(z["gross_m"] for z in zones if z["fluid_code"] == c) for c in (1, 2, 3, 6)}
    has = {"gas": thick[3] >= min_t, "oil": thick[2] >= min_t, "water": thick[1] >= min_t}
    checks = []
    score = 0
    for k, v in has.items():
        checks.append({"check": f"{k.capitalize()} zone identified (>= {min_t:g} m)", "pass": bool(v)})
        score += 2 if v else 0
    n_contact = sum(1 for c in contacts if c["contact"].startswith(("GOC", "OWC", "GWC")))
    checks.append({"check": "At least one fluid contact (GOC/OWC/GWC) picked within a single sand", "pass": n_contact > 0})
    score += 1 if n_contact else 0
    key_ok = all(c in qc["curves"] and qc["curves"][c]["grade"] != "Poor" for c in ("GR", "RHOB", "NPHI", "RT"))
    checks.append({"check": "GR, RHOB, NPHI and RT present with acceptable quality", "pass": key_ok})
    score += 2 if key_ok else 0
    bh = qc["environment"].get("bad_hole_pct", 0)
    checks.append({"check": "Bad-hole fraction below 10%", "pass": bh < 10})
    score += 1 if bh < 10 else 0
    return {
        "suitable": bool(all(has.values()) and key_ok),
        "score": score, "max_score": 9, "has": has,
        "thickness_by_fluid_m": {"gas": round(thick[3], 1), "oil": round(thick[2], 1),
                                 "water": round(thick[1], 1), "hc_undiff": round(thick[6], 1)},
        "checks": checks,
    }
