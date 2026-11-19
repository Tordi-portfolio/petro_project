"""Log quality analysis: completeness, ranges, spikes, flat-lines, depth issues,
borehole/environmental effects (washout, mudcake, DRHO), depth-matching and
cross-curve consistency. Returns a JSON-friendly dict plus per-sample flag arrays."""
from __future__ import annotations

import numpy as np

from .util import rolling_median, runs

PLAUSIBLE = {
    "GR": (0, 300), "CALI": (3, 30), "BS": (3, 30), "RHOB": (1.0, 3.2), "NPHI": (-0.05, 0.8),
    "DT": (30, 250), "PEF": (0, 12), "DRHO": (-0.5, 0.5), "RT": (0.01, 20000),
    "RXO": (0.01, 20000), "RS": (0.01, 20000), "SP": (-300, 300),
}
KEY_CURVES = ("GR", "RHOB", "NPHI", "RT")
LOG_CURVES = ("RT", "RXO", "RS")


def _spikes(x: np.ndarray, window: int, log: bool):
    y = np.log10(x) if log else x
    med = rolling_median(y, window)
    resid = y - med
    good = np.isfinite(resid)
    if good.sum() < 20:
        return np.zeros(len(x), bool)
    mad = np.nanmedian(np.abs(resid[good] - np.nanmedian(resid[good])))
    scale = 1.4826 * mad
    if scale <= 0:
        scale = np.nanstd(resid[good]) * 0.5
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros(len(x), bool)
    return good & (np.abs(resid) > 8 * scale)


def depth_lag(x: np.ndarray, y: np.ndarray, max_lag: int = 20):
    """Best integer-sample lag aligning y to x (positive: y must move shallower... reported as samples)."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 100:
        return 0, 0.0, 0.0
    xs = np.where(ok, (x - np.nanmean(x[ok])) / (np.nanstd(x[ok]) or 1), 0.0)
    ys = np.where(ok, (y - np.nanmean(y[ok])) / (np.nanstd(y[ok]) or 1), 0.0)
    best, best_c, c0 = 0, -2.0, 0.0
    for lag in range(-max_lag, max_lag + 1):
        a = xs[max(0, lag): len(xs) + min(0, lag)]
        b = ys[max(0, -lag): len(ys) + min(0, -lag)]
        n = min(len(a), len(b))
        c = float(np.dot(a[:n], b[:n]) / max(n, 1))
        if lag == 0:
            c0 = c
        if abs(c) > abs(best_c) or best_c == -2.0:
            best, best_c = lag, c
    return best, best_c, c0


def run_qc(sc, p: dict):
    """sc: StdCurves. p: params. Returns (qc_dict, flags_dict, cleaned_data_dict)."""
    n = len(sc.depth)
    issues: list[dict] = []

    def issue(sev, msg):
        issues.append({"severity": sev, "message": msg})

    # ---------- depth ----------
    d = sc.depth
    dd = np.diff(d)
    depth_info = {
        "top": float(np.nanmin(d)), "base": float(np.nanmax(d)), "step": sc.step,
        "unit": sc.depth_unit, "samples": int(n),
        "monotonic": bool(np.all(dd > 0)),
    }
    if not depth_info["monotonic"]:
        n_bad = int((dd <= 0).sum())
        issue("high", f"Depth index is not strictly increasing ({n_bad} reversals/duplicates); "
                      "interval statistics and thicknesses may be wrong.")
    irregular = int((np.abs(dd - sc.step) > 0.05 * max(sc.step, 1e-9)).sum()) if sc.step else 0
    depth_info["irregular_steps"] = irregular
    if irregular > 0.01 * n:
        issue("medium", f"{irregular} irregular depth steps (>5% from the median step {sc.step:g} {sc.depth_unit}).")
    for note in sc.notes:
        issue("info", note)

    # ---------- per curve ----------
    cleaned = {k: v.copy() for k, v in sc.data.items()}
    curve_qc: dict[str, dict] = {}
    spike_flags: dict[str, np.ndarray] = {}
    for name, arr in sc.data.items():
        finite = np.isfinite(arr)
        info = {
            "mnemonic": sc.mapping.get(name, name),
            "completeness_pct": round(100.0 * finite.mean(), 1) if n else 0.0,
            "min": float(np.nanmin(arr)) if finite.any() else None,
            "max": float(np.nanmax(arr)) if finite.any() else None,
            "mean": float(np.nanmean(arr)) if finite.any() else None,
        }
        lo, hi = PLAUSIBLE.get(name, (-np.inf, np.inf))
        out_rng = finite & ((arr < lo) | (arr > hi))
        info["out_of_range"] = int(out_rng.sum())
        info["invalid_nulled"] = int(sc.invalid.get(name, 0))
        notes = []

        if finite.sum() > 30:
            sp = _spikes(arr, p["spike_window"], name in LOG_CURVES)
            info["spikes"] = int(sp.sum())
            spike_flags[name] = sp
            if p["despike"] and sp.any():
                med = rolling_median(np.log10(arr) if name in LOG_CURVES else arr, p["spike_window"])
                repl = 10 ** med if name in LOG_CURVES else med
                cleaned[name] = np.where(sp, repl, arr)
                notes.append(f"{int(sp.sum())} spike(s) replaced by running median")

            # flat-lines (stuck tool / filled data)
            same = np.zeros(n, bool)
            same[1:] = finite[1:] & finite[:-1] & (np.diff(arr) == 0)
            flat_total, flat_segments = 0, []
            for s, e in runs(same):
                if e - s + 1 >= p["flatline_samples"]:
                    flat_total += e - s + 1
                    flat_segments.append((float(d[s]), float(d[min(e, n - 1)])))
            info["flatline_samples"] = flat_total
            info["flatline_segments"] = flat_segments[:5]
        else:
            info["spikes"] = 0
            info["flatline_samples"] = 0
            info["flatline_segments"] = []

        # grade
        pts = 0
        reasons = []
        if info["completeness_pct"] < 60:
            pts += 3; reasons.append(f"only {info['completeness_pct']}% complete")
        elif info["completeness_pct"] < 95:
            pts += 1; reasons.append(f"{info['completeness_pct']}% complete")
        if finite.any() and info["out_of_range"] > 0.01 * finite.sum():
            pts += 2; reasons.append(f"{info['out_of_range']} samples outside plausible range")
        elif info["out_of_range"]:
            pts += 1; reasons.append(f"{info['out_of_range']} samples outside plausible range")
        if info["spikes"] > 0.005 * max(finite.sum(), 1):
            pts += 1; reasons.append(f"{info['spikes']} spikes")
        if info["flatline_samples"] > 0.02 * max(finite.sum(), 1):
            pts += 2; reasons.append(f"{info['flatline_samples']} samples in flat-line runs")
        elif info["flatline_samples"]:
            pts += 1; reasons.append(f"{info['flatline_samples']} samples in flat-line runs")
        info["grade"] = "Good" if pts == 0 else ("Fair" if pts <= 2 else "Poor")
        info["comments"] = reasons + notes
        curve_qc[name] = info

    # ---------- missing key curves ----------
    missing = [c for c in KEY_CURVES if c not in sc.data or not np.isfinite(sc.data[c]).any()]
    for c in missing:
        issue("high" if c in ("RT", "GR") else "medium",
              f"Key curve {c} not found - interpretation steps that need it are degraded or skipped.")
    if "RHOB" not in sc.data and "DT" not in sc.data:
        issue("high", "Neither RHOB nor DT available: porosity cannot be computed.")
    if "PEF" not in sc.data:
        issue("info", "No PEF curve: lithology relies on density-neutron separation only.")

    # ---------- borehole / environmental ----------
    env: dict = {}
    bad_hole = np.zeros(n, bool)
    washout = np.zeros(n, bool)
    cali = cleaned.get("CALI")
    if cali is not None and np.isfinite(cali).any():
        if "BS" in cleaned and np.isfinite(cleaned["BS"]).any():
            bit = np.where(np.isfinite(cleaned["BS"]), cleaned["BS"], np.nanmedian(cleaned["BS"]))
            env["bit_size_source"] = "BS curve"
        else:
            fin = cali[np.isfinite(cali)]
            est = float(np.round(np.percentile(fin, 25) * 4) / 4)   # lower quartile ~ in-gauge hole
            bit = np.full(n, est)
            env["bit_size_source"] = f"estimated from caliper lower quartile ({est:.2f} in)"
        env["bit_size_in"] = float(np.nanmedian(bit))
        diff = cali - bit
        washout = np.isfinite(diff) & (diff > p["wash_thr_in"])
        severe = np.isfinite(diff) & (diff > 3.0)
        mudcake = np.isfinite(diff) & (diff < -0.3)
        step = sc.step or 1.0
        env["washout_samples"] = int(washout.sum())
        env["washout_pct"] = round(100.0 * washout.sum() / n, 1)
        env["severe_washout_pct"] = round(100.0 * severe.sum() / n, 1)
        env["mudcake_pct"] = round(100.0 * mudcake.sum() / n, 1)
        env["washout_intervals"] = [(float(d[s]), float(d[min(e, n - 1)])) for s, e in runs(washout)
                                    if (e - s) * step * sc.m_per_unit >= 0.3][:10]
        bad_hole |= washout
        if env["washout_pct"] > 15:
            issue("high", f"Extensive washout: {env['washout_pct']}% of the logged interval has caliper > bit + "
                          f"{p['wash_thr_in']} in. Density, neutron and PEF are suspect there.")
        elif env["washout_pct"] > 0:
            issue("medium", f"Washout over {env['washout_pct']}% of the interval "
                            f"({len(env['washout_intervals'])} interval(s) >= 0.3 m); density/neutron/PEF flagged unreliable.")
        if env["mudcake_pct"] > 5:
            issue("info", f"Caliper under gauge over {env['mudcake_pct']}% of the interval, indicating mudcake on permeable beds "
                          "(supports permeability; small effect on density).")
    else:
        issue("medium", "No caliper: borehole condition cannot be assessed directly; relying on DRHO only.")

    if "DRHO" in cleaned and np.isfinite(cleaned["DRHO"]).any():
        drho_bad = np.isfinite(cleaned["DRHO"]) & (np.abs(cleaned["DRHO"]) > p["drho_thr"])
        env["drho_bad_pct"] = round(100.0 * drho_bad.sum() / n, 1)
        bad_hole |= drho_bad
        if env["drho_bad_pct"] > 5:
            issue("medium", f"|DRHO| exceeds {p['drho_thr']} g/cc over {env['drho_bad_pct']}% of the interval: "
                            "density correction is large, bulk density unreliable.")
    env["bad_hole_pct"] = round(100.0 * bad_hole.sum() / n, 1)

    # ---------- resistivity separation (invasion) ----------
    inv = {}
    shallow = "RXO" if "RXO" in cleaned else ("RS" if "RS" in cleaned else None)
    if shallow and "RT" in cleaned:
        ok = np.isfinite(cleaned[shallow]) & np.isfinite(cleaned["RT"])
        if ok.any():
            ratio = np.log10(cleaned[shallow][ok] / cleaned["RT"][ok])
            inv["shallow_curve"] = sc.mapping.get(shallow, shallow)
            inv["median_log10_shallow_over_deep"] = float(np.median(ratio))
            inv["separation_pct"] = round(100.0 * float((np.abs(ratio) > 0.1).mean()), 1)
            if shallow == "RS":
                issue("info", f"No micro-resistivity (Rxo): {inv['shallow_curve']} is used only as a shallow-resistivity "
                              "indicator; it is not corrected for invasion and is not used for Sxo.")
    env["invasion"] = inv

    # ---------- depth matching & consistency ----------
    checks = []
    if "GR" in cleaned and "RHOB" in cleaned:
        lag, c, c0 = depth_lag(cleaned["GR"], cleaned["RHOB"])
        checks.append({"pair": "GR vs RHOB", "best_lag_samples": lag, "corr_best": round(c, 2), "corr_zero": round(c0, 2)})
        if lag != 0 and abs(c) - abs(c0) > 0.1:
            shift = abs(lag) * sc.step
            issue("medium", f"Possible depth mismatch between GR and RHOB: best correlation at {lag:+d} samples "
                            f"(~{shift:.2f} {sc.depth_unit}). Check depth shifting.")
    if "GR" in cleaned and "NPHI" in cleaned:
        lag, c, c0 = depth_lag(cleaned["GR"], cleaned["NPHI"])
        checks.append({"pair": "GR vs NPHI", "best_lag_samples": lag, "corr_best": round(c, 2), "corr_zero": round(c0, 2)})
        if lag != 0 and abs(c) - abs(c0) > 0.1:
            issue("medium", f"Possible depth mismatch between GR and NPHI at {lag:+d} samples.")
    if "RHOB" in cleaned and "NPHI" in cleaned:
        ok = np.isfinite(cleaned["RHOB"]) & np.isfinite(cleaned["NPHI"])
        if ok.sum() > 50:
            # informational only: in shaly sequences RHOB and NPHI legitimately correlate positively
            c = float(np.corrcoef(cleaned["RHOB"][ok], cleaned["NPHI"][ok])[0, 1])
            checks.append({"pair": "RHOB vs NPHI", "corr": round(c, 2)})
        nph = cleaned["NPHI"][np.isfinite(cleaned["NPHI"])]
        if len(nph) and np.nanmax(nph) > 0.6:
            issue("info", f"NPHI reaches {np.nanmax(nph):.2f} v/v: check for washout/bound-water effects in shales.")
    if "GR" in cleaned and np.isfinite(cleaned["GR"]).any():
        g = cleaned["GR"][np.isfinite(cleaned["GR"])]
        contrast = float(np.percentile(g, 95) - np.percentile(g, 5))
        checks.append({"pair": "GR contrast (P95-P5)", "value": round(contrast, 1)})
        if contrast < 30:
            issue("medium", f"Low GR contrast ({contrast:.0f} gAPI P95-P5): sand/shale discrimination from GR is weak.")

    # ---------- overall ----------
    grades = [curve_qc[c]["grade"] for c in KEY_CURVES if c in curve_qc]
    if missing or "Poor" in grades or env.get("bad_hole_pct", 0) > 30:
        overall = "Poor"
    elif "Fair" in grades or env.get("bad_hole_pct", 0) > 10:
        overall = "Fair"
    else:
        overall = "Good"

    qc = {
        "overall": overall, "depth": depth_info, "curves": curve_qc, "environment": env,
        "consistency": checks, "issues": issues, "missing_key_curves": missing,
    }
    flags = {"bad_hole": bad_hole, "washout": washout}
    return qc, flags, cleaned
