"""End-to-end evaluation: LAS -> QC -> Vsh -> lithology -> porosity -> Rw -> Sw/Sxo -> fluids -> zones."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import fluids, lithology, porosity, qc, saturation
from .curves import standardise
from .las import LASData, parse_las
from .params import merge
from .util import jsonable


@dataclass
class Evaluation:
    summary: dict
    table: pd.DataFrame
    arrays: dict = field(default_factory=dict)

    def csv(self) -> str:
        return self.table.to_csv(index=False, float_format="%.4f")


def evaluate(source, user_params: dict | None = None) -> Evaluation:
    las = source if isinstance(source, LASData) else parse_las(source)
    p = merge(user_params)
    sc = standardise(las, p["curve_map"])

    for need, why in (("GR", "shale volume / lithology"), ("RT", "resistivity-based saturation")):
        if not sc.has(need):
            raise ValueError(f"Required curve {need} ({why}) was not found or is empty. "
                             f"Curves in file: {', '.join(las.curves)}. Use the curve-mapping parameter to point to it.")
    if not (sc.has("RHOB") or sc.has("NPHI") or sc.has("DT")):
        raise ValueError("No porosity log (RHOB, NPHI or DT) found.")

    depth = sc.depth
    n = len(depth)
    m_u = sc.m_per_unit
    dz_m = np.abs(np.gradient(depth)) * m_u if n > 1 else np.ones(n)
    dz_m = np.where(np.isfinite(dz_m), dz_m, 0.0)
    step_m = float(np.median(dz_m[dz_m > 0])) if (dz_m > 0).any() else 0.1524

    # 1 - quality control
    qc_res, qflags, data = qc.run_qc(sc, p)
    bad_hole = qflags["bad_hole"]

    # 2 - shale volume
    gr_clean, gr_shale, gr_src = lithology.gr_baselines(data["GR"], p)
    vsh = lithology.vsh_from_gr(data["GR"], gr_clean, gr_shale, p["vsh_method"])

    # 3 - lithology
    lith = lithology.classify(data, vsh, p, dz_m, bad_hole)
    codes = lith["codes"]

    # 4 - porosity
    window = int(max(3, min(11, round(p["min_bed_m"] / max(step_m, 1e-6)))))
    por = porosity.compute(data, vsh, codes, bad_hole, p, window)
    have_nd = por["have"]["density"] and por["have"]["neutron"]

    # 5 - saturation
    rxo = data.get("RXO")
    sat = saturation.compute(depth, m_u, data["RT"], rxo, vsh, por["phie"], codes, por["gas_x"], bad_hole, p,
                             step_m)

    # 6 - fluids, zones, contacts
    fcodes, reservoir = fluids.classify(vsh, por["phie"], sat["sw"], por["gas_x"], have_nd, codes,
                                        bad_hole, dz_m, p, step_m)
    zones, contacts = fluids.build_zones(fcodes, depth, dz_m, data, vsh, por["phie"], sat, sat["sxo"],
                                         sat["bvw"], data["RT"], codes, p)
    suit = fluids.suitability(zones, contacts, qc_res, p)

    # per-sample table
    cols = {"DEPT": depth}
    for k in ("GR", "CALI", "BS", "RHOB", "NPHI", "PEF", "DRHO", "DT", "RT", "RXO", "RS", "SP"):
        if k in data:
            cols[k] = data[k]
    cols.update({
        "VSH": vsh, "LITH_CODE": codes, "LITH": [lithology.CODES[int(c)] for c in codes],
        "DPHI": por["dphi"], "NPHI_M": por["nphi_m"], "ND_SEP": por["sep"],
        "PHIT": por["phit"], "PHIE": por["phie"],
        "RW_F": sat["rw_f"], "SW": sat["sw"], "SW_ARCHIE": sat["sw_archie"],
        "SW_SIMANDOUX": sat["sw_simandoux"], "SW_INDONESIA": sat["sw_indonesia"],
        "SXO": sat["sxo"], "BVW": sat["bvw"], "SH": 1 - sat["sw"],
        "FLUID_CODE": fcodes, "FLUID": [fluids.FLUID[int(c)] for c in fcodes],
        "BAD_HOLE": bad_hole.astype(int),
    })
    table = pd.DataFrame(cols)

    # fluid totals
    totals = {}
    for c in (3, 2, 6, 4, 1):
        zs = [z for z in zones if z["fluid_code"] == c]
        if zs:
            g = sum(z["gross_m"] for z in zs)
            totals[fluids.FLUID[c]] = {
                "gross_m": round(g, 1), "net_pay_m": round(sum(z["net_pay_m"] for z in zs), 1),
                "avg_phie": float(np.average([z["phie"] for z in zs], weights=[z["gross_m"] for z in zs])),
                "avg_sw": float(np.average([z["sw"] for z in zs], weights=[z["gross_m"] for z in zs])),
                "hcpv_m": round(sum(z["hcpv_m"] or 0 for z in zs), 3) if c in fluids.HC_CODES else None,
                "n_zones": len(zs),
            }

    lith_intervals = lithology.intervals(codes, depth, dz_m, data, vsh, max(p["min_bed_m"], 1.5))
    well_info = {
        "name": las.well_value("WELL", "WELL NAME", default="(unnamed)"),
        "field": las.well_value("FLD", "FIELD"), "company": las.well_value("COMP"),
        "uwi": las.well_value("UWI", "API"), "las_version": las.version,
        "curves": [{"mnemonic": k, "unit": las.units.get(k, ""), "description": las.descriptions.get(k, "")}
                   for k in las.curves],
        "mapping": sc.mapping, "las_warnings": las.warnings,
    }

    params_used = {k: v for k, v in p.items() if k != "curve_map"}
    params_used.update({
        "gr_clean_used": gr_clean, "gr_shale_used": gr_shale, "gr_baseline_source": gr_src,
        "rw_used_ref": sat["info"]["rw_ref"], "rsh_used": sat["info"]["rsh"],
    })

    summary = {
        "well": well_info,
        "qc": qc_res,
        "vsh": {"method": p["vsh_method"], "gr_clean": gr_clean, "gr_shale": gr_shale, "baseline_source": gr_src},
        "lithology": {
            "stats": lith["stats"], "intervals": lith_intervals,
            "pef_nd_agreement_pct": lith["pef_nd_agreement_pct"], "methods": lith["methods"],
        },
        "porosity": {"method": por["method"], "phit_shale": por["phit_shale"], "have": por["have"],
                     "gas_crossover_pct": round(100.0 * float(por["gas_x"].mean()), 1)},
        "saturation": {"model": p["sat_model"], **sat["info"], "a": p["a"], "m": p["m"], "n": p["n"]},
        "fluids": {"zones": zones, "contacts": contacts, "totals": totals, "have_nd": have_nd,
                   "suitability": suit},
        "params": params_used,
        "depth_unit": sc.depth_unit,
    }
    arrays = {"depth": depth, "codes": codes, "fcodes": fcodes}
    return Evaluation(summary=jsonable(summary), table=table, arrays=arrays)
