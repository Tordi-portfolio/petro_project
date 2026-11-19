"""Porosity: density, neutron (matrix-corrected), combined density-neutron with gas correction,
shale correction, and sonic fallback."""
from __future__ import annotations

import numpy as np

from .lithology import NPHI_OFFSET
from .util import rolling_median

SAND, LIME, DOLO = "sand", "lime", "dolo"


def matrix_arrays(codes: np.ndarray, p: dict):
    rho = np.full(len(codes), p["rho_sand"])
    dt = np.full(len(codes), p["dt_sand"])
    off = np.full(len(codes), NPHI_OFFSET[SAND])
    carb = np.isin(codes, (4, 8))
    dol = codes == 5
    rho[carb], dt[carb], off[carb] = p["rho_lime"], p["dt_lime"], NPHI_OFFSET[LIME]
    rho[dol], dt[dol], off[dol] = p["rho_dolo"], p["dt_dolo"], NPHI_OFFSET[DOLO]
    anh = codes == 7
    rho[anh], dt[anh], off[anh] = 2.96, 50.0, 0.0
    return rho, dt, off


def compute(data: dict, vsh: np.ndarray, codes: np.ndarray, bad_hole: np.ndarray, p: dict, window: int):
    n = len(vsh)
    rho_ma, dt_ma, off = matrix_arrays(codes, p)
    rhob, nphi, dt = data.get("RHOB"), data.get("NPHI"), data.get("DT")
    have_d = rhob is not None and np.isfinite(rhob).any()
    have_n = nphi is not None and np.isfinite(nphi).any()
    have_s = dt is not None and np.isfinite(dt).any()

    dphi_raw = (rho_ma - rhob) / (rho_ma - p["rho_fluid"]) if have_d else np.full(n, np.nan)
    nphi_m = nphi + off if have_n else np.full(n, np.nan)
    sep = dphi_raw - nphi_m            # >0 -> density porosity exceeds neutron: gas-like crossover
    sep_s = rolling_median(sep, window) if (have_d and have_n) else np.full(n, np.nan)
    gas_x = np.isfinite(sep_s) & (sep_s > p["sep_gas"])

    dphi = np.clip(dphi_raw, 0, 0.6)
    nphim = np.clip(nphi_m, 0, 0.6)

    if have_d and have_n:
        liquid = (dphi + nphim) / 2
        gas = np.sqrt((dphi ** 2 + nphim ** 2) / 2)
        phit_raw = np.where(gas_x, gas, liquid)
        method = ("Density-neutron average, matrix set by lithology; RMS (gas-corrected) form used where "
                  "density-neutron crossover exceeds the gas threshold")
    elif have_d:
        phit_raw = dphi
        method = "Density porosity only (no neutron), matrix set by lithology"
    elif have_n:
        phit_raw = nphim
        method = "Neutron porosity only (no density), matrix-corrected"
    elif have_s:
        phit_raw = np.clip(0.625 * (dt - dt_ma) / dt, 0, 0.5)
        method = "Raymer-Hunt-Gardner sonic porosity (no density/neutron)"
    else:
        raise ValueError("No porosity-capable log (RHOB, NPHI or DT) in the file.")

    shale_ref = np.isfinite(vsh) & (vsh > 0.8) & np.isfinite(phit_raw) & ~bad_hole
    phit_sh = float(np.nanmedian(phit_raw[shale_ref])) if shale_ref.sum() > 10 else 0.0
    phie = np.clip(phit_raw - np.nan_to_num(vsh) * phit_sh, 0, 0.45)
    phie = np.where(np.isfinite(phit_raw), phie, np.nan)

    sonic = None
    if have_s and (have_d or have_n):
        sonic = np.clip(0.625 * (dt - dt_ma) / dt, 0, 0.5)

    return {
        "dphi": dphi_raw, "nphi_m": nphi_m, "sep": sep, "sep_s": sep_s, "gas_x": gas_x,
        "phit": phit_raw, "phie": phie, "phit_shale": phit_sh, "method": method,
        "rho_ma": rho_ma, "sonic": sonic,
        "have": {"density": have_d, "neutron": have_n, "sonic": have_s},
    }
