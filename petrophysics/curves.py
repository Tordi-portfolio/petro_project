"""Curve-name aliasing and unit standardisation.

Canonical names: GR, CALI, BS, RHOB, NPHI, DT, PEF, DRHO, RT, RXO, RS, SP
Canonical units: GR gAPI, CALI/BS inches, RHOB g/cc, NPHI v/v (limestone units),
DT us/ft, PEF b/e, resistivities ohm.m, depth in the LAS native unit (m or ft).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ALIASES = {
    "GR": ["GR", "GRC", "GRD", "GAMM", "SGR", "GR_EDTC", "GRN", "CGR", "GAPI"],
    "CALI": ["CALI", "CAL", "CAL1", "CALS", "CALX", "HCAL", "DCAL", "C1", "CAL_1"],
    "BS": ["BS", "BIT", "BITSIZE", "BIT_SIZE", "BSZ"],
    "RHOB": ["RHOB", "RHOZ", "DEN", "DENS", "ZDEN", "RHOM", "DENB", "RHOB_1"],
    "NPHI": ["NPHI", "TNPH", "NPOR", "NEU", "CNL", "NPL", "PHIN", "NPHI_LS", "NPHI_1"],
    "DT": ["DT", "DTC", "AC", "DT24", "DTCO", "SONIC", "DT4P", "DT_1"],
    "PEF": ["PEF", "PE", "PEFZ", "PEFL", "PEF8"],
    "DRHO": ["DRHO", "DRH", "HDRA", "ZCOR", "DCOR"],
    "RT": ["RT", "RDEEP", "ILD", "LLD", "RILD", "RD", "RLLD", "AT90", "AO90", "RLA5",
           "RT90", "HDRS", "M2R9", "RESD", "RDEP", "LLDC", "ILDC", "RT_HRLT", "RES_DEEP"],
    "RXO": ["RXO", "MSFL", "MCFL", "RXOZ", "RXO8", "SFL", "SFLU"],
    "RS": ["LLS", "RLLS", "RS", "AT10", "AT20", "AO10", "AO20", "RLA2", "RSHAL", "M2R1", "ILM", "RILM", "AT60"],
    "SP": ["SP", "SSP", "SPC", "SPBL"],
}

REQUIRED_FOR_FULL_EVAL = ("GR", "RHOB", "NPHI", "RT")


@dataclass
class StdCurves:
    depth: np.ndarray
    depth_unit: str                      # 'm' or 'ft'
    data: dict = field(default_factory=dict)      # canonical -> array (canonical units)
    mapping: dict = field(default_factory=dict)   # canonical -> original mnemonic
    notes: list = field(default_factory=list)     # unit conversions etc.
    invalid: dict = field(default_factory=dict)   # canonical -> count of physically invalid samples nulled

    @property
    def step(self) -> float:
        d = np.diff(self.depth[np.isfinite(self.depth)])
        d = d[d > 0]
        return float(np.median(d)) if len(d) else 0.0

    @property
    def m_per_unit(self) -> float:
        return 0.3048 if self.depth_unit == "ft" else 1.0

    def has(self, *names) -> bool:
        return all(n in self.data and np.isfinite(self.data[n]).any() for n in names)


def _find(las_curves: dict, canonical: str, override: str | None):
    if override and override.upper() in las_curves:
        return override.upper()
    names = list(las_curves)
    for alias in ALIASES[canonical]:
        if alias in names:
            return alias
    return None


def standardise(las, curve_map: dict | None = None) -> StdCurves:
    curve_map = {k.upper(): v for k, v in (curve_map or {}).items()}
    depth = las.depth.astype(float)
    idx_unit = las.units.get(las.index_name, "") or las.well_units.get("STRT", "")
    depth_unit = "ft" if idx_unit.upper().startswith(("F", "FT")) else "m"
    out = StdCurves(depth=depth, depth_unit=depth_unit)
    if not idx_unit:
        out.notes.append("Depth unit not stated in LAS; assumed metres.")

    for canon in ALIASES:
        src = _find(las.curves, canon, curve_map.get(canon))
        if src is None:
            continue
        arr = las.curves[src].astype(float).copy()
        unit = las.units.get(src, "").upper()
        out.mapping[canon] = src

        if canon in ("CALI", "BS"):
            if unit.startswith("MM") or (np.isfinite(arr).any() and np.nanmedian(arr) > 40):
                arr = arr / 25.4
                out.notes.append(f"{src} converted mm -> inches.")
        elif canon == "RHOB":
            if np.isfinite(arr).any() and np.nanmedian(arr) > 100:
                arr = arr / 1000.0
                out.notes.append(f"{src} converted kg/m3 -> g/cc.")
        elif canon == "DRHO":
            if np.isfinite(arr).any() and np.nanmedian(np.abs(arr)) > 5:
                arr = arr / 1000.0
                out.notes.append(f"{src} converted kg/m3 -> g/cc.")
        elif canon == "NPHI":
            finite = arr[np.isfinite(arr)]
            if unit in ("%", "PU", "PERC") or (len(finite) and np.percentile(finite, 99) > 1.5):
                arr = arr / 100.0
                out.notes.append(f"{src} converted percent/pu -> v/v.")
        elif canon == "DT":
            if unit.startswith(("US/M", "USEC/M")) or (np.isfinite(arr).any() and np.nanmedian(arr) > 350):
                arr = arr * 0.3048
                out.notes.append(f"{src} converted us/m -> us/ft.")
        elif canon in ("RT", "RXO", "RS"):
            if "MMHO" in unit or unit.startswith(("MS/M", "MHO")):
                with np.errstate(divide="ignore", invalid="ignore"):
                    arr = np.where(arr > 0, 1000.0 / arr, np.nan)
                out.notes.append(f"{src} converted conductivity -> resistivity.")
            bad = np.isfinite(arr) & (arr <= 0)
            if bad.any():
                out.invalid[canon] = int(bad.sum())
                arr[bad] = np.nan
        out.data[canon] = arr
    return out
