"""Dependency-free LAS 1.2 / 2.0 reader (wrapped or unwrapped data).

Returns a LASData object. Null values (from ~W NULL and common defaults) become NaN.
LAS 3.0 files are not supported (a warning is raised).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

# MNEM.UNIT   VALUE : DESCRIPTION
_LINE = re.compile(r"^\s*([^.:]*?)\s*\.([^\s:]*)\s*(.*?)\s*:\s*(.*)$")
_NULL_DEFAULTS = (-999.25, -999.0, -9999.0, -99999.0)


@dataclass
class LASData:
    version: str = "2.0"
    wrap: bool = False
    well: dict = field(default_factory=dict)         # MNEM -> value string
    well_units: dict = field(default_factory=dict)
    curves: dict = field(default_factory=dict)       # name -> np.ndarray (first = depth/index)
    units: dict = field(default_factory=dict)
    descriptions: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)       # ~P section: MNEM -> value string
    null: float = -999.25
    warnings: list = field(default_factory=list)

    @property
    def index_name(self) -> str:
        return next(iter(self.curves))

    @property
    def depth(self) -> np.ndarray:
        return self.curves[self.index_name]

    def well_value(self, *keys, default=""):
        for k in keys:
            if k in self.well and str(self.well[k]).strip():
                return str(self.well[k]).strip()
        return default


def _to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_las(source) -> LASData:
    """Parse LAS from text, bytes or a file path."""
    if isinstance(source, (bytes, bytearray)):
        text = bytes(source).decode("utf-8", errors="replace")
    elif os.path.isfile(str(source)) if "\n" not in str(source) and len(str(source)) < 4096 else False:
        with open(source, "rb") as fh:
            text = fh.read().decode("utf-8", errors="replace")
    else:
        text = str(source)

    las = LASData()
    section = None
    curve_names: list[str] = []
    ascii_lines: list[str] = []
    seen: dict[str, int] = {}

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("~"):
            section = stripped[1:2].upper()
            continue
        if section == "A":
            ascii_lines.append(stripped)
            continue
        if stripped.startswith("#") or section not in {"V", "W", "C", "P"}:
            continue
        m = _LINE.match(line)
        if not m:
            las.warnings.append(f"Unparsed header line: {stripped[:60]}")
            continue
        mnem, unit, value, desc = (g.strip() for g in m.groups())
        if section == "V":
            if mnem.upper() == "VERS":
                las.version = value
            elif mnem.upper() == "WRAP":
                las.wrap = value.upper().startswith("Y")
        elif section == "W":
            las.well[mnem.upper()] = value
            las.well_units[mnem.upper()] = unit
        elif section == "C":
            key = mnem.upper()
            key = re.sub(r":\d+$", "", key)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > 1:
                key = f"{key}_{seen[key]}"
            curve_names.append(key)
            las.units[key] = unit.upper()
            las.descriptions[key] = desc
        elif section == "P":
            las.params[mnem.upper()] = value

    if las.version.startswith("3"):
        las.warnings.append("LAS 3.0 detected: only LAS 1.2/2.0 layouts are supported; results may be wrong.")
    if not curve_names:
        raise ValueError("No ~Curve section found - is this a LAS file?")
    if not ascii_lines:
        raise ValueError("No ~ASCII data section found.")

    null = _to_float(las.well.get("NULL", ""))
    las.null = null if null is not None else -999.25

    tokens = " ".join(ascii_lines).replace(",", " ").split()
    values = np.empty(len(tokens), dtype=float)
    bad = 0
    for i, t in enumerate(tokens):
        f = _to_float(t)
        if f is None:
            values[i] = np.nan
            bad += 1
        else:
            values[i] = f
    if bad:
        las.warnings.append(f"{bad} non-numeric data tokens replaced with null.")

    ncol = len(curve_names)
    usable = (len(values) // ncol) * ncol
    if usable != len(values):
        las.warnings.append(
            f"Data length ({len(values)}) is not a multiple of curve count ({ncol}); "
            f"trailing {len(values) - usable} values dropped.")
    data = values[:usable].reshape(-1, ncol)

    # null replacement (never touch the index column)
    null_candidates = {las.null, *_NULL_DEFAULTS}
    for j, name in enumerate(curve_names):
        col = data[:, j].copy()
        if j > 0:
            for nv in null_candidates:
                col[np.isclose(col, nv, atol=1e-6)] = np.nan
        las.curves[name] = col
    return las
