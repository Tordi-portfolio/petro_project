from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(x).rolling(window, center=True, min_periods=max(2, window // 2)).median().to_numpy()


def mode_filter(codes: np.ndarray, window: int) -> np.ndarray:
    """Sliding-window majority vote on integer codes (edges handled by padding with edge values)."""
    if window <= 1:
        return codes.copy()
    window = window + 1 if window % 2 == 0 else window
    half = window // 2
    padded = np.pad(codes, half, mode="edge")
    stack = np.stack([padded[i:i + len(codes)] for i in range(window)])
    uniq = np.unique(codes)
    counts = np.stack([(stack == u).sum(axis=0) for u in uniq])
    # tie -> keep the original sample's code
    best = uniq[counts.argmax(axis=0)]
    maxc = counts.max(axis=0)
    orig_idx = np.searchsorted(uniq, codes)
    orig_count = counts[orig_idx, np.arange(len(codes))]
    return np.where(orig_count == maxc, codes, best)


def runs(mask: np.ndarray):
    """Yield (start, stop_exclusive) index pairs of consecutive True values."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return
    d = np.diff(np.concatenate(([0], m.astype(int), [0])))
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        yield int(s), int(e)


def nanmean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else None


def jsonable(obj):
    """Recursively convert numpy/NaN values so json.dumps works."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
