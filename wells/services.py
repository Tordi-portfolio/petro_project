"""Glue between Django models and the framework-free petrophysics package."""
from __future__ import annotations

import base64

from petrophysics import plotting
from petrophysics.pipeline import Evaluation, evaluate

from .models import Analysis, Well


class AnalysisError(Exception):
    pass


def _read_las(well: Well) -> bytes:
    with well.las_file.open("rb") as fh:
        return fh.read()


def evaluate_well(well: Well, params: dict | None) -> Evaluation:
    try:
        return evaluate(_read_las(well), params or {})
    except ValueError as exc:
        raise AnalysisError(str(exc)) from exc


def run_analysis(well: Well, params: dict | None = None) -> Analysis:
    ev = evaluate_well(well, params)
    s = ev.summary
    title = f"{well.name} - composite evaluation"
    log_png = plotting.log_plot(ev.table, s["fluids"]["contacts"], s["depth_unit"], title, s["params"]["sep_gas"])
    xplots = plotting.crossplots(ev.table, s)
    suit = s["fluids"]["suitability"]
    return Analysis.objects.create(
        well=well,
        params=params or {},
        summary=s,
        images={"log_plot": base64.b64encode(log_png).decode(), "crossplots": base64.b64encode(xplots).decode()},
        suitable=bool(suit["suitable"]),
        score=int(suit["score"]),
    )
