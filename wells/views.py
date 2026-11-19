from django.contrib import messages
from django.http import Http404, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.safestring import mark_safe
from django.views.decorators.http import require_POST

from petrophysics import report

from .forms import ParametersForm, WellUploadForm
from .models import Analysis, Well
from .services import AnalysisError, evaluate_well, run_analysis


def well_list(request):
    if request.method == "POST":
        form = WellUploadForm(request.POST, request.FILES)
        if form.is_valid():
            well = form.save()
            messages.success(request, f"Uploaded {well.name} ({len(well.curve_names)} curves).")
            return redirect(well)
    else:
        form = WellUploadForm()
    wells = list(Well.objects.prefetch_related("analyses"))
    ranked = sorted((w for w in wells if w.latest_analysis), key=lambda w: (-w.latest_analysis.score, w.name))
    return render(request, "wells/well_list.html", {"form": form, "wells": wells, "ranked": ranked})


def well_detail(request, pk):
    well = get_object_or_404(Well, pk=pk)
    if request.method == "POST":
        form = ParametersForm(request.POST)
        if form.is_valid():
            try:
                analysis = run_analysis(well, form.to_params())
            except AnalysisError as exc:
                messages.error(request, f"Analysis failed: {exc}")
            else:
                return redirect(analysis)
    else:
        form = ParametersForm()
    return render(request, "wells/well_detail.html", {"well": well, "form": form, "analyses": well.analyses.all()[:10]})


def analysis_detail(request, pk):
    analysis = get_object_or_404(Analysis.objects.select_related("well"), pk=pk)
    sections = report.build_report(analysis.summary)
    body = mark_safe(report.render_html(sections, analysis.images))   # all text is escaped inside render_html
    return render(request, "wells/analysis_detail.html", {"analysis": analysis, "report_html": body,
                                                          "suit": analysis.summary["fluids"]["suitability"]})


def analysis_download(request, pk, fmt):
    analysis = get_object_or_404(Analysis.objects.select_related("well"), pk=pk)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in analysis.well.name)
    if fmt == "csv":
        try:
            ev = evaluate_well(analysis.well, analysis.params)
        except AnalysisError as exc:
            raise Http404(str(exc))
        resp = HttpResponse(ev.csv(), content_type="text/csv")
    elif fmt in ("html", "md"):
        sections = report.build_report(analysis.summary)
        title = f"Petrophysical evaluation - {analysis.well.name}"
        if fmt == "html":
            resp = HttpResponse(report.standalone_html(sections, analysis.images, title), content_type="text/html")
        else:
            resp = HttpResponse(report.render_markdown(sections, title), content_type="text/markdown")
    else:
        raise Http404("Unknown format")
    resp["Content-Disposition"] = f'attachment; filename="{safe_name}_evaluation.{fmt}"'
    return resp


@require_POST
def screen_all(request):
    """Run the default evaluation on every well without an analysis, so wells can be ranked by suitability."""
    done = failed = 0
    for well in Well.objects.all():
        if well.latest_analysis:
            continue
        try:
            run_analysis(well, {})
            done += 1
        except AnalysisError as exc:
            failed += 1
            messages.warning(request, f"{well.name}: {exc}")
    messages.info(request, f"Screened {done} well(s) with default parameters" + (f"; {failed} failed." if failed else "."))
    return redirect("well_list")


@require_POST
def well_delete(request, pk):
    well = get_object_or_404(Well, pk=pk)
    well.las_file.delete(save=False)
    well.delete()
    messages.success(request, "Well deleted.")
    return redirect("well_list")
