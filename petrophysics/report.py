"""Builds the written evaluation from the summary dict.

build_report() returns a list of sections; each section = {"id", "title", "blocks"} and each block is
  {"type": "text", "text": str} | {"type": "list", "items": [str]} |
  {"type": "table", "headers": [str], "rows": [[str]]} | {"type": "image", "key": str, "caption": str}
render_html() / render_markdown() turn that into output, so the web page and downloads always agree.
"""
from __future__ import annotations

import html


def f(x, nd=2, dash="-"):
    if x is None:
        return dash
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def T(text):
    return {"type": "text", "text": text}


def L(items):
    return {"type": "list", "items": items}


def TB(headers, rows):
    return {"type": "table", "headers": headers, "rows": rows}


def IMG(key, caption):
    return {"type": "image", "key": key, "caption": caption}


def build_report(s: dict) -> list:
    du = s["depth_unit"]
    w, qc, p = s["well"], s["qc"], s["params"]
    secs = []

    # ------------------------------------------------------------------ data
    d = qc["depth"]
    secs.append({"id": "data", "title": "1. Well and data summary", "blocks": [
        TB(["Item", "Value"], [
            ["Well", w["name"]], ["Field", w["field"] or "-"], ["UWI", w["uwi"] or "-"],
            ["Logged interval", f"{f(d['top'], 1)} - {f(d['base'], 1)} {du}"],
            ["Sample spacing", f"{f(d['step'], 4)} {du} ({d['samples']} samples)"],
            ["LAS version", w["las_version"]],
        ]),
        TB(["LAS mnemonic", "Unit", "Description", "Used as"],
           [[c["mnemonic"], c["unit"], c["description"],
             next((k for k, v in w["mapping"].items() if v == c["mnemonic"]), "-")] for c in w["curves"]]),
    ] + ([L(w["las_warnings"])] if w["las_warnings"] else [])})

    # ------------------------------------------------------------------- QC
    env = qc["environment"]
    blocks = [T(f"Overall log quality for interpretation: {qc['overall']}. Each curve was checked for completeness, "
                "physically plausible range, isolated spikes (running-median residual > 8 robust sigma), flat-lined "
                "(stuck/filled) runs, and the borehole was assessed from caliper vs bit size and the density correction (DRHO).")]
    blocks.append(TB(["Curve", "Mnemonic", "Complete %", "Min", "Max", "Spikes", "Flat-line samples", "Grade", "Comments"],
                     [[k, c["mnemonic"], f(c["completeness_pct"], 1), f(c["min"], 2), f(c["max"], 2), str(c["spikes"]),
                       str(c["flatline_samples"]), c["grade"], "; ".join(c["comments"]) or "-"]
                      for k, c in qc["curves"].items()]))
    env_rows = []
    if "bit_size_in" in env:
        env_rows += [["Bit size", f"{f(env['bit_size_in'], 2)} in ({env['bit_size_source']})"],
                     ["Washout (caliper > bit + " + f(p['wash_thr_in'], 1) + " in)", f"{env['washout_pct']}% of interval"],
                     ["Severe washout (> 3 in over gauge)", f"{env['severe_washout_pct']}%"],
                     ["Mudcake (caliper < bit - 0.3 in)", f"{env['mudcake_pct']}%"]]
        for a, b in env.get("washout_intervals", []):
            env_rows.append(["Washout interval", f"{f(a, 1)} - {f(b, 1)} {du}"])
    if "drho_bad_pct" in env:
        env_rows.append([f"|DRHO| > {f(p['drho_thr'], 2)} g/cc", f"{env['drho_bad_pct']}%"])
    env_rows.append(["Samples treated as bad hole (density/neutron/PEF unreliable)", f"{env['bad_hole_pct']}%"])
    inv = env.get("invasion") or {}
    if inv:
        env_rows.append([f"Shallow ({inv['shallow_curve']}) vs deep resistivity separation",
                         f"{inv['separation_pct']}% of samples differ by > 25% (median log10 ratio {f(inv['median_log10_shallow_over_deep'], 2)})"])
    blocks += [T("Environmental effects:"), TB(["Check", "Result"], env_rows)]
    if qc["consistency"]:
        blocks.append(TB(["Cross-curve check", "Result"], [
            [c["pair"], ", ".join(f"{k}={v}" for k, v in c.items() if k != "pair")] for c in qc["consistency"]]))
    blocks += [T("Data-quality issues and limitations:"),
               L([f"[{i['severity'].upper()}] {i['message']}" for i in qc["issues"]] or ["No issues detected."]),
               T("Treatment: " + ("isolated spikes were replaced by the running median; " if p["despike"] else "no despiking applied; ")
                 + ("bad-hole samples are excluded from reservoir/fluid classification (flagged 'Unreliable'). " if p["exclude_bad_hole"]
                    else "bad-hole samples were NOT excluded. ")
                 + "No depth shifting, environmental (borehole-size/mud-weight) corrections or tool-specific corrections are applied "
                   "- the software assumes the supplied curves are already environmentally corrected unless stated.")]
    secs.append({"id": "qc", "title": "2. Log quality analysis", "blocks": blocks})

    # ------------------------------------------------------------ lithology
    lith = s["lithology"]
    v = s["vsh"]
    lb = [T(f"Shale volume (Vsh) was computed from gamma ray with the {v['method']} method, using clean = {f(v['gr_clean'], 0)} "
            f"and shale = {f(v['gr_shale'], 0)} gAPI ({v['baseline_source']}). Lithology was then classified per sample:"),
          L(["Vsh >= 0.50: shale.  0.25 <= Vsh < 0.50: shaly sandstone (or argillaceous carbonate if the clean-lithology indicators point to carbonate).  Vsh < 0.25: clean lithology call below.",
             ("PEF thresholds for clean intervals: < 2.5 b/e sandstone, 2.5-4.0 dolomite, > 4.0 limestone (primary indicator)."
              if lith["methods"]["have_pef"] else "No PEF curve: clean lithology relies on density-neutron separation only."),
             ("Neutron-density separation (NPHI on limestone scale minus limestone-scale density porosity), corrected for shale: "
              "< -0.035 sandstone, -0.035 to +0.06 limestone, > +0.06 dolomite. Used as fallback and as a cross-check."
              if lith["methods"]["have_nd"] else "RHOB/NPHI not both available: no density-neutron lithology check."),
             "Coal: RHOB < 1.9 g/cc with NPHI > 0.30 at low Vsh. Anhydrite: RHOB > 2.92 g/cc with NPHI < 0.03 and low Vsh.",
             f"Calls are smoothed with a {p['lith_smooth']}-sample majority filter."])]
    if lith["pef_nd_agreement_pct"] is not None:
        lb.append(T(f"PEF and shale-corrected neutron-density calls agree on {f(lith['pef_nd_agreement_pct'], 0)}% of clean samples."))
    lb.append(TB(["Lithology", "Thickness (m)", "% of interval", "Mean GR", "Mean RHOB", "Mean NPHI", "Mean PEF"],
                 [[r["lithology"], f(r["thickness_m"], 1), f(r["pct"], 1), f(r["GR"], 0), f(r["RHOB"], 2), f(r["NPHI"], 3), f(r["PEF"], 2)]
                  for r in lith["stats"]]))
    lb.append(T("Lithology intervals (>= 1.5 m):"))
    lb.append(TB([f"Top ({du})", f"Base ({du})", "Thickness (m)", "Lithology", "GR", "Vsh", "RHOB", "NPHI", "PEF"],
                 [[f(r["top"], 1), f(r["base"], 1), f(r["thickness_m"], 1), r["lithology"], f(r["GR"], 0), f(r["vsh"], 2),
                   f(r["RHOB"], 2), f(r["NPHI"], 3), f(r["PEF"], 2)] for r in lith["intervals"]]))
    lb.append(L(["Gas in carbonates can mimic a sandstone neutron-density signature, and barite mud or heavy minerals raise PEF; "
                 "treat isolated carbonate calls with caution until checked against cuttings or core.",
                 "GR-based Vsh over-reads in radioactive sands (feldspathic/micaceous/glauconitic) and under-reads where shales are not radioactive."]))
    secs.append({"id": "lith", "title": "3. Lithology determination", "blocks": lb})

    # --------------------------------------------------------------- fluids
    fl = s["fluids"]
    fb = [T("Fluids were identified for reservoir-quality samples (Vsh <= " + f(p["vsh_cut"], 2) + ", PHIE >= " + f(p["phi_cut"], 2) +
            ", not bad hole, not coal/anhydrite) from two lines of evidence:"),
          L([f"Resistivity: Sw <= {f(p['sw_hc_max'], 2)} hydrocarbon-bearing; Sw >= {f(p['sw_water_min'], 2)} water-bearing; in between transition/uncertain.",
             (f"Density-neutron crossover: where NPHI (matrix-scaled) falls below density porosity by more than {f(p['sep_gas'], 2)} v/v inside a hydrocarbon zone, the zone is called gas; without crossover it is called oil."
              if fl["have_nd"] else "RHOB and NPHI are not both available: gas cannot be separated from oil, so hydrocarbon zones are labelled undifferentiated."),
             f"Classification is median-filtered and zones thinner than {f(p['min_bed_m'], 1)} m are discarded. Contacts are placed midway between the last sample of one fluid and the first of the next."])]
    zrows = [[z["sand"], z["fluid"], f(z["top"], 1), f(z["base"], 1), f(z["gross_m"], 1), f(z["net_pay_m"], 1), z["lithology"],
              f(z["vsh"], 2), f(z["phie"], 3), f(z["sw"], 2), f(z["rt"], 1)] for z in fl["zones"]]
    fb.append(TB(["Sand", "Fluid", f"Top ({du})", f"Base ({du})", "Gross (m)", "Net pay (m)", "Lithology", "Vsh", "PHIE", "Sw", "Rt (ohm.m)"], zrows)
              if zrows else T("No reservoir zones met the cutoffs."))
    if fl["contacts"]:
        fb.append(T("Fluid contacts inferred from log response:"))
        fb.append(TB(["Sand", "Contact", f"Depth ({du})"], [[c["sand"], c["contact"], f(c["depth"], 1)] for c in fl["contacts"]]))
    # narrative per sand
    sands = {}
    for z in fl["zones"]:
        sands.setdefault(z["sand"], []).append(z)
    story = []
    for name, zs in sands.items():
        parts = ", ".join(f"{z['fluid'].lower()} {f(z['top'], 1)}-{f(z['base'], 1)} {du} ({f(z['gross_m'], 1)} m)" for z in zs)
        story.append(f"{name}: {parts}.")
    if story:
        fb.append(L(story))
    if not fl["have_nd"]:
        fb.append(T("Note: without a density-neutron pair the oil/gas distinction is not possible."))
    fb.append(T("Contacts and fluid types here come from logs alone. Pressure gradients (formation tester), fluid samples or "
                "production tests are needed to confirm them; low-resistivity pay, fresh formation water, thin beds and "
                "invasion can all mask or mimic hydrocarbons."))
    fb.append(IMG("log_plot", "Composite log display: GR/caliper, resistivity, density-neutron (crossover shaded), PEF, porosity, Sw/Sxo, lithology and fluid columns."))
    secs.append({"id": "fluids", "title": "4. Fluid identification", "blocks": fb})

    # ----------------------------------------------------------- saturation
    sat = s["saturation"]
    model_eq = {
        "archie": "Archie: Sw = [ a*Rw / (PHIE^m * Rt) ]^(1/n)",
        "simandoux": "Simandoux: 1/Rt = PHIE^m*Sw^n / (a*Rw*(1-Vsh)) + Vsh*Sw/Rsh   (solved numerically for Sw)",
        "indonesia": "Indonesian: 1/sqrt(Rt) = [ Vsh^(1-Vsh/2)/sqrt(Rsh) + PHIE^(m/2)/sqrt(a*Rw) ] * Sw^(n/2)",
    }[sat["model"]]
    sb = [T(f"Water saturation (Sw) was calculated with the {sat['model'].capitalize()} model, with Archie and the other shaly-sand models "
            "computed alongside for comparison. Hydrocarbon saturation Sh = 1 - Sw. Bulk volume water BVW = PHIE x Sw."),
          T(model_eq),
          T("Porosity: " + s["porosity"]["method"] + f". Total porosity was corrected for shale: PHIE = PHIT - Vsh x PHIT(shale) with "
            f"PHIT(shale) = {f(s['porosity']['phit_shale'], 3)} taken from shale intervals. Matrix densities: sandstone {f(p['rho_sand'], 2)}, "
            f"limestone {f(p['rho_lime'], 2)}, dolomite {f(p['rho_dolo'], 2)} g/cc; fluid density {f(p['rho_fluid'], 2)} g/cc."),
          TB(["Parameter", "Value", "Source / note"], [
              ["a, m, n", f"{f(sat['a'], 2)}, {f(sat['m'], 2)}, {f(sat['n'], 2)}", "Input parameters (not from core SCAL unless you set them from it)"],
              ["Rw (at " + f(sat["rw_temp_f"], 0) + " F)", f(sat["rw_ref"], 4) + " ohm.m", sat["rw_source"]],
              ["Formation temperature", "depth-dependent", sat["temperature_source"] + "; Rw converted with Arps: R2 = R1 (T1+6.77)/(T2+6.77)"],
              ["Rsh", f(sat["rsh"], 2) + " ohm.m", sat["rsh_source"]],
              ["Rmf", f(p["rmf"], 3) + " ohm.m at " + f(p["rmf_temp_f"], 0) + " F" if p["rmf"] else "not supplied", "Used for Sxo"],
              ["Sxo", "-", sat["sxo_method"]],
              ["Vsh method", v["method"], f"GR clean {f(v['gr_clean'], 0)} / shale {f(v['gr_shale'], 0)} gAPI ({v['baseline_source']})"],
              ["Hydrocarbon / water Sw limits", f"{f(p['sw_hc_max'], 2)} / {f(p['sw_water_min'], 2)}", "Fluid classification"],
              ["Net pay cutoffs", f"Vsh <= {f(p['vsh_cut'], 2)}, PHIE >= {f(p['phi_cut'], 2)}, Sw <= {f(p['sw_pay_cut'], 2)}", "Applied to hydrocarbon zones only"],
          ]),
          T("Results by zone (thickness-weighted averages):")]
    srows = [[z["sand"], z["fluid"], f"{f(z['top'], 1)}-{f(z['base'], 1)}", f(z["phie"], 3), f(z["sw"], 2), f(1 - z["sw"], 2) if z["sw"] is not None else "-",
              f(z["sxo"], 2), f(z["bvw"], 3), f(z["sw_archie"], 2), f(z["sw_simandoux"], 2), f(z["sw_indonesia"], 2), f(z["net_pay_m"], 1),
              f(z["hcpv_m"], 2)] for z in fl["zones"]]
    sb.append(TB(["Sand", "Fluid", f"Interval ({du})", "PHIE", f"Sw ({sat['model']})", "Sh", "Sxo", "BVW", "Sw Archie", "Sw Simandoux",
                  "Sw Indonesian", "Net pay (m)", "PHI*Sh*h (m)"], srows) if srows else T("No zones to summarise."))
    if fl["totals"]:
        sb.append(TB(["Fluid", "Zones", "Gross (m)", "Net pay (m)", "Avg PHIE", "Avg Sw", "Sum PHI*(1-Sw)*h (m)"],
                     [[k, str(t["n_zones"]), f(t["gross_m"], 1), f(t["net_pay_m"], 1), f(t["avg_phie"], 3), f(t["avg_sw"], 2), f(t["hcpv_m"], 2)]
                      for k, t in fl["totals"].items()]))
    sb.append(T("Interpretation notes and assumptions:"))
    sb.append(L([
        "Vertical well assumed (measured depth = true vertical depth) for temperature and thickness.",
        "Rt is the deep-resistivity curve as supplied; no borehole, invasion, shoulder-bed or thin-bed corrections are applied.",
        "Rw " + ("was supplied by the user." if "user" in sat["rw_source"] else
                 "was estimated from the lowest apparent water resistivity and assumes the logged interval contains a clean water-bearing sand. Confirm from a water sample, SP or a regional Rw catalogue."),
        "a, m and n default to Archie's classical values unless changed; use core (SCAL) values where available, since m and n strongly control Sw.",
        "Sw from the shaly-sand models is lower than Archie's wherever Vsh is significant; the spread between the models in the table shows the sensitivity to the shale model.",
        "Bulk volume water that is roughly constant through a hydrocarbon zone indicates it is near irreducible water saturation; a rising BVW toward the base suggests a transition zone with mobile water.",
        "Saturations are computed on log-scale volumes at the supplied sample spacing; they are not core-calibrated.",
    ]))
    sb.append(IMG("crossplots", "Neutron-density crossplot, Pickett plot and the effect of the shale model on Sw."))
    secs.append({"id": "sat", "title": "5. Hydrocarbon saturation analysis", "blocks": sb})

    # ----------------------------------------------------------- suitability
    su = fl["suitability"]
    secs.append({"id": "suit", "title": "6. Well suitability screening", "blocks": [
        T(f"Screening result: {'SUITABLE' if su['suitable'] else 'NOT SUITABLE'} as a well that traverses oil, gas and water zones "
          f"(score {su['score']}/{su['max_score']}). Thickness by fluid (m): gas {su['thickness_by_fluid_m']['gas']}, "
          f"oil {su['thickness_by_fluid_m']['oil']}, water {su['thickness_by_fluid_m']['water']}."),
        TB(["Check", "Result"], [[c["check"], "PASS" if c["pass"] else "FAIL"] for c in su["checks"]])]})
    return secs


# ------------------------------------------------------------------ renderers
def render_html(sections, images: dict | None = None, wrap_images=True) -> str:
    images = images or {}
    esc = html.escape
    out = []
    for sec in sections:
        out.append(f'<section id="{esc(sec["id"])}"><h2>{esc(sec["title"])}</h2>')
        for b in sec["blocks"]:
            if b["type"] == "text":
                out.append(f"<p>{esc(b['text'])}</p>")
            elif b["type"] == "list":
                out.append("<ul>" + "".join(f"<li>{esc(i)}</li>" for i in b["items"]) + "</ul>")
            elif b["type"] == "table":
                head = "".join(f"<th>{esc(h)}</th>" for h in b["headers"])
                rows = "".join("<tr>" + "".join(f"<td>{esc(str(c))}</td>" for c in r) + "</tr>" for r in b["rows"])
                out.append(f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>')
            elif b["type"] == "image":
                b64 = images.get(b["key"])
                if b64:
                    out.append(f'<figure><img alt="{esc(b["caption"])}" src="data:image/png;base64,{b64}">'
                               f'<figcaption>{esc(b["caption"])}</figcaption></figure>')
        out.append("</section>")
    return "\n".join(out)


def render_markdown(sections, title="Petrophysical evaluation") -> str:
    out = [f"# {title}", ""]
    for sec in sections:
        out += [f"## {sec['title']}", ""]
        for b in sec["blocks"]:
            if b["type"] == "text":
                out += [b["text"], ""]
            elif b["type"] == "list":
                out += [f"- {i}" for i in b["items"]] + [""]
            elif b["type"] == "table":
                out.append("| " + " | ".join(b["headers"]) + " |")
                out.append("|" + "---|" * len(b["headers"]))
                for r in b["rows"]:
                    out.append("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |")
                out.append("")
            elif b["type"] == "image":
                out += [f"*[Figure omitted in Markdown export: {b['caption']}]*", ""]
    return "\n".join(out)


STANDALONE_CSS = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:1150px;margin:24px auto;padding:0 16px;color:#1c1c1c;line-height:1.5}
h1{font-size:1.6rem}h2{border-bottom:2px solid #ddd;padding-bottom:4px;margin-top:2rem}
.tbl{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;font-size:.85rem;width:100%}
th,td{border:1px solid #ccc;padding:4px 8px;text-align:left;vertical-align:top}th{background:#f2f2f2}
figure{margin:14px 0}img{max-width:100%}figcaption{font-size:.8rem;color:#555}
"""


def standalone_html(sections, images, title) -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<style>{STANDALONE_CSS}</style></head><body><h1>{html.escape(title)}</h1>"
            f"{render_html(sections, images)}</body></html>")
