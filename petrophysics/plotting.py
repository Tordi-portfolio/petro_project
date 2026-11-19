"""Matplotlib figures (Agg backend, returned as PNG bytes)."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LITH_COLORS = {1: "#8c8c8c", 2: "#d98c3f", 3: "#f5e27a", 4: "#7fb3d5", 5: "#b39ddb",
               6: "#222222", 7: "#e8a0bf", 8: "#9ec9a2", 0: "#ffffff"}
FLUID_COLORS = {0: "#ffffff", 1: "#4a90d9", 2: "#3a9d4a", 3: "#e0453a", 4: "#f0a63a", 5: "#bbbbbb", 6: "#8e5ac8"}


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _column(ax, depth, codes, colors):
    for c in np.unique(codes):
        m = codes == c
        ax.fill_betweenx(depth, 0, 1, where=m, color=colors.get(int(c), "#fff"), step="mid", linewidth=0)
    ax.set_xlim(0, 1)
    ax.set_xticks([])


def log_plot(df, contacts, depth_unit, title, sep_gas=0.04) -> bytes:
    d = df["DEPT"].to_numpy()
    fig, axes = plt.subplots(1, 8, figsize=(13, 11), sharey=True,
                             gridspec_kw={"width_ratios": [1.2, 1.2, 1.4, 1.2, 1.2, 1.2, 0.45, 0.45]})
    ax = axes
    # 1 GR (top axis) / caliper & bit size (bottom axis)
    ax[0].plot(df["GR"], d, color="#2e7d32", lw=0.8)
    ax[0].set_xlabel("GR (gAPI)", color="#2e7d32")
    ax[0].xaxis.tick_top()
    ax[0].xaxis.set_label_position("top")
    ax[0].set_xlim(0, max(150, np.nanpercentile(df["GR"], 99.5)))
    ax[0].fill_betweenx(d, 0, df["GR"], where=df["VSH"] < 0.25, color="#f5e27a", alpha=0.5, lw=0)
    if "CALI" in df:
        t = ax[0].twiny()
        t.plot(df["CALI"], d, color="k", lw=0.7, ls="--")
        if "BS" in df:
            t.plot(df["BS"], d, color="r", lw=0.7)
        t.tick_params(axis="x", top=False, labeltop=False, bottom=True, labelbottom=True, labelsize=7)
        t.xaxis.set_label_position("bottom")
        t.set_xlabel("CALI (dashed) / BS (red), in", fontsize=8)
        t.set_xlim(6, 16)
        ax[0].tick_params(axis="x", bottom=False, labelbottom=False, top=True, labeltop=True)
    # 2 resistivity
    ax[1].set_xscale("log")
    for k, col in (("RT", "#c62828"), ("RXO", "#1565c0"), ("RS", "#6a1b9a")):
        if k in df and np.isfinite(df[k]).any():
            ax[1].plot(df[k], d, color=col, lw=0.8, label=k)
    ax[1].set_xlim(0.2, 2000)
    ax[1].legend(loc="lower right", fontsize=7)
    ax[1].set_xlabel("Resistivity (ohm.m)")
    ax[1].xaxis.set_label_position("top")
    # 3 density porosity vs matrix-scaled neutron porosity on one scale; gas-like crossover shaded
    if "RHOB" in df and "NPHI" in df:
        dp, npm = df["DPHI"].to_numpy(), df["NPHI_M"].to_numpy()
        ax[2].plot(dp, d, color="#c62828", lw=0.8, label="DPHI")
        ax[2].plot(npm, d, color="#1565c0", lw=0.8, label="NPHI (matrix-scaled)")
        ax[2].set_xlim(0.45, -0.15)
        ax[2].set_xlabel("DPHI / NPHI (v/v)")
        ax[2].xaxis.set_label_position("top")
        gas_like = np.isfinite(dp) & np.isfinite(npm) & ((dp - npm) > sep_gas) & (df["VSH"].to_numpy() < 0.5)
        ax[2].fill_betweenx(d, dp, npm, where=gas_like, color="#ffd54f", alpha=0.9, lw=0, label="Gas-like crossover")
        ax[2].legend(loc="lower right", fontsize=6)
    # 4 PEF / DT
    if "PEF" in df and np.isfinite(df["PEF"]).any():
        ax[3].plot(df["PEF"], d, color="#6d4c41", lw=0.8)
        ax[3].set_xlim(0, 8)
        ax[3].set_xlabel("PEF (b/e)")
    elif "DT" in df:
        ax[3].plot(df["DT"], d, color="#00838f", lw=0.8)
        ax[3].set_xlabel("DT (us/ft)")
    ax[3].xaxis.set_label_position("top")
    # 5 porosity
    ax[4].plot(df["PHIE"], d, color="#1565c0", lw=0.9, label="PHIE")
    ax[4].plot(df["PHIT"], d, color="#90a4ae", lw=0.6, label="PHIT")
    ax[4].set_xlim(0, 0.45)
    ax[4].legend(loc="lower right", fontsize=7)
    ax[4].set_xlabel("Porosity (v/v)")
    ax[4].xaxis.set_label_position("top")
    # 6 saturation (display masked in shales, where Sw is not meaningful)
    show = df["VSH"].to_numpy() < 0.5
    ax[5].plot(np.where(show, df["SW"], np.nan), d, color="#1565c0", lw=0.9, label="Sw")
    ax[5].plot(np.where(show, df["SXO"], np.nan), d, color="#43a047", lw=0.6, label="Sxo")
    ax[5].set_xlim(1, 0)
    ax[5].set_xlabel("Sw / Sxo (frac)")
    ax[5].xaxis.set_label_position("top")
    ax[5].fill_betweenx(d, 1, df["SW"], where=(df["FLUID_CODE"].isin([2, 3, 6])), color="#a5d6a7", alpha=0.6, lw=0)
    ax[5].legend(loc="lower right", fontsize=7)
    # 7 lithology, 8 fluid
    _column(ax[6], d, df["LITH_CODE"].to_numpy(), LITH_COLORS)
    ax[6].set_xlabel("Lith")
    ax[6].xaxis.set_label_position("top")
    _column(ax[7], d, df["FLUID_CODE"].to_numpy(), FLUID_COLORS)
    ax[7].set_xlabel("Fluid")
    ax[7].xaxis.set_label_position("top")

    for a in axes:
        a.grid(True, alpha=0.25, lw=0.4)
    for c in contacts:
        for a in axes:
            a.axhline(c["depth"], color="k", lw=0.8, ls="-.", alpha=0.7)
        ax[7].text(1.05, c["depth"], c["contact"].split(" (")[0].split(" ")[0], fontsize=7, va="center")
    ax[0].set_ylim(d.max(), d.min())
    ax[0].set_ylabel(f"Depth ({depth_unit})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=FLUID_COLORS[c]) for c in (3, 2, 1, 4, 5)]
    fig.legend(handles, ["Gas", "Oil", "Water", "Transition", "Bad hole"], loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0.02, 1, 0.985))
    return _png(fig)


def crossplots(df, summary) -> bytes:
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.8))
    res = df["FLUID_CODE"].isin([1, 2, 3, 4, 6])
    # neutron-density crossplot
    if "RHOB" in df and "NPHI" in df:
        ok = np.isfinite(df["RHOB"]) & np.isfinite(df["NPHI"])
        a.scatter(df["NPHI"][ok], df["RHOB"][ok], c=df["VSH"][ok], cmap="YlOrBr", s=4, alpha=0.7)
        phi = np.linspace(0, 0.4, 30)
        a.plot(phi - 0.04 + 0.0, 2.65 - phi * (2.65 - 1.0), "k-", lw=1, label="Sandstone")
        a.plot(phi, 2.71 - phi * (2.71 - 1.0), "b-", lw=1, label="Limestone")
        a.plot(phi + 0.02, 2.87 - phi * (2.87 - 1.0), "m-", lw=1, label="Dolomite")
        a.set_xlim(-0.05, 0.5); a.set_ylim(3.0, 1.8)
        a.set_xlabel("NPHI (v/v, LS)"); a.set_ylabel("RHOB (g/cc)")
        a.set_title("Neutron-density crossplot (colour = Vsh)"); a.legend(fontsize=7)
    # Pickett plot
    ok = res & np.isfinite(df["PHIE"]) & np.isfinite(df["RT"]) & (df["PHIE"] > 0.02)
    colors = df["FLUID_CODE"].map(FLUID_COLORS)
    b.scatter(df["RT"][ok], df["PHIE"][ok], c=colors[ok], s=5, alpha=0.7)
    b.set_xscale("log"); b.set_yscale("log")
    s = summary["saturation"]
    rt_line = np.logspace(-1, 3, 20)
    for sw_l, ls in ((1.0, "-"), (0.5, "--"), (0.25, ":")):
        # Rt = a Rw / (phi^m Sw^n)  ->  phi = (a Rw / (Rt Sw^n))^(1/m)
        rw_f = float(np.nanmedian(df["RW_F"][ok])) if ok.any() else s["rw_ref"]
        phi_l = (s["a"] * rw_f / (rt_line * sw_l ** s["n"])) ** (1.0 / s["m"])
        b.plot(rt_line, phi_l, "k", ls=ls, lw=0.9, label=f"Sw={sw_l:g}")
    b.set_xlim(0.2, 2000); b.set_ylim(0.02, 0.5)
    b.set_xlabel("Rt (ohm.m)"); b.set_ylabel("PHIE (v/v)"); b.set_title("Pickett plot (blue water, green oil, red gas)")
    b.legend(fontsize=7)
    # Sw model comparison
    ok = res & np.isfinite(df["SW_ARCHIE"]) & np.isfinite(df["SW_INDONESIA"])
    c.scatter(df["SW_ARCHIE"][ok], df["SW_INDONESIA"][ok], s=4, c=df["VSH"][ok], cmap="YlOrBr", alpha=0.7)
    c.plot([0, 1], [0, 1], "k--", lw=0.8)
    c.set_xlabel("Sw Archie (clean-sand)"); c.set_ylabel("Sw Indonesian (shaly-sand)")
    c.set_title("Shale effect on Sw (colour = Vsh)"); c.set_xlim(0, 1.05); c.set_ylim(0, 1.05)
    fig.tight_layout()
    return _png(fig)
