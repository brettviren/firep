#!/usr/bin/env python
"""Figures and numbers for the note's section on bandwidth, sampling and the
collection spike (docs/sections/16-bandwidth.tex).

    scripts/bandwidth_figs.py [RUNDIR] [FIGDIR]

RUNDIR (default runs/pdsp) holds the production learned response
(``direct.pt``, and the previous default as ``direct-v1.pt``), the traced
response on the 11-impact grid (``response-traced.npz``) and the experiment
checkpoints in ``exp/``.  Writes ``bandwidth.tex`` (LaTeX macros and the
experiment table), ``spike-experiments.pdf`` and ``direct-compare.pdf`` to
FIGDIR (default docs/figs).  Every number the section quotes comes from here.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdsp_direct_figs as F  # noqa: E402

from firep import direct as D  # noqa: E402
from firep import response as R  # noqa: E402

# (checkpoint, table label, settings) in the order the section discusses them.
RUNS = [
    ("direct-v1.pt", "previous default", r"$\omega_0 = 200$, per-path residual"),
    ("exp/w400.pt", r"$\omega_0 = 400$", r"doubled first-layer frequency"),
    ("exp/w400-dense.pt", r"$\omega_0 = 400$, dense", r"and $4\times$ batch and pool"),
    ("exp/pertime.pt", "per time", "residual per unit time"),
    ("exp/logfeat2.pt", "log input", r"$\ln(1 + \mathrm{relu}(T - t)/\tau_a)$"),
    ("exp/pertime-arrsamp.pt", "per time, near-$T$", r"$\tfrac14$ of times near $T(\mathbf{r})$"),
    ("exp/combined.pt", "per time, log", "both"),
    ("direct.pt", "new default", "per time, log, near-$T$"),
]
PLOT = [("direct-v1.pt", "previous default"), ("exp/pertime.pt", "per time"),
        ("exp/combined.pt", "per time + log"),
        ("direct.pt", "per time + log + near-$T$ (new default)")]
LETTERS = "ABCDEFGHIJ"


def evaluate(path, cm, traced, meta, fine, dev):
    model, cell, spec, blob, tdev = D.load(path, cm, device=dev)
    top = max(e.plane for e in cm.cfg.electrodes)
    ys = cm.geom.yhi - 0.01 * (cm.geom.yhi - top)
    x0 = np.linspace(0, 0.5 * cell.pitch, 11)
    r = D.response(model, cm, x0, ys, 0.1, tdev)
    ra = dict(time=r.time, impact=r.impact, offsets=r.offsets, current=r.current)
    cl, ct, time = F.common(ra, traced)
    tl = R.impact_table(dict(ra, current=cl, time=time), meta, per_side=5)
    tt = R.impact_table(dict(traced, current=ct, time=time), meta, per_side=5)
    iw, cen = 10, slice(1, 10)
    out = dict(w_sum=[float(r.integrated[2, iw, cen].min()), float(r.integrated[2, iw, cen].max())])
    for ip, p in enumerate("uvw"):
        a, b = tl["table"][ip], tt["table"][ip]
        a5, b5 = F.rebin(a, 5), F.rebin(b, 5)
        out[p] = dict(rms=float(np.sqrt(np.mean((a - b) ** 2)) / np.abs(b).max()),
                      rms_adc=float(np.sqrt(np.mean((a5 - b5) ** 2)) / np.abs(b5).max()),
                      peak=float(a.max()), peak_traced=float(b.max()))
    rf = D.response(model, cm, fine["impact"], cell.ytop, 0.02, tdev)
    n = min(rf.current.shape[-1], fine["current"].shape[-1])
    lw = rf.current[2, iw, 1:10, :n]
    lv = rf.current[1, iw, 1:10, :n]
    out["fine"] = dict(w_peak=float(lw.max(1).mean()),
                       w_fwhm=float(np.mean([(q > q.max() / 2).sum() for q in lw]) * 20.0),
                       v_dip=float(lv.min(1).mean()))
    out["curves"] = {ip: rf.current[ip, iw, 5, :n].tolist() for ip in range(3)}
    return out


def fine_reference(rundir, cm):
    path = os.path.join(rundir, "exp", "traced-fine.npz")
    if not os.path.exists(path):
        model, cell = D.build(cm, D.DirectCfg())
        res = R.response(cm, impacts=11, y_start=cell.ytop, tick=0.02, max_time=60.0)
        R.save_npz(path, res, cm)
    return R.load_npz(path)[0]


def fig_spikes(fine, results, out):
    t = np.arange(fine["current"].shape[-1]) * 0.02 + 0.01
    ta = t[int(np.argmax(fine["current"][2, 10, 5]))]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), constrained_layout=True)
    for ax, (ip, lab, lo, hi) in zip(axes, ((2, "W, the collecting wire", 0.6, 0.25),
                                            (1, "V, the plane above", 2.5, 0.25),
                                            (0, "U, two planes above", 8.0, 0.5))):
        ax.plot(t, fine["current"][ip, 10, 5], "k", lw=1.8, label="traced")
        for path, name in PLOT:
            c = np.asarray(results[path]["curves"][ip] if ip in results[path]["curves"]
                           else results[path]["curves"][str(ip)])
            ax.plot(t[: len(c)], c, lw=1.0, label=name)
        ax.set_xlim(ta - lo, ta + hi)
        ax.set_title(f"{lab}, impact 0.25 pitch", fontsize=10)
        ax.set_xlabel(r"time after crossing $y_R$ [$\mu$s]")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"current [e/$\mu$s], 0.02 $\mu$s tick")
    axes[0].legend(fontsize=7.5)
    fig.savefig(out)
    plt.close(fig)


def fig_compare(rundir, out):
    """The production learned response against the traced one, near the wires."""
    ra, rm = R.load_npz(os.path.join(rundir, "response-direct.npz"))
    rt, _ = R.load_npz(os.path.join(rundir, "response-traced.npz"))
    cl, ct, time = F.common(ra, rt)
    tl = R.impact_table(dict(ra, current=cl, time=time), rm, per_side=5)
    tt = R.impact_table(dict(rt, current=ct, time=time), rm, per_side=5)
    t_last = float(time[np.flatnonzero(np.abs(ct).sum(axis=(0, 1, 2)) > 0)[-1]])
    pitch = float(rm["pitch_mm"])
    picks = [(0.05 * pitch, "0.05 pitch"), (0.25 * pitch, "0.25 pitch"),
             (0.45 * pitch, "0.45 pitch"), (1.05 * pitch, "next wire")]
    fig, axes = plt.subplots(3, len(picks), figsize=(12, 7.0), sharex=True,
                             constrained_layout=True)
    for ip, name in enumerate(rm["planes"]):
        for k, (u, lab) in enumerate(picks):
            ax = axes[ip, k]
            j = int(np.argmin(np.abs(tl["impact"] - u)))
            ax.plot(time, tt["table"][ip, j], lw=1.2, label="traced")
            ax.plot(time, tl["table"][ip, j], lw=1.0, ls="--", label="learned")
            ax.axhline(0, color="0.7", lw=0.5)
            ax.set_xlim(t_last - 8.0, t_last + 1.0)
            ax.grid(alpha=0.25)
            if ip == 0:
                ax.set_title(f"launch {lab}", fontsize=9)
            if k == 0:
                ax.set_ylabel(f"{name} [e/$\\mu$s]")
            if ip == 2:
                ax.set_xlabel(r"time [$\mu$s]")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(out)
    plt.close(fig)


def pct(x):
    return f"{100 * x:.2f}"


def main(rundir="runs/pdsp", figdir="docs/figs"):
    os.makedirs(figdir, exist_ok=True)
    dev = os.environ.get("FIREP_DEVICE", "auto")
    cm = R.load_combined(os.path.join(rundir, "drift.pt"),
                         [os.path.join(rundir, f"weight-{p}.pt") for p in "uvw"], device=dev)
    traced, meta = R.load_npz(os.path.join(rundir, "response-traced.npz"))
    fine = fine_reference(rundir, cm)
    results = {}
    for path, _, _ in RUNS:
        results[path] = evaluate(os.path.join(rundir, path), cm, traced, meta, fine, dev)
        print(path, json.dumps({k: v for k, v in results[path].items() if k != "curves"}))

    fig_spikes(fine, results, os.path.join(figdir, "spike-experiments.pdf"))
    fig_compare(rundir, os.path.join(figdir, "direct-compare.pdf"))

    # ---- the traced spike, and the numbers the prose quotes -----------------
    tw = fine["current"][2, 10, 1:10]
    tv = fine["current"][1, 10, 1:10]
    tr = {"w_peak": float(tw.max(1).mean()),
          "w_fwhm": float(np.mean([(q > q.max() / 2).sum() for q in tw]) * 20.0),
          "v_dip": float(tv.min(1).mean())}
    sw = R.load_npz(os.path.join(rundir, "response-traced.npz"))[0]["integrated"][2, 10, 1:10]
    lines = [
        "% generated by scripts/bandwidth_figs.py -- do not edit",
        f"\\newcommand{{\\bwTracedPeak}}{{{tr['w_peak']:.1f}}}",
        f"\\newcommand{{\\bwTracedFwhm}}{{{tr['w_fwhm']:.0f}}}",
        f"\\newcommand{{\\bwTracedDip}}{{{tr['v_dip']:.2f}}}",
        f"\\newcommand{{\\bwTracedSum}}{{{sw.min():.3f}--{sw.max():.3f}}}",
    ]
    for (path, _, _), L in zip(RUNS, LETTERS):
        r = results[path]
        lines += [
            f"\\newcommand{{\\bwPeak{L}}}{{{r['fine']['w_peak']:.1f}}}",
            f"\\newcommand{{\\bwFwhm{L}}}{{{r['fine']['w_fwhm']:.0f}}}",
            f"\\newcommand{{\\bwDip{L}}}{{{r['fine']['v_dip']:.2f}}}",
            f"\\newcommand{{\\bwRmsU{L}}}{{{pct(r['u']['rms'])}}}",
            f"\\newcommand{{\\bwRmsV{L}}}{{{pct(r['v']['rms'])}}}",
            f"\\newcommand{{\\bwRmsW{L}}}{{{pct(r['w']['rms'])}}}",
            f"\\newcommand{{\\bwAdcMax{L}}}{{{pct(max(r[p]['rms_adc'] for p in 'uvw'))}}}",
            f"\\newcommand{{\\bwTickPeak{L}}}{{{r['w']['peak']:.2f}}}",
            f"\\newcommand{{\\bwSum{L}}}{{{r['w_sum'][0]:.3f}--{r['w_sum'][1]:.3f}}}",
        ]
    lines.append(f"\\newcommand{{\\bwTickPeakTraced}}{{{results['direct.pt']['w']['peak_traced']:.2f}}}")
    rows = []
    for (path, label, settings), L in zip(RUNS, LETTERS):
        rows.append(f"{label} & {settings} & \\bwRmsU{L} & \\bwRmsV{L} & \\bwRmsW{L} & "
                    f"\\bwPeak{L} & \\bwFwhm{L} & \\bwDip{L} \\\\")
    lines += ["\\newcommand{\\bandwidthTable}{%",
              "\\begin{tabular}{@{}llrrrrrr@{}}",
              "\\toprule",
              "run & change & \\multicolumn{3}{c}{rms error [\\% of peak]} & "
              "\\multicolumn{2}{c}{W spike} & V dip \\\\",
              " & & $u$ & $v$ & $w$ & [e/$\\mu$s] & [ns] & [e/$\\mu$s] \\\\",
              "\\midrule",
              f"traced & & & & & \\bwTracedPeak & \\bwTracedFwhm & \\bwTracedDip \\\\",
              "\\midrule", *rows, "\\bottomrule", "\\end{tabular}}"]
    with open(os.path.join(figdir, "bandwidth.tex"), "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f"wrote {figdir}/bandwidth.tex")


if __name__ == "__main__":
    main(*sys.argv[1:])
