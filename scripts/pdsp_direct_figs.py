#!/usr/bin/env python
"""Figures for the directly learned PDSP field response, and its comparison
with the traced (drift + weighting + Ramo) response.

    scripts/pdsp_direct_figs.py [RUNDIR]

RUNDIR (default runs/pdsp) must hold

* ``direct.pt``               the learned model (``firep learn-response``)
* ``response-direct.npz``     its response, launched below the cathode
* ``response-traced.npz``     the traced response on the same impact grid

and writes ``direct-*.png`` (the same plots ``pdsp_figs.py`` makes for the
traced response) and ``compare-direct-*.png`` plus ``compare-direct.txt``.

Both responses use the charge form on the same tick grid starting at launch, so
they are compared sample by sample.  The two impacts on the lines ``x = 0`` and
``x = p/2`` are stagnation lines of the drift field: an electron launched
exactly there never reaches a wire in exact arithmetic, and the traced one only
does through rounding.  They are reported, but the comparison statistics use
the centred impacts in between, as ``impact_table`` does by default.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import SymLogNorm

from firep import direct as direct_mod
from firep import plot as plot_mod
from firep import response as resp_mod

G_PLANE = 14.13  # mm, the grid plane of the PDSP example


def common(a, b):
    """Trim or zero-pad two responses onto a common tick grid."""
    n = max(a["current"].shape[-1], b["current"].shape[-1])
    out = []
    for r in (a, b):
        c = r["current"]
        pad = n - c.shape[-1]
        out.append(np.pad(c, [(0, 0)] * (c.ndim - 1) + [(0, pad)]))
    tick = float(a["time"][1] - a["time"][0])
    return out[0], out[1], (np.arange(n) + 0.5) * tick


def time_at_y(arrays, y):
    ty = np.asarray(arrays["traj_y"])[:, 0]
    tt = np.asarray(arrays["traj_t"])
    return float(np.interp(-y, -ty, tt))


def fig_waveforms(tl, tt, time, planes, pitch, out, tmin, tmax, title):
    picks = [(0.05 * pitch, "0.05 pitch"), (0.25 * pitch, "0.25 pitch"),
             (0.45 * pitch, "0.45 pitch"), (1.05 * pitch, "next wire"),
             (2.05 * pitch, "two wires")]
    fig, axes = plt.subplots(len(planes), len(picks), figsize=(3.0 * len(picks), 7.4),
                             sharex=True, constrained_layout=True)
    for ip, name in enumerate(planes):
        for k, (u, lab) in enumerate(picks):
            ax = axes[ip, k]
            j = int(np.argmin(np.abs(tl["impact"] - u)))
            ax.plot(time, tt["table"][ip, j], lw=1.2, label="traced")
            ax.plot(time, tl["table"][ip, j], lw=1.0, ls="--", label="learned")
            ax.axhline(0, color="0.7", lw=0.5)
            ax.set_xlim(tmin, tmax)
            ax.grid(alpha=0.25)
            if ip == 0:
                ax.set_title(f"u = {tl['impact'][j]:.3g} mm ({lab})", fontsize=9)
            if k == 0:
                ax.set_ylabel(f"{name}: current [e/$\\mu$s]")
            if ip == len(planes) - 1:
                ax.set_xlabel("time [$\\mu$s]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_difference(tl, tt, time, planes, out, tmin):
    keep = time >= tmin
    fig, axes = plt.subplots(len(planes), 2, figsize=(13, 8.8), sharex=True,
                             constrained_layout=True)
    for ip, name in enumerate(planes):
        m = float(np.abs(tt["table"][ip]).max()) or 1.0
        norm = SymLogNorm(linthresh=1e-3 * m, vmin=-m, vmax=m, base=10)
        for k, (z, lab) in enumerate(((tl["table"][ip], "learned"),
                                      (tl["table"][ip] - tt["table"][ip],
                                       "learned $-$ traced"))):
            ax = axes[ip, k]
            im = ax.pcolormesh(time[keep], tl["impact"], z[:, keep], cmap="RdBu_r",
                               shading="auto", rasterized=True, norm=norm)
            ax.set_title(f"plane {name}: {lab}", fontsize=10, loc="left")
            ax.set_ylabel("impact offset [mm]")
            fig.colorbar(im, ax=ax, pad=0.015, fraction=0.045)
    for ax in axes[-1]:
        ax.set_xlabel(r"time [$\mu$s]")
    fig.suptitle("learned response and its difference from the traced one "
                 "(colour scale of the traced peak)", fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_charge(ra, rt, planes, out):
    """The Ramo sum rule, which is not in the loss: charge on the central wire."""
    iw = int(np.flatnonzero(ra["offsets"] == 0)[0])
    fig, axes = plt.subplots(1, len(planes), figsize=(13, 3.8), constrained_layout=True)
    for ip, name in enumerate(planes):
        ax = axes[ip]
        ax.plot(rt["impact"], rt["integrated"][ip, iw], "o-", label="traced")
        ax.plot(ra["impact"], ra["integrated"][ip, iw], "s--", label="learned")
        ax.set_title(f"plane {name}: total charge on wire 0", fontsize=10)
        ax.set_xlabel("impact [mm]")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("induced charge [e]")
    axes[0].legend(fontsize=8)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_field(model, cell, cm, device, out, times=(0.0, 1.0, 3.0, 6.0)):
    """The learned charge on the central collection wire, as a field."""
    nx, ny = 120, 360
    xs = np.linspace(0.0, cell.half, nx)
    ys = np.linspace(cell.ylo, cell.ytop, ny)
    X, Y = np.meshgrid(xs, ys)
    xy = np.column_stack((X.ravel(), Y.ravel()))
    inside = cell.distance(xy)[0] <= 0
    d, _ = cell.distance(xy, absorbing_only=True)
    iq = model.index("w", 0)
    phi, _ = direct_mod.weighting(cm, model.planes, model.offsets, xy, cell.pitch, grad=False)
    dist = np.tanh(np.maximum(d, 0.0) / model._ell)
    fig, axes = plt.subplots(1, len(times), figsize=(3.2 * len(times), 6.2),
                             constrained_layout=True, sharey=True)
    for ax, t in zip(axes, times):
        with torch.no_grad():
            xyt = torch.as_tensor(np.column_stack((xy, np.full(len(xy), t))),
                                  dtype=torch.float32, device=device)
            q = model(xyt, torch.as_tensor(phi, dtype=torch.float32, device=device),
                      torch.as_tensor(dist, dtype=torch.float32, device=device))
        z = q[:, iq].cpu().numpy()
        z[inside] = np.nan
        z = z.reshape(ny, nx)
        full = np.concatenate((z[:, ::-1], z[:, 1:]), axis=1)
        fx = np.concatenate((-xs[::-1], xs[1:]))
        im = ax.pcolormesh(fx, ys, full, vmin=0, vmax=1, cmap="viridis", shading="auto",
                           rasterized=True)
        ax.contour(fx, ys, full, levels=[0.1, 0.5, 0.9], colors="w", linewidths=0.6)
        ax.set_aspect("equal")
        ax.set_title(f"t = {t:g} $\\mu$s", fontsize=10)
        ax.set_xlabel("x [mm]")
    axes[0].set_ylabel("launch y [mm]")
    fig.colorbar(im, ax=axes, label="learned charge on wire w0 [e]", fraction=0.04)
    fig.suptitle(r"$\mathcal{Q}_{w0}(\mathbf{r}, t)$: charge induced by time $t$ "
                 "on the central collection wire, by launch point", fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_arrival(model, rt, device, out):
    """The learned arrival time against the traced time-to-go along real paths.

    Validation only: the arrival model was trained on ``v . grad T = -1`` and
    never saw a trajectory.
    """
    t = np.asarray(rt["traj_t"])
    xs, ys, alive = (np.asarray(rt[k]) for k in ("traj_x", "traj_y", "traj_alive"))
    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    worst = 0.0
    for i in range(1, xs.shape[1] - 1):  # skip the two stagnation lines
        last = int(np.argmax(~alive[:, i])) if (~alive[:, i]).any() else len(t) - 1
        sel = np.flatnonzero((ys[:last, i] <= model.ytop) & (ys[:last, i] >= model.ylo))
        if not len(sel):
            continue
        sel = sel[:: max(1, len(sel) // 400)]
        togo = t[last] - t[sel]
        xy = np.column_stack((np.abs(xs[sel, i]), ys[sel, i]))
        with torch.no_grad():
            tl = model.arrival.time(torch.as_tensor(xy, dtype=torch.float32, device=device))
        tl = tl.cpu().numpy()
        worst = max(worst, float(np.abs(tl - togo).max()))
        ax.plot(togo, tl, lw=1.0, label=f"impact {rt['impact'][i]:.2f} mm")
    lim = [0, model.arrival.tc * 0.75]
    ax.plot(lim, lim, color="0.5", lw=0.6, ls=":")
    ax.set_xlabel(r"traced time to arrival [$\mu$s]")
    ax.set_ylabel(r"learned $T(\mathbf{r})$ [$\mu$s]")
    ax.set_title("arrival time along traced paths, below $y_R$", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return worst


def rebin(table, n):
    """Sum ``n`` ticks at a time (charge per ADC period), divided back to a rate."""
    m = table.shape[-1] // n
    return table[..., : m * n].reshape(*table.shape[:-1], m, n).mean(-1)


def fig_history(hist, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    ax.semilogy(hist["step"], hist["loss"], label="transport residual")
    ax.semilogy(hist["step"], hist["weighted"], label="causally weighted", alpha=0.7)
    ax.set_xlabel("step")
    ax.set_ylabel("loss (dimensionless)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def numbers(ra, rt, tl, tt, time, planes, pitch):
    iw = int(np.flatnonzero(ra["offsets"] == 0)[0])
    lines = ["Ramo sum rule (not in the loss): total charge on wire 0 [e]",
             "  plane   impact[mm]: " + " ".join(f"{x:7.3f}" for x in ra["impact"])]
    for ip, name in enumerate(planes):
        lines.append(f"  {name} traced          " + " ".join(
            f"{q:7.4f}" for q in rt["integrated"][ip, iw]))
        lines.append(f"  {name} learned         " + " ".join(
            f"{q:7.4f}" for q in ra["integrated"][ip, iw]))
    lines += ["", "waveforms at the 0.5 us ADC period (5 ticks averaged; the collection spike is",
              "shorter than one 0.1 us tick, so its binned height depends on tick phase)",
              "plane  peak+ traced learned   peak- traced learned   rms diff/peak"]
    for ip, name in enumerate(planes):
        a, b = rebin(tl["table"][ip], 5), rebin(tt["table"][ip], 5)
        pk = np.abs(b).max()
        lines.append(f"  {name}   {b.max():12.4f} {a.max():7.4f}   {b.min():12.4f} {a.min():7.4f}"
                     f"   {np.sqrt(np.mean((a - b) ** 2)) / pk:13.4f}")
    lines += ["", "waveforms on the centred impact table (stagnation lines excluded)",
              "plane  peak+ traced learned   peak- traced learned   "
              "rms diff/peak   max |diff|/peak   time shift [us]"]
    for ip, name in enumerate(planes):
        a, b = tl["table"][ip], tt["table"][ip]
        pk = np.abs(b).max()
        # Timing offset by cross-correlation of the row with the most signal;
        # "time of the largest |i|" jumps between the two lobes of a bipolar
        # induction signal whose lobes are nearly equal.
        j = int(np.argmax((b ** 2).sum(axis=1)))
        xc = np.correlate(a[j], b[j], mode="full")
        shift = (int(np.argmax(xc)) - (len(b[j]) - 1)) * (time[1] - time[0])
        lines.append(
            f"  {name}   {b.max():12.4f} {a.max():7.4f}   {b.min():12.4f} {a.min():7.4f}"
            f"   {np.sqrt(np.mean((a - b) ** 2)) / pk:13.4f}   {np.abs(a - b).max() / pk:15.4f}"
            f"   {shift:+10.2f}")
    return "\n".join(lines)


def main(rundir="runs/pdsp"):
    def out(name):
        return os.path.join(rundir, name)

    ra, rm = resp_mod.load_npz(out("response-direct.npz"))
    rt, _ = resp_mod.load_npz(out("response-traced.npz"))
    planes = list(rm["planes"])
    pitch = float(rm["pitch_mm"])
    y_zoom = G_PLANE + 4 * pitch
    t_zoom = time_at_y(ra, y_zoom)

    # ---- the same plots as for the traced response --------------------------
    for style, extra in (("curves", dict(n_wires=5, impact=1)), ("impacts", {}),
                         ("heatmap", dict(impact=1))):
        plot_mod.plot_response(ra, rm, out(f"direct-response-{style}.png"), style=style,
                               tmin=t_zoom, **extra)
    plot_mod.plot_response(ra, rm, out("direct-response-curves-full.png"),
                           style="curves", n_wires=5, impact=1)
    cl, ct, time = common(ra, rt)
    al = dict(ra, current=cl, time=time)
    at = dict(rt, current=ct, time=time)
    tl = resp_mod.impact_table(al, rm, per_side=5)
    tt = resp_mod.impact_table(at, rm, per_side=5)
    t_end = float(time[-1])
    plot_mod.plot_response_table(tl, out("direct-response-table.png"),
                                 suptitle="PDSP field response, learned directly, whole drift")
    plot_mod.plot_response_table(tl, out("direct-response-table-zoom.png"), tmin=t_zoom,
                                 suptitle=f"the same, from $y = {y_zoom:.1f}$ mm down")
    t_last = float(time[np.flatnonzero(np.abs(ct).sum(axis=(0, 1, 2)) > 0)[-1]])
    plot_mod.plot_response_table(tl, out("direct-response-table-zoom2.png"),
                                 tmin=t_last - 12.0, tmax=t_last + 1.0,
                                 suptitle="the same, last 12 $\\mu$s")

    # ---- comparisons -----------------------------------------------------------
    fig_waveforms(tl, tt, time, planes, pitch, out("compare-direct-waveforms.png"),
                  t_zoom - 1.0, t_last + 1.5,
                  "PDSP field response: learned directly vs traced")
    fig_waveforms(tl, tt, time, planes, pitch, out("compare-direct-waveforms-zoom.png"),
                  t_last - 8.0, t_last + 1.0,
                  "the same, last 8 $\\mu$s")
    fig_difference(tl, tt, time, planes, out("compare-direct-table.png"), t_last - 12.0)
    fig_charge(ra, rt, planes, out("compare-direct-charge.png"))

    blob = torch.load(out("direct.pt"), map_location="cpu", weights_only=False)
    fig_history(blob["history"], out("direct-losses.png"))
    cm = resp_mod.load_combined(blob["drift"], blob["weights"], device=blob["spec"]["device"])
    model, cell, spec, _, dev = direct_mod.load(out("direct.pt"), cm)
    fig_field(model, cell, cm, dev, out("direct-field.png"))
    worst = None
    if model.arrival is not None:
        worst = fig_arrival(model, rt, dev, out("direct-arrival.png"))

    text = numbers(ra, rt, tl, tt, time, planes, pitch)
    if worst is not None:
        text += f"\n\nlearned arrival time vs traced time-to-go below y_R: max |diff| {worst:.3f} us"
    with open(out("compare-direct.txt"), "w") as fp:
        fp.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:])
