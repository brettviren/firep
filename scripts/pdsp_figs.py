#!/usr/bin/env python
"""Field-response figures for the ProtoDUNE-SP example, and the comparison
against the Garfield response that Wire-Cell uses for PDSP / PDHD.

    scripts/pdsp_figs.py [RUNDIR] [GARFIELD_JSON]

RUNDIR (default runs/pdsp) must hold ``response.npz`` (launched just below the
cathode) and ``response-y100.npz`` (launched at y = 100 mm, where the Garfield
paths start).  GARFIELD_JSON defaults to ``dune-garfield-1d565.json.bz2`` found
through ``$WIRECELL_PATH``.

Conventions.  firep reports the current in e/us, positive when an electron
arriving on the wire induces +e on it.  Wire-Cell stores e/ns with the charge of
the electron itself, so a firep current is ``-1000`` times a Wire-Cell one.
Garfield tabulates only the lower half of each wire region; the other half is
its mirror about the readout wire.
"""

from __future__ import annotations

import bz2
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from firep import plot as plot_mod
from firep import response as resp_mod

GARFIELD_NAME = "dune-garfield-1d565.json.bz2"
WCT_TO_FIREP = -1000.0  # e/ns of electron charge -> e/us of induced charge


def find_garfield(name: str = GARFIELD_NAME) -> str:
    for d in os.environ.get("WIRECELL_PATH", "").split(":"):
        p = os.path.join(d, name)
        if d and os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not on WIRECELL_PATH; pass its path explicitly")


def load_garfield(path: str) -> dict:
    """The Garfield response as an ``impact_table``-shaped dict (firep units)."""
    opener = bz2.open if path.endswith(".bz2") else open
    with opener(path, "rt") as fp:
        fr = json.load(fp)["FieldResponse"]
    period = fr["period"] / 1000.0  # ns -> us
    planes, pitch, paths = [], None, []
    for p in sorted(fr["planes"], key=lambda q: -q["PlaneResponse"]["location"]):
        p = p["PlaneResponse"]
        planes.append("uvw"[p["planeid"]])
        pitch = p["pitch"]
        pos = np.array([q["PathResponse"]["pitchpos"] for q in p["paths"]])
        cur = np.array([q["PathResponse"]["current"]["array"]["elements"]
                        for q in p["paths"]]) * WCT_TO_FIREP
        paths.append((pos, cur))
    nt = paths[0][1].shape[1]
    time = fr["tstart"] / 1000.0 + period * np.arange(nt)

    # Each wire region n holds only [(n - 1/2) p, n p]; any signed offset u is
    # read from whichever of u and its mirror -u was tabulated.
    half = int(round(np.abs(paths[0][0]).max() / pitch - 0.5))
    grid = np.arange(-10 * half - 5, 10 * half + 6) * pitch / 10.0
    table = np.zeros((len(planes), len(grid), nt))
    for ip, (pos, cur) in enumerate(paths):
        for m, u in enumerate(grid):
            d = np.minimum(np.abs(pos - u), np.abs(pos + u))
            j = int(np.argmin(d))
            if d[j] > 1e-3:
                raise ValueError(f"no Garfield path at pitchpos {u:g} or its mirror")
            table[ip, m] = cur[j]
    return dict(impact=grid, time=time, table=table, planes=planes, pitch=pitch,
                speed=fr["speed"] * 1000.0, origin=fr["origin"], period=period)


def firep_table(arrays, meta):
    # Six impacts over half a pitch are the bin *edges* of a 0.1-pitch grid.
    return resp_mod.impact_table(arrays, meta, per_side=5, centred=False)


def centroid(t, i):
    w = np.clip(i, 0, None)
    return float((t * w).sum() / w.sum())


def time_at_y(arrays, y):
    ty = np.asarray(arrays["traj_y"])[:, 0]
    tt = np.asarray(arrays["traj_t"])
    return float(np.interp(-y, -ty, tt))


def fig_compare_waveforms(ft, gt, shift, out, tmin, tmax):
    """Selected waveforms, firep over Garfield, one row per plane."""
    pitch = float(ft["pitch"])
    picks = [(0.0, "on the wire"), (0.3 * pitch, "0.3 pitch"),
             (0.5 * pitch, "cell edge"), (1.0 * pitch, "next wire"),
             (2.0 * pitch, "two wires")]
    fig, axes = plt.subplots(3, len(picks), figsize=(3.0 * len(picks), 7.4),
                             sharex=True, constrained_layout=True)
    for ip, name in enumerate(ft["planes"]):
        for k, (u, lab) in enumerate(picks):
            ax = axes[ip, k]
            i_f = int(np.argmin(np.abs(ft["impact"] - u)))
            i_g = int(np.argmin(np.abs(gt["impact"] - u)))
            ax.plot(ft["time"], ft["table"][ip, i_f], lw=1.2, label="firep")
            ax.plot(gt["time"] + shift, gt["table"][ip, i_g], lw=1.0, ls="--",
                    label="Garfield")
            ax.axhline(0, color="0.7", lw=0.5)
            ax.set_xlim(tmin, tmax)
            ax.grid(alpha=0.25)
            if ip == 0:
                ax.set_title(f"u = {u:.3g} mm ({lab})", fontsize=9)
            if k == 0:
                ax.set_ylabel(f"{name}: current [e/$\\mu$s]")
            if ip == 2:
                ax.set_xlabel("time [$\\mu$s]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"PDSP field response, firep vs Garfield "
                 f"(Garfield shifted by {shift:+.2f} $\\mu$s)", fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def compare_numbers(ft, gt, shift):
    """Peak, integral and shape agreement, per plane, over the whole table."""
    lines = ["plane  u[mm]   peak+ firep  Garfield   peak- firep  Garfield"
             "     Q firep  Garfield   rms diff / peak"]
    dt_f = ft["time"][1] - ft["time"][0]
    dt_g = gt["time"][1] - gt["time"][0]
    for ip, name in enumerate(ft["planes"]):
        for u in (0.0, 0.2 * ft["pitch"], 0.4 * ft["pitch"], ft["pitch"]):
            a = ft["table"][ip, int(np.argmin(np.abs(ft["impact"] - u)))]
            b = gt["table"][ip, int(np.argmin(np.abs(gt["impact"] - u)))]
            bi = np.interp(ft["time"], gt["time"] + shift, b, left=0, right=0)
            pk = max(np.abs(a).max(), np.abs(b).max())
            lines.append(
                f"  {name}   {u:5.2f}   {a.max():10.4f} {b.max():9.4f}  "
                f"{a.min():10.4f} {b.min():9.4f}   {a.sum() * dt_f:9.4f} "
                f"{b.sum() * dt_g:9.4f}   {np.sqrt(np.mean((a - bi) ** 2)) / pk:9.3f}")
    return "\n".join(lines)


def main(rundir="runs/pdsp", garfield=None):
    garfield = garfield or find_garfield()

    def out(name):
        return os.path.join(rundir, name)

    # ---- firep's own response, launched just below the cathode -------------
    ra, rm = resp_mod.load_npz(out("response.npz"))
    pitch = float(rm["pitch_mm"])
    y_zoom = 14.13 + 4 * pitch                 # four pitches above the G plane
    t_zoom = time_at_y(ra, y_zoom)
    t_end = float(ra["time"][-1])
    for style, extra in (("curves", dict(n_wires=5)), ("impacts", {}),
                         ("heatmap", {})):
        plot_mod.plot_response(ra, rm, out(f"response-{style}.png"), style=style,
                               tmin=t_zoom, **extra)
    plot_mod.plot_response(ra, rm, out("response-curves-full.png"),
                           style="curves", n_wires=5)
    ftab = firep_table(ra, rm)
    np.savez_compressed(out("response-table.npz"), **{
        k: np.asarray(v) for k, v in ftab.items()})
    plot_mod.plot_response_table(
        ftab, out("response-table.png"),
        suptitle="PDSP field response (firep) over the whole drift")
    plot_mod.plot_response_table(
        ftab, out("response-table-zoom.png"), tmin=t_zoom,
        suptitle=f"the same, from $y = {y_zoom:.1f}$ mm down")
    plot_mod.plot_response_table(
        ftab, out("response-table-zoom2.png"), tmin=t_end - 12.0,
        suptitle="the same, last 12 $\\mu$s")

    # ---- against Garfield, both launched at y = 100 mm ---------------------
    fa, fm = resp_mod.load_npz(out("response-y100.npz"))
    ft = firep_table(fa, fm)
    gt = load_garfield(garfield)
    if not np.isclose(gt["pitch"], ft["pitch"]):
        raise ValueError(f"pitch mismatch: Garfield {gt['pitch']} firep {ft['pitch']}")
    iw = ft["planes"].index("w")
    j0 = int(np.argmin(np.abs(ft["impact"])))
    shift = (centroid(ft["time"], ft["table"][iw, j0])
             - centroid(gt["time"], gt["table"][iw, j0]))
    speed_f = float(resp_mod.drift_speed(np.array([50.0]), fm["velocity"],
                                         fm["temperature_K"],
                                         fm["mobility_cm2_per_Vs"])[0])

    tz = time_at_y(fa, y_zoom)
    tend = float(fa["time"][-1])
    fig_compare_waveforms(ft, gt, shift, out("compare-waveforms.png"),
                          tz - 1.0, tend + 1.0)
    fig_compare_waveforms(ft, gt, shift, out("compare-waveforms-zoom.png"),
                          tend - 8.0, tend + 0.5)
    gshift = dict(gt, time=gt["time"] + shift)
    for tag, t0 in (("", None), ("-zoom", tz)):
        plot_mod.plot_response_table(
            gshift, out(f"garfield-table{tag}.png"), tmin=t0,
            tmax=tend + 1.0,
            suptitle=("Garfield PDSP response" + (f", from y = {y_zoom:.1f} mm"
                      if t0 else "") + f" (shifted {shift:+.2f} $\\mu$s)"))
        plot_mod.plot_response_table(
            ft, out(f"firep-y100-table{tag}.png"), tmin=t0, tmax=tend + 1.0,
            suptitle="firep PDSP response, launched at y = 100 mm"
                     + (f", from y = {y_zoom:.1f} mm" if t0 else ""))

    report = [
        f"Garfield: {garfield}",
        f"  origin {gt['origin']} mm, speed {gt['speed']:.4f} mm/us, "
        f"tick {gt['period']:.5f} us, {gt['table'].shape[-1]} ticks",
        f"firep:    launched at y = {fa['traj_y'][0, 0]:.2f} mm, "
        f"{fm['velocity']} speed at 500 V/cm {speed_f:.4f} mm/us",
        f"time shift applied to Garfield (collection centroid, u = 0): "
        f"{shift:+.3f} us",
        "",
        compare_numbers(ft, gt, shift),
    ]
    text = "\n".join(report)
    with open(out("compare.txt"), "w") as fp:
        fp.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:])
