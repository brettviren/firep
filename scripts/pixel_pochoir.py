#!/usr/bin/env python
"""Pixel anode in 3D: reproduce the pochoir finite-difference field response.

    scripts/pixel_pochoir.py [OUTDIR] [FIGDIR]

Solves the drift cell and the 5x5 and 9x9 weighting domains with the spectral
single layer of ``firep.pixel``, drifts pochoir's 100 launch points, builds its
625 launches around the centre pad, induces the current on that pad, and
compares everything with the stored pochoir results under
``$POCHOIR_STORE`` (default: the colleague's test store).  Solutions are cached
in OUTDIR (default ``runs/pixel``); figures go to OUTDIR as PNG and to FIGDIR
(default ``docs/figs``) as PDF, with every number the note quotes in
``FIGDIR/pixel.tex``.

The geometry and boundary conditions are pochoir's, not a physical anode:
4.4 mm pitch, 3.8 mm pads with ~0.4 mm rounded corners, 0.1 mm thick at
y = 10.0-10.1 mm (two sheets), the cathode at y = 149.9 mm and -7000 V, a
zero-slope bottom face for the drift solve, and for the weighting solves a
grounded bottom with zero-slope top and sides.
"""

from __future__ import annotations

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from firep import pixel as X

STORE = os.environ.get("POCHOIR_STORE",
                       "/nfs/data/1/yousen/nd_field_response/pochoir/test")
DEV = os.environ.get("FIREP_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
PITCH, TOP, V_CATH = 4.4, 149.9, -7000.0
PLANES = (10.0, 10.1)
SHAPE = X.PadShape("rounded_square", 1.9, 0.4)
SC = [0.02, 0.05, 0.1, 0.2, 0.4, 1.0]
POCHOIR_PAD_CENTRE = -0.05  # mm: pochoir's 38-voxel pad is centred on a voxel edge
POCHOIR_W_CENTRE = 19.75  # mm: the centre pad of its 9x9 weighting domain


def poly(d):
    return [(p, q) for p in range(d + 1) for q in range(d + 1 - p)]


def drift_basis(_ring=0):
    return poly(6) + [(p, q, s) for s in SC for (p, q) in poly(6)]


def weight_basis(ring):
    d = {0: 6, 1: 6, 2: 4}.get(ring, 2)
    sc = SC if ring <= 1 else [0.05, 0.2, 1.0]
    return poly(d) + [(p, q, s) for s in sc for (p, q) in poly(d)]


def solve(out, kind, N, grid, modes, basis, count, values):
    path = os.path.join(out, f"{kind}{N}.pt")
    if os.path.exists(path):
        st = torch.load(path, weights_only=False)
        return X.SheetPotential.from_state(st["field"], device=DEV), st["info"]
    arr = X.PadArray(pitch=PITCH, ncell=N, shape=SHAPE, y_pad=PLANES, top=TOP)
    ii, jj, own, dep = X.pad_nodes(arr, grid, count, np.random.default_rng(1))
    t0 = time.time()
    ssl = X.SpectralSingleLayer(arr, kind, modes, grid, basis, device=DEV, nodes=(ii, jj))
    centre = len(arr.pad_centres()) // 2
    target = np.zeros(len(own)) if kind == "drift" else (own == centre).astype(float)
    offset = V_CATH if kind == "drift" else 0.0
    train = np.random.default_rng(2).uniform(size=len(own)) < 0.75
    coef, rms, mx = X.fit_on_nodes(ssl, target, offset, train)
    info = dict(columns=len(ssl.labels), modes=(modes + 1) ** 2, grid=ssl.n, h=ssl.h,
                seconds=time.time() - t0, rms=rms, max=mx, nodes=len(own))
    field = X.SheetPotential.from_fit(ssl, coef, offset)
    # The exact pad-surface residual at every node, from the on-plane design,
    # for the network correction.
    h = ssl.h
    xs, zs = (ii - ssl.n // 2) * h, (jj - ssl.n // 2) * h
    nodes = np.concatenate([np.column_stack((xs, np.full(len(xs), yp), zs)) for yp in ssl.planes])
    resid = (ssl.plane_design @ coef + offset - torch.as_tensor(target, device=DEV)).reshape(-1)
    torch.save(dict(field=field.state(), info=info, nodes=nodes, resid=resid.cpu(),
                    train=np.tile(train, len(ssl.planes))), path)
    print(f"{kind} {N}x{N}: {info['columns']} columns, {info['modes']} modes, "
          f"{info['seconds']:.0f} s; pad-surface residual rms {rms:.2e} max {mx:.2e}")
    return field, info


def load_pochoir():
    z = lambda k: np.load(os.path.join(STORE, "store", k + ".npz"))[k.split("/")[1]]
    return dict(drift=z("potential/drift3d"), paths=z("paths/drift3d"),
                starts=z("starts/drift3d"),
                weight_starts=np.load(os.path.join(STORE, "startpoints.npy")),
                current=np.load(os.path.join(STORE, "fr_4p4pitch_3.8pix_nogrid_10pathsperpixel.npy")))


def wrap(u):
    return (u + PITCH / 2) % PITCH - PITCH / 2


def main(out="runs/pixel", figdir="docs/figs"):
    os.makedirs(out, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)
    V = {}

    def value(name, text):
        # e-notation reads badly in prose: 2.0e-03 -> $2.0\times10^{-3}$
        import re
        m = re.fullmatch(r"(-?[0-9.]+)e([+-]?)0*([0-9]+)", text)
        if m:
            mant, sign, ex = m.groups()
            text = f"${mant}\\times10^{{{'-' if sign == '-' else ''}{ex}}}$"
        V[name] = text

    def save(fig, name):
        fig.savefig(os.path.join(out, name + ".png"), dpi=130)
        fig.savefig(os.path.join(figdir, "pixel-" + name + ".pdf"))
        plt.close(fig)

    drift, dinfo = solve(out, "drift", 1, 1761, 88, drift_basis, 20000, None)
    w9, w9info = solve(out, "weighting", 9, 3961, 400, weight_basis, 60000, None)
    w5, w5info = solve(out, "weighting", 5, 2201, 220, weight_basis, 40000, None)
    ref = load_pochoir()
    t64 = lambda a: torch.as_tensor(np.asarray(a, float), device=DEV, dtype=torch.float64)

    for tag, inf in (("D", dinfo), ("WN", w9info), ("WF", w5info)):
        value(f"pix{tag}Cols", f"{inf['columns']}")
        value(f"pix{tag}Modes", f"{inf['modes']:,}")
        value(f"pix{tag}M", f"{round(inf['modes'] ** 0.5) - 1}")
        value(f"pix{tag}Grid", f"{inf['grid']}")
        value(f"pix{tag}PerSheet", f"{inf['columns'] // len(PLANES)}")
        value(f"pix{tag}Rms", f"{inf['rms']:.1e}")
        value(f"pix{tag}Max", f"{inf['max']:.1e}")
        value(f"pix{tag}Sec", f"{inf['seconds']:.0f}")
    value("pixDGridUm", f"{dinfo['h'] * 1e3:.1f}")
    for tag, N in (("D", 1), ("WN", 9), ("WF", 5)):
        value(f"pix{tag}Classes", f"{len(X.PadArray(pitch=PITCH, ncell=N).classes())}")

    # ---- the network correction, measured -------------------------------------
    st = torch.load(os.path.join(out, "drift1.pt"), weights_only=False)
    if "nodes" in st:
        _, rb, ra, mb, ma = X.train_correction(
            drift, t64(st["nodes"]), st["resid"].to(DEV), torch.as_tensor(st["train"], device=DEV),
            steps=int(os.environ.get("PIXEL_CORRECTION_STEPS", "3000")))
        value("pixSirenBefore", f"{rb:.4f}")
        value("pixSirenAfter", f"{ra:.4f}")
        value("pixSirenMaxBefore", f"{mb:.3f}")
        value("pixSirenMaxAfter", f"{ma:.3f}")

    # ---- the drift potential against pochoir, on its voxel grid -------------
    P = ref["drift"]
    i = np.arange(P.shape[0])
    xs = wrap(i * 0.1 - POCHOIR_PAD_CENTRE)
    Xg, Zg = np.meshgrid(xs, xs, indexing="ij")
    rows = []
    for kz in (102, 105, 110, 120, 150, 200, 400, 1000):
        pts = np.column_stack((Xg.ravel(), np.full(Xg.size, kz * 0.1), Zg.ravel()))
        ours = drift.evaluate(t64(pts)).cpu().numpy().reshape(Xg.shape)
        d = ours - P[:, :, kz]
        rows.append((kz * 0.1, np.sqrt((d ** 2).mean()), np.abs(d).max()))
    value("pixDriftDiffNear", f"{rows[1][1]:.2f}")
    value("pixDriftDiffBulk", f"{rows[4][1]:.2f}")
    value("pixDriftDiffFar", f"{rows[6][1]:.2f}")
    ef = float(-np.gradient(P, 0.1, axis=2)[:, :, 800].mean())
    _, g = drift.evaluate(t64([[0.0, 80.0, 0.0]]), grad=True)
    value("pixEPochoir", f"{ef:.4f}")
    value("pixEOurs", f"{float(-g[0, 1]):.4f}")

    # ---- the weighting potential: plane average against height ---------------
    ys = np.array([10.2, 10.5, 11, 12, 15, 20, 30, 60, 100, 148])
    xs9 = np.linspace(-19.8, 19.8, 199)
    X9, Z9 = np.meshgrid(xs9, xs9, indexing="ij")
    mean_ours, mean_w5 = [], []
    for y in ys:
        pts = np.column_stack((X9.ravel(), np.full(X9.size, y), Z9.ravel()))
        mean_ours.append(float(w9.evaluate(t64(pts), work_dtype=torch.float32).mean()))
    mean_poch = None
    wpath = os.path.join(STORE, "store", "potential", "weight3d.npz")
    if os.path.exists(wpath):
        W = np.load(wpath)["weight3d"]
        mean_poch = np.array([W[:, :, int(round(y * 10))].mean() for y in ys])
        c = int(round(POCHOIR_W_CENTRE * 10))
        axis_p = np.array([W[c, c, int(round(y * 10))] for y in ys])
        del W
    axis_o = w9.evaluate(t64(np.column_stack((np.zeros(len(ys)), ys, np.zeros(len(ys)))))).cpu().numpy()
    value("pixPlateau", f"{mean_ours[-1]:.4f}")
    if mean_poch is not None:
        value("pixPochoirMeanThirty", f"{mean_poch[6]:.1e}")
        value("pixPochoirMeanHundred", f"{mean_poch[8]:.0e}")
        value("pixAxisOursNear", f"{axis_o[0]:.4f}")
        value("pixAxisPochoirNear", f"{axis_p[0]:.4f}")
        value("pixAxisOursTwelve", f"{axis_o[3]:.3f}")
        value("pixAxisPochoirTwelve", f"{axis_p[3]:.3f}")
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
        ax[0].plot(ys - 10.1, mean_ours, "o-", label="firep, 9x9: plane average")
        ax[0].plot(ys - 10.1, mean_poch, "s--", label="pochoir, 9x9: plane average")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("height above the pads [mm]")
        ax[0].set_ylabel("weighting potential")
        ax[0].set_title("the Neumann-top plateau, and pochoir's unconverged far field", fontsize=9)
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)
        ax[1].semilogy(ys - 10.1, axis_o, "o-", label="firep, above the sensing pad")
        ax[1].semilogy(ys - 10.1, np.maximum(axis_p, 1e-9), "s--", label="pochoir")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("height above the pads [mm]")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
        save(fig, "weight-height")

    # ---- paths ---------------------------------------------------------------
    ppath = os.path.join(out, "paths.pt")
    ps = ref["starts"]
    starts = t64(np.column_stack((ps[:, 0] - POCHOIR_PAD_CENTRE, ps[:, 2], ps[:, 1] - POCHOIR_PAD_CENTRE)))
    if os.path.exists(ppath):
        paths = torch.load(ppath, weights_only=False)
    else:
        t0 = time.time()
        pa = X.trace(drift, starts, tick=0.05, t_max=320.0 - 0.05, substeps=4)
        paths = dict(t=pa.t.cpu(), xyz=pa.xyz.cpu(), arrived=pa.arrived.cpu(), pad=pa.pad.cpu(),
                     seconds=time.time() - t0)
        torch.save(paths, ppath)
    PP = ref["paths"]
    tt = paths["t"].numpy()
    t_reach = lambda zt, t, y: np.array([np.interp(-y, -z, t) for z in zt])
    t_top_p = t_reach(PP[:, :, 2], np.arange(PP.shape[1]) * 0.05, max(PLANES))
    arr_o = paths["arrived"].numpy()
    value("pixArrMinOurs", f"{arr_o.min():.2f}")
    value("pixArrMaxOurs", f"{arr_o.max():.2f}")
    value("pixArrMinP", f"{t_top_p.min():.2f}")
    value("pixArrMaxP", f"{t_top_p.max():.2f}")
    t20o = t_reach(paths["xyz"][:, :, 1].numpy(), tt, 20.0)
    t20p = t_reach(PP[:, :, 2], np.arange(PP.shape[1]) * 0.05, 20.0)
    value("pixTwentyOurs", f"{t20o.mean():.3f}")
    value("pixTwentyP", f"{t20p.mean():.3f}")
    value("pixSpread", f"{arr_o.max() - arr_o.min():.2f}")

    # ---- the 625 launches and the induced current ---------------------------
    S = ref["weight_starts"]
    rel = S[:, :2] - POCHOIR_W_CENTRE
    xyz = paths["xyz"]
    loc0 = np.column_stack((wrap(xyz[:, 0, 0].numpy()), wrap(xyz[:, 0, 2].numpy())))
    idx, shift = [], []
    for x, z in rel:
        lx, lz = wrap(x), wrap(z)
        j = int(np.argmin(np.hypot(loc0[:, 0] - lx, loc0[:, 1] - lz)))
        idx.append(j)
        shift.append((x - xyz[j, 0, 0].item(), z - xyz[j, 0, 2].item()))
    idx = np.array(idx)
    shift = torch.as_tensor(np.array(shift))
    full = xyz[idx].clone()
    full[:, :, 0] += shift[:, 0, None]
    full[:, :, 2] += shift[:, 1, None]
    pad = paths["pad"][idx] + shift
    on = torch.as_tensor((np.abs(pad.numpy()) < 1.0).all(1))
    cpath = os.path.join(out, "currents.npz")
    if os.path.exists(cpath):
        cur = np.load(cpath)
        dq, Q = cur["ours"], cur["Q"]
    else:
        Q = X.induced_charge(w9, full.to(DEV), paths["arrived"][idx].to(DEV), paths["t"].to(DEV),
                             on.to(DEV), chunk_paths=4).cpu().numpy()
        dq = np.diff(Q, axis=1)
        np.savez(cpath, ours=dq, Q=Q, starts=rel, on=on.numpy())
    # the same launches with the 5x5 weighting domain
    cpath5 = os.path.join(out, "currents5.npz")
    if os.path.exists(cpath5):
        dq5 = np.load(cpath5)["ours"]
    else:
        Q5 = X.induced_charge(w5, full.to(DEV), paths["arrived"][idx].to(DEV), paths["t"].to(DEV),
                              on.to(DEV), chunk_paths=4).cpu().numpy()
        dq5 = np.diff(Q5, axis=1)
        np.savez(cpath5, ours=dq5, Q=Q5)
    refq = ref["current"] * 5e-8  # e/s over 0.05 us ticks -> e per tick
    on = on.numpy()
    # One definition of "collected" for both codes: the time by which 99% of
    # the charge the launch will induce on the sensing pad has been induced.
    def t99(q_per_tick):
        c = np.cumsum(q_per_tick, axis=1)
        return (np.argmax(c >= 0.99 * c[:, -1:], axis=1) + 1) * 0.05
    t99o, t99p = t99(dq[on]), t99(refq[on])
    value("pixColMinOurs", f"{t99o.min():.2f}")
    value("pixColMaxOurs", f"{t99o.max():.2f}")
    value("pixColMinP", f"{t99p.min():.2f}")
    value("pixColMaxP", f"{t99p.max():.2f}")
    value("pixColSpreadOurs", f"{t99o.max() - t99o.min():.2f}")
    value("pixColSpreadP", f"{t99p.max() - t99p.min():.2f}")
    value("pixCollectOurs", f"{dq[on].sum(1).mean():.4f}")
    value("pixCollectP", f"{refq[on].sum(1).mean():.4f}")
    value("pixNOn", f"{int(on.sum())}")
    tc = np.arange(dq.shape[1]) * 0.05 + 0.025

    def pick(x, z):
        return int(np.argmin(np.hypot(rel[:, 0] - x, rel[:, 1] - z)))

    picks = [(0.27, 0.27, "centre pad, near its centre"), (1.59, 0.27, "centre pad, near an edge"),
             (2.03, 2.03, "the gap crossing"), (-2.37, 0.27, "over the next pad"),
             (-4.13, -4.13, "over the diagonal pad"), (-8.53, -8.53, "two pads away")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for a, (x, z, lab) in zip(axes.ravel(), picks):
        j = pick(x, z)
        a.plot(tc, refq[j] / 0.05, lw=1.2, label="pochoir (finite differences)")
        a.plot(tc, dq[j] / 0.05, "--", lw=1.2, label="firep, $9\\times9$ weighting domain")
        a.plot(tc, dq5[j] / 0.05, ":", lw=1.5, color="C3", label="firep, $5\\times5$ weighting domain")
        a.set_xlim(72, 88)
        a.set_title(f"{lab}: launch ({rel[j, 0]:.2f}, {rel[j, 1]:.2f}) mm", fontsize=9)
        a.set_xlabel(r"time [$\mu$s]")
        a.set_ylabel(r"current [e/$\mu$s]")
        a.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    save(fig, "currents")
    jc = pick(2.03, 2.03)
    value("pixCornerPeakOurs", f"{dq[jc].max() / 0.05:.1f}")
    value("pixCornerPeakP", f"{refq[jc].max() / 0.05:.1f}")
    j0 = pick(0.27, 0.27)
    value("pixCentrePeakOurs", f"{dq[j0].max() / 0.05:.2f}")
    value("pixCentrePeakP", f"{refq[j0].max() / 0.05:.2f}")
    near = on & (np.abs(rel[:, 0]) < 1.5) & (np.abs(rel[:, 1]) < 1.5)
    d = (dq - refq)[near]
    # 5x5 against 9x9: the far-field induction the smaller domain misses
    w5far = [float(np.abs(dq5[pick(x, z)] - dq[pick(x, z)]).max() / np.abs(dq[pick(x, z)]).max())
             for x, z, _ in picks]
    value("pixFiveCentreDiff", f"{100 * w5far[0]:.0f}")
    value("pixFiveTwoAwayDiff", f"{100 * w5far[-1]:.0f}")
    value("pixFiveNextDiff", f"{100 * w5far[3]:.0f}")
    value("pixFiveDiagDiff", f"{100 * w5far[4]:.0f}")
    early = tc < 80.0
    value("pixCollectFive", f"{dq5[on].sum(1).mean():.4f}")
    value("pixFiveEarlyCentre", f"{dq5[pick(0.27, 0.27)][early].sum():.3f}")
    value("pixNineEarlyCentre", f"{dq[pick(0.27, 0.27)][early].sum():.3f}")
    value("pixInteriorRms", f"{np.sqrt((d ** 2).mean()) / np.abs(refq[near]).max() * 100:.2f}")

    # ---- the time dispersion over the cell: arrival and peak ----------------
    cell = on & (rel[:, 0] >= 0) & (rel[:, 1] >= 0)
    order = np.argsort(rel[cell, 0] * 100 + rel[cell, 1])
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    col_o = np.full(len(rel), np.nan)
    col_p = np.full(len(rel), np.nan)
    col_o[on], col_p[on] = t99o, t99p
    for a, (vals, lab) in zip(ax, ((col_o, r"firep: 99% collected [$\mu$s]"),
                                   (col_p, r"pochoir: 99% collected [$\mu$s]"),
                                   (dq.max(1) / 0.05, r"firep peak current [e/$\mu$s]"))):
        sc = a.scatter(rel[on, 0], rel[on, 1], c=vals[on], s=60, marker="s", cmap="viridis")
        fig.colorbar(sc, ax=a, label=lab)
        a.set_aspect("equal")
        a.set_xlabel("launch x [mm]")
        a.set_ylabel("launch z [mm]")
    fig.suptitle("launches that collect on the centre pad (pochoir's launch grid covers "
                 "it up to 2.03 mm on each axis)", fontsize=10)
    save(fig, "dispersion")

    # ---- slices of the drift field, with streamlines -------------------------
    xs3 = np.linspace(-1.5 * PITCH, 1.5 * PITCH, 331)
    ysl = np.linspace(8.0, 18.0, 301)
    XS, YS = np.meshgrid(xs3, ysl, indexing="xy")
    pts = np.column_stack((wrap(XS.ravel()), YS.ravel(), np.zeros(XS.size)))
    v, g = drift.evaluate(t64(pts), grad=True)
    v = v.cpu().numpy().reshape(XS.shape)
    g = g.cpu().numpy()
    ex, ey = -g[:, 0].reshape(XS.shape), -g[:, 1].reshape(XS.shape)
    fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    im = ax.pcolormesh(XS, YS, v, cmap="viridis", shading="auto", rasterized=True)
    ax.contour(XS, YS, v, levels=30, colors="k", linewidths=0.3)
    ax.streamplot(XS, YS, -ex, -ey, color="w", density=1.6, linewidth=0.6, arrowsize=0.6)
    for kx in (-1, 0, 1):
        ax.plot([kx * PITCH - 1.9, kx * PITCH + 1.9], [10.05, 10.05], color="orange", lw=3)
    fig.colorbar(im, ax=ax, label="potential [V]")
    ax.set_xlabel("x [mm] (z = 0, through the pad centres)")
    ax.set_ylabel("y [mm]")
    ax.set_title("drift potential and electron drift lines near the pads", fontsize=10)
    ax.set_ylim(8, 18)
    save(fig, "drift-slice")

    # ---- slices of the weighting potential -----------------------------------
    for tagw, wf, half in (("9", w9, 4.5), ("5", w5, 2.5)):
        xs4 = np.linspace(-half * PITCH, half * PITCH, 401)
        ysl = np.linspace(8.0, 30.0, 221)
        XS, YS = np.meshgrid(xs4, ysl, indexing="xy")
        pts = np.column_stack((XS.ravel(), YS.ravel(), np.zeros(XS.size)))
        v = wf.evaluate(t64(pts), work_dtype=torch.float32).cpu().numpy().reshape(XS.shape)
        fig, ax = plt.subplots(figsize=(10, 3.6), constrained_layout=True)
        im = ax.pcolormesh(XS, YS, np.log10(np.clip(v, 1e-4, 1)), cmap="magma",
                           shading="auto", rasterized=True, vmin=-4, vmax=0)
        ax.contour(XS, YS, v, levels=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.9], colors="w", linewidths=0.5)
        for kx in range(-int(half), int(half) + 1):
            ax.plot([kx * PITCH - 1.9, kx * PITCH + 1.9], [10.05, 10.05],
                    color="c" if kx else "orange", lw=2)
        fig.colorbar(im, ax=ax, label="log$_{10}$ weighting potential")
        ax.set_xlabel("x [mm] (z = 0)")
        ax.set_ylabel("y [mm]")
        ax.set_title(f"weighting potential of the centre pad, {tagw}x{tagw} domain", fontsize=10)
        save(fig, f"weight{tagw}-slice")
    # the 5x5 and 9x9 weighting potentials along the axis: the domain size matters
    ya = np.linspace(10.2, 60, 200)
    a5 = w5.evaluate(t64(np.column_stack((np.zeros(len(ya)), ya, np.zeros(len(ya)))))).cpu().numpy()
    a9 = w9.evaluate(t64(np.column_stack((np.zeros(len(ya)), ya, np.zeros(len(ya)))))).cpu().numpy()
    value("pixPlateauFive", f"{float(a5[-1]):.4f}")

    # ---- response table: current on the centre pad vs launch x, time ----------
    line = np.flatnonzero(np.abs(rel[:, 1] - 0.27) < 1e-6)
    line = line[np.argsort(rel[line, 0])]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True, sharey=True)
    from matplotlib.colors import SymLogNorm
    m = np.abs(refq[line]).max() / 0.05
    for a, (arrz, lab) in zip(ax, ((refq, "pochoir"), (dq, "firep"), (dq - refq, "firep - pochoir"))):
        im = a.pcolormesh(tc, rel[line, 0], arrz[line] / 0.05, cmap="RdBu_r", shading="auto",
                          norm=SymLogNorm(linthresh=1e-3 * m, vmin=-m, vmax=m), rasterized=True)
        a.set_xlim(76, 88)
        a.set_title(lab, fontsize=10)
        a.set_xlabel(r"time [$\mu$s]")
    ax[0].set_ylabel("launch x [mm], z = 0.27 mm")
    fig.colorbar(im, ax=ax, label=r"current on the centre pad [e/$\mu$s]")
    save(fig, "table")

    with open(os.path.join(figdir, "pixel.tex"), "w") as fp:
        fp.write("% generated by scripts/pixel_pochoir.py -- do not edit\n")
        for k, v in V.items():
            fp.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    for k, v in V.items():
        print(f"{k:24s} {v}")


if __name__ == "__main__":
    main(*sys.argv[1:])
