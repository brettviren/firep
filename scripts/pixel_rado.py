#!/usr/bin/env python
"""Against the newer pochoir calculation with a PCB (Sec. 17).

The store ``store_larpix_v2a_5x5_15cm_pathsongrid`` (R. Fan's pochoir fork)
models LArPix-v2a pads: 3.5 mm squares with 0.7 mm corner radius at 4.4 mm
pitch, the PCB between them a no-flux (zero-slope) surface flush with the
pad tops at y = 10.0 mm, in both the drift and the weighting solve.  Drift:
cathode -7500 V at y = 160.0 mm, periodic sides.  Weighting: 5 x 5 pads, the
far face grounded at 159.9 mm, zero-slope sides.  Paths start at y = 159.9 mm
on a 0.1 mm grid; currents are in e/ns.

In firep that is the charged-PCB slab (``pcb = 0``) with a single pad sheet
on the surface, and the weighting solve uses the same slab (zero slope below,
fixed top).  A 9 x 9 weighting solve of the same model is added.

    python scripts/pixel_rado.py [runs/pixel/rado] [docs/figs]
"""

import os
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from firep import pixel as X  # noqa: E402
from pixel_pochoir import SC, poly  # noqa: E402

STORE = os.environ.get("RADO_STORE", "/nfs/data/1/bviren/rado/results/store_larpix_v2a_5x5_15cm_pathsongrid")
DEV = os.environ.get("FIREP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64
PITCH, Y0 = 4.4, 10.0
SHAPE = X.PadShape("rounded_square", 1.75, 0.7)
V_CATH, T_DRIFT, T_WEIGHT = -7500.0, 160.0, 159.9
Y_START, TICK, NT = 159.9, 0.05, 4000
W_CENTRE = 10.9  # mm: the sensing pad's centre in the store's weighting domain


def drift_basis(_ring=0):
    return poly(6) + [(p, q, s) for s in SC for (p, q) in poly(6)]


def weight_basis(ring):
    d = {0: 6, 1: 6, 2: 4}.get(ring, 2)
    sc = SC if ring <= 1 else [0.05, 0.2, 1.0]
    return poly(d) + [(p, q, s) for s in sc for (p, q) in poly(d)]


def solve(out, name, N, grid, modes, top, target_kind):
    path = os.path.join(out, f"{name}.pt")
    if os.path.exists(path):
        st = torch.load(path, weights_only=False)
        return X.SheetPotential.from_state(st["field"], device=DEV), st["info"]
    # the slab of both solves: zero slope on the PCB surface (charged, pcb=0),
    # the far face fixed
    arr = X.PadArray(pitch=PITCH, ncell=N, shape=SHAPE, y_pad=Y0, top=top, pcb=0.0)
    ii, jj, own, _ = X.pad_nodes(arr, grid, {1: 20000, 5: 40000}.get(N, 60000), np.random.default_rng(1))
    t0 = time.time()
    ssl = X.SpectralSingleLayer(arr, "drift", modes, grid, drift_basis if N == 1 else weight_basis,
                                device=DEV, nodes=(ii, jj))
    centre = len(arr.pad_centres()) // 2
    target = np.zeros(len(own)) if target_kind == "drift" else (own == centre).astype(float)
    offset = V_CATH if target_kind == "drift" else 0.0
    train = np.random.default_rng(2).uniform(size=len(own)) < 0.75
    coef, rms, mx = X.fit_on_nodes(ssl, target, offset, train)
    field = X.SheetPotential.from_fit(ssl, coef, offset)
    info = dict(columns=len(ssl.labels), modes=(modes + 1) ** 2, rms=rms, max=mx, seconds=time.time() - t0)
    torch.save(dict(field=field.state(), info=info), path)
    print(f"{name}: {info['columns']} columns, fit rms {rms:.2e} max {mx:.2e}, {info['seconds']:.0f} s",
          flush=True)
    return field, info


def wrap(u):
    return (u + PITCH / 2) % PITCH - PITCH / 2


def main(out="runs/pixel/rado", figdir="docs/figs"):
    os.makedirs(out, exist_ok=True)
    V = {}

    def value(name, text):
        m = re.fullmatch(r"(-?[0-9.]+)e([+-]?)0*([0-9]+)", text)
        if m:
            mant, sign, ex = m.groups()
            text = f"${mant}\\times10^{{{'-' if sign == '-' else ''}{ex}}}$"
        V[name] = text

    t64 = lambda a: torch.as_tensor(np.asarray(a, float), device=DEV, dtype=DT)
    drift, di = solve(out, "drift", 1, 1761, 88, T_DRIFT, "drift")
    w5, w5i = solve(out, "weighting5", 5, 2201, 220, T_WEIGHT, "weight")
    w9, w9i = solve(out, "weighting9", 9, 3961, 400, T_WEIGHT, "weight")
    for tag, inf in (("D", di), ("WF", w5i), ("WN", w9i)):
        value(f"rado{tag}Rms", f"{inf['rms']:.1e}")
        value(f"rado{tag}Max", f"{inf['max']:.1e}")
        value(f"rado{tag}Cols", f"{inf['columns']}")

    # ---- fields against the store ------------------------------------------
    ld = lambda k: np.load(os.path.join(STORE, k + ".npz"))[k.split("/")[1]]
    P = ld("potential/drift3d")  # (44, 44, 1601), pad centred on index 0
    xs = wrap(np.arange(44) * 0.1)
    Xg, Zg = np.meshgrid(xs, xs, indexing="ij")
    rows = []
    for kz in (101, 102, 105, 110, 120, 150, 200, 400, 1000):
        ours = drift.evaluate(t64(np.column_stack((Xg.ravel(), np.full(Xg.size, kz * 0.1), Zg.ravel())))
                              ).cpu().numpy().reshape(Xg.shape)
        d = ours - P[:, :, kz]
        rows.append((kz * 0.1, float(np.sqrt((d ** 2).mean())), float(np.abs(d).max())))
        print(f"drift potential at y = {kz * 0.1:6.1f}: rms diff {rows[-1][1]:.3f} V, max {rows[-1][2]:.3f} V")
    value("radoDriftDiffNear", f"{rows[1][1]:.2f}")
    value("radoDriftDiffBulk", f"{rows[5][1]:.2f}")
    value("radoDriftDiffFar", f"{rows[7][1]:.2f}")
    ef = float(-np.gradient(P, 0.1, axis=2)[:, :, 800].mean())
    _, g = drift.evaluate(t64([[0.0, 80.0, 0.0]]), grad=True)
    value("radoEPochoir", f"{ef:.3f}")
    value("radoEOurs", f"{float(-g[0, 1]):.3f}")
    W = np.load(os.path.join(STORE, "potential", "weight3d.npz"))["weight3d"]
    c = int(round(W_CENTRE / 0.1))
    ys = np.array([10.1, 10.2, 10.5, 11.0, 12.0, 15.0, 20.0, 30.0, 60.0, 100.0, 150.0])
    axis_p = np.array([W[c, c, int(round(y * 10))] for y in ys])
    mean_p = np.array([W[:, :, int(round(y * 10))].mean() for y in ys])
    # Gauss: with zero-slope sides and PCB surface and a grounded top, the plane
    # average must fall linearly with height above the pads
    mz = W.mean(axis=(0, 1))
    slope_p = {y: float(-(mz[int(y * 10) + 1] - mz[int(y * 10) - 1]) / 0.2) for y in (20.0, 150.0)}
    del W
    axis5 = w5.evaluate(t64(np.column_stack((np.zeros(len(ys)), ys, np.zeros(len(ys)))))).cpu().numpy()
    axis9 = w9.evaluate(t64(np.column_stack((np.zeros(len(ys)), ys, np.zeros(len(ys)))))).cpu().numpy()
    xs5 = np.linspace(-11.0, 11.0, 111)
    X5, Z5 = np.meshgrid(xs5, xs5, indexing="ij")
    mean5 = np.array([float(w5.evaluate(t64(np.column_stack((X5.ravel(), np.full(X5.size, y), Z5.ravel()))),
                                        work_dtype=torch.float32).mean()) for y in ys])
    slope_o = float((mean5[8] - mean5[9]) / (ys[9] - ys[8]))
    value("radoSlopeTwentyP", f"{slope_p[20.0]:.1e}")
    value("radoSlopeOneFiftyP", f"{slope_p[150.0]:.1e}")
    value("radoSlopeOurs", f"{slope_o:.1e}")
    value("radoMeanHundredP", f"{mean_p[9]:.4f}")
    value("radoMeanHundredOurs", f"{mean5[9]:.4f}")
    for y, a, b, e in zip(ys, axis_p, axis5, axis9):
        print(f"weighting on the axis at y = {y:6.1f}: pochoir {a:.5f}  firep 5x5 {b:.5f}  9x9 {e:.5f}")
    value("radoAxisNearP", f"{axis_p[0]:.4f}")
    value("radoAxisNearOurs", f"{axis5[0]:.4f}")
    value("radoAxisTwoP", f"{axis_p[4]:.4f}")
    value("radoAxisTwoOurs", f"{axis5[4]:.4f}")
    value("radoAxisTwoNine", f"{axis9[4]:.4f}")
    value("radoAxisTwentyP", f"{axis_p[6]:.4f}")
    value("radoAxisTwentyOurs", f"{axis5[6]:.4f}")
    value("radoAxisTwentyNine", f"{axis9[6]:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    ax[0].plot(ys - Y0, mean_p, "s--", label="pochoir (R. Fan), 5x5: plane average")
    ax[0].plot(ys - Y0, mean5, "o-", label="firep, 5x5: plane average")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("height above the pads [mm]")
    ax[0].set_ylabel("weighting potential")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].semilogy(ys - Y0, axis_p, "s--", label="pochoir, 5x5")
    ax[1].semilogy(ys - Y0, axis5, "o-", label="firep, 5x5")
    ax[1].semilogy(ys - Y0, axis9, "^:", label="firep, 9x9")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("height above the pads [mm]")
    ax[1].set_ylabel("weighting potential above the sensing pad")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.savefig(os.path.join(figdir, "pixel-rado-weight.pdf"))
    fig.savefig(os.path.join(out, "weight.png"), dpi=120)
    plt.close(fig)

    # ---- paths: the store's start grid over one cell, by symmetry the wedge --
    PP = ld("paths/drift3d")  # (1936, 4000, 3), pochoir coordinates (x, z-transverse, y-height)
    S0 = ld("starts/drift3d")
    loc = np.column_stack((wrap(S0[:, 0]), wrap(S0[:, 1])))
    wedge = (loc[:, 0] >= 0) & (loc[:, 1] >= 0) & (loc[:, 1] <= loc[:, 0] + 1e-9)
    starts = loc[wedge]
    ppath = os.path.join(out, "paths.pt")
    if os.path.exists(ppath):
        pa = torch.load(ppath, weights_only=False)
    else:
        t0 = time.time()
        tr = X.trace(drift, t64(np.column_stack((starts[:, 0], np.full(len(starts), Y_START), starts[:, 1]))),
                     tick=TICK, t_max=(NT - 1) * TICK, substeps=4, landing="conductor", slide=True)
        pa = dict(t=tr.t.cpu(), xyz=tr.xyz.cpu(), arrived=tr.arrived.cpu(), pad=tr.pad.cpu(),
                  seconds=time.time() - t0)
        torch.save(pa, ppath)
        print(f"{len(starts)} paths, {pa['seconds']:.0f} s", flush=True)
    arr_o = pa["arrived"].numpy()
    # pochoir's arrival: the first tick at or below the pad top
    PPw = PP[wedge]
    arr_p = np.array([np.argmax(p[:, 2] <= Y0 + 1e-9) * TICK for p in PPw])
    value("radoArrMinOurs", f"{arr_o.min():.2f}")
    value("radoArrMaxOurs", f"{arr_o.max():.2f}")
    value("radoArrMinP", f"{arr_p.min():.2f}")
    value("radoArrMaxP", f"{arr_p.max():.2f}")
    t20o = np.array([np.interp(-20.0, -z, pa["t"].numpy()) for z in pa["xyz"][:, :, 1].numpy()])
    t20p = np.array([np.interp(-20.0, -p[:, 2], np.arange(NT) * TICK) for p in PPw])
    value("radoTwentyOurs", f"{t20o.mean():.3f}")
    value("radoTwentyP", f"{t20p.mean():.3f}")

    # ---- currents: launches over the sensing pad's quarter, and a few beyond --
    SP = np.load(os.path.join(STORE, "startpoints.npy"))
    rel = SP[:, :2] - W_CENTRE
    refI = np.load(os.path.join(STORE, "current", "induced_current.npz"))["induced_current"] * 1e3  # e/us
    picks = [(0.05, 0.05, "centre pad, near its centre"), (1.55, 0.05, "centre pad, near an edge"),
             (2.15, 2.15, "the gap crossing"), (2.25, 0.05, "over the next pad"),
             (4.45, 4.45, "over the diagonal pad"), (8.85, 8.85, "two pads away")]
    cell = (rel[:, 0] < PITCH / 2) & (rel[:, 1] < PITCH / 2)
    want = np.flatnonzero(cell)
    want = np.unique(np.concatenate((want, [int(np.argmin(np.hypot(rel[:, 0] - x, rel[:, 1] - z)))
                                            for x, z, _ in picks])))
    xyz = pa["xyz"]
    sidx, shift = [], []
    for j in want:
        x, z = rel[j]
        lx, lz = wrap(x), wrap(z)
        # the wedge path with the same local start, mirrored into place
        swap = abs(lz) > abs(lx)
        ax_, az_ = (abs(lz), abs(lx)) if swap else (abs(lx), abs(lz))
        k = int(np.argmin(np.hypot(starts[:, 0] - ax_, starts[:, 1] - az_)))
        sidx.append((k, swap, np.sign(lx) or 1.0, np.sign(lz) or 1.0, x - lx, z - lz))
    full = torch.empty((len(want), xyz.shape[1], 3), dtype=xyz.dtype)
    padc = torch.empty((len(want), 2), dtype=xyz.dtype)
    for n, (k, swap, sx, sz, ox, oz) in enumerate(sidx):
        p = xyz[k].clone()
        pc = pa["pad"][k].clone()
        if swap:
            p = p[:, [2, 1, 0]]
            pc = pc[[1, 0]]
        full[n, :, 0] = sx * p[:, 0] + ox
        full[n, :, 1] = p[:, 1]
        full[n, :, 2] = sz * p[:, 2] + oz
        padc[n] = torch.tensor([sx * pc[0] + ox, sz * pc[1] + oz])
    on = (padc.abs() < 1.0).all(1) & torch.isfinite(pa["arrived"][[s[0] for s in sidx]])
    arrived = pa["arrived"][[s[0] for s in sidx]]
    curr = {}
    for tag, wf in (("5", w5), ("9", w9)):
        cp = os.path.join(out, f"currents{tag}.npz")
        if os.path.exists(cp):
            curr[tag] = np.load(cp)["I"]
        else:
            Q = X.induced_charge(wf, full.to(DEV), arrived.to(DEV), pa["t"].to(DEV), on.to(DEV),
                                 chunk_paths=4).cpu().numpy()
            curr[tag] = np.diff(Q, axis=1) / TICK  # e/us
            np.savez(cp, I=curr[tag], Q=Q, want=want)
    ref = refI[want]
    on = on.numpy()
    tc = np.arange(NT - 1) * TICK + 0.5 * TICK

    def t99(I):
        c = np.cumsum(I, axis=1)
        return (np.argmax(c >= 0.99 * c[:, -1:], axis=1) + 1) * TICK

    oc = cell[want] & on
    t99o, t99p = t99(curr["5"][oc]), t99(ref[oc])
    for k, v in (("ColMinOurs", t99o.min()), ("ColMaxOurs", t99o.max()), ("ColMinP", t99p.min()),
                 ("ColMaxP", t99p.max()), ("ColSpreadOurs", t99o.max() - t99o.min()),
                 ("ColSpreadP", t99p.max() - t99p.min())):
        value("rado" + k, f"{v:.2f}")
    value("radoCollectOurs", f"{curr['5'][oc].sum(1).mean() * TICK:.4f}")
    value("radoCollectP", f"{ref[oc].sum(1).mean() * TICK:.4f}")
    value("radoCollectNine", f"{curr['9'][oc].sum(1).mean() * TICK:.4f}")

    def pick(x, z):
        return int(np.argmin(np.hypot(rel[want, 0] - x, rel[want, 1] - z)))

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for a, (x, z, lab) in zip(axes.ravel(), picks):
        j = pick(x, z)
        a.plot(tc, ref[j], lw=1.2, label="pochoir (R. Fan), 5x5")
        a.plot(tc, curr["5"][j], "--", lw=1.2, label="firep, 5x5")
        a.plot(tc, curr["9"][j], ":", lw=1.5, color="C3", label="firep, 9x9")
        a.set_xlim(tc[np.argmax(np.abs(ref[j]))] - 12, tc[np.argmax(np.abs(ref[j]))] + 3)
        a.set_title(f"{lab}: launch ({rel[want][j, 0]:.2f}, {rel[want][j, 1]:.2f}) mm", fontsize=9)
        a.set_xlabel(r"time [$\mu$s]")
        a.set_ylabel(r"current [e/$\mu$s]")
        a.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(os.path.join(figdir, "pixel-rado-currents.pdf"))
    fig.savefig(os.path.join(out, "currents.png"), dpi=120)
    plt.close(fig)
    jc, j0 = pick(2.15, 2.15), pick(0.05, 0.05)
    value("radoCornerPeakOurs", f"{curr['5'][jc].max():.2f}")
    value("radoCornerPeakP", f"{ref[jc].max():.2f}")
    value("radoCentrePeakOurs", f"{curr['5'][j0].max():.3f}")
    value("radoCentrePeakP", f"{ref[j0].max():.3f}")
    near = oc & (np.abs(rel[want, 0]) < 1.3) & (np.abs(rel[want, 1]) < 1.3)
    d = (curr["5"] - ref)[near]
    value("radoInteriorRms", f"{np.sqrt((d ** 2).mean()) / np.abs(ref[near]).max() * 100:.2f}")
    for j, (x, z, lab) in enumerate(picks):
        k = pick(x, z)
        print(f"{lab:28s} peak pochoir {np.abs(ref[k]).max():.4f}  firep5 {np.abs(curr['5'][k]).max():.4f}  "
              f"firep9 {np.abs(curr['9'][k]).max():.4f} e/us")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    rr = rel[want][oc]
    for a, (vals, lab) in zip(ax, ((t99o, r"firep: 99% collected [$\mu$s]"),
                                   (t99p, r"pochoir: 99% collected [$\mu$s]"),
                                   (curr["5"][oc].max(1), r"firep peak current [e/$\mu$s]"))):
        sc = a.scatter(rr[:, 0], rr[:, 1], c=vals, s=18, marker="s", cmap="viridis")
        fig.colorbar(sc, ax=a, label=lab)
        a.set_aspect("equal")
        a.set_xlabel("launch x [mm]")
        a.set_ylabel("launch z [mm]")
    fig.savefig(os.path.join(figdir, "pixel-rado-dispersion.pdf"))
    fig.savefig(os.path.join(out, "dispersion.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(figdir, "pixel-rado.tex"), "w") as f:
        f.write("% generated by scripts/pixel_rado.py -- do not edit\n")
        for k, v in V.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    for k, v in V.items():
        print(f"{k:24s} {v}")


if __name__ == "__main__":
    main(*sys.argv[1:])
