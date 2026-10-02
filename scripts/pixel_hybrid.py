#!/usr/bin/env python
"""Against R. Fan's latest pochoir solves of the standard LArPix-v2a pad
(Sec. 17): the "task13 hybrid" stores, one each with a 5 x 5, 9 x 9 and
17 x 17 pad weighting domain.

The model, from the stores: 3.5 mm pads with 0.7 mm corner radius at 4.4 mm
pitch, three voxels thick about y = 10.0 mm; the PCB a no-flux surface on the
pads' top plane, y = 10.1 mm, in both solves.  Drift: cathode -7500 V at
y = 160.0 mm, periodic sides.  Weighting: centre pad at 1, others 0, the far
face (y = 160.0 mm) and the bottom face grounded, zero-slope side walls.  Each
field is finite differences on two grids: 0.4 mm over the whole depth, 0.1 mm
over the first 40 mm, stitched.  100 launches per pixel on a 0.44 mm grid from
y = 159.9 mm; currents in e/ns.

In firep that is the charged-PCB slab (``pcb = 0``) with one pad sheet on its
surface at y = 10.1 mm, the drift and the weighting solve sharing the slab
(zero slope below, fixed far face).

    python scripts/pixel_hybrid.py [runs/pixel/hybrid] [docs/figs]
"""

import os
import re
import sys
import time

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from firep import pixel as X  # noqa: E402
from pixel_pochoir import SC, poly  # noqa: E402

ROOT = os.environ.get("RADO_RESULTS", "/nfs/data/1/bviren/rado/results")
STORES = {5: "store_task13_hybrid_5x5pix", 9: "store_task13_hybrid_9x9pix", 17: "store_task13_hybrid_17x17pix"}
DEV = os.environ.get("FIREP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64
PITCH, Y0 = 4.4, 10.1  # the PCB surface, flush with the pad tops
SHAPE = X.PadShape("rounded_square", 1.75, 0.7)
V_CATH, TOP = -7500.0, 160.0
Y_START, TICK, NT = 159.9, 0.05, 4000
# fine grid, modes and fit nodes per weighting domain
RES = {1: (1761, 88, 20000), 5: (2201, 220, 40000), 9: (3961, 400, 60000), 17: (6401, 640, 120000)}


def drift_basis(_ring=0):
    return poly(6) + [(p, q, s) for s in SC for (p, q) in poly(6)]


def weight_basis(ring):
    d = {0: 6, 1: 6, 2: 4}.get(ring, 2)
    sc = SC if ring <= 1 else [0.05, 0.2, 1.0]
    return poly(d) + [(p, q, s) for s in sc for (p, q) in poly(d)]


def solve(out, N):
    name = "drift" if N == 1 else f"weighting{N}"
    path = os.path.join(out, f"{name}.pt")
    if os.path.exists(path):
        st = torch.load(path, weights_only=False)
        return X.SheetPotential.from_state(st["field"], device=DEV), st["info"]
    grid, modes, count = RES[N]
    arr = X.PadArray(pitch=PITCH, ncell=N, shape=SHAPE, y_pad=Y0, top=TOP, pcb=0.0)
    ii, jj, own, _ = X.pad_nodes(arr, grid, count, np.random.default_rng(1))
    t0 = time.time()
    ssl = X.SpectralSingleLayer(arr, "drift", modes, grid, drift_basis if N == 1 else weight_basis,
                                device=DEV, nodes=(ii, jj))
    centre = len(arr.pad_centres()) // 2
    target = np.zeros(len(own)) if N == 1 else (own == centre).astype(float)
    offset = V_CATH if N == 1 else 0.0
    train = np.random.default_rng(2).uniform(size=len(own)) < 0.75
    coef, rms, mx = X.fit_on_nodes(ssl, target, offset, train)
    field = X.SheetPotential.from_fit(ssl, coef, offset)
    info = dict(columns=len(ssl.labels), modes=(modes + 1) ** 2, rms=rms, max=mx, seconds=time.time() - t0)
    torch.save(dict(field=field.state(), info=info), path)
    print(f"{name}: {info['columns']} columns, fit rms {rms:.2e} max {mx:.2e}, {info['seconds']:.0f} s", flush=True)
    del ssl
    torch.cuda.empty_cache() if DEV.startswith("cuda") else None
    return field, info


def wrap(u):
    return (u + PITCH / 2) % PITCH - PITCH / 2


def main(out="runs/pixel/hybrid", figdir="docs/figs"):
    os.makedirs(out, exist_ok=True)
    V = {}

    def value(name, text):
        m = re.fullmatch(r"(-?[0-9.]+)e([+-]?)0*([0-9]+)", text)
        if m:
            mant, sign, ex = m.groups()
            text = f"${mant}\\times10^{{{'-' if sign == '-' else ''}{ex}}}$"
        V[name] = text

    t64 = lambda a: torch.as_tensor(np.asarray(a, float), device=DEV, dtype=DT)
    tag = {5: "Five", 9: "Nine", 17: "Seventeen"}
    drift, di = solve(out, 1)
    W, wi = {}, {}
    for N in STORES:
        W[N], wi[N] = solve(out, N)
    value("hybDRms", f"{di['rms']:.1e}")
    value("hybDMax", f"{di['max']:.1e}")
    for N in STORES:
        value(f"hyb{tag[N]}Rms", f"{wi[N]['rms']:.1e}")
        value(f"hyb{tag[N]}Cols", f"{wi[N]['columns']}")

    # ---- the drift field against the store ---------------------------------
    s9 = os.path.join(ROOT, STORES[9])
    P = np.load(os.path.join(s9, "potential", "drift3d.npz"))["drift3d"]
    xs = wrap(np.arange(44) * 0.1)
    Xg, Zg = np.meshgrid(xs, xs, indexing="ij")
    rows = []
    for kz in (102, 103, 105, 110, 150, 200, 350, 450, 1000):
        ours = drift.evaluate(t64(np.column_stack((Xg.ravel(), np.full(Xg.size, kz * 0.1), Zg.ravel())))
                              ).cpu().numpy().reshape(Xg.shape)
        d = ours - P[:, :, kz]
        rows.append((kz * 0.1, float(np.sqrt((d ** 2).mean())), float(np.abs(d).max())))
        print(f"drift at y = {kz * 0.1:6.1f}: rms diff {rows[-1][1]:.3f} V, max {rows[-1][2]:.3f} V", flush=True)
    value("hybDriftNear", f"{rows[0][1]:.2f}")
    value("hybDriftBulk", f"{rows[4][1]:.2f}")
    value("hybDriftStitch", f"{rows[6][1]:.2f}")
    value("hybDriftAbove", f"{rows[7][1]:.2f}")
    value("hybDriftFar", f"{rows[8][1]:.2f}")
    ef = float(-np.gradient(P, 0.1, axis=2)[:, :, 800].mean())
    _, g = drift.evaluate(t64([[0.0, 80.0, 0.0]]), grad=True)
    value("hybEP", f"{ef:.3f}")
    value("hybEOurs", f"{float(-g[0, 1]):.3f}")
    del P

    # ---- the weighting fields: axis and plane average ------------------------
    ys = np.array([10.2, 10.3, 10.6, 11.1, 12.1, 15.1, 20.1, 30.1, 40.1, 60.1, 100.1, 150.1])
    wz = {}
    for N, st in STORES.items():
        cache = os.path.join(out, f"pochoir-weight{N}.npz")
        if os.path.exists(cache):
            wz[N] = dict(np.load(cache))
            continue
        Wp = np.load(os.path.join(ROOT, st, "potential", "weight3d.npz"))["weight3d"]
        sp = np.load(os.path.join(ROOT, st, "startpoints.npy"))
        centre = sp[:, 0].min() - 0.22
        c = int(round(centre / 0.1))
        mz = Wp.mean(axis=(0, 1))
        wz[N] = dict(axis=Wp[c, c, :].copy(), mean=mz, centre=centre)
        np.savez(cache, **wz[N])
        del Wp
    for N in STORES:
        wz[N]["ours_axis"] = W[N].evaluate(t64(np.column_stack((np.zeros(len(ys)), ys, np.zeros(len(ys)))))
                                           ).cpu().numpy()
        # the plane average of a cosine series is its (0, 0) mode, exactly
        k0 = torch.zeros(1, device=DEV, dtype=DT)
        wz[N]["ours_mean"] = sum(float(W[N].A[ip, 0, 0]) * W[N].arr.green(k0, t64(ys), yp, W[N].kind)[:, 0].cpu().numpy()
                                 for ip, yp in enumerate(W[N].planes)) + W[N].offset
        pa = np.array([wz[N]["axis"][int(round(y * 10))] for y in ys])
        pm = np.array([wz[N]["mean"][int(round(y * 10))] for y in ys])
        wz[N]["p_axis"], wz[N]["p_mean"] = pa, pm
        for y, a, b, cm, dm in zip(ys, pa, wz[N]["ours_axis"], pm, wz[N]["ours_mean"]):
            print(f"{N}x{N} y = {y:6.1f}: axis pochoir {a:.5f} firep {b:.5f}; plane mean pochoir {cm:.5f} firep {dm:.5f}")
        m = wz[N]["mean"]
        sl = lambda y: float(-(m[int(y * 10) + 1] - m[int(y * 10) - 1]) / 0.2)
        value(f"hyb{tag[N]}SlopeTwentyP", f"{sl(20.0):.2e}")
        value(f"hyb{tag[N]}SlopeOneFiftyP", f"{sl(150.0):.2e}")
        value(f"hyb{tag[N]}SlopeOurs", f"{(wz[N]['ours_mean'][9] - wz[N]['ours_mean'][10]) / 40.0:.2e}")
        value(f"hyb{tag[N]}AxisNearP", f"{pa[0]:.4f}")
        value(f"hyb{tag[N]}AxisNearOurs", f"{wz[N]['ours_axis'][0]:.4f}")
        value(f"hyb{tag[N]}AxisTwoP", f"{pa[4]:.4f}")
        value(f"hyb{tag[N]}AxisTwoOurs", f"{wz[N]['ours_axis'][4]:.4f}")
        value(f"hyb{tag[N]}AxisTwentyP", f"{pa[7]:.4f}")
        value(f"hyb{tag[N]}AxisTwentyOurs", f"{wz[N]['ours_axis'][7]:.4f}")
        value(f"hyb{tag[N]}MeanHundredP", f"{pm[10]:.4f}")
        value(f"hyb{tag[N]}MeanHundredOurs", f"{wz[N]['ours_mean'][10]:.4f}")

    # the field magnitudes: firep's exactly (the plane average's by a central
    # difference of its (0, 0) mode), pochoir's by central differences on its
    # 0.1 mm voxels; on the axis the field is along y by symmetry
    hd = np.geomspace(0.1, TOP - Y0 - 0.1, 200)
    k0 = torch.zeros(1, device=DEV, dtype=DT)

    def plane_mean(N, y):
        return sum(float(W[N].A[ip, 0, 0]) * W[N].arr.green(k0, t64(y), yp, W[N].kind)[:, 0].cpu().numpy()
                   for ip, yp in enumerate(W[N].planes))

    kv = np.unique(np.round(np.geomspace(1, int(round((TOP - Y0) * 10)) - 2, 28)).astype(int)) + int(round(Y0 * 10))
    for N in STORES:
        d = 5e-3
        wz[N]["ours_emean"] = np.abs(plane_mean(N, Y0 + hd - d) - plane_mean(N, Y0 + hd + d)) / (2 * d)
        _, g = W[N].evaluate(t64(np.column_stack((np.zeros(len(hd)), Y0 + hd, np.zeros(len(hd))))), grad=True)
        wz[N]["ours_eaxis"] = g.norm(dim=1).cpu().numpy()
        for key, arr in (("p_emean", wz[N]["mean"]), ("p_eaxis", wz[N]["axis"])):
            wz[N][key] = np.abs(arr[kv + 1] - arr[kv - 1]) / 0.2
    hp = kv * 0.1 - Y0

    cols = {5: "C0", 9: "C1", 17: "C2"}
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.6), constrained_layout=True, sharex=True)
    for N in STORES:
        ax[0, 0].plot(ys - Y0, wz[N]["p_mean"], "s--", color=cols[N], ms=4, label=f"pochoir, {N}x{N}")
        ax[0, 0].plot(ys - Y0, wz[N]["ours_mean"], "o-", color=cols[N], mfc="none", ms=5, label=f"firep, {N}x{N}")
        ax[0, 1].semilogy(ys - Y0, wz[N]["p_axis"], "s--", color=cols[N], ms=4, label=f"pochoir, {N}x{N}")
        ax[0, 1].semilogy(ys - Y0, wz[N]["ours_axis"], "o-", color=cols[N], mfc="none", ms=5, label=f"firep, {N}x{N}")
        for j, key in enumerate(("emean", "eaxis")):
            ax[1, j].plot(hp, wz[N]["p_" + key], "s", color=cols[N], ms=4, label=f"pochoir, {N}x{N}")
            ax[1, j].plot(hd, wz[N]["ours_" + key], "-", color=cols[N], label=f"firep, {N}x{N}")
    for a in ax.ravel():
        a.set_xscale("log")
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    for a in ax[1]:
        a.set_xlabel("height above the pads [mm]")
        a.set_ylim(bottom=0)
    # the far field above the pad, where it should meet the plane average's
    ins = ax[1, 1].inset_axes([0.42, 0.28, 0.54, 0.36])
    far, farp = hd >= 20.0, hp >= 20.0
    for N in STORES:
        ins.plot(hp[farp], wz[N]["p_eaxis"][farp], "s", color=cols[N], ms=3)
        ins.plot(hd[far], wz[N]["ours_eaxis"][far], "-", color=cols[N])
        ins.axhline(float(wz[N]["ours_emean"].mean()), color=cols[N], ls=":", lw=0.8)
    ins.set_xscale("log")
    ins.set_ylim(0, 1.6 * max(float(wz[N]["ours_emean"].mean()) for N in STORES))
    ins.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ins.tick_params(labelsize=7)
    ins.grid(alpha=0.3)
    ins.set_title("above 20 mm; dotted: firep plane average", fontsize=7)
    ax[1, 1].legend(fontsize=7, loc="upper right")
    ax[0, 0].set_ylabel("plane-averaged weighting potential")
    ax[0, 1].set_ylabel("weighting potential above the sensing pad")
    ax[1, 0].set_ylabel("|plane-averaged weighting field| [1/mm]")
    ax[1, 1].set_ylabel("|weighting field| above the sensing pad [1/mm]")
    fig.savefig(os.path.join(figdir, "pixel-hybrid-weight.pdf"))
    fig.savefig(os.path.join(out, "weight.png"), dpi=120)
    plt.close(fig)

    # ---- paths from the store's 100 starts over one cell ---------------------
    S0 = np.load(os.path.join(s9, "starts", "drift3d.npz"))["drift3d"]
    loc = np.column_stack((wrap(S0[:, 0]), wrap(S0[:, 1])))
    ppath = os.path.join(out, "paths.pt")
    if os.path.exists(ppath):
        pa = torch.load(ppath, weights_only=False)
    else:
        t0 = time.time()
        tr = X.trace(drift, t64(np.column_stack((loc[:, 0], np.full(len(loc), Y_START), loc[:, 1]))),
                     tick=TICK, t_max=(NT - 1) * TICK, substeps=4, landing="conductor", slide=True)
        pa = dict(t=tr.t.cpu(), xyz=tr.xyz.cpu(), arrived=tr.arrived.cpu(), pad=tr.pad.cpu(),
                  seconds=time.time() - t0)
        torch.save(pa, ppath)
        print(f"{len(loc)} paths, {pa['seconds']:.0f} s", flush=True)
    PP = np.load(os.path.join(s9, "paths", "drift3d.npz"))["drift3d"]
    t20o = np.array([np.interp(-20.0, -z, pa["t"].numpy()) for z in pa["xyz"][:, :, 1].numpy()])
    t20p = np.array([np.interp(-20.0, -p[:, 2], np.arange(NT) * TICK) for p in PP])
    value("hybTwentyOurs", f"{t20o.mean():.3f}")
    value("hybTwentyP", f"{t20p.mean():.3f}")
    del PP

    # ---- currents for every launch of each store ------------------------------
    tc = np.arange(NT - 1) * TICK + 0.5 * TICK
    xyz, arrived = pa["xyz"], pa["arrived"]
    picks = [(0.22, 0.22, "over the pad, near its centre"), (1.54, 0.22, "over the pad, near an edge"),
             (1.98, 1.98, "near the gap crossing"), (2.42, 0.22, "over the gap, next pad's side"),
             (4.62, 4.62, "over the diagonal pad"), (9.02, 9.02, "two pads away")]
    res = {}
    for N, st in STORES.items():
        SP = np.load(os.path.join(ROOT, st, "startpoints.npy"))
        rel = SP[:, :2] - wz[N]["centre"]
        refI = np.load(os.path.join(ROOT, st, "current", "induced_current.npz"))["induced_current"] * 1e3
        cp = os.path.join(out, f"currents{N}.npz")
        if os.path.exists(cp):
            I = np.load(cp)["I"]
        else:
            idx, shift = [], []
            for x, z in rel:
                j = int(np.argmin(np.hypot(loc[:, 0] - wrap(x), loc[:, 1] - wrap(z))))
                idx.append(j)
                shift.append((x - xyz[j, 0, 0].item(), z - xyz[j, 0, 2].item()))
            idx = np.array(idx)
            shift = torch.as_tensor(np.array(shift))
            full = xyz[idx].clone()
            full[:, :, 0] += shift[:, 0, None]
            full[:, :, 2] += shift[:, 1, None]
            padc = pa["pad"][idx] + shift
            on = (padc.abs() < 1.0).all(1) & torch.isfinite(arrived[idx])
            Q = X.induced_charge(W[N], full.to(DEV), arrived[idx].to(DEV), pa["t"].to(DEV), on.to(DEV),
                                 chunk_paths=4).cpu().numpy()
            I = np.diff(Q, axis=1) / TICK
            np.savez(cp, I=I, Q=Q, on=on.numpy())
            print(f"{N}x{N}: currents for {len(rel)} launches", flush=True)
        res[N] = dict(rel=rel, ref=refI, ours=I)

    def pick(N, x, z):
        r = res[N]["rel"]
        return int(np.argmin(np.hypot(r[:, 0] - x, r[:, 1] - z)))

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for a, (x, z, lab) in zip(axes.ravel(), picks):
        for N in STORES:
            j = pick(N, x, z)
            a.plot(tc, res[N]["ref"][j], "-", color=cols[N], lw=1.2, label=f"pochoir, {N}x{N}")
            a.plot(tc, res[N]["ours"][j], "--", color=cols[N], lw=1.2, label=f"firep, {N}x{N}")
        j = pick(9, x, z)
        tp = tc[np.argmax(np.abs(res[9]["ref"][j]))]
        a.set_xlim(tp - 12, tp + 3)
        a.set_title(f"{lab}: launch ({x:.2f}, {z:.2f}) mm", fontsize=9)
        a.set_xlabel(r"time [$\mu$s]")
        a.set_ylabel(r"current [e/$\mu$s]")
        a.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(os.path.join(figdir, "pixel-hybrid-currents.pdf"))
    fig.savefig(os.path.join(out, "currents.png"), dpi=120)
    plt.close(fig)

    def t99(I):
        c = np.cumsum(I, axis=1)
        return (np.argmax(c >= 0.99 * c[:, -1:], axis=1) + 1) * TICK

    for N in STORES:
        r = res[N]
        own = (r["rel"][:, 0] < PITCH / 2) & (r["rel"][:, 1] < PITCH / 2)
        to, tp = t99(r["ours"][own]), t99(r["ref"][own])
        value(f"hyb{tag[N]}ColSpreadOurs", f"{to.max() - to.min():.2f}")
        value(f"hyb{tag[N]}ColSpreadP", f"{tp.max() - tp.min():.2f}")
        value(f"hyb{tag[N]}CollectOurs", f"{r['ours'][own].sum(1).mean() * TICK:.4f}")
        value(f"hyb{tag[N]}CollectP", f"{r['ref'][own].sum(1).mean() * TICK:.4f}")
        near = own & (np.abs(r["rel"][:, 0]) < 1.3) & (np.abs(r["rel"][:, 1]) < 1.3)
        d = (r["ours"] - r["ref"])[near]
        value(f"hyb{tag[N]}InteriorRms", f"{np.sqrt((d ** 2).mean()) / np.abs(r['ref'][near]).max() * 100:.2f}")
        for x, z, lab in picks:
            j = pick(N, x, z)
            pr, po = np.abs(r["ref"][j]).max(), np.abs(r["ours"][j]).max()
            print(f"{N}x{N} {lab:32s} peak pochoir {pr:.4f} firep {po:.4f} e/us", flush=True)
        j0, jc, j2 = pick(N, 0.22, 0.22), pick(N, 1.98, 1.98), pick(N, 9.02, 9.02)
        value(f"hyb{tag[N]}CentrePeakP", f"{r['ref'][j0].max():.3f}")
        value(f"hyb{tag[N]}CentrePeakOurs", f"{r['ours'][j0].max():.3f}")
        value(f"hyb{tag[N]}CornerPeakP", f"{r['ref'][jc].max():.2f}")
        value(f"hyb{tag[N]}CornerPeakOurs", f"{r['ours'][jc].max():.2f}")
        value(f"hyb{tag[N]}TwoAwayP", f"{np.abs(r['ref'][j2]).max():.4f}")
        value(f"hyb{tag[N]}TwoAwayOurs", f"{np.abs(r['ours'][j2]).max():.4f}")

    with open(os.path.join(figdir, "pixel-hybrid.tex"), "w") as f:
        f.write("% generated by scripts/pixel_hybrid.py -- do not edit\n")
        for k, v in V.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    for k, v in V.items():
        print(f"{k:28s} {v}")


if __name__ == "__main__":
    main(*sys.argv[1:])
