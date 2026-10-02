#!/usr/bin/env python
"""Drift lines through one inter-pad gap, and what lies under the pads.

The pochoir reference, and the reproduction in Sec. 17, put free space under
the pads down to a zero-slope face 10 mm below them.  In a detector that space
is a PCB.  This script solves the drift cell three ways and traces electrons
through one gap on an *exact* field lattice (spectral, from the full fine
grid; no finite-difference error):

  free     free space under the pads (the pochoir geometry);
  charged  an insulating PCB charged up by the electrons that land on it: its
           surface is then zero-slope wherever there is no pad (pcb = 0);
  fr4      a fresh, uncharged FR4 PCB (eps_r 4.4 against LAr's 1.505).

It also resamples the free-space field at pochoir's 0.1 mm voxels, to separate
the effect of sampling from that of the model.

    python scripts/pixel_gap.py [runs/pixel/gap] [docs/figs]

Wants a GPU (FIREP_DEVICE, default cuda when available).
"""

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon, Rectangle

sys.path.insert(0, os.path.dirname(__file__))
from firep import pixel as X  # noqa: E402
from pixel_pochoir import PITCH, PLANES, SHAPE, TOP, V_CATH, drift_basis  # noqa: E402

DEV = os.environ.get("FIREP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64
EPS_FR4, EPS_LAR = 4.4, 1.505
MODELS = {"free": None, "charged": 0.0, "fr4": EPS_FR4 / EPS_LAR}
TITLES = {"free": "free space under the pads (pochoir)",
          "charged": "charged-up PCB (zero-slope gap surface)",
          "fr4": "uncharged FR4 PCB ($\\varepsilon_r = 4.4$)"}
GRID, MODES, COUNT = 1761, 88, 20000
Y0, Y1 = PLANES  # pad bottom (= PCB surface) and top faces
HALF, RAD = SHAPE.half, SHAPE.radius
Y_LAUNCH = 12.5  # mm: where the drift time is counted from
Z_CORNER = 1.8  # mm: the slice through the rounded corners
X_LO, X_HI = 1.2, 3.2  # mm: the window around the gap at x = pitch/2
E_FAR = -V_CATH / (TOP - 0.5 * (Y0 + Y1))


def corner_x(z):
    """Half-extent in x of the pad at height z in the pad plane."""
    a = HALF - RAD
    z = abs(z)
    return HALF if z <= a else a + np.sqrt(max(RAD ** 2 - (z - a) ** 2, 0.0))


# --------------------------------------------------------------------------
# the solves
# --------------------------------------------------------------------------


def solve(out, name):
    pcb = MODELS[name]
    arr = X.PadArray(pitch=PITCH, ncell=1, shape=SHAPE, y_pad=PLANES, top=TOP, pcb=pcb)
    path = os.path.join(out, f"drift-{name}.pt")
    if os.path.exists(path):
        st = torch.load(path, weights_only=False)
        ssl = X.SpectralSingleLayer(arr, "drift", MODES, GRID, drift_basis, device=DEV)
        return ssl, st["coef"].to(DEV), st["info"]
    ii, jj, own, _ = X.pad_nodes(arr, GRID, COUNT, np.random.default_rng(1))
    t0 = time.time()
    ssl = X.SpectralSingleLayer(arr, "drift", MODES, GRID, drift_basis, device=DEV, nodes=(ii, jj))
    coef, rms, mx = X.fit_on_nodes(ssl, np.zeros(len(own)), V_CATH)
    info = dict(rms=rms, max=mx, seconds=time.time() - t0)
    torch.save(dict(coef=coef.cpu(), info=info), path)
    print(f"{name}: fit rms {rms:.2e} V, max {mx:.2e} V, {info['seconds']:.0f} s")
    return ssl, coef, info


class Lattice:
    """E on a regular (y, x, z) lattice, trilinearly interpolated."""

    def __init__(self, ssl, coef, ys, xs, zs, cache=None):
        self.ys, self.xs, self.zs = (np.asarray(v, float) for v in (ys, xs, zs))
        if cache and os.path.exists(cache):
            st = torch.load(cache)
            self.V, self.E = st["V"].to(DEV), st["E"].to(DEV)
        else:
            V, Ex, Ey, Ez = ssl.field_layers(coef, ys, xs, zs, V_CATH)
            self.V = V.float()
            self.E = torch.stack((Ex, Ey, Ez))[None].float()  # (1, 3, ny, nx, nz)
            del V, Ex, Ey, Ez
            if cache:
                torch.save(dict(V=self.V.cpu(), E=self.E.cpu()), cache)
        self.lo = torch.tensor([self.zs[0], self.xs[0], self.ys[0]], device=DEV, dtype=DT)
        self.span = torch.tensor([self.zs[-1] - self.zs[0], self.xs[-1] - self.xs[0],
                                  self.ys[-1] - self.ys[0]], device=DEV, dtype=DT)

    def __call__(self, p):
        """E at points p = (x, y, z).  z is folded into the table through the
        mirror planes z = 0 and z = pitch/2, flipping E_z at each fold."""
        flip = torch.where(p[:, 2] < 0, -1.0, 1.0).to(DT)
        z = p[:, 2].abs()
        far = z > PITCH / 2
        z = torch.where(far, PITCH - z, z)
        flip = torch.where(far, -flip, flip)
        q = torch.stack((z, p[:, 0], p[:, 1]), 1)
        g = (2 * (q - self.lo) / self.span - 1).float()
        e = torch.nn.functional.grid_sample(self.E, g[None, :, None, None, :], mode="bilinear",
                                            padding_mode="border", align_corners=True)[0, :, :, 0, 0].T
        e = e.to(DT)
        return torch.stack((e[:, 0], e[:, 1], e[:, 2] * flip), 1)


def sdf_t(x, z):
    """The rounded-square outline's signed distance, in torch."""
    q = torch.stack((x.abs(), z.abs()), 1) - (HALF - RAD)
    return torch.linalg.vector_norm(q.clamp_min(0), dim=1) + q.max(1).values.clamp_max(0) - RAD


TOL = 1e-3  # mm: within this of a pad face counts as landed (below the lattice spacing)


def on_pad(p, tol=TOL):
    x = torch.remainder(p[:, 0] + PITCH / 2, PITCH) - PITCH / 2
    z = torch.remainder(p[:, 2] + PITCH / 2, PITCH) - PITCH / 2
    return (sdf_t(x, z) < tol) & (p[:, 1] >= Y0 - tol) & (p[:, 1] <= Y1 + tol)


def trace(field, seeds, ds, backward=False, pcb=True, slide=False, ylim=(0.0, Y_LAUNCH),
          dt_max=0.01, t_max=400.0, keep=2):
    """RK4 in time on the electron velocity v = -mu(|E|) E (or +v backwards),
    each step bounded by the length ``ds`` and the time ``dt_max``, so that
    a line slows down honestly near a stagnation point.

    Stops on a pad, on the PCB surface (``pcb``), on leaving ``ylim``, or at
    ``t_max`` us.  With ``slide`` the PCB surface is zero-slope (a charged
    insulator): a step that dips below it is put back on it, so the electron
    slides along the surface.  Returns the kept points of each path, its end
    point, how it ended, and the drift time in us."""
    sgn = -1.0 if backward else 1.0

    def v(p):
        e = field(p)
        mag = torch.linalg.vector_norm(e, dim=1, keepdim=True).clamp_min(1e-30)
        return -sgn * (X.bnl_speed(mag) / mag) * e

    p = seeds.clone().to(DEV, DT)
    n = len(p)
    alive = torch.ones(n, dtype=torch.bool, device=DEV)
    how = np.array(["stalled"] * n, dtype=object)
    t = torch.zeros(n, device=DEV, dtype=DT)
    pts = [p.clone()]
    it = 0
    while alive.any():
        a = torch.nonzero(alive)[:, 0]
        q = p[a]
        k1 = v(q)
        dt = (ds / torch.linalg.vector_norm(k1, dim=1).clamp_min(1e-30)).clamp_max(dt_max)[:, None]
        k2 = v(q + 0.5 * dt * k1)
        k3 = v(q + 0.5 * dt * k2)
        k4 = v(q + dt * k3)
        new = q + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if slide:
            new[:, 1] = new[:, 1].clamp_min(Y0)
        hitpad = on_pad(new)
        if hitpad.any():  # bisect the step onto the pad surface
            lo_, hi_ = torch.zeros(int(hitpad.sum()), device=DEV, dtype=DT), torch.ones(int(hitpad.sum()), device=DEV, dtype=DT)
            q0, q1 = q[hitpad], new[hitpad]
            for _ in range(12):
                mid = 0.5 * (lo_ + hi_)
                inside = on_pad(q0 + mid[:, None] * (q1 - q0))
                hi_ = torch.where(inside, mid, hi_)
                lo_ = torch.where(inside, lo_, mid)
            new[hitpad] = q0 + hi_[:, None] * (q1 - q0)
            dt[hitpad] *= hi_[:, None]
        t[a] += dt[:, 0]
        hitpcb = (new[:, 1] < Y0) & ~hitpad if pcb else torch.zeros_like(hitpad)
        out = (new[:, 1] > ylim[1]) | (new[:, 1] < ylim[0])
        late = t[a] > t_max
        p[a] = new
        for mask, label in ((hitpad, "pad"), (hitpcb, "pcb"), (out & ~hitpad & ~hitpcb, "out"),
                            (late & ~hitpad & ~hitpcb & ~out, "stalled")):
            how[a[mask].cpu().numpy()] = label
            alive[a[mask]] = False
        if it % keep == 0:
            pts.append(p.clone())
        it += 1
    pts.append(p.clone())
    return torch.stack(pts, 1).cpu().numpy(), p.cpu().numpy(), how, t.cpu().numpy()


def full_lines(field, seeds, ds, pcb, ylim=(0.0, Y_LAUNCH), slide=False):
    """Lines through ``seeds``: back up to the launch height, then down to
    wherever they end; the time is counted from the launch height."""
    bp, bend, bhow, bt = trace(field, seeds, ds, backward=True, pcb=False, ylim=ylim)
    fp, fend, fhow, ft = trace(field, seeds, ds, pcb=pcb, slide=slide, ylim=ylim)
    lines = [np.concatenate((bp[i, ::-1], fp[i]), 0) for i in range(len(seeds))]
    ok = bhow == "out"  # reached the launch height going back
    return lines, fend, fhow, np.where(ok, bt + ft, np.nan)


def where_on_pad(end):
    """Landing position around the right-hand edge of the centre pad (or,
    mirrored, the left-hand edge of its neighbour) at z = 0, as one
    coordinate: negative on the top face, 0..0.1 down the side, then
    0.1 + distance in along the underside.  The face is the nearest one."""
    x = np.where(end[:, 0] > PITCH / 2, PITCH - end[:, 0], end[:, 0])
    y = end[:, 1]
    d = np.stack((np.abs(y - Y1), np.abs(x - HALF), np.abs(y - Y0)))
    face = np.argmin(d, 0)
    return np.choose(face, (x - HALF, Y1 - y, (Y1 - Y0) + (HALF - x)))


# --------------------------------------------------------------------------


def main(out="runs/pixel/gap", figdir="docs/figs"):
    os.makedirs(out, exist_ok=True)
    V = {}
    solves = {m: solve(out, m) for m in MODELS}
    h = solves["free"][0].h

    # ---- how much flux reaches the gap surface, from the field alone ---------
    for m, (ssl, coef, info) in solves.items():
        xs = (np.arange(ssl.n) - ssl.n // 2) * h
        _, _, Ey, _ = ssl.field_layers(coef, [Y0 + 2e-6], xs, xs, V_CATH)
        XX, ZZ = np.meshgrid(xs, xs, indexing="ij")
        gap = torch.as_tensor(SHAPE.sdf(XX, ZZ) > 0, device=DEV)
        frac = float((Ey[0] * gap).sum() * h * h / (E_FAR * PITCH ** 2))
        V[f"gapFlux{m}"] = frac
        V[f"gapFit{m}"] = info["rms"]
        print(f"{m}: fraction of the drift flux through the gap surface {frac:.4f}")

    # ---- lattices ------------------------------------------------------------
    t0 = time.time()
    xs = np.arange(round(X_LO / h), round(X_HI / h) + 1) * h
    ys_fine = np.arange(9.0, Y_LAUNCH + 0.1, 0.0025)
    ys_free = np.arange(0.0, Y_LAUNCH + 0.1, 0.0025)
    lat0, latc = {}, {}
    for m, (ssl, coef, _) in solves.items():
        ys = ys_free if m == "free" else ys_fine
        lat0[m] = Lattice(ssl, coef, ys, xs, np.array([0.0, h, 2 * h]), os.path.join(out, f"lat0-{m}.pt"))
        zc = np.arange(0, round(PITCH / 2 / h) + 3, 2) * h  # all of 0 <= z <= pitch/2
        latc[m] = Lattice(ssl, coef, np.arange(9.0 if m != "free" else 6.0, Y_LAUNCH + 0.1, 0.005),
                          xs[::2], zc, os.path.join(out, f"latc-{m}.pt"))
    # pochoir's voxels: the same exact field, sampled every 0.1 mm
    ssl, coef, _ = solves["free"]
    xv = np.arange(round(X_LO / 0.1), round(X_HI / 0.1) + 1) * 0.1
    xv = np.round(xv / h) * h
    lat_vox = Lattice(ssl, coef, np.arange(0.0, Y_LAUNCH + 0.1, 0.1), xv, np.array([0.0, 0.1, 0.2]),
                      os.path.join(out, "lat-vox.pt"))
    lat_half = Lattice(ssl, coef, ys_free[::2], xs[::2], np.array([0.0, 2 * h, 4 * h]),
                       os.path.join(out, "lat-half.pt"))
    print(f"lattices: {time.time() - t0:.0f} s")
    # free space: the stagnation point on the gap's midline, below the pads
    ey = lat0["free"].E[0, 1, :, int(np.argmin(np.abs(xs - PITCH / 2))), 0].cpu().numpy()
    below = ys_free < Y0 - 0.05
    flips = np.nonzero(np.diff(np.sign(ey[below])))[0]
    V["gapSaddle"] = float(ys_free[below][flips[-1]]) if len(flips) else float("nan")
    print(f"free space: midline stagnation point at y = {V['gapSaddle']:.2f} mm")

    # ---- seeds ---------------------------------------------------------------
    def gap_seeds(z, n=24):
        xa, xb = corner_x(z), PITCH - corner_x(z)
        mid = PITCH / 2
        u = xa + (xb - xa) * (np.arange(n) + 0.5) / n
        u = np.sort(np.concatenate((u, mid + np.array([-1e-2, -1e-3, -1e-4, 1e-4, 1e-3, 1e-2]))))
        return torch.tensor(np.column_stack((u, np.full(len(u), 0.5 * (Y0 + Y1)), np.full(len(u), z))),
                            device=DEV, dtype=DT)

    def top_seeds(z):
        u = np.arange(X_LO + 0.05, X_HI, 0.1)
        return torch.tensor(np.column_stack((u, np.full(len(u), Y_LAUNCH - 1e-3), np.full(len(u), z))),
                            device=DEV, dtype=DT)

    runs = {}
    for m in MODELS:
        pcb = m == "fr4"  # electrons land on a fresh insulator; a charged one repels them
        slide = m == "charged"
        ylim = (0.0 if m == "free" else Y0 - 0.01, Y_LAUNCH)
        runs[m, 0] = full_lines(lat0[m], gap_seeds(0.0), 0.002, pcb, ylim, slide)
        runs[m, 0, "top"] = trace(lat0[m], top_seeds(0.0), 0.002, pcb=pcb, slide=slide, ylim=ylim)
        ylc = (6.0 if m == "free" else Y0 - 0.01, Y_LAUNCH)
        runs[m, "c"] = full_lines(latc[m], gap_seeds(Z_CORNER), 0.004, pcb, ylc, slide)
        runs[m, "c", "top"] = trace(latc[m], top_seeds(Z_CORNER), 0.004, pcb=pcb, slide=slide, ylim=ylc)
    # precision: the free-space lines with a finer step, a coarser lattice, and voxels
    prec = {"exact, 2 µm steps": runs["free", 0],
            "exact, 0.5 µm steps": full_lines(lat0["free"], gap_seeds(0.0), 0.0005, False),
            "5 µm lattice, 2 µm steps": full_lines(lat_half, gap_seeds(0.0), 0.002, False),
            "0.1 mm voxels, 2 µm steps": full_lines(lat_vox, gap_seeds(0.0), 0.002, False),
            "0.1 mm voxels, 20 µm steps": full_lines(lat_vox, gap_seeds(0.0), 0.02, False)}
    seedx = gap_seeds(0.0)[:, 0].cpu().numpy()
    regular = np.abs(seedx - PITCH / 2) > 0.02  # the separatrix seeds are decided by 1e-4 mm
    ref = where_on_pad(prec["exact, 2 µm steps"][1])
    for k, (_, end, how, tt) in prec.items():
        d = np.abs(where_on_pad(end) - ref)[regular]
        side = np.sign(end[~regular, 0] - PITCH / 2)
        print(f"{k:28s} landing shift max {np.nanmax(d) * 1e3:7.2f} um over the regular seeds; "
              f"separatrix seeds land on sides {side.astype(int).tolist()}, {how[~regular].tolist()}; "
              f"max time {np.nanmax(tt):.2f} us")
        V["gapShift" + {"exact, 0.5 µm steps": "Fine", "5 µm lattice, 2 µm steps": "Half",
                        "0.1 mm voxels, 2 µm steps": "Vox", "0.1 mm voxels, 20 µm steps": "VoxCoarse"}
          .get(k, "Ref")] = np.nanmax(d) * 1e3
    np.savez(os.path.join(out, "gap.npz"), **{f"{k}": np.asarray(v) for k, v in V.items()})

    # ---- figures -------------------------------------------------------------
    tnorm = LogNorm(1.2, 300.0)
    cmap = plt.cm.plasma

    def draw_lines(ax, lines, times, proj=(0, 1), lw=0.8):
        segs = [ln[:, proj] for ln in lines]
        cols = [cmap(tnorm(t)) if np.isfinite(t) else (0.5, 0.5, 0.5, 1) for t in times]
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw))

    def draw_pads(ax, z, pcb):
        xa = corner_x(z)
        for x0, x1 in ((-HALF, xa), (PITCH - xa, PITCH + HALF)):
            ax.add_patch(Rectangle((x0, Y0), x1 - x0, Y1 - Y0, fc="0.25", ec="k", lw=0.5, zorder=5))
        if xa < HALF:  # the rounded corner, projected: pad present at smaller |z|
            for x0 in (xa, PITCH - HALF):
                ax.add_patch(Rectangle((x0, Y0), HALF - xa, Y1 - Y0, fc="none", ec="0.25", lw=0.6,
                                       hatch="////", zorder=5))
        if pcb:
            ax.add_patch(Rectangle((-10, -10), 30, 10 + Y0, fc="#d9c89a", ec="none", zorder=0))

    def potential(ax, lat, zi=0):
        V_ = lat.V[:, :, zi].cpu().numpy()
        ax.contour(lat.xs, lat.ys, V_, levels=np.linspace(-120, 0, 25), colors="0.75", linewidths=0.4,
                   zorder=1)

    def finish(fig, axes):
        sm = plt.cm.ScalarMappable(norm=tnorm, cmap=cmap)
        fig.colorbar(sm, ax=axes, shrink=0.85, label=f"drift time from y = {Y_LAUNCH} mm (µs)")

    # (1) through the pad centres
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True, sharey=True)
    for ax, m in zip(axes, MODELS):
        lines, end, how, tt = runs[m, 0]
        tp = runs[m, 0, "top"]
        potential(ax, lat0[m])
        draw_lines(ax, [ln for ln in tp[0]], tp[3], lw=0.5)
        draw_lines(ax, lines, tt)
        draw_pads(ax, 0.0, m != "free")
        pc = end[how == "pcb"]
        ax.plot(pc[:, 0], pc[:, 1], "x", color="C2", ms=4, zorder=6)
        ax.set_xlim(1.4, 3.0)
        ax.set_ylim(8.9, 11.0)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_title(TITLES[m], fontsize=9)
    axes[0].set_ylabel("y (mm)")
    finish(fig, axes)
    fig.suptitle("z = 0: through the pad centres; one gap, x = 1.9 to 2.5 mm. "
                 "Grey: equipotentials every 5 V; ×: lands on the PCB", fontsize=10)
    fig.savefig(os.path.join(figdir, "pixel-gap-centre.pdf"))
    fig.savefig(os.path.join(out, "gap-centre.png"), dpi=120)

    # (1b) free space: how deep the lines under the pads go
    fig, ax = plt.subplots(figsize=(6.5, 5.2), constrained_layout=True)
    lines, end, how, tt = runs["free", 0]
    potential(ax, lat0["free"])
    draw_lines(ax, lines, tt)
    draw_pads(ax, 0.0, False)
    ax.axhline(0.0, color="k", lw=2)
    ax.text(1.0, 0.2, "zero-slope bottom face", fontsize=8)
    ax.set_xlim(0.2, 4.2)
    ax.set_ylim(-0.2, 12.5)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    finish(fig, [ax])
    ax.axhline(V["gapSaddle"], color="C2", ls=":", lw=0.8)
    ax.plot([PITCH / 2], [V["gapSaddle"]], "o", color="C2", ms=4)
    ax.text(PITCH / 2 + 0.08, V["gapSaddle"] - 0.35, f"stagnation point, y = {V['gapSaddle']:.1f} mm",
            color="C2", fontsize=8)
    ax.set_title("free space under the pads, z = 0: lines through the gap near its midline\n"
                 "dive towards a stagnation point, then turn back up onto the pad undersides", fontsize=9)
    fig.savefig(os.path.join(figdir, "pixel-gap-deep.pdf"))
    fig.savefig(os.path.join(out, "gap-deep.png"), dpi=120)

    # (2) through the rounded corners
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6), constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[1, 1]))
    for j, m in enumerate(MODELS):
        lines, end, how, tt = runs[m, "c"]
        tp = runs[m, "c", "top"]
        ax = axes[0, j]
        zi = int(np.argmin(np.abs(latc[m].zs - Z_CORNER)))
        potential(ax, latc[m], zi)
        draw_lines(ax, tp[0], tp[3], lw=0.5)
        draw_lines(ax, lines, tt)
        draw_pads(ax, Z_CORNER, m != "free")
        pc = end[how == "pcb"]
        ax.plot(pc[:, 0], pc[:, 1], "x", color="C2", ms=4, zorder=6)
        ax.set_xlim(1.4, 3.0)
        ax.set_ylim(8.9, 11.0)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_title(TITLES[m] + f"\nprojected on z = {Z_CORNER} mm", fontsize=9)
        # plan view
        b = axes[1, j]
        for cx in (0.0, PITCH):
            for cz in (0.0, PITCH):
                t_ = np.linspace(0, 2 * np.pi, 800)
                a_ = HALF - RAD
                b.add_patch(Polygon(np.column_stack((cx + np.sign(np.cos(t_)) * a_ + RAD * np.cos(t_),
                                                     cz + np.sign(np.sin(t_)) * a_ + RAD * np.sin(t_))),
                                    closed=True, fc="0.85", ec="k", lw=0.6, zorder=0))
        b.axhline(Z_CORNER, color="C0", ls="--", lw=0.8)
        b.text(1.45, Z_CORNER + 0.03, f"launch line z = {Z_CORNER}", color="C0", fontsize=8)
        draw_lines(b, tp[0], tp[3], proj=(0, 2), lw=0.5)
        draw_lines(b, lines, tt, proj=(0, 2))
        b.plot(end[how == "pad", 0], end[how == "pad", 2], ".", color="k", ms=2, zorder=6)
        b.plot(pc[:, 0], pc[:, 2], "x", color="C2", ms=4, zorder=6)
        b.set_xlim(1.4, 3.0)
        b.set_ylim(0.9, 2.45)
        b.set_aspect("equal")
        b.set_xlabel("x (mm)")
        b.set_title("plan view (x, z): pad outlines and the same lines", fontsize=9)
    axes[0, 0].set_ylabel("y (mm)")
    axes[1, 0].set_ylabel("z (mm)")
    finish(fig, axes)
    fig.suptitle(f"Lines launched along z = {Z_CORNER} mm, 0.1 mm inside the pad edge, where the corner "
                 f"(radius {RAD} mm) trims the pad to |x| ≤ {corner_x(Z_CORNER):.3f} mm. "
                 "Hatched: pad present only at smaller |z|", fontsize=10)
    fig.savefig(os.path.join(figdir, "pixel-gap-corner.pdf"))
    fig.savefig(os.path.join(out, "gap-corner.png"), dpi=120)

    # (3) precision and timing
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    a = ax[0]
    for (k, (lines, end, how, tt)), c in zip(prec.items(), ("k", "C0", "C1", "C3", "C4")):
        for ln in lines:
            a.plot(ln[:, 0], ln[:, 1], color=c, lw=0.6, alpha=0.8)
        a.plot([], [], color=c, label=k)
    draw_pads(a, 0.0, False)
    a.set_xlim(1.6, 2.8)
    a.set_ylim(9.2, 10.6)
    a.set_aspect("equal")
    a.legend(fontsize=7, loc="upper center")
    a.set_xlabel("x (mm)")
    a.set_ylabel("y (mm)")
    a.set_title("free space, z = 0: the same field, traced five ways", fontsize=9)
    a = ax[1]
    for (k, (lines, end, how, tt)), c, mk in zip(prec.items(), ("k", "C0", "C1", "C3", "C4"), "o.sx+"):
        a.plot(seedx, where_on_pad(end), mk, color=c, ms=4, mfc="none", label=k)
    a.axhspan(0, Y1 - Y0, color="0.9")
    a.text(1.92, 0.03, "side", fontsize=8)
    a.text(1.92, 0.3, "underside", fontsize=8)
    a.text(1.92, -0.05, "top face", fontsize=8)
    a.set_xlabel("x at mid-thickness in the gap (mm)")
    a.set_ylabel("where it lands, around the pad edge (mm)")
    a.set_title("landing point against launch point", fontsize=9)
    a.legend(fontsize=7)
    a = ax[2]
    for m, c in zip(MODELS, ("k", "C0", "C2")):
        lines, end, how, tt = runs[m, 0]
        a.semilogy(seedx, tt, "o-", color=c, ms=3, label=TITLES[m])
        lost = how == "pcb"
        a.semilogy(seedx[lost], tt[lost], "x", color="C3", ms=6)
    a.set_xlabel("x at mid-thickness in the gap (mm)")
    a.set_ylabel(f"drift time from y = {Y_LAUNCH} mm (µs)")
    a.set_title("time to land; red ×: lands on the PCB, not a pad", fontsize=9)
    a.legend(fontsize=7)
    fig.savefig(os.path.join(figdir, "pixel-gap-precision.pdf"))
    fig.savefig(os.path.join(out, "gap-precision.png"), dpi=120)

    for m in MODELS:
        lines, end, how, tt = runs[m, 0]
        print(m, "z=0 gap lines:", {k: int((how == k).sum()) for k in ("pad", "pcb", "out", "stalled")},
              "time range", np.nanmin(tt), np.nanmax(tt), "deepest y", min(ln[:, 1].min() for ln in lines))
    with open(os.path.join(figdir, "pixel-gap.tex"), "w") as f:
        f.write("% generated by scripts/pixel_gap.py -- do not edit\n")
        tag = {"free": "Free", "charged": "Charged", "fr4": "Fresh"}  # no digits in TeX names
        for m in MODELS:
            lines, end, how, tt = runs[m, 0]
            f.write(f"\\newcommand{{\\gapFlux{tag[m]}}}{{{100 * V[f'gapFlux{m}']:.1f}}}\n")
            f.write(f"\\newcommand{{\\gapTmax{tag[m]}}}{{{np.nanmax(tt):.1f}}}\n")
            f.write(f"\\newcommand{{\\gapDeep{tag[m]}}}{{{min(ln[:, 1].min() for ln in lines):.1f}}}\n")
        f.write(f"\\newcommand{{\\gapSaddle}}{{{V['gapSaddle']:.1f}}}\n")
        f.write(f"\\newcommand{{\\gapSaddleDepth}}{{{Y0 - V['gapSaddle']:.1f}}}\n")
        for k in ("gapShiftFine", "gapShiftHalf", "gapShiftVox", "gapShiftVoxCoarse"):
            f.write(f"\\newcommand{{\\{k}}}{{{V[k]:.1f}}}\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
