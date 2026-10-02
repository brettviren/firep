#!/usr/bin/env python
"""Pad shape and size against the time dispersion and sharpness of the
field response (Sec. 17).

Seven pads at 4.4 mm pitch: the 3.8 mm rounded square of the pochoir
reference, flat disks and hemispheres of 3.8, 2.5 and 1.0 mm diameter.  All
use the physical treatment of the PCB found in Sec. 17 (the gap study): the
drift field over a charged-up board (zero-slope surface between pads), and
the weighting field with the FR4 dielectric under the pads and ground 10 mm
below them.  The weighting domain is 9 x 9 pads.

For each pad: currents on the centre pad from launches above its centre out
along the diagonal to the corner shared with its three neighbours, and the
average over launches spread uniformly over one pitch square.

    python scripts/pixel_shapes.py [runs/pixel/shapes] [docs/figs]

Wants a GPU (FIREP_DEVICE).
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
from pixel_pochoir import PITCH, SC, TOP, V_CATH, drift_basis, poly, weight_basis  # noqa: E402

DEV = os.environ.get("FIREP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64
EPS_FR4, EPS_LAR = 4.4, 1.505
Y0, THICK = 10.0, 0.1  # the PCB surface, and the flat pads' thickness
Y_START = 148.0  # launch height, as pochoir's
TICK, T_MAX = 0.05, 110.0
FACE_STEP = 0.05  # mm: the launch grid over the pitch square (one eighth of it, by symmetry)
SHAPES = {
    "square3.8": X.PadShape("rounded_square", 1.9, 0.4),
    "disk3.8": X.PadShape("disk", 1.9, 0.0),
    "disk2.5": X.PadShape("disk", 1.25, 0.0),
    "disk1.0": X.PadShape("disk", 0.5, 0.0),
    "hemi3.8": X.PadShape("hemisphere", 1.9, 0.0),
    "hemi2.5": X.PadShape("hemisphere", 1.25, 0.0),
    "hemi1.0": X.PadShape("hemisphere", 0.5, 0.0),
    # four 1.1 mm dots per pixel, ganged, on a uniform 2.2 mm lattice: 1.1 mm
    # between dots within a pixel and across its boundary
    "quad1.1": X.PadShape("quad", 0.55, 1.1),
    # the same with a fifth dot at the pixel centre
    "quint1.1": X.PadShape("quint", 0.55, 1.1),
    # a flat ring, 3.3 mm outside and 1.1 mm inside: along the axes through the
    # pixel centres the gap between rings (1.1 mm) equals the hole, and the
    # line is half conductor, 2 x 1.1 mm per 4.4 mm pitch
    "ring3.3": X.PadShape("annulus", 1.65, 0.55),
    # the standard LArPix-v2a pad, without and with a focusing grid
    "std3.5": X.PadShape("rounded_square", 1.75, 0.7),
}
# Focusing grids over the standard pad: (height above the pad tops, drift-solve
# voltage, hole radius), holes centred over the pads.  The grid is at 0 V in
# the weighting solve, so designs differing only in voltage share one.
GRIDS = {
    "grid": (3.2, -2000.0, 1.9),
    "gridV1000": (3.2, -1000.0, 1.9), "gridV3000": (3.2, -3000.0, 1.9),
    "gridH1.6": (1.6, -2000.0, 1.9), "gridH4.8": (4.8, -2000.0, 1.9),
    "gridR1.5": (3.2, -2000.0, 1.5), "gridR2.1": (3.2, -2000.0, 2.1),
}
# The cathode stays at V_CATH unless given here: holding the drift field above
# the nominal grid at its no-grid value (50.05 V/mm) needs a cathode at
# -2000 - 50.05 * (149.9 - 13.3) V.
# The nominal drift field, 500 V/cm, is a hard limit: a real design keeps it
# above the grid and gets its grid-to-pad field from the grid and pad biases.
# Only differences matter here, so each "E50" design holds the field above the
# grid at its no-grid value by setting the cathode to
# V_grid - 50.05 V/mm * (149.9 mm - y_grid), which is the same as keeping the
# cathode and raising the grid and pad biases together.
E_DRIFT = 50.05
for _g in list(GRIDS):
    _h, _v, _r = GRIDS[_g]
    GRIDS["gridE50" + _g[4:]] = (_h, _v, _r)
# The optimum of scripts/pixel_optimize.py, when it exists: its own pad size.
_OPTDIR = os.path.join(os.path.dirname(__file__), "..", "runs", "pixel", "opt")
for _name, _file in (("gridE50opt", "optimum.json"), ("gridE50optc", "optimum-combined.json"),
                     ("gridE50optn", "optimum-neighbour.json")):
    _p = os.path.join(_OPTDIR, _file)
    if os.path.exists(_p):
        import json as _json
        _o = _json.load(open(_p))
        GRIDS[_name] = (round(_o["h"], 3), -round(_o["dV"], 1), round(_o["r"], 3))
        SHAPES[_name] = X.PadShape("rounded_square", round(_o["a"], 3), round(0.4 * _o["a"], 3))
# designs for which the neighbours' induced charge is computed (a second trace)
NEIGHBOURS = ["square3.8", "std3.5", "gridE50", "gridE50opt", "gridE50optc", "gridE50optn"]
CATHODE = {g: GRIDS[g][1] - E_DRIFT * (TOP - (10.1 + GRIDS[g][0])) for g in GRIDS if g.startswith("gridE50")}
for _g in GRIDS:
    SHAPES[_g] = SHAPES["std3.5"]
LABEL = {"square3.8": "3.8 mm rounded square", "disk3.8": "3.8 mm flat disk", "disk2.5": "2.5 mm flat disk",
         "disk1.0": "1.0 mm flat disk", "hemi3.8": "3.8 mm hemisphere", "hemi2.5": "2.5 mm hemisphere",
         "hemi1.0": "1.0 mm hemisphere", "quad1.1": "four 1.1 mm dots", "quint1.1": "five 1.1 mm dots",
         "ring3.3": "3.3/1.1 mm ring", "std3.5": "3.5 mm standard pad"}
for _g, (_h, _v, _r) in GRIDS.items():
    LABEL[_g] = f"3.5 mm pad, grid {_h:g} mm, {_v:g} V, r {_r:g}"
    if _g in CATHODE:
        LABEL[_g] += f", cathode {CATHODE[_g]:.0f} V"


def pad_array(shape, ncell, pcb):
    hemi = shape.kind == "hemisphere"
    return X.PadArray(pitch=PITCH, ncell=ncell, shape=shape, y_pad=Y0 if hemi else (Y0, Y0 + THICK),
                      top=TOP, pcb=pcb)


def dome_basis(ring):
    return poly({0: 8, 1: 8, 2: 6}.get(ring, 3))


def dome_count(ring):
    return {0: 5000, 1: 3000, 2: 1200}.get(ring, 300)


def weight_name(name):
    """Grid designs differing only in voltage share a weighting solve."""
    if name in GRIDS:
        h, _, r = GRIDS[name]
        for g, (h2, _, r2) in GRIDS.items():
            if (h2, r2) == (h, r) and SHAPES.get(g) == SHAPES.get(name):
                return g
    return name


def grid_y(name):
    return Y0 + THICK + GRIDS[name][0]


def solve(out, name, kind):
    """Fit one field; cached.  Returns the SheetPotential and fit info."""
    if kind == "weighting":
        name = weight_name(name)
    path = os.path.join(out, f"{name}-{kind}.pt")
    if os.path.exists(path):
        st = torch.load(path, weights_only=False)
        return X.SheetPotential.from_state(st["field"], device=DEV), st["info"]
    shape = SHAPES[name]
    hemi = shape.kind == "hemisphere"
    drift = kind == "drift"
    N, grid, modes = (1, 1761, 88) if drift else (9, 3961, 400)
    arr = pad_array(shape, N, 0.0 if drift else EPS_FR4 / EPS_LAR)
    offset = CATHODE.get(name, V_CATH) if drift else 0.0
    rng = np.random.default_rng(1)
    t0 = time.time()
    centre = len(arr.pad_centres()) // 2
    if name in GRIDS:
        _, vg, rh = GRIDS[name]
        sheets = [X.Sheet(Y0, shape), X.Sheet(Y0 + THICK, shape),
                  X.Sheet(grid_y(name), X.PadShape("hole", PITCH / 2, rh), "edge", tile=True)]
        ii, jj, own, _ = X.pad_nodes(arr, grid, 20000 if drift else 60000, rng)
        ig, jg = X.grid_nodes(arr, rh, grid, 20000 if drift else 60000, np.random.default_rng(3))
        nodes = {0: (ii, jj), 1: (ii, jj), 2: (ig, jg)}
        ssl = X.SpectralSingleLayer(arr, kind, modes, grid, drift_basis if drift else weight_basis,
                                    device=DEV, nodes=nodes, sheets=sheets)
        tpad = np.zeros(len(own)) if drift else (own == centre).astype(float)
        target = np.concatenate((tpad, tpad, np.full(len(ig), vg if drift else 0.0)))
        train = np.random.default_rng(2).uniform(size=len(target)) < 0.75
        coef, rms, mx = X.fit_rows(ssl, target, offset, train)
    elif hemi:
        ssl = X.SpectralSingleLayer(arr, kind, modes, grid, poly(8) if drift else dome_basis, device=DEV)
        xyz, own = X.dome_nodes(arr, 6000 if drift else dome_count, rng)
        target = np.zeros(len(own)) if drift else (own == centre).astype(float)
        D = ssl.evaluate(torch.as_tensor(xyz, device=DEV, dtype=DT))
        train = np.random.default_rng(2).uniform(size=len(own)) < 0.75
        coef, rms, mx = X.fit_design(D, target, offset, train)
        del D
    else:
        ii, jj, own, _ = X.pad_nodes(arr, grid, 20000 if drift else 60000, rng)
        ssl = X.SpectralSingleLayer(arr, kind, modes, grid, drift_basis if drift else weight_basis,
                                    device=DEV, nodes=(ii, jj))
        target = np.zeros(len(own)) if drift else (own == centre).astype(float)
        train = np.random.default_rng(2).uniform(size=len(own)) < 0.75
        coef, rms, mx = X.fit_on_nodes(ssl, target, offset, train)
    field = X.SheetPotential.from_fit(ssl, coef, offset)
    info = dict(columns=len(ssl.labels), rms=rms, max=mx, seconds=time.time() - t0)
    torch.save(dict(field=field.state(), info=info), path)
    print(f"{name} {kind}: {info['columns']} columns, fit rms {rms:.2e} max {mx:.2e}, {info['seconds']:.0f} s",
          flush=True)
    return field, info


def launches(name=None):
    """The diagonal family and the face grid (one eighth of the pitch square,
    by symmetry, with the weights that make it the whole square).  For the
    four-dot pixel the centre, between its dots, is a stagnation point on the
    charged board, so the first diagonal launch is nudged off it."""
    f = np.linspace(0.0, 1.0, 11)
    if name is not None and SHAPES[name].kind in ("quad", "annulus"):
        f[0] = 0.02 / (PITCH / 2)
    diag = np.column_stack((f * PITCH / 2, f * PITCH / 2))
    n = int(round(PITCH / 2 / FACE_STEP))
    c = (np.arange(n) + 0.5) * FACE_STEP
    face, w = [], []
    for i, x in enumerate(c):
        for z in c[: i + 1]:
            face.append((x, z))
            w.append(4.0 if x == z else 8.0)
    return f, diag, np.array(face), np.array(w) / np.sum(w)


def responses(out, name, drift, weight):
    path = os.path.join(out, f"{name}-currents.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    f, diag, face, w = launches(name)
    xz = np.concatenate((diag, face))
    starts = torch.tensor(np.column_stack((xz[:, 0], np.full(len(xz), Y_START), xz[:, 1])), device=DEV, dtype=DT)
    t0 = time.time()
    # a grid lowers the field above it, and a more negative grid more so: allow a longer drift
    pa = X.trace(drift, starts, tick=TICK, t_max=T_MAX if name not in GRIDS else 160.0, substeps=4,
                 landing="conductor", slide=True,
                 grid=(grid_y(name), GRIDS[name][2]) if name in GRIDS else None)
    on = torch.isfinite(pa.arrived) & (pa.pad.abs() < 1.0).all(1)
    Q = X.induced_charge(weight, pa.xyz, pa.arrived, pa.t, on, chunk_paths=4).cpu().numpy()
    res = dict(t=pa.t.cpu().numpy(), Q=Q, arrived=pa.arrived.cpu().numpy(), on=on.cpu().numpy(),
               on_grid=torch.isnan(pa.pad).any(1).cpu().numpy(),
               end=pa.xyz[:, -1].cpu().numpy(), ndiag=len(diag), f=f, w=w)
    np.savez(path, **res)
    print(f"{name}: {len(xz)} paths, {int(on.sum())} collected, {time.time() - t0:.0f} s", flush=True)
    return res


def neighbour_signals(out, name, drift, weight):
    """The charge the face launches (collected on the centre pad) induce on its
    edge and diagonal neighbours, averaged over the pitch square: by
    translation, the centre pad's weighting potential along the paths shifted
    by a pitch, and 0 after landing.  Cached; needs its own trace."""
    path = os.path.join(out, f"{name}-neighbours.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    f, diag, face, w = launches(name)
    starts = torch.tensor(np.column_stack((face[:, 0], np.full(len(face), Y_START), face[:, 1])), device=DEV, dtype=DT)
    t0 = time.time()
    pa = X.trace(drift, starts, tick=TICK, t_max=T_MAX if name not in GRIDS else 160.0, substeps=4,
                 landing="conductor", slide=True, grid=(grid_y(name), GRIDS[name][2]) if name in GRIDS else None)
    none = torch.zeros(len(face), dtype=torch.bool, device=DEV)
    res = {}
    for key, sh in (("edge", (PITCH, 0.0, 0.0)), ("diag", (PITCH, 0.0, PITCH))):
        xyz = pa.xyz + torch.tensor(sh, device=DEV, dtype=DT)
        Q = X.induced_charge(weight, xyz, pa.arrived, pa.t, none, chunk_paths=4).cpu().numpy()
        res[key] = (w[:, None] * Q).sum(0)
    out_ = dict(t=pa.t.cpu().numpy(), Qe=res["edge"], Qd=res["diag"])
    np.savez(path, **out_)
    print(f"{name}: neighbour signals, {time.time() - t0:.0f} s", flush=True)
    return out_


def neighbour_metric(nb, m):
    """Peak-to-peak swing of the neighbour's charge, smeared by diffusion,
    within 1.5 us of the collecting pad's smeared peak."""
    sig = np.sqrt(2 * 4.0e-4 * 86.0) / 1.60
    ker_t = np.arange(-int(4 * sig / TICK), int(4 * sig / TICK) + 1) * TICK
    ker = np.exp(-0.5 * (ker_t / sig) ** 2)
    ker /= ker.sum()
    out = {}
    tpk = m["tc"][int(np.argmax(np.convolve(m["avg"], ker, mode="same")))]
    for key in ("Qe", "Qd"):
        q = nb[key]
        qp = np.pad(q, len(ker) // 2, mode="edge")
        qs = np.convolve(qp, ker, mode="valid")
        win = np.abs(nb["t"] - tpk) <= 1.5
        out[key] = float(qs[win].max() - qs[win].min())
        out[key + "_series"] = qs
    return out


def metrics(r):
    dq = np.diff(r["Q"], axis=1) / TICK  # e/us, per tick
    tc = r["t"][:-1] + 0.5 * TICK
    nd = r["ndiag"]
    face = dq[nd:]
    avg = (r["w"][:, None] * face).sum(0)
    on = r["on"][nd:]
    c = np.cumsum(face, 1) * TICK
    t99 = tc[np.argmax(c >= 0.99 * c[:, -1:], axis=1)]
    tpk = tc[np.argmax(face, axis=1)]  # when the current peaks
    peak = np.where(r["on_grid"][nd:], -np.inf, face.max(1)) if "on_grid" in r else face.max(1)
    half = avg >= 0.5 * avg.max()
    fwhm = (np.nonzero(half)[0][-1] - np.nonzero(half)[0][0] + 1) * TICK
    w = r["w"]
    # a launch caught by a grid induces a bipolar pulse and no net charge: the
    # timing statistics are over the launches that reach the pad
    caught = r["on_grid"][nd:] if "on_grid" in r else np.zeros(len(w), bool)
    w = np.where(caught, 0.0, w)
    w = w / w.sum()
    wm = lambda v: float((w * v).sum())
    wsd = lambda v: float(np.sqrt((w * (v - wm(v)) ** 2).sum()))
    ok = ~caught

    def wq(v, q):  # area-weighted quantile over the pitch square
        o = np.argsort(v)
        cw = np.cumsum(w[o])
        return float(v[o][np.searchsorted(cw, q * cw[-1])])
    # the optimiser's objective (scripts/pixel_optimize.py): the rms duration of
    # the average smeared by longitudinal diffusion (4 cm^2/s over the drift),
    # within 3 us of its peak so that the far-field induction is left out
    sig = np.sqrt(2 * 4.0e-4 * 86.0) / 1.60
    ker_t = np.arange(-int(4 * sig / TICK), int(4 * sig / TICK) + 1) * TICK
    ker = np.exp(-0.5 * (ker_t / sig) ** 2)
    sm = np.convolve(avg, ker / ker.sum(), mode="same")
    ip = int(np.argmax(sm))
    win = np.abs(tc - tc[ip]) <= 3.0
    tb = (tc[win] * sm[win]).sum() / sm[win].sum()
    width_diff = float(np.sqrt(((tc[win] - tb) ** 2 * sm[win]).sum() / sm[win].sum()))
    return dict(dq=dq, tc=tc, avg=avg, t99=t99, tpk=tpk, peak=peak, width_diff=width_diff,
                peak_diff=float(sm.max()),
                tpk_sd=wsd(tpk), tpk_range=float(tpk[ok].max() - tpk[ok].min()),
                t99_sd=wsd(t99), t99_range=float(t99[ok].max() - t99[ok].min()),
                t99_w90=wq(t99, 0.95) - wq(t99, 0.05), peak_min=float(peak[np.isfinite(peak)].min()),
                peak_max=float(peak.max()), avg_peak=float(avg.max()), fwhm=fwhm,
                collected=float((r["w"] * c[:, -1]).sum()), frac_on=float(on.mean()),
                lost=float((r["w"] * caught).sum()))


def design_figure(name, dims, M, R, panel, launches, t_lo, t_hi, ymax, out, figdir):
    """A pad design's layout, its diagonal family and average, and its average
    beside those of the 3.8 mm square and the 1.0 mm disk."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True,
                           gridspec_kw=dict(width_ratios=[0.8, 1, 1]))
    a = ax[0]
    sh = SHAPES[name]
    g = np.linspace(-6.6, 6.6, 1321)
    GX, GZ = np.meshgrid(g, g, indexing="xy")
    lx, lz = (GX + PITCH / 2) % PITCH - PITCH / 2, (GZ + PITCH / 2) % PITCH - PITCH / 2
    cond = sh.sdf(lx, lz) < 0
    centre = (np.abs(GX) < PITCH / 2) & (np.abs(GZ) < PITCH / 2)
    img = np.where(cond & centre, 2, np.where(cond, 1, 0))
    a.contourf(GX, GZ, img, levels=[0.5, 1.5, 2.5], colors=["0.6", "C2"])
    if name in GRIDS:  # the grid: shaded bars, outlined holes
        h_, v_, r_ = GRIDS[name]
        bars = np.hypot(lx, lz) > r_
        a.contourf(GX, GZ, bars.astype(float), levels=[0.5, 1.5], colors=["C5"], alpha=0.35)
        a.contour(GX, GZ, np.hypot(lx, lz), levels=[r_], colors=["C5"], linewidths=0.8)
    for k in (-1.5, -0.5, 0.5, 1.5):
        a.axvline(k * PITCH, color="k", ls=":", lw=0.6)
        a.axhline(k * PITCH, color="k", ls=":", lw=0.6)
    _, diag, _, _ = launches(name)
    a.plot(diag[:, 0], diag[:, 1], "k.-", ms=3, lw=0.6)
    for (x0, x1), z, lab, zt in dims:
        a.annotate("", (x0, z), (x1, z), arrowprops=dict(arrowstyle="<->", lw=0.8))
        if lab:
            a.text(0.5 * (x0 + x1), zt, lab, ha="center", va="center", fontsize=7, backgroundcolor="w")
    a.set_xlim(-6.6, 6.6)
    a.set_ylim(-6.6, 6.6)
    a.set_aspect("equal")
    a.set_xlabel("x (mm)")
    a.set_ylabel("z (mm)")
    extra = (f"; brown: the grid, {GRIDS[name][0]:g} mm above the pads, {-GRIDS[name][1]:g} V below them,"
             f" 500 V/cm above it" if name in GRIDS else "")
    a.set_title(f"{LABEL[name] if name not in GRIDS else '3.5 mm standard pad'} per 4.4 mm pixel\n"
                f"green: the sensing pixel; black: the diagonal launches{extra}".replace("; brown", "\nbrown"),
                fontsize=8)
    panel(ax[1], name, legend=True)
    ax[1].set_yscale("log")
    ax[1].set_ylim(0.01, ymax)
    ax[1].legend(fontsize=7, loc="upper left")
    ax[2].plot(M["square3.8"]["tc"], M["square3.8"]["avg"], "k", label=LABEL["square3.8"])
    ax[2].plot(M["disk1.0"]["tc"], M["disk1.0"]["avg"], "C0:", label=LABEL["disk1.0"])
    if name != "quad1.1":
        ax[2].plot(M["quad1.1"]["tc"], M["quad1.1"]["avg"], "C2", lw=1, label=LABEL["quad1.1"])
    if name in GRIDS:
        ax[2].plot(M["std3.5"]["tc"], M["std3.5"]["avg"], "C7", lw=1.5, label=LABEL["std3.5"] + ", no grid")
    ax[2].plot(M[name]["tc"], M[name]["avg"], "C3" if name != "quad1.1" else "C2", lw=2, label=LABEL[name])
    ax[2].set_xlim(t_lo, t_hi)
    ax[2].set_xlabel("time (µs)")
    ax[2].set_ylabel("average current (e/µs)")
    ax[2].set_title("averaged over the pitch square", fontsize=9)
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3)
    stem = {"quad1.1": "quad", "quint1.1": "quint", "ring3.3": "ring", "gridE50": "grid"}[name]
    fig.savefig(os.path.join(figdir, f"pixel-shapes-{stem}.pdf"))
    fig.savefig(os.path.join(out, f"shapes-{stem}.png"), dpi=110)
    plt.close(fig)


def main(out="runs/pixel/shapes", figdir="docs/figs"):
    os.makedirs(out, exist_ok=True)
    R, M, info = {}, {}, {}
    only = os.environ.get("PIXEL_SHAPES_ONLY")  # solve and trace just these, no figures (to share GPUs)
    for name in (only.split(",") if only else SHAPES):
        drift, di = solve(out, name, "drift")
        weight, wi = solve(out, name, "weighting")
        info[name] = (di, wi)
        R[name] = responses(out, name, drift, weight)
        M[name] = metrics(R[name])
        if name in NEIGHBOURS and (not only or name in only.split(",")):
            nb = neighbour_signals(out, name, drift, weight)
            M[name]["nb"] = nb
            M[name].update({"x_" + k[1]: v for k, v in neighbour_metric(nb, M[name]).items()
                            if not k.endswith("series")})
        del drift, weight
        torch.cuda.empty_cache() if DEV.startswith("cuda") else None
    if only:
        return

    # a common window: from before the earliest peak to after the last 99%
    shown = {n: m for n, m in M.items() if n not in GRIDS or n.startswith("gridE50")}
    t_lo = min(m["tpk"].min() for m in shown.values()) - 3.0
    t_hi = max(m["t99"].max() for m in shown.values()) + 0.5
    cmap = plt.cm.viridis

    def panel(ax, name, legend=False, window=(t_lo, t_hi)):
        m, r = M[name], R[name]
        nd = r["ndiag"]
        for k in range(nd):
            lab = None
            if legend and k in (0, nd // 2, nd - 2, nd - 1):
                lab = f"{r['f'][k]:.1f} of the way to the corner" if k < nd - 1 else "the shared corner"
            ls = ":" if k == nd - 1 else "-"
            ax.plot(m["tc"], m["dq"][k], ls, color=cmap(k / (nd - 1)), lw=0.9, label=lab)
        ax.plot(m["tc"], m["avg"], "k", lw=2.0, label="average over the pitch square" if legend else None)
        ax.set_xlim(*window)
        ax.set_title(f"{LABEL[name]}: single-launch peaks {m['peak_min']:.1f}–{m['peak_max']:.1f} e/µs\n"
                     f"99% times: rms {m['t99_sd']:.2f} µs, 90% of the square within {m['t99_w90']:.2f} µs",
                     fontsize=8.5)
        ax.set_xlabel("time (µs)")
        ax.set_ylabel("current on the pixel (e/µs)")
        ax.grid(alpha=0.3)

    for name in SHAPES:  # one figure each
        fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
        panel(ax, name, legend=True)
        ax.legend(fontsize=7)
        fig.savefig(os.path.join(out, f"{name}.png"), dpi=120)
        plt.close(fig)

    # the grid: disks on top, hemispheres below; the square on its own
    ymax = max(max(M[n]["dq"][: R[n]["ndiag"] - 1].max(), M[n]["avg"].max()) for n in SHAPES) * 1.5
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), constrained_layout=True, sharex=True)
    for j, d in enumerate(("3.8", "2.5", "1.0")):
        for i, kind in enumerate(("disk", "hemi")):
            panel(axes[i, j], kind + d, legend=(i == 0 and j == 0))
            axes[i, j].set_yscale("log")
            axes[i, j].set_ylim(0.01, ymax)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.savefig(os.path.join(figdir, "pixel-shapes-waveforms.pdf"))
    fig.savefig(os.path.join(out, "shapes-waveforms.png"), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    panel(ax, "square3.8", legend=True)
    ax.set_yscale("log")
    ax.set_ylim(0.01, ymax)
    ax.legend(fontsize=7, loc="upper left")
    fig.savefig(os.path.join(figdir, "pixel-shapes-square.pdf"))
    fig.savefig(os.path.join(out, "shapes-square.png"), dpi=110)
    plt.close(fig)

    # the multi-conductor pixels: layout, and responses beside the square's
    for name, dims in (("quad1.1", [((2.75, 3.85), -5.5, "1.1 mm", -4.55), ((-0.55, 0.55), -5.5, "1.1 mm gap", -6.25)]),
                       ("quint1.1", [((2.75, 3.85), -5.5, "1.1 mm", -4.55), ((-0.55, 0.55), -5.5, "1.1 mm gap", -6.25)]),
                       ("ring3.3", [((-1.65, 1.65), -4.4, "3.3 mm", -3.55), ((-0.55, 0.55), -4.4, "", 0),
                                    ((1.65, 2.75), -4.4, "1.1 mm gap", -5.35)]),
                       ("gridE50", [((-1.9, 1.9), -4.4, "3.8 mm hole", -5.35)])):
        design_figure(name, dims, M, R, panel, launches, t_lo, t_hi, ymax, out, figdir)

    # the grid scan: one parameter at a time about the nominal design
    scans = (("grid bias relative to the pads (V)", ["gridV1000", "grid", "gridV3000"], lambda g: GRIDS[g][1]),
             ("grid height above the pads (mm)", ["gridH1.6", "grid", "gridH4.8"], lambda g: GRIDS[g][0]),
             ("hole radius (mm)", ["gridR1.5", "grid", "gridR2.1"], lambda g: GRIDS[g][2]))
    fig, ax = plt.subplots(3, 3, figsize=(13, 8.5), constrained_layout=True, sharey="row")
    for j, (xl, names, xf) in enumerate(scans):
        xv = [xf(g) for g in names]
        for i, (key, yl, scale) in enumerate((("t99_sd", "rms spread of 99% times (µs)", 1.0),
                                              ("avg_peak", "peak of the average (e/µs)", 1.0),
                                              ("peak_max", "sharpest single-launch peak (e/µs)", 1.0))):
            a = ax[i, j]
            held = ["gridE50" + g[4:] for g in names]
            if all(h in M for h in held):
                a.plot(xv, [scale * M[h][key] for h in held], "o-", color="C3",
                       label="drift field held at 500 V/cm above the grid")
            a.plot(xv, [scale * M[g][key] for g in names], "s--", color="C5", mfc="none",
                   label="cathode fixed at -7000 V (field above the grid drops)")
            a.axhline(scale * M["std3.5"][key], color="C7", ls="--", lw=1, label="no grid")
            a.axhline(scale * M["square3.8"][key], color="k", ls=":", lw=1, label="3.8 mm square, no grid")
            a.set_xlabel(xl)
            if j == 0:
                a.set_ylabel(yl)
            a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("Focusing grid over the 3.5 mm standard pad: nominal 3.2 mm up, 2000 V below the pads, "
                 "1.9 mm holes; one parameter varied at a time", fontsize=10)
    fig.savefig(os.path.join(figdir, "pixel-shapes-gridscan.pdf"))
    fig.savefig(os.path.join(out, "shapes-gridscan.png"), dpi=110)
    plt.close(fig)

    # the summary: average responses and the dispersion against the peak
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    sty = {"square3.8": ("k", "-"), "disk3.8": ("C0", "-"), "disk2.5": ("C0", "--"), "disk1.0": ("C0", ":"),
           "hemi3.8": ("C3", "-"), "hemi2.5": ("C3", "--"), "hemi1.0": ("C3", ":"), "quad1.1": ("C2", "-"),
           "quint1.1": ("C2", "--"), "ring3.3": ("C4", "-"), "std3.5": ("C7", "-")}
    sty.update({g: ("C5", "-" if g == "gridE50" else ":") for g in GRIDS})
    short = {n: LABEL[n].replace(" mm", "") for n in SHAPES}
    short.update({"std3.5": "3.5 std", "grid": "grid (nominal)", "gridV1000": "grid -1 kV",
                  "gridV3000": "grid -3 kV", "gridH1.6": "grid 1.6 up", "gridH4.8": "grid 4.8 up",
                  "gridR1.5": "grid r 1.5", "gridR2.1": "grid r 2.1"})
    short.update({"gridE50" + k[4:]: v for k, v in list(short.items()) if k in GRIDS
                  and not k.startswith("gridE50")})
    short.update({"gridE50V1000": "grid 1 kV", "gridE50V3000": "grid 3 kV", "gridE50opt": "grid, optimised",
                  "gridE50optc": "grid, opt. combined", "gridE50optn": "grid, opt. neighbour"})
    fixed = [g for g in GRIDS if not g.startswith("gridE50")]  # shown in the grid scan only
    for name in SHAPES:
        if name in fixed:
            continue
        c, ls = sty[name]
        if name not in GRIDS or name == "gridE50":
            ax[0].plot(M[name]["tc"], M[name]["avg"], color=c, ls=ls,
                       label=LABEL[name] if name != "gridE50" else "3.5 mm pad with the nominal grid")
        ax[1].plot(M[name]["t99_sd"], M[name]["avg_peak"], "o", color=c, mfc="none" if ls != "-" else c,
                   ms={"-": 8, "--": 6, ":": 4}[ls])
        ax[1].annotate(short[name], (M[name]["t99_sd"], M[name]["avg_peak"]),
                       fontsize=7, xytext=(4, 2) if name != "hemi2.5" else (-60, 6), textcoords="offset points")
        ax[2].plot(M[name]["t99_w90"], M[name]["peak_max"], "s", color=c, mfc="none" if ls != "-" else c,
                   ms={"-": 8, "--": 6, ":": 4}[ls])
        ax[2].annotate(short[name], (M[name]["t99_w90"], M[name]["peak_max"]),
                       fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax[0].set_xlim(t_lo, t_hi)
    ax[0].set_xlabel("time (µs)")
    ax[0].set_ylabel("average current (e/µs)")
    ax[0].legend(fontsize=7)
    ax[0].set_title("the response averaged over the pitch square", fontsize=9)
    ax[1].set_xlabel("rms spread of the 99% time over the square (µs)")
    ax[1].set_ylabel("peak of the average (e/µs)")
    ax[1].set_title("average response: height against dispersion", fontsize=9)
    ax[2].set_xlabel("width holding 90% of the 99% times, over the square (µs)")
    ax[2].set_ylabel("largest single-launch peak (e/µs)")
    ax[2].set_title("single launches: sharpest peak against spread", fontsize=9)
    for a in ax:
        a.grid(alpha=0.3)
    fig.savefig(os.path.join(figdir, "pixel-shapes-summary.pdf"))
    fig.savefig(os.path.join(out, "shapes-summary.png"), dpi=110)
    plt.close(fig)

    tags = {"square3.8": "Sq", "disk3.8": "DiskL", "disk2.5": "DiskM", "disk1.0": "DiskS",
            "hemi3.8": "HemiL", "hemi2.5": "HemiM", "hemi1.0": "HemiS", "quad1.1": "Quad", "quint1.1": "Quint", "ring3.3": "Ring",
            "std3.5": "Std", "grid": "Grid", "gridV1000": "GridVlow", "gridV3000": "GridVhigh",
            "gridH1.6": "GridHlow", "gridH4.8": "GridHhigh", "gridR1.5": "GridRlow", "gridR2.1": "GridRhigh",
            "gridE50": "GridEfifty", "gridE50opt": "GridOpt", "gridE50optc": "GridOptComb",
            "gridE50optn": "GridOptNbr"}
    tags.update({"gridE50" + k[4:]: v.replace("Grid", "GridEfifty") for k, v in list(tags.items())
                 if k in GRIDS and not k.startswith("gridE50") and k != "grid"})
    with open(os.path.join(figdir, "pixel-shapes.tex"), "w") as fh:
        fh.write("% generated by scripts/pixel_shapes.py -- do not edit\n")
        for name, tg in tags.items():
            if name not in M:
                continue
            m, (di, wi) = M[name], info[name]
            for k, v in (("Sd", f"{m['t99_sd']:.2f}"), ("Range", f"{m['t99_range']:.2f}"), ("Ninety", f"{m['t99_w90']:.2f}"),
                         ("PeakTimeSd", f"{m['tpk_sd']:.2f}"), ("PeakTimeRange", f"{m['tpk_range']:.2f}"), ("AvgPeak", f"{m['avg_peak']:.2f}"),
                         ("PeakMax", f"{m['peak_max']:.1f}"), ("PeakMin", f"{m['peak_min']:.2f}"),
                         ("Fwhm", f"{m['fwhm']:.2f}"), ("Collected", f"{m['collected']:.3f}"),
                         ("WidthDiff", f"{m['width_diff']:.3f}"), ("PeakDiff", f"{m['peak_diff']:.2f}"),
                         ("Xe", f"{100 * m['x_e']:.2f}" if "x_e" in m else "--"),
                         ("Xd", f"{100 * m['x_d']:.2f}" if "x_d" in m else "--"),
                         ("Lost", f"{100 * m['lost']:.1f}"),
                         ("DriftRms", f"{di['rms']:.1e}"), ("WeightRms", f"{wi['rms']:.1e}"),
                         ("WeightMax", f"{wi['max']:.1e}"), ("Cols", f"{wi['columns']}")):
                mm = re.fullmatch(r"(-?[0-9.]+)e([+-]?)0*([0-9]+)", v)
                if mm:  # e-notation reads badly in prose
                    v = f"${mm.group(1)}\\times10^{{{'-' if mm.group(2) == '-' else ''}{mm.group(3)}}}$"
                fh.write(f"\\newcommand{{\\shape{tg}{k}}}{{{v}}}\n")
    for name in SHAPES:
        m = M[name]
        print(f"{name:10s} t99 sd {m['t99_sd']:.3f} w90 {m['t99_w90']:.3f} range {m['t99_range']:.3f} peak time sd {m['tpk_sd']:.3f} "
              f"range {m['tpk_range']:.3f} smeared width {m['width_diff']:.3f} peak {m['peak_diff']:.3f} "
              f"X edge {m.get('x_e', float('nan')):.4f} diag {m.get('x_d', float('nan')):.4f} "
              f"avg peak {m['avg_peak']:.3f} fwhm {m['fwhm']:.2f} peaks {m['peak_min']:.2f}-{m['peak_max']:.2f} "
              f"collected {m['collected']:.4f} on {m['frac_on']:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
