#!/usr/bin/env python
"""Regenerate every figure and every quoted number used by the technical note.

    scripts/make_figs.py [RUNDIR] [FIGDIR]

Reads the solved models and simulation outputs in ``RUNDIR`` (default ``runs``)
and writes PDF figures plus ``values.tex`` into ``FIGDIR`` (default
``docs/figs``).  ``values.tex`` defines a LaTeX macro for every number the note
quotes, so re-running this script after improving a model updates the prose as
well as the pictures -- nothing in the document is transcribed by hand.
"""

from __future__ import annotations

import json
import math
import re
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from firep import config as config_mod  # noqa: E402
from firep import bias as bias_mod  # noqa: E402
from firep import perf as perf_mod  # noqa: E402
from firep import electronics as elec_mod  # noqa: E402
from firep import ionization as ion_mod  # noqa: E402
from firep import plot as plot_mod  # noqa: E402
from firep import response as resp_mod  # noqa: E402
from firep import sample as sample_mod  # noqa: E402
from firep import signal as sig_mod  # noqa: E402
from firep import train as train_mod  # noqa: E402

VALUES: dict[str, str] = {}


def _texify(text: str) -> str:
    """Make a formatted number safe and well set in both text and math mode.

    Commas become thin spaces (a comma is punctuation in math mode, and gets
    the wrong spacing) and exponent notation becomes a proper power of ten.
    """
    text = str(text)
    m = re.fullmatch(r"([-+]?[0-9.]+)[eE]([-+]?)0*([0-9]+)", text)
    if m:
        mant, sign, expo = m.groups()
        return f"{mant}\\times 10^{{{'-' if sign == '-' else ''}{expo}}}"
    return text.replace(",", "\\,")


def value(name: str, text) -> None:
    """Record a number for ``values.tex`` under LaTeX macro ``\\name``.

    The name must be letters only: a TeX control sequence cannot contain
    digits, so ``\\valFoo2`` would silently parse as ``\\valFoo`` followed by
    a ``2`` and fail far from here.
    """
    if not name.isalpha():
        raise ValueError(
            f"macro name {name!r} must be letters only -- TeX control "
            "sequences cannot contain digits or punctuation")
    VALUES[name] = _texify(text)




def fig_warmstart(cfg, geom, model, out: str):
    """What the warm-start RMS is an average over, and what it looks like.

    The fit of Eq. (warmstart) is a least squares over points drawn on the
    conductor surfaces -- ``n_surface`` per wire, uniform in angle.  The
    residual it reports is the RMS of ``B_c - g`` over exactly those points,
    pooled across every conductor.  Here that residual is drawn as a function of
    position around each wire, with and without the dipole columns.
    """
    import torch

    from firep.geometry import Sampler
    from firep.siren import HarmonicBaseline

    sampler = Sampler(geom, cfg.sampling.seed, cfg.sampling.near_factor)
    xy, vals = sampler.surfaces(cfg.sampling.n_surface)
    t_xy = torch.as_tensor(xy, dtype=torch.float64)
    t_v = torch.as_tensor(vals, dtype=torch.float64)

    fits = {}
    for mp in (0, 1):
        base = HarmonicBaseline(geom, "wires", multipole=mp).double()
        fits[mp] = (base, base.warm_start(t_xy, t_v))

    # the residual as a continuous function of angle around each wire
    th = np.linspace(0.0, 2 * np.pi, 721)
    curves = {}
    for mp, (base, _) in fits.items():
        curves[mp] = []
        for c in geom.conductors:
            pts = np.column_stack((c.x + c.radius * np.cos(th),
                                   c.y + c.radius * np.sin(th)))
            with torch.no_grad():
                v = base(torch.as_tensor(pts, dtype=torch.float64))[:, 0].numpy()
            curves[mp].append(v - c.potential)

    # per-wire RMS over the sampled points themselves
    n_per = cfg.sampling.n_surface
    with torch.no_grad():
        resid = (fits[1][0](t_xy)[:, 0] - t_v).numpy()
    per_wire = [float(np.sqrt((resid[i * n_per:(i + 1) * n_per] ** 2).mean()))
                for i in range(len(geom.conductors))]

    value("valWarmRms", f"{fits[1][1]:.4f}")
    value("valWarmRmsMono", f"{fits[0][1]:.2f}")
    value("valWarmPoints", f"{len(xy):,d}")
    for c, r in zip(geom.conductors, per_wire):
        value(f"valWarmRms{c.electrode.upper()}", f"{r:.4f}")

    # ---- draw --------------------------------------------------------------
    fig = plt.figure(figsize=(10.5, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.60, 1.0], wspace=0.42, hspace=0.62)
    ax = fig.add_subplot(gs[:, 0])

    ylo, yhi = -2.0, 12.5
    gx = np.linspace(geom.xlo, geom.xhi, 240)
    gy = np.linspace(ylo, yhi, 420)
    GX, GY = np.meshgrid(gx, gy)
    with torch.no_grad():
        pot = model(torch.as_tensor(np.column_stack((GX.ravel(), GY.ravel())),
                                    dtype=torch.get_default_dtype()))[:, 0]
    pot = pot.numpy().reshape(GX.shape)
    im = ax.pcolormesh(GX, GY, pot, cmap="RdBu_r", shading="auto", rasterized=True)
    ax.contour(GX, GY, pot, levels=18, colors="k", linewidths=0.3, alpha=0.5)
    cb = fig.colorbar(im, ax=ax, pad=0.04, fraction=0.06, shrink=0.8)
    cb.ax.set_title("$\\Phi$ [V]", fontsize=8, pad=5)
    cb.ax.tick_params(labelsize=7)

    r_show = 0.95                                  # display radius of a wire
    scale = 0.55 / max(np.abs(np.concatenate(curves[1])).max(), 1e-12)
    for c, e in zip(geom.conductors, curves[1]):
        ax.plot(c.x + r_show * np.cos(th), c.y + r_show * np.sin(th),
                color="0.25", lw=0.9, ls=(0, (2, 2)))
        rr = r_show + scale * e
        ax.plot(c.x + rr * np.cos(th), c.y + rr * np.sin(th), color="k", lw=1.4)
        ax.plot([c.x], [c.y], "o", ms=2.0, color="k")
        ax.text(c.x + 1.15, c.y + 0.35, c.electrode, fontsize=9, color="k")
    ax.set_xlim(geom.xlo, geom.xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("(a) $\\Phi$ near the wires, with the\nwarm-start error drawn "
                 f"around each\nwire (dashed = zero, $\\times${scale:.0f} mm/V)",
                 fontsize=9)

    colours = {"u": "tab:blue", "v": "tab:orange", "w": "tab:green"}
    for k, (mp, title) in enumerate(((0, "(b) monopoles only"),
                                     (1, "(c) monopoles + dipoles"))):
        a = fig.add_subplot(gs[k, 1])
        for c, e in zip(geom.conductors, curves[mp]):
            a.plot(np.degrees(th), e, lw=1.3, color=colours[c.electrode],
                   label=f"{c.electrode}")
        a.axhline(0.0, color="0.6", lw=0.6)
        a.set_xlim(0, 360)
        a.set_xticks([0, 90, 180, 270, 360])
        a.set_xlabel(r"angle around the wire $\vartheta$ [deg]")
        a.set_ylabel(r"$B_c - g$ [V]")
        a.set_title(f"{title}\nRMS over all "
                    f"{len(xy)} points = {fits[mp][1]:.4g} V", fontsize=9)
        a.legend(fontsize=8, ncol=3, loc="upper right", frameon=False)
        a.grid(alpha=0.25, lw=0.4)

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def weighting_bounds(rundir, n=200000):
    """A weighting potential must satisfy 0 <= phi <= 1.  Do the models?

    The Dirichlet data of Eq.~(weighting) is 0 or 1, so by the maximum
    principle no interior value can lie outside that range.  Any excursion is
    unambiguously error, with no interpretation needed -- which makes this the
    cheapest way to find where a weighting solution has stopped being physics.
    """
    import torch as _torch

    from firep.geometry import Sampler as _S

    worst = {}
    agg = {"analytic": [], "trained": []}
    for plane in ("u", "v", "w"):
        for tag, path in (("analytic",
                           os.path.join(rundir, f"weight-{plane}-analytic.pt")),
                          ("trained",
                           os.path.join(rundir, f"weight-{plane}.pt"))):
            if not os.path.exists(path):
                continue
            _cfg, _geom, _mdl, _ = train_mod.load(path)
            _mdl.eval()
            s = _S(_geom, 7, _cfg.sampling.near_factor)
            pts = np.vstack((s.bulk(2 * n // 3), s.near(n // 3)))
            with _torch.no_grad():
                v = _mdl(_torch.tensor(pts, dtype=_torch.float32))[:, 0].numpy()
            # boundary residuals, on the trainer's own sample sets
            fx = np.vstack([s.face(4000, y) for y in (_geom.ylo, _geom.yhi)])
            sx, sv = s.surfaces(512)
            with _torch.no_grad():
                fe = _mdl(_torch.tensor(fx, dtype=_torch.float32))[:, 0].numpy()
                se = _mdl(_torch.tensor(sx, dtype=_torch.float32))[:, 0].numpy()
            agg[tag].append((float(v.min()),
                             float(np.sqrt((fe ** 2).mean())),
                             float(np.sqrt(((se - sv) ** 2).mean()))))
            if tag == "analytic":
                worst[plane] = (float(v.min()), float(v.max()))
                _viol = abs(min(float(v.min()), 0.0))
                value(f"valPhiMin{plane.upper()}",
                      f"{_viol:.1e}" if _viol else "0")
    if worst:
        lo = min(w[0] for w in worst.values())
        value("valPhiMinWorst", f"{abs(lo):.1e}" if lo < 0 else "0")
        value("valPhiMinPlane",
              r"\mathrm{" + min(worst, key=lambda k: worst[k][0]) + "}")
    for tag, rows in agg.items():
        if not rows:
            continue
        t = tag.capitalize()
        _v = abs(min(min(r[0] for r in rows), 0.0))
        value(f"valWeightPhiMin{t}", f"{_v:.1e}" if _v else "0")
        value(f"valWeightFace{t}", f"{max(r[1] for r in rows):.1e}")
        value(f"valWeightSurf{t}", f"{max(r[2] for r in rows):.1e}")
    return worst


def ramo_trained(rundir, tick=0.1):
    """The Ramo residual of a response built from the *trained* weighting models.

    The note claims that training degrades a weighting solution for response
    work.  That claim should be generated, not remembered, so recompute the
    conservation check both ways.
    """
    paths = [os.path.join(rundir, f"weight-{p}.pt") for p in ("u", "v", "w")]
    if not all(os.path.exists(q) for q in paths):
        return
    cm = resp_mod.load_combined(os.path.join(rundir, "drift.pt"), paths)
    r = resp_mod.response(cm, impacts=11, tick=tick, n_wires=21)
    iw = int(np.argmin(np.abs(r.offsets)))
    ipw = r.planes.index("w")
    value("valRamoCollectTrained", f"{r.integrated[ipw, iw, 0]:.4f}")
    value("valRamoInductionTrained",
          f"{max(abs(r.integrated[r.planes.index(p)]).max() for p in 'uv'):.1e}")


def fig_response_table(arrays, meta, figdir, fig):
    """The impact/time tabulation, over the full drift and zoomed near the wires."""
    table = resp_mod.impact_table(arrays, meta, per_side=5)
    u = np.asarray(table["impact"])
    tab = np.asarray(table["table"])
    t = np.asarray(table["time"])
    pitch = float(table["pitch"])
    tick = float(t[1] - t[0])

    # Zoom from the moment the reference trajectory is four pitches above the
    # topmost plane; everything of interest happens after that.
    ty = np.asarray(arrays["traj_y"])[:, 0]
    tt = np.asarray(arrays["traj_t"])
    y_zoom = 4.0 * pitch
    t_zoom = float(np.interp(-y_zoom, -ty, tt))

    plot_mod.plot_response_table(
        table, fig("response-table.pdf"),
        suptitle="field response over the whole drift")
    plot_mod.plot_response_table(
        table, fig("response-table-zoom.pdf"), tmin=t_zoom,
        suptitle=f"the same, from $y = {y_zoom:g}$ mm down")

    value("valTableImpacts", f"{len(u)}")
    value("valTableStep", f"{pitch / 10:g}")
    value("valTableRegions", f"{len(np.asarray(arrays['offsets']))}")
    value("valTableSpan", f"{u.max():g}")
    value("valTableZoom", f"{t_zoom:.1f}")
    value("valTableZoomY", f"{y_zoom:g}")

    # Ramo, applied to the re-indexed table: the collecting wire integrates to
    # one electron inside its own region and to zero everywhere else.
    q = tab.sum(axis=-1) * tick
    inside = np.abs(u) < 0.5 * pitch - 1e-9
    iw = table["planes"].index("w") if "w" in table["planes"] else -1
    value("valTableQIn", f"{np.abs(q[iw, inside]).min():.4f}")
    value("valTableQInMax", f"{np.abs(q[iw, inside]).max():.4f}")
    value("valTableQOut", f"{np.abs(q[iw, ~inside]).max():.4f}")
    value("valTableQInd", f"{np.abs(q[:iw]).max():.4f}")

    # Where the tabulation stops being physics: the peak per wire region falls
    # steeply and then flattens, or turns back up, as the periodic images of the
    # sensing wire in the 21-wire weighting solve take over.
    n_cells = int(np.abs(np.asarray(arrays["offsets"])).max())
    peaks = np.zeros((tab.shape[0], n_cells + 1))
    for n in range(n_cells + 1):
        sel = (np.abs(u) >= n * pitch - 0.5 * pitch + 1e-9) & \
              (np.abs(u) <= n * pitch + 0.5 * pitch - 1e-9)
        peaks[:, n] = np.abs(tab[:, sel, :]).max(axis=(1, 2))
    turn = int(min(int(np.argmin(row)) for row in peaks))
    value("valTableTrust", f"{turn}")
    value("valTableRegionsOut", f"{n_cells}")
    value("valTableOuter", f"{peaks[:, -1].max():.0e}")
    value("valTableDecades",
          f"{np.log10(peaks[:, 0].max() / max(peaks[:, -1].max(), 1e-30)):.0f}")
    value("valTableFloor", f"{peaks[:, turn:].max():.0e}")
    value("valTableFloorFrac",
          f"{100 * peaks[:, turn:].max() / peaks[:, 0].max():.1f}")
    value("valTablePeakW", f"{peaks[iw, 0]:.2f}")
    return table

# --------------------------------------------------------------------------
# what a plain network can reach
# --------------------------------------------------------------------------


def _harmonic_families(geom, pitch, x, y):
    """Explicit harmonic basis columns, grouped by what they can represent.

    ``source_free(N)`` spans every x-periodic harmonic function that extends
    harmonically *through* the wires -- the functions a globally smooth network
    is confined to.  The log and dipole families add the singular branches that
    only a conductor carrying charge can supply.
    """
    ylo, yhi = geom.ylo, geom.yhi

    def source_free(n_modes):
        cols = [np.ones_like(y), y]
        for n in range(1, n_modes + 1):
            k = 2 * np.pi * n / pitch
            cols.append(np.exp(np.clip(k * (y - yhi), -700, 0)) * np.cos(k * x))
            cols.append(np.exp(np.clip(-k * (y - ylo), -700, 0)) * np.cos(k * x))
        return cols

    def psi(xi, yi):
        u = 2 * np.pi * (y - yi) / pitch
        v = 2 * np.pi * (x - xi) / pitch
        au = np.abs(u)
        val = -0.5 * (au - np.log(2.0)
                      + np.log1p(-2 * np.exp(-au) * np.cos(v) + np.exp(-2 * au)))
        far = lambda Y: -np.pi * np.abs(Y - yi) / pitch + 0.5 * np.log(2.0)  # noqa: E731
        a, b = far(ylo), far(yhi)
        return val - (a + (b - a) * (y - ylo) / (yhi - ylo))

    logs, dipoles = [], []
    h = 1e-4
    for c in geom.conductors:
        logs.append(psi(c.x, c.y))
        dipoles.append((psi(c.x + h, c.y) - psi(c.x - h, c.y)) / (2 * h) * c.radius)
        dipoles.append((psi(c.x, c.y + h) - psi(c.x, c.y - h)) / (2 * h) * c.radius)
    return source_free, logs, dipoles


def reachability(cfg, geom, sol, model, n_modes=64, steps=1000):
    """How close each function space can get to the boundary data, and what a
    plain SIREN actually does.

    The first half is pure linear algebra: the least-squares projection of the
    Dirichlet data onto explicit harmonic families.  It sets a floor that no
    optimiser can beat, and it involves no network at all.  The second half
    trains a plain SIREN and measures the line charge it ends up carrying.
    """
    import torch as _torch

    from firep.geometry import Sampler
    from firep.siren import gradient as _grad

    pitch = cfg.electrode("w").lattice.pitch
    span = geom.potential_span
    s = Sampler(geom, 12345, cfg.sampling.near_factor)
    nf, ns = 2000, 1000
    pts, tgt, wt = [], [], []
    for yy, vv in ((geom.ylo, geom.v_ground), (geom.yhi, geom.v_cathode)):
        p = s.face(nf, yy)
        pts.append(p); tgt.append(np.full(nf, vv)); wt.append(np.full(nf, 0.5 / nf))
    sx, sv = s.surfaces(ns)
    pts.append(sx); tgt.append(sv); wt.append(np.full(len(sx), 1.0 / len(sx)))
    X = np.vstack(pts)
    g = np.concatenate(tgt)
    w = np.concatenate(wt)
    nface = 2 * nf

    source_free, logs, dipoles = _harmonic_families(geom, pitch, X[:, 0], X[:, 1])

    def project(cols):
        """Weighted least squares; returns (rms on faces, rms on conductors)."""
        A = np.column_stack(cols)
        nrm = np.sqrt((w[:, None] * A ** 2).sum(0))
        A = A / np.where(nrm > 0, nrm, 1.0)
        sw = np.sqrt(w)[:, None]
        c, *_ = np.linalg.lstsq(A * sw, g * np.sqrt(w), rcond=1e-13)
        e = A @ c - g
        return (float(np.sqrt(np.mean(e[:nface] ** 2))),
                float(np.sqrt(np.mean(e[nface:] ** 2))))

    f_lin, c_lin = project(source_free(0))
    f_mod, c_mod = project(source_free(n_modes))
    f_log, c_log = project(source_free(0) + logs)
    f_dip, c_dip = project(source_free(0) + logs + dipoles)
    value("valProjFaceLin", f"{f_lin:.0f}")
    value("valProjWireLin", f"{c_lin:.0f}")
    value("valProjFaceModes", f"{f_mod:.0f}")
    value("valProjWireModes", f"{c_mod:.0f}")
    value("valProjNModes", f"{n_modes}")
    value("valProjDimModes", f"{2 + 2 * n_modes}")
    value("valProjFaceLogs", f"{f_log:.4f}")
    value("valProjWireLogs", f"{c_log:.3f}")
    value("valProjWireDip", f"{c_dip:.4f}")
    # the trainer's own dimensionless objective, for the best straight line
    value("valLossLine", f"{10 * ((f_lin / span) ** 2 + (c_lin / span) ** 2):.4f}")
    # how tightly a transverse mode is locked between the collection row and
    # the ground plane one plane spacing below it
    lock = np.exp(2 * np.pi * (geom.conductors[-1].y - geom.ylo) / pitch)
    value("valModeLock", f"{lock:.0e}")
    value("valScaleQuartic",
          f"{(pitch / cfg.electrode('w').shape.radius) ** 4:.0e}")

    # -- and what a plain SIREN actually converges to ------------------------
    prev = _torch.get_default_dtype()
    cfg2 = config_mod.load_with_overrides(
        None, ("model.baseline.mode=linear", f"train.steps={steps}"))
    model2, geom2, hist, sol2 = train_mod.train(cfg2, bias_mod.solve(cfg2))
    model2.eval()
    rec = hist.records[-1]
    value("valPlainFace", f"{rec.rms_face_v:.0f}")
    value("valPlainWire", f"{rec.rms_surface_v:.0f}")
    value("valPlainPde", f"{rec.pde:.0e}")
    value("valLossPlain", f"{rec.total:.4f}")

    # Gauss's law: the line charge each trained network ended up carrying.
    th = np.linspace(0, 2 * np.pi, 4096, endpoint=False)

    def flux(mdl, g_, radius):
        """Contour integral of grad(Phi).n around every conductor of ``g_``."""
        out = []
        for c in g_.conductors:
            xy = np.column_stack((c.x + radius * np.cos(th),
                                  c.y + radius * np.sin(th)))
            t = _torch.tensor(xy, dtype=_torch.get_default_dtype(),
                              requires_grad=True)
            gr = _grad(mdl, t).detach().numpy()
            en = -(gr[:, 0] * np.cos(th) + gr[:, 1] * np.sin(th))
            out.append(float(en.mean() * 2 * np.pi * radius))
        return np.array(out)

    plain = np.abs(flux(model2, geom2, 0.3)).max()
    value("valPlainFlux", f"{plain:.0e}" if plain else "0")
    value("valGaussW", f"{2 * np.pi * sol.gamma['w']:.0f}")
    # ...and the same integral on the hybrid model, which must return 2 pi gamma
    # at every contour radius.
    exact = np.array([2 * np.pi * sol.gamma[c.electrode] for c in geom.conductors])
    err = max(np.abs(flux(model, geom, r) / exact - 1).max() for r in (0.3, 1.0))
    value("valHybridFluxErr", f"{100 * err:.1f}")

    # The two numbers the boundary-condition argument turns on: how far the
    # collection wire sits from the straight line a source-free harmonic
    # function is confined to, and the mean Laplacian a smooth network would
    # have to sustain inside a wire to carry the charge that closes the gap.
    ramp = geom.v_ground + ((geom.v_cathode - geom.v_ground)
                            * (geom.conductors[-1].y - geom.ylo)
                            / (geom.yhi - geom.ylo))
    value("valRampError", f"{abs(geom.conductors[-1].potential - ramp):.0f}")
    a_w = geom.conductors[-1].radius
    value("valWireLap", f"{2 * np.pi * abs(sol.gamma['w']) / (np.pi * a_w ** 2):.0e}")
    _torch.set_default_dtype(prev)

# --------------------------------------------------------------------------
# conceptual diagrams
# --------------------------------------------------------------------------


def fig_flow(out: str) -> str:
    """How the four models compose into a readout waveform."""
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    fc=fc, ec="0.35", lw=1.0))
        ax.text(x + w / 2, y + h - 0.20, title, ha="center", va="top",
                fontsize=9.5, fontweight="bold")
        ax.text(x + w / 2, y + h - 0.52, body, ha="center", va="top", fontsize=7.6)

    def arrow(x0, y0, x1, y1, label="", dx=0.0, dy=0.16):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=12, lw=1.1, color="0.3"))
        if label:
            ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, label, ha="center",
                    fontsize=7.4, color="0.25", style="italic")

    sir = "#dbe9f6"
    cmp = "#e8f3e2"
    inp = "#fdf0dc"

    box(0.15, 3.0, 2.2, 1.3, "drift model  $\\Phi$",
        "SIREN + analytic basis\nelectrodes at bias\n"
        "$\\nabla^2\\Phi=0$", sir)
    box(0.15, 1.2, 2.2, 1.3, "weighting  $\\varphi_p$",
        "SIREN + analytic basis\none wire at 1 V\n"
        "$\\nabla^2\\varphi_p=0$", sir)
    box(2.95, 2.1, 2.3, 1.3, "field response  $r_p$",
        "drift trajectories\n+ Shockley--Ramo\n"
        "$i=q\\,\\mathbf{v}\\cdot\\mathbf{E}_{w}$", cmp)
    box(2.95, 0.15, 2.3, 1.1, "ionization  $n(x,t)$",
        "Gaussian groups\ndiffusion, attachment", inp)
    box(5.85, 1.3, 1.9, 1.3, "current  $I_{p,w}$",
        "$n \\ast r_p$\nper wire", cmp)
    box(8.05, 1.3, 1.8, 1.3, "ADC  $a_{p,w,k}$",
        "shaper $\\times$ gain\nsample, quantise", inp)

    arrow(2.35, 3.65, 2.95, 3.0, "$\\mathbf{E}=-\\nabla\\Phi$", dy=0.22)
    arrow(2.35, 1.85, 2.95, 2.4, "$\\varphi_p$", dy=0.20)
    arrow(5.25, 2.75, 5.85, 2.3, "", )
    arrow(5.25, 0.70, 5.85, 1.6, "")
    arrow(7.75, 1.95, 8.05, 1.95, "")

    ax.text(1.25, 4.42, "trained", ha="center", fontsize=8.5, style="italic",
            color="0.35")
    ax.text(6.9, 4.42, "composed, no new parameters", ha="center", fontsize=8.5,
            style="italic", color="0.35")
    ax.plot([2.7, 2.7], [0.05, 4.3], color="0.8", lw=1.0, ls="--")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_siren(out: str) -> str:
    """What a SIREN is and why the sine matters."""
    import torch

    from firep.siren import Siren

    fig = plt.figure(figsize=(10.5, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.42,
                          wspace=0.30)

    # (a) architecture
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    xs = [0.6, 2.6, 4.6, 6.6, 8.6]
    labels = [
        ("$\\mathbf{r}=(x,y)$", "input\n2 numbers"),
        ("$\\gamma(\\mathbf{r})$", "encoding\n$\\sin,\\cos$ of $2\\pi m x/P$\n$n_0=2M{+}1$ numbers"),
        ("$h^{(1)}$", "$\\sin(\\omega_1(W^{(1)}\\gamma+b^{(1)}))$\n$W^{(1)}$ is $n_1\\times n_0$"),
        ("$h^{(L)}$", "$\\sin(\\omega_L(W^{(L)}h^{(L-1)}+b^{(L)}))$\n$n_L\\times n_{L-1}$"),
        ("$N_\\theta$", "linear read-out\n$1\\times n_L$"),
    ]
    for x, (name, sub) in zip(xs, labels):
        ax.add_patch(FancyBboxPatch((x - 0.55, 1.05), 1.1, 0.7,
                                    boxstyle="round,pad=0.05",
                                    fc="#dbe9f6", ec="0.35", lw=1.0))
        ax.text(x, 1.40, name, ha="center", va="center", fontsize=10)
        ax.text(x, 0.88, sub, ha="center", va="top", fontsize=7.0, color="0.3")
    for a, b in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch((a + 0.58, 1.40), (b - 0.58, 1.40),
                                     arrowstyle="-|>", mutation_scale=11,
                                     lw=1.0, color="0.35"))
    ax.text(5.6, 2.30, "$\\cdots$ $L$ sine layers $\\cdots$", ha="center",
            fontsize=8.5, color="0.35")
    ax.text(0.05, 0.25,
            "every layer's activation is a sine, so every derivative of the "
            "network is again a network of the same form",
            fontsize=8.2, style="italic", color="0.3")
    ax.set_title("(a) architecture", fontsize=10, loc="left")

    # (b) composing sines widens the spectrum
    ax = fig.add_subplot(gs[1, 0])
    n = 1 << 14
    t = np.linspace(-1, 1, n, endpoint=False)
    w = 6.0
    sigs = {
        "1 layer": np.sin(w * t),
        "2 layers": np.sin(w * np.sin(w * t)),
        "3 layers": np.sin(w * np.sin(w * np.sin(w * t))),
    }
    freq = np.fft.rfftfreq(n, d=(t[1] - t[0]))
    for lab, y in sigs.items():
        mag = np.abs(np.fft.rfft(y)) / n
        ax.semilogy(freq, np.maximum(mag, 1e-8), lw=1.1, label=lab)
    ax.set_xlim(0, 40)
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel("frequency [cycles per unit input]")
    ax.set_ylabel("amplitude")
    ax.set_title("(b) depth builds bandwidth", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which="both")

    # (c) derivatives stay smooth, unlike ReLU
    ax = fig.add_subplot(gs[1, 1])
    x = np.linspace(-1, 1, 1500)
    for name, act, style in (("sine", torch.sin, "-"),
                             ("ReLU", torch.relu, "--")):
        xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        w = torch.tensor([[2.7], [-4.1], [3.3], [-2.2], [5.0]],
                         dtype=torch.float64)
        b = torch.tensor([[0.3], [-0.7], [1.1], [0.2], [-0.4]],
                         dtype=torch.float64)
        h = act(w * xt[None, :] + b)
        f = (torch.tensor([[1.0, -0.8, 0.6, -1.2, 0.5]], dtype=torch.float64)
             @ h).squeeze()
        g = torch.autograd.grad(f.sum(), xt, create_graph=True)[0]
        d2 = torch.autograd.grad(g.sum(), xt, create_graph=True)[0]
        ax.plot(x, d2.detach().numpy(), style, lw=1.3, label=f"{name}")
    ax.set_xlabel("input")
    ax.set_ylabel("$d^2 f/dx^2$")
    ax.set_title("(c) second derivative", fontsize=10)
    ax.annotate("ReLU: identically zero", xy=(0.0, 0.0), xytext=(-0.95, -12),
                fontsize=7.5, color="C1",
                arrowprops=dict(arrowstyle="->", color="C1", lw=0.9))
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.25)

    # (d) periodicity is exact by construction
    ax = fig.add_subplot(gs[1, 2])
    torch.manual_seed(3)
    net = Siren(in_features=5, hidden=48, layers=3, omega0=8.0, omega_hidden=8.0)
    period = 5.0
    xx = np.linspace(-1.5 * period, 1.5 * period, 1200)
    modes = np.arange(1, 3)
    feats = np.concatenate(
        [np.sin(2 * np.pi * np.outer(xx, modes) / period),
         np.cos(2 * np.pi * np.outer(xx, modes) / period),
         np.zeros((len(xx), 1))], axis=1)
    with torch.no_grad():
        vals = net(torch.tensor(feats, dtype=torch.float32)).numpy()[:, 0]
    ax.plot(xx, vals, lw=1.3)
    for k in (-1, 0, 1):
        ax.axvline((k + 0.5) * period, color="0.7", lw=0.8, ls="--")
    ax.set_xlabel("x [mm]")
    ax.set_title("(d) exact periodicity from $\\gamma$", fontsize=10)
    ax.grid(alpha=0.25)

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_ionization(groups, props, out: str) -> str:
    """The line source: where it is, and what reaches the response plane."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)

    ax = axes[0]
    ax.plot(groups.x, groups.t, lw=2.0, color="C0")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("arrival time [$\\mu$s]")
    ax.invert_yaxis()
    ax.set_title("(a) track at the response plane", fontsize=10)
    ax.grid(alpha=0.25)
    ax.text(0.04, 0.05,
            f"{len(groups)} groups\n{groups.total:,.0f} e after attachment",
            transform=ax.transAxes, fontsize=7.5,
            bbox=dict(fc="white", ec="0.8", alpha=0.9))

    ax = axes[1]
    d = np.linspace(1.0, 3500.0, 400)
    s_l, s_t, s_time = props.spread(d)
    ax.plot(d / 1000, s_t, lw=1.5, label="$\\sigma_T$ transverse")
    ax.plot(d / 1000, s_l, lw=1.5, label="$\\sigma_L$ longitudinal")
    ax.plot(d / 1000, 100 * props.survival(d), lw=1.5, ls="--",
            label="survival [%]")
    ax.axvline(1.0, color="0.6", lw=0.9, ls=":")
    ax.set_xlabel("drift distance [m]")
    ax.set_ylabel("$\\sigma$ [mm]   /   survival [%]")
    ax.set_title("(b) diffusion and attachment", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = axes[2]
    xe = np.linspace(groups.x.min() - 6, groups.x.max() + 6, 241)
    te = np.linspace(groups.t.min() - 4, groups.t.max() + 4, 241)
    dens = ion_mod.arrival_grid(groups, xe, te)
    im = ax.pcolormesh(0.5 * (xe[:-1] + xe[1:]), 0.5 * (te[:-1] + te[1:]),
                       dens.T, cmap="magma_r", shading="auto", rasterized=True)
    fig.colorbar(im, ax=ax, label="electrons / bin", pad=0.02)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("arrival time [$\\mu$s]")
    ax.invert_yaxis()
    ax.set_title("(c) arrival density $n(x,t)$", fontsize=10)
    fig.savefig(out)
    plt.close(fig)
    return out


def _lattice_shapes(cfg, e, ndim=3):
    from firep import shapes as shp

    return shp.lattice_shapes(e, ndim)


def fig_dune_vd_detail(out: str) -> str:
    """How the hole lattice aligns with the strips on a DUNE VD anode."""
    from firep import config as cfg_mod
    from firep import shapes as shp

    cfg = cfg_mod.load(os.path.join(REPO, "examples", "geom-dune-vd.yaml"))
    by = {e.name: e for e in cfg.electrodes}
    holes = ("holes", "#ffffff", shp.lattice_shapes(by["holes2"], 3),
             by["holes2"].plane, "aperture")
    lim = (-9.0, 9.0)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3), constrained_layout=True)
    panels = [
        (["w"], "(a) collection, pitch 5.1 mm = 2d", ["#2ca02c"]),
        (["u"], "(b) induction, pitch 7.65 mm = 3d", ["#1f77b4"]),
        (["grid", "u", "v", "w"], "(c) all four views", 
         ["#7f7f7f", "#1f77b4", "#d62728", "#2ca02c"]),
    ]
    for ax, (names, title, colours) in zip(axes, panels):
        groups = [(f"{n} ({by[n].kind})", c, shp.lattice_shapes(by[n], 3),
                   by[n].plane, "fill") for n, c in zip(names, colours)]
        groups.append(holes)
        plot_mod.plot_electrode_plan(groups, None, lim, lim, title=title,
                                     n=520, ax=ax, legend=False)
        if len(names) > 1:
            ax.legend(fontsize=6.5, loc="upper center", ncol=3,
                      bbox_to_anchor=(0.5, -0.13), frameon=False)
    d = 2.55
    for ax in axes[:2]:
        for k in range(-3, 4):
            ax.axvline(k * d, color="0.1", lw=0.7, ls=(0, (4, 3)), zorder=7)
    fig.suptitle("DUNE vertical drift: every strip gap bisects a row of holes",
                 fontsize=11)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_geometries(figdir: str) -> list:
    """Plan views of the three detector styles the method must generalise to."""
    from firep import config as cfg_mod

    written = []
    specs = [
        ("geom-microboone.pdf", "geom-microboone-3d.yaml",
         "(a) MicroBooNE: crossing wire planes", (-15, 15), (-15, 15),
         {"u": "#1f77b4", "v": "#d62728", "y": "#2ca02c"}),
        ("geom-dune-vd.pdf", "geom-dune-vd.yaml",
         "(b) DUNE vertical drift: strips and holes", (-16, 16), (-16, 16),
         {"u": "#1f77b4", "v": "#d62728", "w": "#2ca02c", "holes": "#ffffff"}),
        ("geom-pixels.pdf", "geom-pixels.yaml",
         "(c) pixels: an array of pads", (-12, 12), (-12, 12),
         {"pad": "#2ca02c"}),
    ]
    for name, path, title, xlim, zlim, colours in specs:
        cfg = cfg_mod.load(os.path.join(REPO, "examples", path))
        groups = []
        for e in cfg.electrodes:
            style = "aperture" if e.kind == "hole" else "fill"
            groups.append((f"{e.name} ({e.kind})", colours.get(e.name, "0.5"),
                           _lattice_shapes(cfg, e), e.plane, style))
        out = os.path.join(figdir, name)
        plot_mod.plot_electrode_plan(groups, out, xlim, zlim, title=title)
        written.append(out)
    return written


def fig_periodicity(rundir: str, out: str):
    """What the encoding's periodicity means for a weighting solve."""
    import torch

    from firep import bias as bias_mod
    from firep import config as cfg_mod
    from firep.geometry import Geometry, Sampler
    from firep.siren import FieldModel

    torch.set_default_dtype(torch.float32)
    cfg, geom, model, _ = train_mod.load(os.path.join(rundir,
                                                      "weight-w-analytic.pt"))
    pitch = cfg.electrode("w").lattice.pitch
    nw = cfg.electrode("w").lattice.count

    def ev(mdl, x, y):
        with torch.no_grad():
            return mdl(torch.tensor(np.column_stack((x, np.full(len(x), y))),
                                    dtype=torch.float32))[:, 0].numpy()

    # an independent solve with the WHOLE collection plane at unit potential
    half = nw * pitch / 2
    sets = (f"domain.bounds.x=[{-half}, {half}]", "bias.mode=explicit",
            "domain.boundaries.yhi.potential=0.0",
            "domain.boundaries.ylo.potential=0.0",
            "electrodes.0.bias=0.0", "electrodes.1.bias=0.0",
            "electrodes.2.bias=1.0") + tuple(
            f"electrodes.{i}.lattice.count={nw}" for i in range(3))
    c2 = cfg_mod.load_with_overrides(None, sets)
    g2 = Geometry(c2, bias_mod.solve(c2))
    m2 = FieldModel(c2, g2)
    sp = Sampler(g2, 3, c2.sampling.near_factor)
    xy, vals = sp.surfaces(64)
    m2.baseline.warm_start(torch.tensor(xy, dtype=torch.float32),
                           torch.tensor(vals, dtype=torch.float32))
    with torch.no_grad():
        m2.output_scale.zero_()

    xs = np.linspace(-11.0, 11.0, 601)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3), constrained_layout=True)

    ax = axes[0]
    for k, y in enumerate((0.5, 1.5, 4.0)):
        ax.plot(xs, ev(model, xs, y), lw=1.4, color=f"C{k}",
                label=f"y = {y:g} mm")
    for n in range(-2, 3):
        ax.axvline(n * pitch, color="0.8", lw=0.7, ls=":")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("$\\varphi_w$")
    ax.set_title("(a) one sensing wire: not $p$-periodic", fontsize=9.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for k, y in enumerate((0.5, 1.5, 4.0)):
        tot = sum(ev(model, xs - n * pitch, y)
                  for n in range(-(nw // 2), nw // 2 + 1))
        ax.plot(xs, tot, lw=1.8, color=f"C{k}", alpha=0.45)
        ax.plot(xs, ev(m2, xs, y), "--", lw=1.0, color=f"C{k}")
    for n in range(-2, 3):
        ax.axvline(n * pitch, color="0.8", lw=0.7, ls=":")
    ax.plot([], [], color="0.4", lw=1.8, alpha=0.45, label="sum over all wires")
    ax.plot([], [], "--", color="0.2", lw=1.0, label="whole plane at 1 V")
    ax.set_xlabel("x [mm]")
    ax.set_title("(b) summed: $p$-periodic", fontsize=9.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # (c) how good is the finite-array approximation?
    ax = axes[2]
    # Heights above the collection plane, avoiding the interiors of the u and
    # v wires at y = 10 and 5 mm where the basis is log-singular.
    ys = np.array([0.3, 0.6, 1.2, 2.0, 3.0, 3.8, 7.0, 12.0, 20.0, 35.0])
    planes = sorted(e.plane for e in cfg.electrodes)
    ys = np.array([y for y in ys
                   if min(abs(y - q) for q in planes) > 0.5])
    ref = None
    curves = {}
    for n_wires in (11, 21, 41, 161):
        hw = n_wires * pitch / 2
        st = (f"domain.bounds.x=[{-hw}, {hw}]", "problem.kind=weighting",
              "problem.weighting.electrode=w") + tuple(
              f"electrodes.{i}.lattice.count={n_wires}" for i in range(3))
        c3 = cfg_mod.load_with_overrides(None, st)
        g3 = Geometry(c3, bias_mod.solve(c3))
        m3 = FieldModel(c3, g3)
        s3 = Sampler(g3, 5, c3.sampling.near_factor)
        a3, v3 = s3.surfaces(48)
        m3.baseline.warm_start(torch.tensor(a3, dtype=torch.float32),
                               torch.tensor(v3, dtype=torch.float32))
        with torch.no_grad():
            m3.output_scale.zero_()
        curves[n_wires] = np.array([ev(m3, np.array([0.0]), y)[0] for y in ys])
    ref = curves[161]
    for n_wires in (11, 21, 41):
        ax.loglog(ys, np.abs(curves[n_wires] - ref), "o-", ms=3, lw=1.1,
                  label=f"$N_w$ = {n_wires}")
    ax.set_xlabel("height above the plane [mm]")
    ax.set_ylabel("$|\\varphi_w - \\varphi_w^{N_w=161}|$")
    ax.set_title("(c) cost of a finite array", fontsize=9.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which="both")

    fig.savefig(out)
    plt.close(fig)
    worst21 = float(np.abs(curves[21] - ref).max())
    return out, worst21


def fig_trajectory(rundir: str, out: str):
    """How a drift path is stepped: adaptively, on the distance to a conductor."""
    cm = resp_mod.load_combined(
        os.path.join(rundir, "drift.pt"),
        [os.path.join(rundir, f"weight-{p}-analytic.pt") for p in ("u", "v", "w")])
    x0 = np.array([0.15, 1.25, 2.30])
    traj = resp_mod.trajectories(cm, x0, y_start=16.0, tick=0.1, substeps=4)
    xy, tt = traj.xy, traj.t

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), constrained_layout=True)

    ax = axes[0]
    for i in range(len(x0)):
        live = traj.alive[:, i]
        last = int(np.argmax(~live)) if (~live).any() else len(live) - 1
        ax.plot(xy[:last + 1, i, 0], xy[:last + 1, i, 1], lw=1.0, color=f"C{i}",
                label=f"$x_0$ = {x0[i]:.2f} mm")
        ax.plot(xy[:last + 1:12, i, 0], xy[:last + 1:12, i, 1], ".", ms=2.0,
                color=f"C{i}")
    for c in cm.geom.conductors:
        ax.add_patch(plt.Circle((c.x, c.y), c.radius, fc="0.3", ec="k", lw=0.6,
                                zorder=5))
    ax.set_xlim(cm.geom.xlo, cm.geom.xhi)
    ax.set_ylim(-1.5, 16.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("(a) paths, every 12th step marked", fontsize=9.5)
    ax.legend(fontsize=6.2, loc="lower left", framealpha=0.92,
              borderpad=0.3, handlelength=1.2)

    ax = axes[1]
    i = 0
    live = traj.alive[:, i]
    last = int(np.argmax(~live)) if (~live).any() else len(live) - 1
    p_ = xy[:last + 1, i, :]
    gap = cm.geom.min_gap_distance(
        np.column_stack((cm.geom.wrap_x(p_[:, 0]), p_[:, 1])))
    step = np.linalg.norm(np.diff(p_, axis=0), axis=1)
    ax.loglog(gap[:-1], np.maximum(step, 1e-6), ".", ms=2.5)
    g = np.logspace(np.log10(max(gap.min(), 1e-4)), np.log10(gap.max()), 50)
    ax.loglog(g, 0.3 * g, "--", color="0.5", lw=1.0,
              label="$0.3\\,d$ (curvature limit)")
    ax.axhline(0.1 / 4 * cm.speed(np.array([62.0]))[0], color="C3", ls=":",
               lw=1.1, label="tick/4 (resolution limit)")
    ax.set_xlabel("distance to nearest conductor $d$ [mm]")
    ax.set_ylabel("step length [mm]")
    ax.set_title("(b) the adaptive step", fontsize=9.5)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.25, which="both")

    ax = axes[2]
    for i in range(len(x0)):
        live = traj.alive[:, i]
        last = int(np.argmax(~live)) if (~live).any() else len(live) - 1
        e = cm.drift_field(xy[:last + 1, i, :])
        ax.semilogy(tt[:last + 1], np.hypot(e[:, 0], e[:, 1]) * 10 / 1e3, lw=1.1,
                    color=f"C{i}")
    ax.axhspan(*resp_mod.WALKOWIAK_E_RANGE, color="C2", alpha=0.10)
    ax.set_xlabel("time since launch [$\\mu$s]")
    ax.set_ylabel("$|E|$ on the path [kV/cm]")
    ax.set_title("(c) field along the path", fontsize=9.5)
    ax.grid(alpha=0.25, which="both")

    fig.savefig(out)
    plt.close(fig)
    return out, int(last), float(np.median(step))


def fig_surrogate(arrays, meta, out: str):
    """Learning the response directly, and conditioned on a diffusion width."""
    import torch

    from firep import surrogate as sur

    tick = float(arrays["time"][1] - arrays["time"][0])
    ip = meta["planes"].index("w")
    impact = np.asarray(arrays["impact"], float)
    iw0 = int(np.argmin(np.abs(arrays["offsets"])))
    cur = np.asarray(arrays["current"])[ip]
    held = 1.1

    # (1) plain surrogate of the tabulated response
    x0, c0, y0 = sur.training_set(arrays, (), plane=ip)
    spec = sur.SurrogateSpec(n_wires=y0.shape[1], n_time=y0.shape[2])
    m0, r0 = sur.fit(spec, x0, c0, y0, steps=3000)
    with torch.no_grad():
        p0 = m0(torch.tensor(x0, dtype=torch.float32)).numpy()

    # (2) conditioned on sigma_t, with one width held out of training
    def train_conditioned(n_sigma, hidden, steps):
        sig = [v for v in np.linspace(0.0, 2.0, n_sigma) if abs(v - held) > 1e-9]
        xs, cs, ts = [], [], []
        for v in sig:
            ts.append(np.transpose(sur.smear(cur, tick, float(v)), (1, 0, 2)))
            xs.append(impact)
            cs.append(np.full((len(impact), 1), v))
        sp = sur.SurrogateSpec(n_wires=arrays["current"].shape[1],
                               n_time=arrays["current"].shape[-1],
                               conditions=("sigma_t",), hidden=hidden)
        mdl, res = sur.fit(sp, np.concatenate(xs), np.concatenate(cs),
                           np.concatenate(ts), steps=steps, batch=256)
        return mdl, res, len(sig)

    truth_held = np.transpose(sur.smear(cur, tick, held), (1, 0, 2))

    def held_error(mdl):
        with torch.no_grad():
            pr = mdl(torch.tensor(impact, dtype=torch.float32),
                     torch.full((len(impact), 1), held)).numpy()
        return pr, float(np.sqrt(((pr - truth_held) ** 2).mean())
                         / np.abs(truth_held).max())

    sweep = []
    m1 = r1 = None
    for n_sigma, hidden, steps in ((11, 160, 4000), (21, 192, 6000),
                                   (41, 192, 8000)):
        mdl, res, n_used = train_conditioned(n_sigma, hidden, steps)
        pred, rel = held_error(mdl)
        sweep.append((2.0 / (n_sigma - 1), rel))
        m1, r1, pred_held, rel_held = mdl, res, pred, rel

    t = arrays["time"]
    lo = max(int(np.argmax(np.abs(y0[0, iw0]) > 1e-3 * np.abs(y0).max())) - 40, 0)
    sl = slice(lo, min(lo + 260, len(t)))

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.2), constrained_layout=True)

    ax = axes[0]
    for k, ii in enumerate((0, len(impact) // 2, len(impact) - 1)):
        ax.plot(t[sl], y0[ii, iw0, sl], lw=1.8, color=f"C{k}", alpha=0.40,
                label=f"tabulated, {impact[ii]:.2f} mm")
        ax.plot(t[sl], p0[ii, iw0, sl], "--", lw=1.0, color=f"C{k}",
                label="surrogate" if k == 0 else None)
    ax.set_xlabel("time [$\\mu$s]")
    ax.set_ylabel("current [e/$\\mu$s]")
    ax.set_title(f"(a) learned response, rel.~rms {r0.rel_rms:.1e}", fontsize=9.5)
    ax.legend(fontsize=6.3)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for k, v in enumerate((0.0, 0.5, 1.5)):
        tr = np.transpose(sur.smear(cur, tick, v), (1, 0, 2))[0, iw0]
        with torch.no_grad():
            pr = m1(torch.tensor(impact[:1], dtype=torch.float32),
                    torch.full((1, 1), v)).numpy()[0, iw0]
        ax.plot(t[sl], tr[sl], lw=1.8, color=f"C{k}", alpha=0.40,
                label=f"$\\sigma_t$ = {v:g} $\\mu$s")
        ax.plot(t[sl], pr[sl], "--", lw=1.0, color=f"C{k}")
    ax.set_xlabel("time [$\\mu$s]")
    ax.set_title("(b) conditioned on $\\sigma_t$", fontsize=9.5)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(t[sl], truth_held[0, iw0][sl], lw=2.0, color="0.35", alpha=0.6,
            label="truth")
    ax.plot(t[sl], pred_held[0, iw0][sl], "--", lw=1.2, color="C3",
            label="surrogate")
    ax.set_xlabel("time [$\\mu$s]")
    ax.set_title(f"(c) held-out $\\sigma_t$ = {held} $\\mu$s, "
                 f"rel.~rms {rel_held:.1e}", fontsize=9.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = axes[3]
    ds = np.array([d for d, _ in sweep])
    er = np.array([e for _, e in sweep])
    ax.loglog(ds, er, "o-", lw=1.3)
    ax.set_xlabel("$\\sigma_t$ grid spacing [$\\mu$s]")
    ax.set_ylabel("held-out rel. rms")
    ax.set_title("(d) interpolation converges", fontsize=9.5)
    ax.grid(alpha=0.25, which="both")

    fig.savefig(out)
    plt.close(fig)
    return out, r0, r1, rel_held, sweep


# --------------------------------------------------------------------------


def main(rundir="runs", figdir="docs/figs") -> None:
    with perf_mod.step("figures and values", "figs") as _d:
        _main(rundir, figdir, _d)


def _main(rundir, figdir, _perf) -> None:
    os.makedirs(figdir, exist_ok=True)

    def fig(name):
        return os.path.join(figdir, name)

    # ---- geometry, bias ---------------------------------------------------
    cfg, geom, model, sol = train_mod.load(os.path.join(rundir, "drift.pt"))
    plot_mod.plot_geometry(sample_mod.metadata(cfg, geom), fig("geometry.pdf"),
                           figsize=(4.6, 6.4))
    plot_mod.plot_bias_profile(cfg, sol, fig("bias-profile.pdf"))
    plot_mod.plot_basis(model, geom, fig("basis.pdf"))

    value("valPitch", f"{cfg.electrode('w').lattice.pitch:g}")
    value("valRadius", f"{cfg.electrode('w').shape.radius:g}")
    value("valCathode", f"{sol.cathode:,.0f}")
    value("valDriftNominal", f"{sol.drift_field_nominal:.0f}")
    value("valDriftActual", f"{sol.drift_field_actual:.1f}")
    rho = 2 * np.pi * cfg.electrode("w").shape.radius / cfg.electrode("w").lattice.pitch
    value("valRho", f"{rho:.4f}")
    value("valTransparencyRatio", f"{(1 + rho) / (1 - rho):.4f}")
    for name in ("u", "v", "w"):
        value(f"valV{name.upper()}", f"{sol.electrodes[name]:.1f}")
        value(f"valGamma{name.upper()}", f"{sol.gamma[name]:.1f}")
        value(f"valOffset{name.upper()}", f"{sol.offset(name):+.0f}")
    value("valLambda", f"{sol.lam['w']:.4f}")
    for _k, _g in zip(("CU", "UV", "VW", "WG"), sol.gaps):
        value(f"valField{_k}", f"{_g.field:.1f}")
    value("valSpan", f"{geom.potential_span:,.0f}")

    # Solver settings the outline section quotes, taken from the same config
    # and checkpoint the figures come from.
    value("valNBulk", f"{cfg.sampling.n_bulk}")
    value("valNNear", f"{cfg.sampling.n_near}")
    value("valNFace", f"{cfg.sampling.n_face}")
    value("valNSurface", f"{cfg.sampling.n_surface}")
    value("valNearFactor", f"{cfg.sampling.near_factor:g}")
    value("valSteps", f"{cfg.train.steps:,d}")
    value("valLrInit", f"{cfg.train.lr:.0e}")
    value("valLrFinal", f"{cfg.train.lr_final:.0e}")
    value("valWBoundary", f"{cfg.train.w_face:g}")
    value("valNParams",
          f"{sum(q.numel() for q in model.net.parameters()):,d}")
    value("valNCoeff", f"{model.baseline.coeff.numel()}")
    value("valOutputScale", f"{float(model.output_scale):.2f}")

    # ---- drift solution ---------------------------------------------------
    arrays, meta = sample_mod.load_npz(os.path.join(rundir, "drift.npz"))
    plot_mod.plot_solution(arrays, meta, fig("drift-potential.pdf"),
                           contours=30, drift=11, figsize=(4.6, 6.4))
    zarr, zmeta = sample_mod.load_npz(os.path.join(rundir, "drift-zoom.npz"))
    plot_mod.plot_solution(zarr, zmeta, fig("drift-zoom.pdf"), contours=40,
                           streamlines=1.5, equal=True, figsize=(4.4, 7.0))

    # How harmless is it to evaluate the analytic basis' Laplacian numerically?
    # Analytically zero; in float32, near a wire, it is not.
    import torch as _torch

    from firep.siren import laplacian as _lap

    _probe = np.column_stack((np.zeros(1), [cfg.electrode("w").shape.radius + 0.01]))
    for _prec, _tag, _dt in (("64", "Double", _torch.float64),
                             ("32", "Single", _torch.float32)):
        _cfg2 = config_mod.load_with_overrides(None, (f"train.precision=float{_prec}",))
        _sol2 = bias_mod.solve(_cfg2)
        from firep.geometry import Geometry as _G
        from firep.siren import FieldModel as _FM
        _torch.set_default_dtype(_dt)
        _g2 = _G(_cfg2, _sol2)
        _m2 = _FM(_cfg2, _g2).to(_dt)
        with _torch.no_grad():
            _m2.baseline.coeff[:3] = _torch.tensor(
                [sol.gamma[n] for n in ("u", "v", "w")], dtype=_dt)
        _t = _torch.tensor(_probe, dtype=_dt)
        _l = float(abs(_lap(_m2.baseline, _t, create_graph=False)[0, 0]))
        value(f"valBasisLap{_tag}", f"{_l:.1e}" if _l else "0")
        if _prec == "64":
            _tt = _t.clone().requires_grad_(True)
            _gr = _torch.autograd.grad(_m2.baseline(_tt).sum(), _tt)[0].numpy()[0]
            value("valBasisGrad", f"{np.hypot(*_gr):.0f}")
    _torch.set_default_dtype(_torch.float32)

    # The fitted basis coefficients have physical readings: the monopole is the
    # static line charge (checked against Gauss's law in the bias solver) and
    # the y-dipole is the induced dipole of a cylinder in the local field.
    _c = model.baseline.coeff.detach().numpy()
    _nc = len(geom.conductors)
    _names = [k.electrode for k in geom.conductors]
    _rw = cfg.electrode("w").shape.radius
    _f = [g.field for g in sol.gaps]
    for _i, _n in enumerate(_names):
        _mean = 0.5 * (_f[_i] + _f[_i + 1]) / 10.0          # V/cm -> V/mm
        value(f"valDipolePred{_n.upper()}", f"{_rw * _mean:.3f}")
        value(f"valDipoleFit{_n.upper()}", f"{_c[2 * _nc + _i]:.3f}")
        value(f"valMono{_n.upper()}", f"{_c[_i]:.2f}")
    value("valDipoleX", f"{np.abs(_c[_nc:2 * _nc]).max():.0e}")

    # A transverse dipole is zero here because of a symmetry, not because the
    # column is useless: in a weighting solve the sensing wire breaks the
    # mirror symmetry about its neighbours and the same column is needed.
    try:
        _wcfg, _wgeom, _wmodel, _ = train_mod.load(
            os.path.join(rundir, "weight-u.pt"))
    except FileNotFoundError:
        pass
    else:
        _wc = _wmodel.baseline.coeff.detach().numpy()
        _wn = len(_wgeom.conductors)
        _mono, _dx = _wc[:_wn], _wc[_wn:2 * _wn]
        _k = int(np.argmax(np.abs(_dx)))
        _cond = _wgeom.conductors[_k]
        value("valDipoleXWeight", f"{abs(_dx[_k]):.1e}")
        value("valDipoleXWeightFrac",
              f"{100 * abs(_dx[_k] / _mono[_k]):.0f}")
        value("valDipoleXWeightOffset", f"{abs(_cond.x):g}")
        _axis = [i for i, c_ in enumerate(_wgeom.conductors) if abs(c_.x) < 1e-9]
        value("valDipoleXWeightAxis", f"{np.abs(_dx[_axis]).max():.0e}")

    # What a plain network can and cannot reach (Sec. "A hybrid ansatz").
    reachability(cfg, geom, sol, model)
    fig_warmstart(cfg, geom, model, fig("warm-start.pdf"))

    diag = sample_mod.diagnostics(cfg, geom, model, 50000)
    value("valDriftResidual", f"{diag['residual_rms_scaled']:.1e}")
    value("valDriftResidualAbs", f"{diag['residual_rms']:.1e}")
    value("valDriftSurfaceRMS", f"{diag['surface_rms_V']:.3f}")
    value("valDriftSurfaceMax", f"{diag['surface_max_V']:.3f}")

    # ---- weighting solutions ---------------------------------------------
    wfigs = []
    for plane in ("u", "v", "w"):
        a, m = sample_mod.load_npz(os.path.join(rundir, f"weight-{plane}-zoom.npz"))
        p = fig(f"weight-{plane}.pdf")
        plot_mod.plot_solution(a, m, p, contours=30, equal=True,
                               figsize=(6.2, 3.4),
                               title=f"weighting potential, central {plane} wire")
        wfigs.append(p)
        wcfg, wgeom, wmodel, _ = train_mod.load(
            os.path.join(rundir, f"weight-{plane}-analytic.pt"))
        d = sample_mod.diagnostics(wcfg, wgeom, wmodel, 20000)
        value(f"valWeight{plane.upper()}Face", f"{d['face_ground_rms_V']:.4f}")
        value(f"valWeight{plane.upper()}Cond", f"{d['surface_rms_V']:.1e}")

    # ---- field response ---------------------------------------------------
    ra, rm = resp_mod.load_npz(os.path.join(rundir, "response.npz"))
    plot_mod.plot_response(ra, rm, fig("response.pdf"), style="curves",
                           n_wires=5, tmin=48.0)
    plot_mod.plot_response(ra, rm, fig("response-impacts.pdf"), style="impacts",
                           tmin=48.0)
    iw0 = int(np.argmin(np.abs(ra["offsets"])))
    ipw = rm["planes"].index("w")
    value("valRamoCollect", f"{ra['integrated'][ipw, iw0, 0]:.4f}")
    value("valRamoInduction",
          f"{max(abs(ra['integrated'][rm['planes'].index(p)]).max() for p in 'uv'):.1e}")
    value("valRamoAgreement", f"{rm['agreement']:.1e}")
    value("valNImpacts", f"{len(ra['impact'])}")
    value("valNWires", f"{len(ra['offsets'])}")
    value("valTick", f"{ra['time'][1] - ra['time'][0]:g}")
    value("valEMax", f"{rm['e_max_V_per_mm'] * 10 / 1e3:.1f}")

    weighting_bounds(rundir)
    ramo_trained(rundir)
    fig_response_table(ra, rm, figdir, fig)

    # ---- transport and electronics ---------------------------------------
    ic, ec = cfg.ionization, cfg.electronics
    speed = float(resp_mod.drift_speed(np.array([ic.drift_field * 0.1]),
                                       cfg.response.velocity,
                                       cfg.response.temperature, cfg.response.mobility)[0])
    props = ion_mod.ArgonProperties(drift_speed=speed, d_long=ic.d_long,
                                    d_tran=ic.d_tran, lifetime=ic.lifetime)
    elec = elec_mod.Electronics(gain=ec.gain, mode=ec.mode, filter=ec.filter,
                                order=ec.order, tau=ec.tau, decay=ec.decay,
                                adc_rate=ec.adc_rate, adc_bits=ec.adc_bits,
                                adc_range=ec.adc_range, baseline=ec.baseline)
    plot_mod.plot_electronics(elec, fig("electronics.pdf"))
    plot_mod.plot_drift_velocity(
        props, fig("drift-velocity.pdf"), velocity=cfg.response.velocity,
        temperature=cfg.response.temperature,
        marks=((ic.drift_field / 1000.0, "bulk drift"),
               (sol.drift_field_actual / 1000.0, "cathode gap")))

    value("valVDrift", f"{speed:.4f}")
    value("valDLong", f"{ic.d_long:g}")
    value("valDTran", f"{ic.d_tran:g}")
    value("valLifetime", f"{ic.lifetime / 1000:g}")
    value("valGain", f"{ec.gain:g}")
    value("valTau", f"{ec.tau:g}")
    value("valAdcRate", f"{ec.adc_rate:g}")
    value("valAdcBits", f"{ec.adc_bits:d}")
    value("valAdcRange", f"{ec.adc_range / 1000:g}")
    value("valLsbUv", f"{elec.lsb * 1e3:.0f}")
    value("valLsbE", f"{elec.lsb / elec.mv_per_electron:.1f}")
    value("valUbPeakX", f"{elec_mod.UBOONE_PEAK_X:.4f}")

    # ---- ionization and readout ------------------------------------------
    groups = ion_mod.line_source(1000.0, props, length=60.0, angle=45.0,
                                 per_mm=5000.0, step=0.1)
    fig_ionization(groups, props, fig("ionization.pdf"))
    s_l, s_t, s_time = props.spread(1000.0)
    value("valSigmaT", f"{s_t:.2f}")
    value("valSigmaL", f"{s_l:.2f}")
    value("valSigmaTime", f"{s_time:.2f}")
    value("valSurvival", f"{100 * props.survival(1000.0):.1f}")
    value("valDriftTime", f"{props.drift_time(1000.0):.0f}")
    value("valNGroups", f"{len(groups)}")
    value("valNElectrons", f"{groups.total:,.0f}")

    sa, sm = sig_mod.load_npz(os.path.join(rundir, "signal.npz"))
    plot_mod.plot_adc(sa, sm, fig("adc.pdf"), style="image", tmin=25.0, tmax=78.0)
    plot_mod.plot_adc(sa, sm, fig("adc-waveforms.pdf"), style="waveforms",
                      tmin=25.0, tmax=78.0)
    tick = sa["time"][1] - sa["time"][0]
    for ip, p in enumerate(sm["planes"]):
        value(f"valAdcPeak{p.upper()}", f"{int(np.abs(sa['adc'][ip]).max())}")
        value(f"valVPeak{p.upper()}", f"{np.abs(sa['volts'][ip]).max():.1f}")
        value(f"valNet{p.upper()}", f"{sa['current'][ip].sum() * tick:,.0f}")

    # ---- spectrum of the response ----------------------------------------
    plot_mod.plot_response_spectrum(ra, rm, fig("response-spectrum.pdf"),
                                    elec=elec, sigma_t=s_time)
    n_t = ra["current"].shape[-1]
    frq = np.fft.rfftfreq(n_t, d=float(ra["time"][1] - ra["time"][0]))
    iw = int(np.argmin(np.abs(ra["offsets"])))
    for ip, p in enumerate(rm["planes"]):
        mg = np.abs(np.fft.rfft(ra["current"][ip, iw, 0, :]))
        value(f"valSpecNyq{p.upper()}", f"{mg[-1] / mg.max():.1e}")
    value("valNyquist", f"{0.5 / float(ra['time'][1] - ra['time'][0]):g}")

    # ---- trajectories -----------------------------------------------------
    _, w21 = fig_periodicity(rundir, fig("periodicity.pdf"))
    value("valWeightArrayErr", f"{w21:.1e}")

    _, n_steps, med_step = fig_trajectory(rundir, fig("trajectory.pdf"))
    value("valTrajSteps", f"{n_steps}")
    value("valTrajMedianStep", f"{med_step * 1000:.0f}")

    # ---- generalisations --------------------------------------------------
    fig_geometries(figdir)
    fig_dune_vd_detail(fig("dune-vd-detail.pdf"))
    _, sur0, sur1, sur_held, sur_sweep = fig_surrogate(ra, rm, fig("surrogate.pdf"))
    value("valSurrogateSigmaStep", f"{sur_sweep[-1][0]:.2f}")
    value("valSurrogateHeldCoarse", f"{sur_sweep[0][1]:.1e}")
    value("valSurrogateRms", f"{sur0.rel_rms:.1e}")
    value("valSurrogateCharge", f"{sur0.charge_error * float(ra['time'][1] - ra['time'][0]):.1e}")
    value("valSurrogateCondRms", f"{sur1.rel_rms:.1e}")
    value("valSurrogateHeldRms", f"{sur_held:.1e}")

    # ---- conceptual diagrams ---------------------------------------------
    fig_flow(fig("flow.pdf"))
    fig_siren(fig("siren.pdf"))

    # ---- training history -------------------------------------------------
    import torch

    blob = torch.load(os.path.join(rundir, "drift.pt"), map_location="cpu",
                      weights_only=False)
    if blob.get("history", {}).get("step") is not None:
        plot_mod.plot_history(blob["history"], fig("losses.pdf"))

    # ---- values.tex -------------------------------------------------------
    path = os.path.join(figdir, "values.tex")
    with open(path, "w") as fp:
        fp.write("% Generated by scripts/make_figs.py -- do not edit.\n")
        fp.write("% Every number the note quotes is defined here, so that\n")
        fp.write("% re-running the pipeline updates the prose as well as the plots.\n")
        for k in sorted(VALUES):
            fp.write(f"\\newcommand{{\\{k}}}{{\\ensuremath{{{VALUES[k]}}}}}\n")
    _perf["macros"] = len(VALUES)
    _perf["figures"] = sum(1 for n in os.listdir(figdir) if n.endswith(".pdf"))
    print(f"wrote {path} with {len(VALUES)} macros")
    # keep the note buildable even if no timing log exists yet
    stamp = os.path.join(figdir, "perf.tex")
    if not os.path.exists(stamp):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import perf_table
        perf_table.stub(stamp, "not measured on this run")
    for name in sorted(os.listdir(figdir)):
        if name.endswith(".pdf"):
            print(f"  {os.path.join(figdir, name)}")


if __name__ == "__main__":
    main(*(sys.argv[1:3] or ["runs", "docs/figs"]))
