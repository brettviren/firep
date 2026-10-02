#!/usr/bin/env python
"""Diagram of the spectral single layer of Sec. 17.3 (docs/figs/pixel-basis.pdf).

Every panel is computed with firep.pixel on the real geometry: a 5x5
weighting domain of 3.8 mm rounded-square pads at 4.4 mm pitch.

    python scripts/pixel_basis_fig.py [figdir]
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8})
import numpy as np
import torch
from matplotlib.patches import Polygon, Rectangle

from firep import pixel as X

PITCH, HALF, RAD = 4.4, 1.9, 0.4
PLANES, TOP = (10.0, 10.1), 149.9
NCELL, GRID, MODES = 5, 2201, 220
TERM = (1, 0, 0.2)  # the example column: an odd term in a 0.2 mm edge band


def outline(cx, cz, n=200):
    """The rounded-square outline as a closed polygon."""
    t = np.linspace(0, 2 * np.pi, 4 * n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    a = HALF - RAD
    x = np.sign(c) * a + RAD * c
    z = np.sign(s) * a + RAD * s
    return np.column_stack((cx + x, cz + z))


def main(figdir):
    shape = X.PadShape("rounded_square", HALF, RAD)
    arr = X.PadArray(pitch=PITCH, ncell=NCELL, shape=shape, y_pad=PLANES, top=TOP)
    ssl = X.SpectralSingleLayer(arr, "weighting", MODES, GRID, [TERM])
    centres = arr.pad_centres()
    classes = arr.classes()
    L = arr.width
    colours = plt.cm.tab10(np.arange(len(classes)))

    fig, ax = plt.subplots(2, 3, figsize=(11.5, 8.2), constrained_layout=True)

    # (a) plan view: domain, pads, classes, representative pads, coordinates
    a = ax[0, 0]
    for ic, cls in enumerate(classes):
        for j, i in enumerate(cls):
            a.add_patch(Polygon(outline(*centres[i]), closed=True, fc=colours[ic],
                                alpha=0.85 if j == 0 else 0.3, ec="k", lw=0.6))
        cx, cz = centres[cls[0]]
        a.text(cx - 1.55, cz + 1.5, f"$c_{{{ic}}}$", ha="left", va="top", fontsize=9)
    a.add_patch(Polygon([[0, 0], [L / 2, 0], [L / 2, L / 2]], closed=True, fc="none",
                        ec="k", ls="--", lw=1.0, hatch="///", alpha=0.5))
    a.add_patch(Rectangle((-L / 2, -L / 2), L, L, fc="none", ec="k", lw=2))
    a.annotate("", (L / 2 + 0.2, -L / 2), (L / 2 + 0.2, L / 2),
               arrowprops=dict(arrowstyle="<->"))
    a.text(L / 2 + 0.45, 0, f"$L = {NCELL}\\times{PITCH}$ mm", rotation=90, va="center")
    a.annotate("", (0, -L / 2 - 0.35), (PITCH, -L / 2 - 0.35), arrowprops=dict(arrowstyle="<->"))
    a.text(PITCH / 2, -L / 2 - 0.8, "pitch", ha="center")
    # local pad coordinates on one representative pad
    cx, cz = centres[classes[1][0]]
    for dx, dz, lab in ((1.1, 0, r"$\xi$"), (0, 1.1, r"$\zeta$")):
        a.annotate("", (cx + dx, cz + dz), (cx, cz), arrowprops=dict(arrowstyle="->", lw=1.2, color="k"))
        a.text(cx + dx + 0.12 * (dx > 0), cz + dz + 0.1 * (dz > 0) - 0.25 * (dx > 0), lab,
               color="k", fontsize=9)
    # global axes at the origin
    a.annotate("", (-L / 2 + 2.2, -L / 2 - 1.4), (-L / 2, -L / 2 - 1.4), arrowprops=dict(arrowstyle="->"))
    a.annotate("", (-L / 2, -L / 2 + 0.8), (-L / 2, -L / 2 - 1.4), arrowprops=dict(arrowstyle="->"))
    a.text(-L / 2 + 2.3, -L / 2 - 1.5, "$x$", va="center")
    a.text(-L / 2 - 0.1, -L / 2 + 0.9, "$z$", ha="right")
    a.text(-L / 2, L / 2 + 0.25, "zero-slope side walls", ha="left", fontsize=7.5)
    a.annotate("fit wedge $0 \\leq z \\leq x$", (7.5, 3.2), (4.0, 12.3), fontsize=8,
               ha="center", arrowprops=dict(arrowstyle="->", lw=0.6))
    a.set_xlim(-L / 2 - 1.2, L / 2 + 1.2)
    a.set_ylim(-L / 2 - 2.0, L / 2 + 2.0)
    a.set_aspect("equal")
    a.axis("off")
    a.set_title("(a) plan view: classes $c$ (orbits of $D_4$)\n"
                "solid = representative pad, pale = its symmetry images", fontsize=9)

    # (b) side view: faces and sheets, both kinds of solve
    b = ax[0, 1]
    ybreak = 22.0
    b.axhline(0, color="k", lw=2)
    b.axhline(ybreak + 3, color="k", lw=2)
    for i in range(-2, 3):
        b.add_patch(Rectangle((i * PITCH - HALF, PLANES[0]), 2 * HALF, 0.5, fc="0.3", ec="k"))
    b.text(0, PLANES[0] + 1.0, "pad: two sheets, $y_p = 10.0,\\ 10.1$ mm\n(thickness drawn 5$\\times$)",
           ha="center", fontsize=7.5)
    b.text(-L / 2 + 0.2, 0.6, "$y = 0$:  drift, zero slope;   weighting, $V = 0$", fontsize=7.5)
    b.text(-L / 2 + 0.2, ybreak + 3.6,
           "$y = T = 149.9$ mm:  drift, $V = -7000$ V;   weighting, zero slope", fontsize=7.5)
    b.text(L / 2 - 0.2, ybreak - 1.5, "($y$ not to scale)", ha="right", fontsize=8)
    for s in (-1, 1):
        b.axvline(s * L / 2, color="k", lw=1, ls="--")
    b.text(-L / 2 + 0.2, 5, "side walls:\nzero slope", fontsize=7.5)
    b.annotate("", (L / 2 - 1, 16), (L / 2 - 1, 11.5), arrowprops=dict(arrowstyle="->"))
    b.text(L / 2 - 1.2, 16.3, "$y$", ha="right")
    b.annotate("", (L / 2 - 1 + 2.3, 11.5), (L / 2 - 1, 11.5), arrowprops=dict(arrowstyle="->"))
    b.text(L / 2 + 1.4, 12.0, "$x$")
    b.set_xlim(-L / 2 - 0.5, L / 2 + 2)
    b.set_ylim(-1, ybreak + 6)
    b.set_xlabel("$x$ (mm)")
    b.set_yticks([])
    b.set_title("(b) side view: the slab and its faces\n(every column meets the faces and sides exactly)",
                fontsize=9)

    # (c) the edge factor and the local profiles across one pad
    c = ax[0, 2]
    xi = np.linspace(-HALF - 0.4, HALF + 0.4, 4001)
    d = -shape.sdf(xi, np.zeros_like(xi))
    inside = d > 0
    dd = np.where(inside, d, np.nan)
    edge = 1 / np.sqrt(dd)
    c.plot(xi, edge, "k", lw=2, label=r"edge factor $1/\sqrt{d}$  ($P = 1$, no band)")
    for s, col in ((1.0, "C0"), (0.2, "C1"), (0.05, "C2")):
        c.plot(xi, edge * np.exp(-dd / s), color=col, label=f"band $s = {s}$ mm")
    c.plot(xi, edge * (xi / HALF), "C3", ls="--", label=r"$P = \xi/a$ (odd), no band")
    c.axvspan(-HALF - 0.4, -HALF, color="0.9")
    c.axvspan(HALF, HALF + 0.4, color="0.9")
    c.annotate("", (HALF, 7.3), (HALF - 1.0, 7.3), arrowprops=dict(arrowstyle="<->"))
    c.text(HALF - 0.5, 7.6, "$d$", ha="center")
    c.text(HALF + 0.2, 1, "gap", ha="center", rotation=90)
    c.set_ylim(-3, 9)
    c.set_xlabel(r"$\xi$ (mm), across the pad at $\zeta = 0$")
    c.set_ylabel(r"$\sigma$ (arb.)")
    c.legend(fontsize=7, loc="lower center")
    c.set_title("(c) local densities: $P_j(\\xi,\\zeta)\\,e^{-d/s_j}/\\sqrt{d}$\n"
                "$d$ = distance in from the outline", fontsize=9)

    # (d) one column: one term on one class, symmetrised
    ic = 1
    cls = classes[ic]
    dens = ssl._density(cls, TERM)
    xs = (np.arange(ssl.n) - ssl.n // 2) * ssl.h
    step = 4
    e = ax[1, 0]
    v = np.nanmax(np.abs(dens)) * 0.25
    im = e.imshow(dens[::step, ::step].T, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v,
                  extent=[xs[0], xs[-1], xs[0], xs[-1]])
    for i in range(len(centres)):
        e.add_patch(Polygon(outline(*centres[i]), closed=True, fc="none", ec="0.6", lw=0.5))
    e.set_aspect("equal")
    e.set_xlabel("$x$ (mm)")
    e.set_ylabel("$z$ (mm)")
    fig.colorbar(im, ax=e, shrink=0.8, label=r"$\sigma$ (clipped)")
    e.set_title(f"(d) one column's density: term $(p,q,s) = {TERM}$\n"
                f"on class $c_{{{ic}}}$, symmetrised over $D_4$", fontsize=9)

    # (e) its cosine coefficients: the modes
    f = ax[1, 1]
    j = [t for t, (cl, term) in enumerate(ssl._terms) if cl == cls][0]
    A = ssl.coef[j].cpu().numpy()  # (M+1, M+1), sheet 0
    full = np.fft.fft2(np.fft.ifftshift(dens)).real * ssl.h ** 2 / L ** 2
    show = 320
    F = np.abs(full[:show, :show]) * 4
    im = f.imshow(np.log10(F + 1e-12).T, origin="lower", cmap="viridis",
                  vmin=np.log10(F.max()) - 5, vmax=np.log10(F.max()), extent=[0, show, 0, show])
    f.add_patch(Rectangle((0, 0), MODES + 1, MODES + 1, fc="none", ec="w", lw=1.5))
    f.text(MODES - 4, MODES - 14, f"kept: $0 \\leq m,n \\leq M = {MODES}$\n$(M+1)^2$ modes",
           color="w", ha="right", va="top", fontsize=7.5)
    f.text(show - 4, show - 8, f"fine grid: {ssl.n}$^2$\n(on-sheet values)", color="w",
           ha="right", va="top", fontsize=7.5)
    f.set_xlabel("$m$")
    f.set_ylabel("$n$")
    fig.colorbar(im, ax=f, shrink=0.8, label=r"$\log_{10}|a_{mn}|$")
    f.set_title("(e) its cosine coefficients $a_{mn}$, one per mode\n"
                r"mode $(m,n)$: $\cos(mkx)\cos(nkz)$, $\kappa = k\sqrt{m^2+n^2}$", fontsize=9)
    assert np.allclose(A[:5, :5], (full[:5, :5] * np.outer(*(2 * [np.r_[1.0, [2.0] * 4]]))), rtol=1e-6)

    # (f) how each mode leaves the sheet: the slab Green's function
    g = ax[1, 2]
    k = 2 * np.pi / L
    y = torch.linspace(0, 40, 4001, dtype=torch.float64)
    for (m, n), col in (((0, 0), "k"), ((1, 0), "C0"), ((3, 2), "C1"), ((10, 0), "C2"), ((40, 0), "C3")):
        kap = torch.tensor([k * np.hypot(m, n)], dtype=torch.float64)
        gv = X.slab_green(kap, y, PLANES[1], TOP, "weighting")[:, 0].numpy()
        lab = f"$(m,n)=({m},{n})$, $1/\\kappa$ = " + ("$\\infty$" if m == n == 0 else f"{1 / kap.item():.2f} mm")
        g.plot(y.numpy(), gv / gv.max(), color=col, label=lab)
    g.axvline(PLANES[1], color="0.5", ls=":")
    g.text(PLANES[1] + 0.3, 0.5, "sheet $y_p$", color="0.4", fontsize=7.5)
    g.set_xlabel("$y$ (mm)")
    g.set_ylabel(r"$g_\kappa(y, y_p)$ / max")
    g.set_xlim(0, 40)
    g.legend(fontsize=7, loc="upper right")
    g.set_title("(f) height profiles $g_\\kappa(y, y_p)$, weighting kind:\n"
                "$V = 0$ at $y = 0$, falling as $e^{-\\kappa|y-y_p|}$", fontsize=9)

    fig.savefig(f"{figdir}/pixel-basis.pdf")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/figs")
