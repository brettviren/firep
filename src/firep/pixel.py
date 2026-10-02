"""Pixel-pad anodes in three dimensions: the analytic part of the potential.

A pixel anode is a plane of thin conductors.  Each pad carries a surface charge
whose potential is a *single layer*: exactly harmonic everywhere except on the
pad itself.  This module builds that single layer spectrally.

The cell and its symmetry
-------------------------
The transverse domain is a square of side ``L`` centred on a pad: one pitch for
a drift solve, ``N`` pitches for an ``N x N`` weighting solve.  Every quantity is
invariant under the dihedral group of the square about the cell centre (90
degree rotations and the mirrors about the axes and diagonals), and its normal
derivative vanishes on the cell sides.  Both follow at once from writing the
transverse dependence as a double cosine series of period ``L``,

.. math::

    f(x, z) = \\sum_{m, n \\ge 0} a_{mn} \\cos(m k x) \\cos(n k z),
    \\qquad k = 2\\pi/L, \\quad a_{mn} = a_{nm}.

For a drift solve the sides are periodic *and* mirror planes, so the cosine
series is exact.  For a weighting solve the sides are the zero-slope (Neumann)
walls of the pochoir finite-difference reference, which is again exactly what a
cosine series of period ``L`` satisfies.

The slab
--------
Each transverse mode ``(m, n)`` with ``kappa = k sqrt(m^2 + n^2)`` gets the
Green's function of the slab for the face types at hand, so every column meets
the face conditions exactly: a sheet of density ``cos(mkx) cos(nkz)`` at
``y = y_p`` contributes ``g_kappa(y, y_p) cos(mkx) cos(nkz)`` with

* drift (zero-slope bottom at ``y = 0``, fixed top at ``y = T``):
  ``g = 2 cosh(kappa y<) sinh(kappa (T - y>)) / (kappa cosh(kappa T))``,
  and ``g_0 = 2 (T - y>)``;
* weighting (fixed bottom, zero-slope top):
  ``g = 2 sinh(kappa y<) cosh(kappa (T - y>)) / (kappa cosh(kappa T))``,
  and ``g_0 = 2 y<``.

The pad densities
-----------------
A thin conductor's surface density goes as ``1/sqrt(d)`` at its edge, with
``d`` the distance from the edge, whatever the outline.  The densities are
``d^{-1/2}`` times low-order polynomials, supported on the pad, symmetrised over
the dihedral group, and turned into cosine coefficients by a fine-grid FFT.  A
rounded outline has no corners, so the edge factor is the whole singular
structure.  Only the outline changes between a rounded square, a disk and an
annulus.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

# --------------------------------------------------------------------------
# pad outlines (signed distance in the pad plane, negative inside)
# --------------------------------------------------------------------------


def rounded_square_sdf(x: np.ndarray, z: np.ndarray, half: float, radius: float) -> np.ndarray:
    """Signed distance to a square of half-side ``half`` with corners rounded
    at ``radius``, centred on the origin."""
    qx = np.abs(x) - (half - radius)
    qz = np.abs(z) - (half - radius)
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qz, 0.0))
    inside = np.minimum(np.maximum(qx, qz), 0.0)
    return outside + inside - radius


def disk_sdf(x, z, radius):
    return np.hypot(x, z) - radius


def annulus_sdf(x, z, r_in, r_out):
    r = np.hypot(x, z)
    return np.maximum(r - r_out, r_in - r)


@dataclass
class PadShape:
    kind: str = "rounded_square"  # rounded_square | disk | annulus | hemisphere | quad | quint
    half: float = 1.9  # mm: half side (square), radius (disk, hemisphere, each quad dot), outer radius (annulus)
    radius: float = 0.4  # mm: corner radius (square), inner radius (annulus), dot offset (quad, quint)

    def sdf(self, x, z):
        """Signed distance in the pad plane to the outline (a hemisphere's is
        its footprint)."""
        if self.kind == "rounded_square":
            return rounded_square_sdf(x, z, self.half, self.radius)
        if self.kind in ("disk", "hemisphere"):
            return disk_sdf(x, z, self.half)
        if self.kind == "annulus":
            return annulus_sdf(x, z, self.radius, self.half)
        if self.kind == "hole":  # a grid tile: conductor outside a hole of this radius (distance to the hole's rim)
            return self.radius - np.hypot(x, z)
        if self.kind in ("quad", "quint"):  # four dots at (+-radius, +-radius), ganged; quint adds one at the centre
            o = self.radius
            d = [disk_sdf(x - sx * o, z - sz * o, self.half) for sx in (-1, 1) for sz in (-1, 1)]
            if self.kind == "quint":
                d.append(disk_sdf(x, z, self.half))
            return np.minimum.reduce(d)
        raise ValueError(f"pad kind {self.kind!r} unknown")

    @property
    def extent(self) -> float:
        return self.half + self.radius if self.kind in ("quad", "quint") else self.half

    def inside(self, x, y, z, y_lo: float, y_hi: float):
        """Whether points (local to the pad centre) are in the conductor: a
        flat pad fills ``y_lo <= y <= y_hi`` over its outline; a hemisphere is
        the dome of radius ``half`` standing on ``y_lo``.  Works on torch
        tensors."""
        if self.kind == "hemisphere":
            return (x * x + z * z + (y - y_lo) ** 2 <= self.half ** 2) & (y >= y_lo)
        if self.kind in ("quad", "quint"):
            o = self.radius
            ok = (x * x + z * z <= self.half ** 2) if self.kind == "quint" else torch.zeros_like(x, dtype=torch.bool)
            for sx in (-1, 1):
                for sz in (-1, 1):
                    ok = ok | ((x - sx * o) ** 2 + (z - sz * o) ** 2 <= self.half ** 2)
            return ok & (y >= y_lo) & (y <= y_hi)
        if self.kind in ("disk", "annulus"):
            r = torch.sqrt(x * x + z * z)
            ok = r <= self.half if self.kind == "disk" else (r <= self.half) & (r >= self.radius)
        else:
            a = self.half - self.radius
            q = torch.stack((x.abs() - a, z.abs() - a), -1)
            sd = torch.linalg.vector_norm(q.clamp_min(0), dim=-1) + q.max(-1).values.clamp_max(0) - self.radius
            ok = sd <= 0
        return ok & (y >= y_lo) & (y <= y_hi)


@dataclass
class Sheet:
    """A source sheet at height ``y``: densities supported on ``shape``'s
    outline, with the ``1/sqrt(d)`` edge profile of a thin conductor ("edge")
    or a smooth taper for a source inside a thick one ("smooth")."""

    y: float
    shape: PadShape
    profile: str = "edge"
    tile: bool = False  # confine each density to the pad's cell, |xi|, |zeta| < pitch/2 (a grid electrode)


def dome_sheets(arr, n: int = 6, depth: float = 0.75, shrink: float = 0.8) -> list:
    """Sources for hemispherical pads: ``n`` smooth disk sheets inside each
    dome, from its base up to ``depth`` of its radius, each of ``shrink``
    times the dome's radius at that height.  On a zero-slope PCB surface the
    dome and its image are a whole sphere, so interior sources represent the
    exterior potential with no edge to resolve."""
    R, y0 = arr.shape.half, arr.y_floor
    out = []
    for u in np.linspace(0.0, depth, n):
        rho = shrink * R * math.sqrt(max(1.0 - u * u, 0.0))
        out.append(Sheet(y0 + u * R, PadShape("disk", rho, 0.0), "smooth"))
    return out


def dome_nodes(arr, count, rng: np.random.Generator, wedge: bool = True, rim_weight: float = 2.0):
    """Fit points on the dome surfaces of hemispherical pads, uniform in solid
    angle with extra weight near the rim.  ``count`` is a number per pad or a
    callable of the ring.  Returns ``(xyz, owner)`` in the wedge
    ``0 <= z <= x`` (when ``wedge``)."""
    R, y0 = arr.shape.half, arr.y_floor
    centres = arr.pad_centres()
    cc = centres / arr.pitch
    pts, own = [], []
    for i, (cx, cz) in enumerate(centres):
        ring = int(round(np.abs(cc[i]).max()))
        m = count(ring) if callable(count) else count
        if m <= 0:
            continue
        m = int(m) * (8 if wedge else 1)  # most fall outside the wedge
        # polar angle from the top: cos(theta) uniform is uniform in area;
        # mix in a share concentrated near the rim (theta -> 90 degrees)
        c = rng.uniform(size=m)
        c = np.where(rng.uniform(size=m) < 1.0 / (1.0 + rim_weight), c, c ** 3)
        phi = rng.uniform(0, 2 * np.pi, size=m)
        st = np.sqrt(1 - c * c)
        x, y, z = cx + R * st * np.cos(phi), y0 + R * c, cz + R * st * np.sin(phi)
        ok = (z >= 0) & (z <= x + 1e-12) if wedge else np.ones(m, bool)
        pts.append(np.column_stack((x[ok], y[ok], z[ok])))
        own.append(np.full(ok.sum(), i))
    return np.concatenate(pts), np.concatenate(own)


def fit_rows(ssl: "SpectralSingleLayer", target, offset: float = 0.0, train=None):
    """Least squares on the exact on-plane design built from per-plane nodes
    (``nodes`` a dict): one target per row, rows running plane by plane."""
    return fit_design(ssl.plane_design, target, offset, train)


def grid_nodes(arr, r_hole: float, grid: int, count: int, rng: np.random.Generator,
               edge_scale: float = 0.1, wedge: bool = True):
    """Fine-grid nodes on a grid electrode (outside round holes of radius
    ``r_hole`` centred over the pads), denser near the holes' rims."""
    n = int(grid) | 1
    h = arr.width / n
    xs = (np.arange(n) - n // 2) * h
    X, Z = np.meshgrid(xs, xs, indexing="ij")
    rho = np.full((n, n), np.inf)
    for cx, cz in arr.pad_centres():
        rho = np.minimum(rho, np.hypot(X - cx, Z - cz))
    ok = rho > r_hole
    if wedge:
        ok &= (Z >= 0) & (Z <= X + 1e-12)
    ii, jj = np.nonzero(ok)
    depth = rho[ii, jj] - r_hole
    w = 0.02 + np.exp(-depth / edge_scale)
    keep = rng.uniform(size=len(ii)) < np.minimum(1.0, w * count / w.sum())
    return ii[keep], jj[keep]


def fit_design(D: torch.Tensor, target, offset: float = 0.0, train=None):
    """Least squares on a design ``D`` (nodes, columns) evaluated at fit
    points: ``D @ coef + offset = target``.  Returns ``(coef, rms, max)`` over
    the points not in ``train`` (all points when ``train`` is None)."""
    b = torch.as_tensor(np.asarray(target, float) - offset, device=D.device, dtype=D.dtype)
    sel = torch.ones(len(b), dtype=torch.bool, device=D.device) if train is None else \
        torch.as_tensor(train, device=D.device)
    A = D[sel]
    scale = A.abs().max(0).values.clamp_min(1e-30)
    sol = torch.linalg.lstsq((A / scale).cpu(), b[sel].cpu()[:, None], driver="gelsd").solution[:, 0]
    coef = sol.to(D.device) / scale
    test = ~sel if train is not None else sel
    r = D[test] @ coef - b[test]
    return coef, float(r.pow(2).mean().sqrt()), float(r.abs().max())


# --------------------------------------------------------------------------
# the dihedral group of the square, on a centred grid
# --------------------------------------------------------------------------


def d4_symmetrise(f: np.ndarray) -> np.ndarray:
    """Average a centred square grid over the 8 elements of D4."""
    acc = np.zeros_like(f)
    for g in (f, f.T):
        for r in range(4):
            acc += np.rot90(g, r)
    return acc / 8.0


# --------------------------------------------------------------------------
# slab Green's functions, in overflow-safe form
# --------------------------------------------------------------------------


def slab_green(kappa: torch.Tensor, y: torch.Tensor, yp: float, top: float,
               kind: str, pcb: float | None = None, y0: float | None = None) -> torch.Tensor:
    """``g_kappa(y, y_p)`` for every point (rows) and mode (columns).

    ``kind`` is "drift" (zero-slope bottom at 0, fixed top at ``top``) or
    "weighting" (fixed bottom at 0, zero-slope top at ``top``).

    With ``pcb`` the slab below ``y0`` (the pads' bottom face) is a dielectric
    of permittivity ``pcb`` relative to the medium above; ``V`` and
    ``eps dV/dy`` are continuous at ``y0``, and every sheet must lie at or
    above it.  ``pcb = 1`` is free space; ``pcb = 0`` makes the surface
    ``y = y0`` zero-slope wherever it carries no sheet charge, the steady
    state of an insulator charged up by the electrons that land on it.
    """
    if pcb is None:
        return _slab_green_free(kappa, y, yp, top, kind)
    if yp < y0 - 1e-12:
        raise ValueError("with a PCB every sheet must lie on or above its surface")
    y = y[:, None]
    below = y < y0
    ye = torch.where(below, torch.full_like(y, y0), y)
    ylo = torch.minimum(ye, torch.full_like(ye, yp))
    yhi = torch.maximum(ye, torch.full_like(ye, yp))
    k = kappa[None, :]
    safe = torch.where(k > 0, k, torch.ones_like(k))
    # u_low: the solution meeting the bottom face and the interface, as
    # cosh(k s) + rho sinh(k s) above it (s = y - y0); u_up meets the top.
    t = torch.tanh(safe * y0) if kind == "drift" else 1.0 / torch.tanh(safe * y0)
    rho = pcb * t
    s_ = ylo - y0
    e = torch.exp(-safe * (yhi - ylo))
    low = (1 + rho) + (1 - rho) * torch.exp(-2 * safe * s_)
    b = torch.exp(-2.0 * safe * (top - yhi))
    c = torch.exp(-2.0 * safe * (top - y0))
    if kind == "drift":
        g = e * low * (1 - b) / ((1 + rho) + (1 - rho) * c) / safe
        g0 = 2.0 * (top - yhi)
        # inside the dielectric: cosh(k y) / cosh(k y0) times the value at y0
        ratio = torch.exp(-safe * (y0 - y)) * (1 + torch.exp(-2 * safe * y)) / (1 + torch.exp(-2 * safe * y0))
        ratio0 = torch.ones_like(y)
    elif kind == "weighting":
        g = e * low * (1 + b) / ((1 + rho) - (1 - rho) * c) / safe
        if pcb <= 0:
            raise ValueError("a weighting solve needs the PCB's true permittivity, not 0")
        g0 = 2.0 * (y0 / pcb + (ylo - y0))
        ratio = torch.exp(-safe * (y0 - y)) * (1 - torch.exp(-2 * safe * y)) / (1 - torch.exp(-2 * safe * y0))
        ratio0 = y / y0
    else:
        raise ValueError(f"slab kind {kind!r} unknown")
    out = torch.where(k > 0, g, g0.expand_as(g))
    fac = torch.where(k > 0, ratio.expand_as(g), ratio0.expand_as(g))
    return torch.where(below.expand_as(g), out * fac, out)


def _slab_green_free(kappa, y, yp, top, kind):
    y = y[:, None]
    ylo = torch.minimum(y, torch.full_like(y, yp))
    yhi = torch.maximum(y, torch.full_like(y, yp))
    k = kappa[None, :]
    safe = torch.where(k > 0, k, torch.ones_like(k))
    e = torch.exp(-safe * (yhi - ylo))
    a = torch.exp(-2.0 * safe * ylo)
    b = torch.exp(-2.0 * safe * (top - yhi))
    c = torch.exp(-2.0 * safe * top)
    if kind == "drift":
        g = e * (1.0 + a) * (1.0 - b) / (1.0 + c) / safe
        g0 = 2.0 * (top - yhi)
    elif kind == "weighting":
        g = e * (1.0 - a) * (1.0 + b) / (1.0 + c) / safe
        g0 = 2.0 * ylo
    else:
        raise ValueError(f"slab kind {kind!r} unknown")
    return torch.where(k > 0, g, g0.expand_as(g))


# --------------------------------------------------------------------------
# the spectral single layer
# --------------------------------------------------------------------------


@dataclass
class PadArray:
    """A square array of ``ncell x ncell`` pads centred on the origin."""

    pitch: float = 4.4
    ncell: int = 1  # 1: a drift cell; N: an N x N weighting domain
    shape: PadShape = field(default_factory=PadShape)
    y_pad: float | tuple = 10.05  # mm: the pad plane, or (bottom, top) faces of a thick pad
    top: float = 149.9  # mm, the far face
    pcb: float | None = None  # permittivity below the pads, relative to above; None: free space

    @property
    def y_floor(self) -> float:
        """The pads' bottom face: the PCB surface when there is one."""
        return min(self.y_pad) if isinstance(self.y_pad, (tuple, list)) else float(self.y_pad)

    def green(self, kappa, y, yp, kind):
        return slab_green(kappa, y, yp, self.top, kind, self.pcb, self.y_floor)

    @property
    def width(self) -> float:
        return self.pitch * self.ncell

    def pad_centres(self) -> np.ndarray:
        h = (self.ncell - 1) // 2
        i = np.arange(-h, h + 1) * self.pitch
        xx, zz = np.meshgrid(i, i, indexing="ij")
        return np.column_stack((xx.ravel(), zz.ravel()))

    def classes(self) -> list[list[int]]:
        """Pads grouped into orbits of D4 about the centre pad."""
        c = self.pad_centres() / self.pitch
        key = [tuple(sorted((abs(round(a)), abs(round(b))))) for a, b in c]
        out: dict = {}
        for i, k in enumerate(key):
            out.setdefault(k, []).append(i)
        return [out[k] for k in sorted(out)]


class SpectralSingleLayer:
    """Columns of sheet potentials on the pad plane, for a least-squares fit.

    ``densities`` are functions of local pad coordinates ``(xi, zeta)`` (mm)
    returning the density on a fine grid; each column is the D4-symmetrised sum
    of one density over one class of pads.
    """

    def __init__(self, arr: PadArray, kind: str, modes: int, grid: int,
                 local_basis: list[tuple], edge_floor: float | None = None,
                 device="cpu", dtype=torch.float64, nodes=None, sheets=None):
        """``nodes``: optional ``(i, j)`` fine-grid indices at which to tabulate
        every column on every sheet plane *exactly*, by an inverse FFT over the
        whole fine grid.  On a sheet the truncated series converges only by
        cancellation (its terms fall as kappa^-3/2), so the pad-surface fit must
        use these values, not ``evaluate``.

        ``sheets``: the source sheets, a list of :class:`Sheet`.  By default a
        flat pad is one sheet per face in ``arr.y_pad`` with the pad outline
        and the ``1/sqrt(d)`` edge profile, and a hemisphere is a stack of
        smooth sheets inside the dome (:func:`dome_sheets`)."""
        self.arr, self.kind, self.modes = arr, kind, int(modes)
        self.device, self.dtype = torch.device(device), dtype
        self._nodes = nodes
        L = arr.width
        self.k = 2.0 * math.pi / L
        n = int(grid) | 1  # odd, so that the D4 pivot is the grid origin
        h = L / n
        self.h, self.n = h, n
        if sheets is None:
            if arr.shape.kind == "hemisphere":
                sheets = dome_sheets(arr)
            else:
                planes0 = arr.y_pad if isinstance(arr.y_pad, (tuple, list)) else (arr.y_pad,)
                sheets = [Sheet(float(y), arr.shape, "edge") for y in planes0]
        self.sheets = list(sheets)
        self.planes = [sh.y for sh in self.sheets]
        if nodes is not None:
            f = torch.fft.fftfreq(n, d=1.0 / n, device=self.device).to(dtype)
            kfull = (2.0 * math.pi / L) * torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2).reshape(-1)
            self._gfull = {(yq, yp): arr.green(kfull, torch.tensor([float(yq)], device=self.device, dtype=dtype),
                                               float(yp), kind)[0].reshape(n, n)
                           for yq in self.planes for yp in self.planes}
            # nodes: one (i, j) set evaluated on every sheet plane, or a dict
            # {plane index: (i, j)} with a set per plane (a grid and pads on
            # different planes); the design's rows then run plane by plane
            per = nodes if isinstance(nodes, dict) else {q: nodes for q in range(len(self.planes))}
            self._rows = {q: (torch.as_tensor(per[q][0], device=self.device),
                              torch.as_tensor(per[q][1], device=self.device)) for q in sorted(per)}
            self._flat_rows = isinstance(nodes, dict)
            if not self._flat_rows:
                self._ni, self._nj = self._rows[0]
        floor = edge_floor if edge_floor is not None else 0.5 * h
        centres = arr.pad_centres()
        # Each pad's density is built on a window of the fine grid around it
        # (origin at index n//2), then the sum is symmetrised over D4.
        self._geoms = []
        for sh in self.sheets:
            a = sh.shape.extent
            half_w = int(math.ceil(a / h)) + 2
            loc = np.arange(-half_w, half_w + 1) * h
            XI, ZE = np.meshgrid(loc, loc, indexing="ij")
            d = sh.shape.sdf(XI, ZE)
            if sh.profile == "edge":
                base = np.where(d < 0, 1.0 / np.sqrt(np.maximum(-d, floor)), 0.0)
            elif sh.profile == "smooth":  # a source inside a conductor: no edge, a smooth taper
                t = np.clip(1.0 - (XI ** 2 + ZE ** 2) / a ** 2, 0.0, None)
                base = t * t
            else:
                raise ValueError(f"sheet profile {sh.profile!r} unknown")
            if sh.tile:
                base = base * ((np.abs(XI) < 0.5 * arr.pitch) & (np.abs(ZE) < 0.5 * arr.pitch))
            self._geoms.append((half_w, XI, ZE, base, np.maximum(-d, 0.0), a))
        self._centres = centres
        self.labels, cols, plane_of, plane_cols = [], [], [], []
        self._cols = []  # (sheet, class, term) for every column, in order
        centres_c = centres / arr.pitch
        for ip, sh in enumerate(self.sheets):
            for cls in arr.classes():
                # A callable basis may depend on the ring (Chebyshev distance, in
                # pitches) of the class from the centre pad: the pads nearest the
                # sensing one carry the sharpest charge and need the richest basis.
                ring = int(round(np.abs(centres_c[cls[0]]).max()))
                terms = local_basis(ring) if callable(local_basis) else local_basis
                for term in terms:
                    if sh.profile == "smooth" and len(term) > 2:
                        continue  # edge bands mean nothing inside a conductor
                    dens, raw = self._density(ip, cls, term, with_raw=True)
                    if np.abs(dens).max() < 1e-9 * raw:
                        continue  # a term the symmetry removes (odd, on the centre pad);
                        # keeping its round-off would let the fit give it a huge weight
                    cols.append(self._cosine_coefficients(dens))
                    self.labels.append((tuple(cls), *term, sh.y))
                    plane_of.append(ip)
                    self._cols.append((ip, cls, term))
                    pv = self._plane_values(dens, ip)
                    if pv is not None:
                        plane_cols.append(pv)  # (n_eval_planes, nodes) from sheet ip
        self.plane_design = (torch.stack(plane_cols, -1) if plane_cols else None)
        self.coef = torch.as_tensor(np.stack(cols), device=self.device, dtype=dtype)
        self.plane_of = torch.as_tensor(plane_of, device=self.device)
        m = torch.arange(self.modes + 1, device=self.device, dtype=dtype)
        self.m = m
        self.kappa = (self.k * torch.sqrt(m[:, None] ** 2 + m[None, :] ** 2)).reshape(-1)

    def _density(self, ip, cls, term, with_raw=False) -> np.ndarray:
        """One local term on one representative pad of ``cls`` on sheet ``ip``,
        symmetrised: the symmetrisation then places the correctly rotated and
        mirrored copy on every pad of the orbit.  (Placing the same local term
        on all of them makes the odd terms cancel under the 180-degree
        rotation.)  ``(p, q)`` is a pad-wide polynomial; ``(p, q, s)`` confines
        it to a band of width ~s inside the edge, where a neighbour at a
        different potential piles the charge up."""
        half_w, XI, ZE, base, dist_in, a = self._geoms[ip]
        n, h = self.n, self.h
        prof = base if len(term) == 2 or term[2] is None else base * np.exp(-dist_in / term[2])
        dens = np.zeros((n, n))
        i = cls[0]
        ci = n // 2 + int(round(self._centres[i, 0] / h))
        cj = n // 2 + int(round(self._centres[i, 1] / h))
        val = prof * (XI / a) ** term[0] * (ZE / a) ** term[1]
        if ci - half_w >= 0 and ci + half_w < n and cj - half_w >= 0 and cj + half_w < n:
            dens[ci - half_w:ci + half_w + 1, cj - half_w:cj + half_w + 1] += val
        else:  # a tile reaching the cell side: the domain is periodic (drift) or mirrored (cosine) anyway
            ix = (ci + np.arange(-half_w, half_w + 1)) % n
            jx = (cj + np.arange(-half_w, half_w + 1)) % n
            np.add.at(dens, (ix[:, None], jx[None, :]), val)
        raw = np.abs(dens).max()
        dens = d4_symmetrise(dens)
        return (dens, raw) if with_raw else dens

    def _combined_spectra(self, coef):
        """Per sheet, the unshifted FFT of the weighted sum of its densities,
        normalised so that ``ifft2(S * g)`` is the potential."""
        n, L = self.n, self.arr.width
        key = tuple(np.asarray(torch.as_tensor(coef).cpu(), float).round(15).tolist())
        cache = getattr(self, "_spectra", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        dens = [np.zeros((n, n)) for _ in self.sheets]
        for c, (ip, cls, term) in zip(np.asarray(torch.as_tensor(coef).cpu(), float), self._cols):
            if c != 0.0:
                dens[ip] += c * self._density(ip, cls, term)
        spectra = []
        for ip, sh in enumerate(self.sheets):
            S = torch.fft.fft2(torch.fft.ifftshift(torch.as_tensor(dens[ip], device=self.device,
                                                                   dtype=self.dtype)))
            spectra.append((sh.y, S * (self.h * self.h / L ** 2) * (n * n)))
        self._spectra = (key, spectra)
        return spectra

    def layers(self, coef: torch.Tensor, ys, region: float | None = None):
        """The fitted potential on whole planes ``y`` (fine grid, exact).

        Returns ``(len(ys), n', n')`` over the centred square of half-width
        ``region`` (default: the whole cell), and the grid coordinates.  This
        is the reference near the sheets, where the truncated series of
        ``evaluate`` is slow to converge.
        """
        n, L = self.n, self.arr.width
        f = torch.fft.fftfreq(n, d=1.0 / n, device=self.device).to(self.dtype)
        kfull = (2.0 * math.pi / L) * torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2).reshape(-1)
        spectra = self._combined_spectra(coef)
        xs = (np.arange(n) - n // 2) * self.h
        sl = slice(None) if region is None else np.flatnonzero(np.abs(xs) <= region + 1e-12)
        sl_t = sl if region is None else torch.as_tensor(sl, device=self.device)
        out = []
        for y in ys:
            acc = torch.zeros((n, n), device=self.device, dtype=self.dtype)
            for yp, S in spectra:
                g = self.arr.green(kfull, torch.tensor([float(y)], device=self.device, dtype=self.dtype),
                                   yp, self.kind)[0].reshape(n, n)
                acc += torch.fft.fftshift(torch.fft.ifft2(S * g).real)
            out.append(acc if region is None else acc[sl_t][:, sl_t])
        return torch.stack(out), (xs if region is None else xs[sl])

    def field_layers(self, coef: torch.Tensor, ys, xs, zs, offset: float = 0.0, h_y: float = 1e-6):
        """``V`` and ``E = -grad V`` of the fitted potential on the lattice
        ``ys x xs x zs``, exactly: one inverse FFT over the whole fine grid per
        plane and component, with spectral transverse derivatives.  ``xs`` and
        ``zs`` are rounded to fine-grid nodes and wrapped periodically, so a box
        may straddle the cell side.  Returns four ``(ny, nx, nz)`` tensors."""
        n, L = self.n, self.arr.width
        f = torch.fft.fftfreq(n, d=1.0 / n, device=self.device).to(self.dtype)
        kx = (2.0 * math.pi / L) * f
        kfull = torch.sqrt(kx[:, None] ** 2 + kx[None, :] ** 2).reshape(-1)
        spectra = self._combined_spectra(coef)
        ix = torch.as_tensor(np.round(np.asarray(xs) / self.h).astype(int) % n, device=self.device)
        iz = torch.as_tensor(np.round(np.asarray(zs) / self.h).astype(int) % n, device=self.device)
        out = [torch.empty((len(ys), len(ix), len(iz)), device=self.device, dtype=self.dtype) for _ in range(4)]
        ikx = (1j * kx)[:, None]
        ikz = (1j * kx)[None, :]
        for iy, y in enumerate(ys):
            phi = torch.zeros((n, n), device=self.device, dtype=torch.complex128 if self.dtype == torch.float64
                              else torch.complex64)
            dphi = torch.zeros_like(phi)
            for yp, S in spectra:
                gy = lambda v: self.arr.green(kfull, torch.tensor([float(v)], device=self.device, dtype=self.dtype),
                                              yp, self.kind)[0].reshape(n, n)
                phi += S * gy(y)
                dphi += S * (gy(y + h_y) - gy(y - h_y)) / (2 * h_y)
            for c, spec in enumerate((phi, -ikx * phi, -dphi, -ikz * phi)):
                plane = torch.fft.ifft2(spec).real
                out[c][iy] = plane[ix][:, iz]
        out[0] += offset
        return tuple(out)

    def _plane_values(self, dens: np.ndarray, ip: int):
        """This density's potential, as a source on sheet ``ip``, at the fit
        nodes on every sheet plane: ``(n_eval_planes, nodes)``."""
        if self._nodes is None:
            return None
        n, L = self.n, self.arr.width
        S = torch.fft.fft2(torch.fft.ifftshift(torch.as_tensor(dens, device=self.device, dtype=self.dtype)))
        S = S * (self.h * self.h / L ** 2)
        yp = self.planes[ip]
        vals = []
        for q, (ni, nj) in self._rows.items():
            phi = torch.fft.fftshift(torch.fft.ifft2(S * self._gfull[(self.planes[q], yp)]).real) * (n * n)
            vals.append(phi[ni, nj])
        return torch.cat(vals) if self._flat_rows else torch.stack(vals)

    def _cosine_coefficients(self, dens: np.ndarray) -> np.ndarray:
        """``a_mn`` with ``f = sum a_mn cos(mkx) cos(nkz)``, for an even ``f``
        sampled on the centred grid."""
        n = dens.shape[0]
        f = np.fft.fft2(np.fft.ifftshift(dens)).real * (self.h ** 2)  # int f e^{-i k.r}
        M = self.modes
        a = f[: M + 1, : M + 1] / (self.arr.width ** 2)
        w = np.full(M + 1, 2.0)
        w[0] = 1.0
        return a * w[:, None] * w[None, :]

    def evaluate(self, xyz: torch.Tensor, coef: torch.Tensor | None = None,
                 chunk: int | None = None, grad: bool = False, work_dtype=None):
        """Columns (or the combination ``coef``) at points ``(x, y, z)``.

        Returns ``(n, ncol)`` values, or ``(n,)`` for a combination; with
        ``grad`` also the gradient ``(n, 3)`` of the combination.  Each chunk of
        points is one matrix product over the ``(M+1)^2`` transverse modes, done
        in ``work_dtype`` (float32 on a GPU is ample for a basis column).
        """
        k, m = self.k, self.m
        nm = m.numel()
        wd = work_dtype or self.dtype
        if chunk is None:
            chunk = max(64, int(2.0e8 // (nm * nm)))
        combos = []
        for ip, yp in enumerate(self.planes):
            sel = self.plane_of == ip
            A = self.coef[sel] if coef is None else (coef[sel][:, None, None] * self.coef[sel]).sum(0, keepdim=True)
            combos.append((yp, sel, A.reshape(A.shape[0], -1).to(wd)))
        ncol = self.coef.shape[0] if coef is None else 1
        outs, grads = [], []
        mw = m.to(wd)
        kap = self.kappa.to(wd)
        for s in range(0, xyz.shape[0], chunk):
            p = xyz[s:s + chunk].to(wd)
            x, y, z = p[:, 0], p[:, 1], p[:, 2]
            cx, cz = torch.cos(k * x[:, None] * mw), torch.cos(k * z[:, None] * mw)
            val = torch.zeros((len(p), ncol), device=self.device, dtype=wd)
            g3 = torch.zeros((len(p), 3), device=self.device, dtype=wd)
            if grad:
                sx = -k * mw * torch.sin(k * x[:, None] * mw)
                sz = -k * mw * torch.sin(k * z[:, None] * mw)
            for yp, sel, A in combos:
                G = self.arr.green(kap, y, yp, self.kind).reshape(-1, nm, nm)
                basis = (cx[:, :, None] * cz[:, None, :] * G).reshape(len(p), -1)
                v = basis @ A.T
                if coef is None:
                    val[:, sel] = v
                else:
                    val += v
                if grad:
                    h = 1e-4
                    dG = ((self.arr.green(kap, y + h, yp, self.kind)
                           - self.arr.green(kap, y - h, yp, self.kind)) / (2 * h)
                          ).reshape(-1, nm, nm)
                    g3[:, 0] += ((sx[:, :, None] * cz[:, None, :] * G).reshape(len(p), -1) @ A.T)[:, 0]
                    g3[:, 1] += ((cx[:, :, None] * cz[:, None, :] * dG).reshape(len(p), -1) @ A.T)[:, 0]
                    g3[:, 2] += ((cx[:, :, None] * sz[:, None, :] * G).reshape(len(p), -1) @ A.T)[:, 0]
            outs.append(val.to(self.dtype))
            grads.append(g3.to(self.dtype))
        val = torch.cat(outs)
        if coef is not None:
            val = val[:, 0]
        return (val, torch.cat(grads)) if grad else val


def pad_surface_points(arr: PadArray, n: int, rng: np.random.Generator,
                       edge_fraction: float = 0.6, pads: list[int] | None = None) -> np.ndarray:
    """Points on the pads (in the pad plane), dense toward the edges.

    Returns ``(n, 3)`` points ``(x, y_pad, z)`` in the symmetry wedge
    ``0 <= z <= x`` of the cell (enough, because everything is D4-symmetric),
    and the index of the pad each lies on.
    """
    centres = arr.pad_centres()
    pads = list(range(len(centres))) if pads is None else pads
    a = arr.shape.extent
    out, which = [], []
    need = n
    while need > 0:
        m = 4 * need + 64
        i = rng.choice(pads, m)
        xi = rng.uniform(-a, a, m)
        ze = rng.uniform(-a, a, m)
        d = arr.shape.sdf(xi, ze)
        # half uniform, half concentrated near the edge (accept by distance)
        near = rng.uniform(0, 1, m) < edge_fraction
        keep = (d < 0) & (~near | (-d < 0.05 * a * rng.exponential(1.0, m) + 1e-4))
        x, z = centres[i, 0] + xi, centres[i, 1] + ze
        wedge = (z >= 0) & (z <= x)
        keep &= wedge
        planes = arr.y_pad if isinstance(arr.y_pad, (tuple, list)) else (arr.y_pad,)
        yy = rng.choice(np.asarray(planes, float), keep.sum())
        out.append(np.column_stack((x[keep], yy, z[keep])))
        which.append(i[keep])
        need = n - sum(len(o) for o in out)
    xyz = np.concatenate(out)[:n]
    return xyz, np.concatenate(which)[:n]


def fit_single_layer(ssl: SpectralSingleLayer, pts: np.ndarray, target: np.ndarray,
                     offset: float = 0.0, ridge: float = 0.0):
    """Least-squares coefficients so that ``offset + sum c_j col_j = target``."""
    t = torch.as_tensor(pts, device=ssl.device, dtype=ssl.dtype)
    A = ssl.evaluate(t)
    b = torch.as_tensor(target - offset, device=ssl.device, dtype=ssl.dtype)
    scale = A.abs().max(0).values.clamp_min(1e-30)
    An = (A / scale).cpu()
    if ridge > 0:
        An = torch.cat((An, ridge * torch.eye(An.shape[1], dtype=An.dtype)))
        bb = torch.cat((b.cpu(), torch.zeros(An.shape[1], dtype=An.dtype)))
    else:
        bb = b.cpu()
    sol = torch.linalg.lstsq(An, bb[:, None], driver="gelsd").solution[:, 0]
    coef = (sol.to(ssl.device) / scale)
    res = A @ coef - b
    return coef, float(res.pow(2).mean().sqrt()), float(res.abs().max())


def pad_nodes(arr: PadArray, grid: int, count: int, rng: np.random.Generator,
              edge_scale: float = 0.1, wedge: bool = True):
    """Fine-grid nodes on the pads for the surface fit, denser near the edges.

    Returns ``(i, j, owner, depth)``: grid indices (on the odd grid ``grid|1``
    with the origin at its centre), the pad each lies on, and its distance
    inside the pad edge.  With ``wedge`` only the symmetry wedge
    ``0 <= z <= x`` is used, which is enough for a D4-symmetric problem.
    """
    n = int(grid) | 1
    h = arr.width / n
    xs = (np.arange(n) - n // 2) * h
    X, Z = np.meshgrid(xs, xs, indexing="ij")
    sdf = np.full((n, n), np.inf)
    owner = np.full((n, n), -1)
    for i, (cx, cz) in enumerate(arr.pad_centres()):
        d = arr.shape.sdf(X - cx, Z - cz)
        m = d < sdf
        sdf[m], owner[m] = d[m], i
    ok = sdf < 0
    if wedge:
        ok &= (Z >= 0) & (Z <= X + 1e-12)
    ii, jj = np.nonzero(ok)
    depth = -sdf[ii, jj]
    w = 0.02 + np.exp(-depth / edge_scale)
    keep = rng.uniform(size=len(ii)) < np.minimum(1.0, w * count / w.sum())
    return ii[keep], jj[keep], owner[ii[keep], jj[keep]], depth[keep]


def fit_on_nodes(ssl: SpectralSingleLayer, target: np.ndarray, offset: float = 0.0,
                 train: np.ndarray | None = None):
    """Least squares on the exact on-plane design: every sheet plane at every
    node must equal ``target`` (per node).  Returns ``(coef, rms, max)`` over
    the nodes not in ``train`` (all nodes when ``train`` is None)."""
    A = ssl.plane_design  # (n_planes, nodes, ncol)
    npl = A.shape[0]
    b = torch.as_tensor(np.asarray(target, float) - offset, device=ssl.device, dtype=ssl.dtype)
    sel = torch.ones(A.shape[1], dtype=torch.bool, device=ssl.device) if train is None else \
        torch.as_tensor(train, device=ssl.device)
    Af = A[:, sel].reshape(-1, A.shape[-1])
    bf = b[sel].repeat(npl)
    scale = Af.abs().max(0).values.clamp_min(1e-30)
    sol = torch.linalg.lstsq((Af / scale).cpu(), bf.cpu()[:, None], driver="gelsd").solution[:, 0]
    coef = sol.to(ssl.device) / scale
    test = ~sel if train is not None else sel
    r = A[:, test].reshape(-1, A.shape[-1]) @ coef - b[test].repeat(npl)
    return coef, float(r.pow(2).mean().sqrt()), float(r.abs().max())


# --------------------------------------------------------------------------
# drift and induction
# --------------------------------------------------------------------------


def bnl_speed(e_v_per_mm: torch.Tensor, temperature: float = 87.0) -> torch.Tensor:
    """Electron drift speed in mm/us from the BNL liquid-argon mobility
    parameterisation (https://lar.bnl.gov/properties/trans.html), as used by
    the pochoir finite-difference reference."""
    e = e_v_per_mm * 10.0 / 1000.0  # kV/cm
    trel = temperature / 89.0
    a0, a1, a2, a3, a4, a5 = 551.6, 7158.3, 4440.43, 4.29, 43.63, 0.2053
    mu = (a0 + a1 * e + a2 * e ** 1.5 + a3 * e ** 2.5) / (
        (1 + (a1 / a0) * e + a4 * e ** 2 + a5 * e ** 3) * trel ** 1.5)  # cm^2/(V s)
    return mu * (e * 1000.0) * 1e-5  # (cm/s) -> mm/us


@dataclass
class Paths:
    t: torch.Tensor  # (nt,) us
    xyz: torch.Tensor  # (npath, nt, 3) mm; frozen after arrival
    arrived: torch.Tensor  # (npath,) us, arrival time (inf if none)
    pad: torch.Tensor  # (npath, 2) mm, centre of the pad each landed on (local to the drift cell)


def trace(field: "SheetPotential", starts: torch.Tensor,
          tick: float, t_max: float, substeps: int = 8, temperature: float = 87.0,
          landing: str | None = None, slide: bool = False, grid: tuple | None = None) -> Paths:
    """Drift electrons through the drift-cell potential (RK4, ``substeps``
    per tick), stopping on a pad.

    ``landing`` "top" (the default for flat pads, as in the pochoir
    reproduction) stops a path once it is inside the outline and below the
    top face, interpolated to the top plane; "conductor" (the default for a
    hemisphere) stops it on entering the conductor, bisected onto its
    surface.  With ``slide`` the PCB surface ``arr.y_floor`` is zero-slope (a
    charged insulator), and a step that dips below it is put back on it.
    ``grid = (y, r_hole)`` is a thin grid electrode at height ``y`` with round
    holes over the pads: a step that crosses it outside a hole ends there,
    and the path's pad is reported as NaN.

    Positions are reported at every tick; after arrival they stay put.  The
    drift cell is periodic, so transverse coordinates are wrapped for the
    field and unwrapped for the path.
    """
    arr = field.arr
    y_pads = arr.y_pad if isinstance(arr.y_pad, (tuple, list)) else (arr.y_pad,)
    # the pads' top face (not the highest sheet, which may be a grid)
    y_top = max(y_pads) if arr.shape.kind != "hemisphere" else arr.y_floor + arr.shape.half
    landing = landing or ("conductor" if arr.shape.kind == "hemisphere" else "top")
    L = arr.width
    dt = tick / substeps

    def velocity(p):
        wrapped = p.clone()
        wrapped[:, 0] = torch.remainder(p[:, 0] + 0.5 * L, L) - 0.5 * L
        wrapped[:, 2] = torch.remainder(p[:, 2] + 0.5 * L, L) - 0.5 * L
        _, g = field.evaluate(wrapped, grad=True)
        e = -g  # V/mm
        mag = torch.linalg.vector_norm(e, dim=1).clamp_min(1e-12)
        return -(bnl_speed(mag, temperature) / mag)[:, None] * e  # electrons move against E

    def inside(p):
        wx = torch.remainder(p[:, 0] + 0.5 * L, L) - 0.5 * L
        wz = torch.remainder(p[:, 2] + 0.5 * L, L) - 0.5 * L
        return arr.shape.inside(wx, p[:, 1], wz, arr.y_floor, y_top)

    def landed(p):
        if landing == "conductor":
            return inside(p)
        wx = (torch.remainder(p[:, 0] + 0.5 * L, L) - 0.5 * L).cpu().numpy()
        wz = (torch.remainder(p[:, 2] + 0.5 * L, L) - 0.5 * L).cpu().numpy()
        on = torch.as_tensor(arr.shape.sdf(wx, wz) < 0, device=p.device)
        return on & (p[:, 1] <= y_top)

    def settle(p, new, hit):
        """Put the paths that hit a conductor on its surface."""
        if landing == "top":
            frac = ((p[hit, 1] - y_top) / (p[hit, 1] - new[hit, 1]).clamp_min(1e-12)).clamp(0, 1)
            return p[hit] + frac[:, None] * (new[hit] - p[hit])
        q0, q1 = p[hit], new[hit]
        lo = torch.zeros(len(q0), device=p.device, dtype=p.dtype)
        hi = torch.ones_like(lo)
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            ins = inside(q0 + mid[:, None] * (q1 - q0))
            hi, lo = torch.where(ins, mid, hi), torch.where(ins, lo, mid)
        return q0 + hi[:, None] * (q1 - q0)

    nt = int(round(t_max / tick)) + 1
    pos = starts.clone().to(field.dtype)
    out = torch.empty((len(pos), nt, 3), device=pos.device, dtype=pos.dtype)
    out[:, 0] = pos
    alive = torch.ones(len(pos), dtype=torch.bool, device=pos.device)
    grid_hit = torch.zeros(len(pos), dtype=torch.bool, device=pos.device)
    arrived = torch.full((len(pos),), float("inf"), device=pos.device, dtype=pos.dtype)
    for it in range(1, nt):
        for _ in range(substeps):
            if not alive.any():
                break
            p = pos[alive]
            k1 = velocity(p)
            k2 = velocity(p + 0.5 * dt * k1)
            k3 = velocity(p + 0.5 * dt * k2)
            k4 = velocity(p + dt * k3)
            new = p + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            if slide:
                new[:, 1] = new[:, 1].clamp_min(arr.y_floor)
            hit = landed(new)
            if hit.any():
                new[hit] = settle(p, new, hit)
            if grid is not None:
                yg, rh = grid
                cross = ((p[:, 1] - yg) * (new[:, 1] - yg) <= 0) & (p[:, 1] != new[:, 1]) & ~hit
                if cross.any():
                    fr = ((p[cross, 1] - yg) / (p[cross, 1] - new[cross, 1])).clamp(0, 1)
                    at = p[cross] + fr[:, None] * (new[cross] - p[cross])
                    wx = torch.remainder(at[:, 0] + 0.5 * arr.pitch, arr.pitch) - 0.5 * arr.pitch
                    wz = torch.remainder(at[:, 2] + 0.5 * arr.pitch, arr.pitch) - 0.5 * arr.pitch
                    on_grid = wx * wx + wz * wz > rh * rh
                    gi = torch.nonzero(cross)[:, 0][on_grid]
                    new[gi] = at[on_grid]
                    hit = hit.clone()
                    hit[gi] = True
                    grid_hit[torch.nonzero(alive)[:, 0][gi]] = True
            idx = torch.nonzero(alive)[:, 0]
            pos[idx] = new
            t_now = (it - 1) * tick + dt * (_ + 1)
            arrived[idx[hit]] = t_now
            alive[idx[hit]] = False
        out[:, it] = pos
        if not alive.any():
            out[:, it + 1:] = pos[:, None, :]
            break
    wx = torch.remainder(pos[:, 0] + 0.5 * L, L) - 0.5 * L
    wz = torch.remainder(pos[:, 2] + 0.5 * L, L) - 0.5 * L
    pad = torch.stack((pos[:, 0] - wx, pos[:, 2] - wz), 1)
    pad[grid_hit] = float("nan")
    return Paths(torch.arange(nt, device=pos.device, dtype=pos.dtype) * tick, out, arrived, pad)


def induced_charge(weight: "SheetPotential", xyz: torch.Tensor,
                   arrived: torch.Tensor, t: torch.Tensor, landed_on_sensing: torch.Tensor,
                   chunk_paths: int = 8) -> torch.Tensor:
    """Charge ``Q = -q phi_w(x(t))`` (e, with q = -e) induced on the sensing
    pad along each path, at every tick.  After arrival the charge is the
    landing pad's weighting potential exactly: 1 on the sensing pad, 0
    elsewhere (the boundary condition, not the truncated series)."""
    npath, nt, _ = xyz.shape
    Q = torch.empty((npath, nt), device=xyz.device, dtype=xyz.dtype)
    for s in range(0, npath, chunk_paths):
        p = xyz[s:s + chunk_paths]
        before = t[None, :] < arrived[s:s + chunk_paths, None]
        vals = torch.zeros(p.shape[:2], device=xyz.device, dtype=xyz.dtype)
        pts = p[before]
        if len(pts):
            vals[before] = weight.evaluate(pts, work_dtype=torch.float32)
        final = landed_on_sensing[s:s + chunk_paths].to(xyz.dtype)[:, None].expand_as(vals)
        Q[s:s + chunk_paths] = torch.where(before, vals, final)
    return Q


class SheetPotential:
    """A fitted single layer, reduced to one set of cosine coefficients per
    sheet: small, saveable, and all that evaluation needs.

    ``offset`` is added (the fixed-face potential of a drift solve).
    """

    def __init__(self, arr: PadArray, kind: str, k: float, planes, A, offset=0.0,
                 device="cpu", dtype=torch.float64):
        self.arr, self.kind, self.k = arr, kind, float(k)
        self.planes = [float(p) for p in planes]
        self.A = torch.as_tensor(A, device=device, dtype=dtype)  # (nplanes, M+1, M+1)
        self.offset = float(offset)
        self.device, self.dtype = torch.device(device), dtype
        M = self.A.shape[-1] - 1
        self.m = torch.arange(M + 1, device=self.device, dtype=dtype)
        self.kappa = (self.k * torch.sqrt(self.m[:, None] ** 2 + self.m[None, :] ** 2)).reshape(-1)

    @classmethod
    def from_fit(cls, ssl: SpectralSingleLayer, coef: torch.Tensor, offset=0.0):
        A = torch.stack([(coef[ssl.plane_of == i][:, None, None] * ssl.coef[ssl.plane_of == i]).sum(0)
                         for i in range(len(ssl.planes))])
        return cls(ssl.arr, ssl.kind, ssl.k, ssl.planes, A, offset, ssl.device, ssl.dtype)

    def state(self) -> dict:
        from dataclasses import asdict
        return dict(arr=asdict(self.arr), kind=self.kind, k=self.k, planes=self.planes,
                    A=self.A.cpu(), offset=self.offset)

    @classmethod
    def from_state(cls, st: dict, device="cpu", dtype=torch.float64):
        a = dict(st["arr"])
        a["shape"] = PadShape(**a["shape"])
        if isinstance(a["y_pad"], list):
            a["y_pad"] = tuple(a["y_pad"])
        return cls(PadArray(**a), st["kind"], st["k"], st["planes"], st["A"], st["offset"], device, dtype)

    def evaluate(self, xyz: torch.Tensor, grad: bool = False, work_dtype=None,
                 chunk: int | None = None):
        """``V`` (and ``grad V``) at points ``(x, y, z)``."""
        k, m = self.k, self.m
        nm = m.numel()
        wd = work_dtype or self.dtype
        chunk = chunk or max(64, int(2.0e8 // (nm * nm)))
        mw, kap = m.to(wd), self.kappa.to(wd)
        A = self.A.reshape(len(self.planes), -1).to(wd)
        vals, grads = [], []
        for s in range(0, xyz.shape[0], chunk):
            p = xyz[s:s + chunk].to(wd)
            x, y, z = p[:, 0], p[:, 1], p[:, 2]
            cx, cz = torch.cos(k * x[:, None] * mw), torch.cos(k * z[:, None] * mw)
            v = torch.zeros(len(p), device=self.device, dtype=wd)
            g = torch.zeros((len(p), 3), device=self.device, dtype=wd)
            if grad:
                sx = -k * mw * torch.sin(k * x[:, None] * mw)
                sz = -k * mw * torch.sin(k * z[:, None] * mw)
            for ip, yp in enumerate(self.planes):
                G = self.arr.green(kap, y, yp, self.kind).reshape(-1, nm, nm)
                v += (cx[:, :, None] * cz[:, None, :] * G).reshape(len(p), -1) @ A[ip]
                if grad:
                    h = 1e-4
                    dG = ((self.arr.green(kap, y + h, yp, self.kind)
                           - self.arr.green(kap, y - h, yp, self.kind)) / (2 * h)
                          ).reshape(-1, nm, nm)
                    g[:, 0] += (sx[:, :, None] * cz[:, None, :] * G).reshape(len(p), -1) @ A[ip]
                    g[:, 1] += (cx[:, :, None] * cz[:, None, :] * dG).reshape(len(p), -1) @ A[ip]
                    g[:, 2] += (cx[:, :, None] * sz[:, None, :] * G).reshape(len(p), -1) @ A[ip]
            vals.append((v + self.offset).to(self.dtype))
            grads.append(g.to(self.dtype))
        v = torch.cat(vals)
        return (v, torch.cat(grads)) if grad else v


# --------------------------------------------------------------------------
# the network correction: a SIREN on a D4-invariant encoding
# --------------------------------------------------------------------------


class D4Encoding(torch.nn.Module):
    """Features invariant under the dihedral group of the square, with zero
    slope on the cell sides:

    ``C_mn = cos(m k x) cos(n k z) + cos(n k x) cos(m k z)``, ``0 <= m <= n <= H``,

    plus an affine height.  Any function of these has the symmetry of the
    problem and meets the side conditions exactly, as the periodic encoding
    does for the 2D wires.
    """

    def __init__(self, period: float, harmonics: int, y_center: float, y_scale: float):
        super().__init__()
        pairs = [(m, n) for n in range(harmonics + 1) for m in range(n + 1) if (m, n) != (0, 0)]
        self.register_buffer("mm", torch.tensor([p[0] for p in pairs], dtype=torch.float64))
        self.register_buffer("nn", torch.tensor([p[1] for p in pairs], dtype=torch.float64))
        self.k = 2.0 * math.pi / period
        self.yc, self.ys = float(y_center), float(y_scale)
        self.out_features = len(pairs) + 1

    def forward(self, xyz):
        x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
        k = self.k
        c = (torch.cos(self.mm * k * x) * torch.cos(self.nn * k * z)
             + torch.cos(self.nn * k * x) * torch.cos(self.mm * k * z))
        return torch.cat((c, (y - self.yc) / self.ys), 1)


def train_correction(field: SheetPotential, node_xyz: torch.Tensor, node_resid: torch.Tensor,
                     train: torch.Tensor, harmonics: int = 4, hidden: int = 64, layers: int = 3,
                     omega0: float = 30.0, steps: int = 3000, y_center: float = 12.0,
                     y_scale: float = 3.0, seed: int = 0, log=None):
    """Fit a SIREN correction ``N`` so that ``V = B + N`` is harmonic and meets
    the pad surfaces where the single layer ``B`` misses them.

    ``node_resid`` is ``B - target`` at ``node_xyz`` (exact, from the on-plane
    design).  The faces take ``N = 0`` at the fixed one and zero slope at the
    other; the sides and the D4 symmetry are exact through the encoding.
    Returns ``(N, rms_before, rms_after, max_before, max_after)`` over the
    held-out nodes.
    """
    from .siren import Siren

    dev, dt = node_xyz.device, node_xyz.dtype
    torch.manual_seed(seed)
    arr = field.arr
    sigma = float(node_resid.abs().max()) or 1.0
    enc = D4Encoding(arr.width, harmonics, y_center, y_scale).to(dev)
    net = Siren(enc.out_features, hidden, layers, omega0, 30.0, out_features=1,
                final_init_scale=0.1).to(dev).to(dt)

    def N(p):
        return sigma * net(enc(p))[:, 0]

    fixed, free = (arr.top, 0.0) if field.kind == "drift" else (0.0, arr.top)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, (1e-5 / 1e-3) ** (1.0 / steps))
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed + 1)
    L = arr.width
    yp = sum(field.planes) / len(field.planes)
    idx = torch.nonzero(train)[:, 0]

    def uniform(n):
        return (torch.rand(n, device=dev, generator=gen, dtype=dt) - 0.5) * L

    def colloc(n):
        u = torch.rand(n, device=dev, generator=gen, dtype=dt)
        y = yp + torch.sign(u - 0.3) * 0.01 * torch.exp(
            torch.rand(n, device=dev, generator=gen, dtype=dt) * math.log(1000.0))
        return torch.stack((uniform(n), y.clamp(0.0, arr.top), uniform(n)), 1)

    for step in range(1, steps + 1):
        p = colloc(4096).requires_grad_(True)
        v = N(p)
        (g,) = torch.autograd.grad(v.sum(), p, create_graph=True)
        lap = sum(torch.autograd.grad(g[:, i].sum(), p, create_graph=True)[0][:, i] for i in range(3))
        pde = ((lap * arr.pitch ** 2 / sigma) ** 2).mean()
        b = idx[torch.randint(0, len(idx), (4096,), device=dev, generator=gen)]
        bc = (((N(node_xyz[b]) + node_resid[b]) / sigma) ** 2).mean()
        fx = torch.stack((uniform(512), torch.full((512,), fixed, device=dev, dtype=dt), uniform(512)), 1)
        fr = torch.stack((uniform(512), torch.full((512,), free, device=dev, dtype=dt), uniform(512)), 1)
        fr.requires_grad_(True)
        (gf,) = torch.autograd.grad(N(fr).sum(), fr, create_graph=True)
        face = ((N(fx) / sigma) ** 2).mean() + ((gf[:, 1] * arr.pitch / sigma) ** 2).mean()
        loss = pde + 10.0 * bc + 10.0 * face
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if log and step % max(1, steps // 5) == 0:
            log(f"correction step {step}: pde {float(pde):.2e} surface {float(bc):.2e}")
    with torch.no_grad():
        held = ~train
        before = node_resid[held]
        after = N(node_xyz[held]) + before
    return (N, float(before.pow(2).mean().sqrt()), float(after.pow(2).mean().sqrt()),
            float(before.abs().max()), float(after.abs().max()))
