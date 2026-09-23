"""Electrode shapes as signed distance functions, in 2D and 3D.

Everything the solver needs to know about an electrode's *form* is here:

``sdf(points)``
    signed distance to the surface, negative inside;
``sample_surface(n, rng)``
    points drawn on the surface;
``bounds()``
    an axis-aligned box that contains it (``None`` along directions in which
    the shape is unbounded).

Nothing else in the package knows that a conductor is a circle.  Adding a new
electrode kind means adding a class here and naming it in the schema; the
sampler, the boundary loss, the drift integrator and the Ramo bookkeeping all
work unchanged, because they only ever ask those three questions.

The one part that is *not* shape-agnostic is the analytic basis of
:mod:`firep.siren`, which is specific to round wires between parallel planes.
For any other shape the basis falls back to ``mode: linear`` and the network has
to do the work -- see the generalisation discussion in the technical note.

Coordinates
-----------
``x`` transverse (across the pitch), ``y`` drift, ``z`` along the electrode.  A
2D model uses ``(x, y)`` and every shape below degenerates correctly: an
infinite cylinder becomes a circle, an infinite strip becomes a segment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _as_points(p: np.ndarray, ndim: int) -> np.ndarray:
    p = np.atleast_2d(np.asarray(p, dtype=float))
    if p.shape[-1] != ndim:
        raise ValueError(f"expected {ndim}-D points, got shape {p.shape}")
    return p


def _rot2(angle_deg: float) -> np.ndarray:
    """Rotation taking detector ``(x, z)`` into electrode-local (across, along).

    The electrode axis is ``e_along = (sin a, 0, cos a)`` and the transverse
    direction is ``e_across = (cos a, 0, -sin a)``, so the across-coordinate of
    a displacement is ``dx cos a - dz sin a``.
    """
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


@dataclass
class Shape:
    """Base class: a conductor's form, independent of its potential."""

    ndim: int = 2

    def sdf(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample_surface(self, n: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def bounds(self):
        """``(lo, hi)`` arrays; entries may be ``+-inf`` where unbounded."""
        raise NotImplementedError

    # -- conveniences -------------------------------------------------------

    def inside(self, points: np.ndarray, margin: float = 0.0) -> np.ndarray:
        return self.sdf(points) < margin

    def check_surface(self, n: int = 2000, seed: int = 0) -> float:
        """Largest |sdf| over sampled surface points; a self-consistency test."""
        rng = np.random.default_rng(seed)
        pts = self.sample_surface(n, rng)
        return float(np.abs(self.sdf(pts)).max())


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------


@dataclass
class Wire(Shape):
    """A cylinder of radius ``radius`` along a direction in the readout plane.

    In 2D (``ndim = 2``) this is a circle.  In 3D the axis lies in the plane
    ``y = centre[1]`` at ``angle`` degrees from the ``z`` axis, which is how a
    stereo wire plane is described: MicroBooNE's induction wires sit at
    :math:`\\pm 60^\\circ` to its collection wires.
    """

    centre: tuple = (0.0, 0.0, 0.0)
    radius: float = 0.075
    angle: float = 0.0  # degrees from +z, about the drift axis

    def sdf(self, points: np.ndarray) -> np.ndarray:
        p = _as_points(points, self.ndim)
        c = np.asarray(self.centre, float)[: self.ndim]
        d = p - c
        if self.ndim == 2:
            return np.hypot(d[:, 0], d[:, 1]) - self.radius
        # distance to an infinite line: drop the component along the axis
        a = math.radians(self.angle)
        axis = np.array([math.sin(a), 0.0, math.cos(a)])
        along = d @ axis
        perp = d - along[:, None] * axis[None, :]
        return np.linalg.norm(perp, axis=1) - self.radius

    def sample_surface(self, n: int, rng: np.random.Generator) -> np.ndarray:
        c = np.asarray(self.centre, float)[: self.ndim]
        th = rng.uniform(0.0, 2.0 * np.pi, n)
        if self.ndim == 2:
            return c + self.radius * np.column_stack((np.cos(th), np.sin(th)))
        a = math.radians(self.angle)
        axis = np.array([math.sin(a), 0.0, math.cos(a)])
        # two unit vectors perpendicular to the axis
        e1 = np.array([math.cos(a), 0.0, -math.sin(a)])
        e2 = np.array([0.0, 1.0, 0.0])
        s = rng.uniform(-0.5, 0.5, n) * self._extent
        ring = (self.radius * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2))
        return c + ring + s[:, None] * axis[None, :]

    _extent: float = 100.0  # sampling length along the axis (it is infinite)

    def bounds(self):
        c = np.asarray(self.centre, float)[: self.ndim]
        r = self.radius
        if self.ndim == 2:
            return c - r, c + r
        lo = np.array([-np.inf, c[1] - r, -np.inf])
        hi = np.array([np.inf, c[1] + r, np.inf])
        return lo, hi


@dataclass
class Strip(Shape):
    """A long flat electrode: a rounded slab, infinite along its own axis.

    ``width`` is across the strip, ``thickness`` through it (along the drift).
    This is the DUNE vertical-drift anode element; with ``thickness`` small
    compared with ``width`` it is a good model of an etched copper trace on a
    PCB.
    """

    centre: tuple = (0.0, 0.0, 0.0)
    width: float = 5.0
    thickness: float = 0.035
    angle: float = 0.0  # degrees from +z

    def sdf(self, points: np.ndarray) -> np.ndarray:
        p = _as_points(points, self.ndim)
        c = np.asarray(self.centre, float)[: self.ndim]
        d = p - c
        if self.ndim == 2:
            local = np.column_stack((d[:, 0], d[:, 1]))
        else:
            r = _rot2(self.angle)
            across = d[:, [0, 2]] @ r.T  # (across, along)
            local = np.column_stack((across[:, 0], d[:, 1]))
        half = np.array([0.5 * self.width, 0.5 * self.thickness])
        q = np.abs(local) - half[None, :]
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside

    def sample_surface(self, n: int, rng: np.random.Generator) -> np.ndarray:
        # Sample the perimeter of the cross-section in proportion to its length.
        w, th = self.width, self.thickness
        peri = 2 * (w + th)
        u = rng.uniform(0.0, peri, n)
        ax = np.empty(n)
        ay = np.empty(n)
        m = u < w
        ax[m] = u[m] - 0.5 * w
        ay[m] = 0.5 * th
        m2 = (u >= w) & (u < w + th)
        ax[m2] = 0.5 * w
        ay[m2] = 0.5 * th - (u[m2] - w)
        m3 = (u >= w + th) & (u < 2 * w + th)
        ax[m3] = 0.5 * w - (u[m3] - w - th)
        ay[m3] = -0.5 * th
        m4 = u >= 2 * w + th
        ax[m4] = -0.5 * w
        ay[m4] = -0.5 * th + (u[m4] - 2 * w - th)

        c = np.asarray(self.centre, float)[: self.ndim]
        if self.ndim == 2:
            return c + np.column_stack((ax, ay))
        s = rng.uniform(-0.5, 0.5, n) * self._extent
        a = math.radians(self.angle)
        e_across = np.array([math.cos(a), 0.0, -math.sin(a)])
        e_along = np.array([math.sin(a), 0.0, math.cos(a)])
        return (c + ax[:, None] * e_across[None, :]
                + np.column_stack((np.zeros(n), ay, np.zeros(n)))
                + s[:, None] * e_along[None, :])

    _extent: float = 100.0

    def bounds(self):
        c = np.asarray(self.centre, float)[: self.ndim]
        if self.ndim == 2:
            h = np.array([0.5 * self.width, 0.5 * self.thickness])
            return c - h, c + h
        lo = np.array([-np.inf, c[1] - 0.5 * self.thickness, -np.inf])
        hi = np.array([np.inf, c[1] + 0.5 * self.thickness, np.inf])
        return lo, hi


@dataclass
class Hole(Shape):
    """A circular aperture through a conducting sheet.

    The *conductor* is the sheet; the hole is where it is absent.  The signed
    distance is therefore that of the slab minus a cylinder.  This is the second
    half of the DUNE vertical-drift "strips and holes" anode: charge passes
    through the holes of the upper planes to reach the lower ones.
    """

    centre: tuple = (0.0, 0.0, 0.0)
    radius: float = 2.5
    thickness: float = 0.035
    sheet_extent: float = 50.0  # how far the sheet is drawn/sampled

    def __post_init__(self):
        if self.ndim != 3:
            raise ValueError("a hole is intrinsically 3D; set dimension: 3")

    def sdf(self, points: np.ndarray) -> np.ndarray:
        p = _as_points(points, 3)
        c = np.asarray(self.centre, float)
        d = p - c
        # slab in y
        slab = np.abs(d[:, 1]) - 0.5 * self.thickness
        # cylinder about the y axis through the hole centre
        rho = np.hypot(d[:, 0], d[:, 2])
        cyl = self.radius - rho  # negative outside the hole
        # sheet minus cylinder: max(slab, -cyl) is the usual CSG subtraction
        return np.maximum(slab, cyl)

    def sample_surface(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Points on the two faces and on the barrel of the aperture."""
        c = np.asarray(self.centre, float)
        th = self.thickness
        face_area = 2.0 * (self.sheet_extent ** 2 - np.pi * self.radius ** 2)
        barrel_area = 2.0 * np.pi * self.radius * th
        p_barrel = barrel_area / (face_area + barrel_area)
        is_barrel = rng.uniform(0, 1, n) < p_barrel

        out = np.empty((n, 3))
        nb = int(is_barrel.sum())
        if nb:
            a = rng.uniform(0, 2 * np.pi, nb)
            out[is_barrel] = c + np.column_stack(
                (self.radius * np.cos(a), rng.uniform(-0.5, 0.5, nb) * th,
                 self.radius * np.sin(a)))
        nf = n - nb
        if nf:
            xs = np.empty(nf)
            zs = np.empty(nf)
            todo = np.ones(nf, bool)
            while todo.any():
                k = int(todo.sum())
                cand = rng.uniform(-0.5, 0.5, (k, 2)) * self.sheet_extent
                ok = np.hypot(cand[:, 0], cand[:, 1]) > self.radius
                idx = np.flatnonzero(todo)[ok]
                xs[idx] = cand[ok, 0]
                zs[idx] = cand[ok, 1]
                todo[idx] = False
            ys = np.where(rng.uniform(0, 1, nf) < 0.5, -0.5 * th, 0.5 * th)
            out[~is_barrel] = c + np.column_stack((xs, ys, zs))
        return out

    def bounds(self):
        c = np.asarray(self.centre, float)
        lo = np.array([-np.inf, c[1] - 0.5 * self.thickness, -np.inf])
        hi = np.array([np.inf, c[1] + 0.5 * self.thickness, np.inf])
        return lo, hi


@dataclass
class Pad(Shape):
    """A finite rectangular pad: the pixel electrode.

    Unlike a wire or a strip it is bounded in every direction, so a pixel plane
    has no translational symmetry to exploit along an electrode -- which is
    exactly what makes a pixel field response a genuinely 3D calculation.
    """

    centre: tuple = (0.0, 0.0, 0.0)
    width: float = 3.0  # along x
    length: float = 3.0  # along z
    thickness: float = 0.035  # along y

    def __post_init__(self):
        if self.ndim != 3:
            raise ValueError("a pad is intrinsically 3D; set dimension: 3")

    def _half(self) -> np.ndarray:
        return 0.5 * np.array([self.width, self.thickness, self.length])

    def sdf(self, points: np.ndarray) -> np.ndarray:
        p = _as_points(points, 3)
        d = np.abs(p - np.asarray(self.centre, float)) - self._half()[None, :]
        outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
        inside = np.minimum(np.max(d, axis=1), 0.0)
        return outside + inside

    def sample_surface(self, n: int, rng: np.random.Generator) -> np.ndarray:
        h = self._half()
        areas = np.array([h[1] * h[2], h[0] * h[2], h[0] * h[1]]) * 4.0
        face = rng.choice(3, size=n, p=areas / areas.sum())
        sign = rng.choice([-1.0, 1.0], size=n)
        out = rng.uniform(-1.0, 1.0, (n, 3)) * h[None, :]
        out[np.arange(n), face] = sign * h[face]
        return out + np.asarray(self.centre, float)[None, :]

    def bounds(self):
        c = np.asarray(self.centre, float)
        h = self._half()
        return c - h, c + h


# --------------------------------------------------------------------------
# construction from the config schema
# --------------------------------------------------------------------------


KINDS = {"wire": Wire, "strip": Strip, "hole": Hole, "pad": Pad}


def make(kind: str, ndim: int, centre, shape_cfg, angle: float = 0.0) -> Shape:
    """Build a :class:`Shape` from a config ``electrodes[].shape`` block."""
    if kind not in KINDS:
        raise ValueError(f"unknown electrode kind {kind!r}; "
                         f"known kinds are {sorted(KINDS)}")
    centre = tuple(np.asarray(centre, float).tolist())
    if kind == "wire":
        if not shape_cfg.radius:
            raise ValueError("a wire needs shape.radius")
        return Wire(ndim=ndim, centre=centre, radius=shape_cfg.radius, angle=angle)
    if kind == "strip":
        if not shape_cfg.width:
            raise ValueError("a strip needs shape.width")
        return Strip(ndim=ndim, centre=centre, width=shape_cfg.width,
                     thickness=shape_cfg.thickness or 0.035, angle=angle)
    if kind == "hole":
        if not shape_cfg.radius:
            raise ValueError("a hole needs shape.radius")
        return Hole(ndim=ndim, centre=centre, radius=shape_cfg.radius,
                    thickness=shape_cfg.thickness or 0.035)
    if not (shape_cfg.width and shape_cfg.length):
        raise ValueError("a pad needs shape.width and shape.length")
    return Pad(ndim=ndim, centre=centre, width=shape_cfg.width,
               length=shape_cfg.length, thickness=shape_cfg.thickness or 0.035)


def lattice_shapes(electrode, ndim: int = 3) -> list[Shape]:
    """Every :class:`Shape` of one electrode's lattice.

    The lattice runs *across* a wire or strip (its own axis is unbounded, so
    only the transverse offset means anything) and in two directions for a pad
    or a hole array.  ``lattice.stagger`` offsets alternate rows by half the
    second pitch, which turns a rectangular array into a triangular
    close-packed one.
    """
    from .geometry import lattice_positions

    lat = electrode.lattice
    xs = lattice_positions(lat.pitch, lat.count, lat.offset)
    if lat.pitch_z and lat.count_z:
        zs = lattice_positions(lat.pitch_z, lat.count_z, lat.offset_z)
    else:
        zs = np.array([0.0])
    a = math.radians(lat.angle)
    out = []
    for ix, x in enumerate(xs):
        dz = 0.5 * lat.pitch_z if (lat.stagger and ix % 2) else 0.0
        for z0 in zs:
            if electrode.kind in ("wire", "strip"):
                # an offset along the electrode's own axis is meaningless
                centre = (x * math.cos(a), electrode.plane, -x * math.sin(a))
            else:
                centre = (x, electrode.plane, z0 + dz)
            out.append(make(electrode.kind, ndim, centre, electrode.shape,
                            lat.angle))
    return out


def describe(cfg) -> str:
    """A text summary of an electrode layout, solvable or not."""
    lines = [f"{cfg.name}: {cfg.domain.dimension}D"]
    b = cfg.domain.bounds
    lines.append(f"  x in [{b.x[0]}, {b.x[1]}] mm" + (
        f"   z in [{b.z[0]}, {b.z[1]}] mm" if b.z else ""))
    lines.append(f"  y in [{b.y[0]}, {b.y[1]}] mm")
    for e in cfg.electrodes_by_y:
        shp = lattice_shapes(e, cfg.domain.dimension)
        lat = e.lattice
        extra = f" x {lat.count_z}" if lat.count_z else ""
        lines.append(
            f"  {e.name:<8} {e.kind:<6} {e.role:<11} y = {e.plane:7.3f} mm   "
            f"{lat.count}{extra} at pitch {lat.pitch}"
            + (f" x {lat.pitch_z}" if lat.pitch_z else "")
            + (f", {lat.angle:+g} deg" if lat.angle else "")
            + (", staggered" if lat.stagger else "")
            + f"   ({len(shp)} shapes)")
    return "\n".join(lines)
