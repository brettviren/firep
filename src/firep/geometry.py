"""Concrete geometry: electrodes expanded into individual conductors, and the
point sampling that feeds the PINN residual and boundary losses.

Everything here is NumPy and framework-free; :mod:`firep.train` moves the
sampled points onto the torch device.

The only electrode ``kind`` implemented is ``wire``, a circle of radius ``r`` in
the x-y plane.  The sampling interface (a signed distance and a surface
sampler per conductor) is what a strip / hole / pad implementation would have to
provide, so adding one is local to this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bias import BiasSolution
from .config import Config


@dataclass(frozen=True)
class Conductor:
    """One individual electrode instance (one wire) with its potential."""

    electrode: str  # parent electrode name, e.g. "u"
    index: int  # position within the lattice, 0-based
    kind: str
    x: float
    y: float
    radius: float
    potential: float

    @property
    def label(self) -> str:
        return f"{self.electrode}{self.index}"


def lattice_positions(pitch: float, count: int, offset: float) -> np.ndarray:
    """Transverse centres of a ``count``-fold lattice centred on ``offset``."""
    return offset + (np.arange(count) - 0.5 * (count - 1)) * pitch


class Geometry:
    """The domain box plus the conductors inside it."""

    def __init__(self, cfg: Config, sol: BiasSolution):
        self.cfg = cfg
        self.sol = sol
        self.xlo, self.xhi = cfg.domain.bounds.x
        self.ylo, self.yhi = cfg.domain.bounds.y
        self.width = self.xhi - self.xlo
        self.height = self.yhi - self.ylo

        weighting = cfg.problem.kind == "weighting"
        target = _weighting_target(cfg) if weighting else None

        conductors: list[Conductor] = []
        for e in cfg.electrodes_by_y:
            xs = lattice_positions(e.lattice.pitch, e.lattice.count, e.lattice.offset)
            for i, x in enumerate(xs):
                if weighting:
                    pot = cfg.problem.weighting.unit if (e.name, i) == target else 0.0
                else:
                    pot = sol.electrodes[e.name]
                conductors.append(
                    Conductor(
                        electrode=e.name,
                        index=i,
                        kind=e.kind,
                        x=float(x),
                        y=float(e.plane),
                        radius=float(e.shape.radius),
                        potential=float(pot),
                    )
                )
        self.conductors = conductors

        # Face potentials for the problem actually being solved.
        if weighting:
            self.v_cathode = 0.0
            self.v_ground = 0.0
        else:
            self.v_cathode = sol.cathode
            self.v_ground = sol.ground

        self.cx = np.array([c.x for c in conductors])
        self.cy = np.array([c.y for c in conductors])
        self.cr = np.array([c.radius for c in conductors])
        self.cv = np.array([c.potential for c in conductors])

    # -- queries ------------------------------------------------------------

    @property
    def potential_span(self) -> float:
        vals = [self.v_cathode, self.v_ground, *self.cv.tolist()]
        span = max(vals) - min(vals)
        return span if span > 0 else 1.0

    def wrap_x(self, x: np.ndarray) -> np.ndarray:
        """Fold x back into the periodic domain."""
        return self.xlo + np.mod(x - self.xlo, self.width)

    def min_gap_distance(self, xy: np.ndarray) -> np.ndarray:
        """Distance to the nearest conductor surface, negative inside.

        Distances are measured with the transverse coordinate taken modulo the
        domain period, so a conductor near one edge is correctly "close" to
        points near the other.
        """
        dx = xy[:, None, 0] - self.cx[None, :]
        dx = dx - self.width * np.round(dx / self.width)
        dy = xy[:, None, 1] - self.cy[None, :]
        return np.min(np.hypot(dx, dy) - self.cr[None, :], axis=1)

    def inside(self, xy: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Boolean mask of points inside (or within ``margin`` of) a conductor."""
        return self.min_gap_distance(xy) < margin

    def describe(self) -> str:
        lines = [
            f"domain: x in [{self.xlo}, {self.xhi}] mm (periodic, period {self.width} mm)",
            f"        y in [{self.ylo}, {self.yhi}] mm",
            f"faces:  cathode (y={self.yhi}) = {self.v_cathode:.3f} V, "
            f"ground (y={self.ylo}) = {self.v_ground:.3f} V",
            f"conductors: {len(self.conductors)}",
        ]
        for e in self.cfg.electrodes_by_y:
            mine = [c for c in self.conductors if c.electrode == e.name]
            pots = sorted({round(c.potential, 6) for c in mine})
            xs = [c.x for c in mine]
            lines.append(
                f"  {e.name:<6} y={e.plane:<8.3f} n={len(mine):<4d} r={e.shape.radius} mm "
                f"x in [{min(xs):.3f}, {max(xs):.3f}] "
                f"V = {pots if len(pots) <= 3 else f'{len(pots)} distinct values'}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


class Sampler:
    """Draws collocation and boundary points for one training step."""

    def __init__(self, geom: Geometry, seed: int, near_factor: float = 20.0,
                 surface_margin: float = 1e-3):
        self.geom = geom
        self.rng = np.random.default_rng(seed)
        self.near_factor = near_factor
        # Keep collocation points a hair off the conductor surfaces so that the
        # log-singular baseline terms stay well conditioned.
        self.surface_margin = surface_margin

    def bulk(self, n: int) -> np.ndarray:
        """Uniform points in the domain, excluding conductor interiors."""
        g = self.geom
        out = np.empty((0, 2))
        for _ in range(32):
            need = n - len(out)
            if need <= 0:
                break
            draw = min(max(int(need * 1.2), 64), 4 * n + 1024)
            xy = np.column_stack(
                (
                    self.rng.uniform(g.xlo, g.xhi, draw),
                    self.rng.uniform(g.ylo, g.yhi, draw),
                )
            )
            keep = xy[~g.inside(xy, self.surface_margin)]
            out = np.vstack((out, keep))
        return out[:n]

    def near(self, n: int) -> np.ndarray:
        """Points in log-radial annuli around each conductor.

        The potential near a wire varies as ``ln r``, so sampling uniformly in
        ``ln r`` puts equal weight on each decade of approach -- exactly where a
        uniformly sampled PINN starves.
        """
        g = self.geom
        nc = len(g.conductors)
        if nc == 0 or n <= 0:
            return np.empty((0, 2))
        which = self.rng.integers(0, nc, n)
        r_in = g.cr[which] * (1.0 + self.surface_margin)
        r_out = g.cr[which] * self.near_factor
        u = self.rng.uniform(0.0, 1.0, n)
        r = r_in * (r_out / r_in) ** u
        th = self.rng.uniform(0.0, 2.0 * np.pi, n)
        xy = np.column_stack((g.cx[which] + r * np.cos(th), g.cy[which] + r * np.sin(th)))
        xy[:, 0] = g.wrap_x(xy[:, 0])
        # Annuli around the outermost planes can poke through a face.
        keep = (xy[:, 1] > g.ylo) & (xy[:, 1] < g.yhi)
        xy = xy[keep]
        # And they can poke into a *neighbouring* conductor.
        return xy[~g.inside(xy, self.surface_margin)]

    def face(self, n: int, y: float) -> np.ndarray:
        g = self.geom
        return np.column_stack((self.rng.uniform(g.xlo, g.xhi, n), np.full(n, y)))

    def surfaces(self, n_per: int) -> tuple[np.ndarray, np.ndarray]:
        """Points on every conductor surface and the potential each should have."""
        g = self.geom
        nc = len(g.conductors)
        if nc == 0 or n_per <= 0:
            return np.empty((0, 2)), np.empty((0,))
        which = np.repeat(np.arange(nc), n_per)
        th = self.rng.uniform(0.0, 2.0 * np.pi, nc * n_per)
        xy = np.column_stack(
            (g.cx[which] + g.cr[which] * np.cos(th), g.cy[which] + g.cr[which] * np.sin(th))
        )
        xy[:, 0] = g.wrap_x(xy[:, 0])
        return xy, g.cv[which]


def _weighting_target(cfg: Config) -> tuple[str, int]:
    """(electrode name, lattice index) of the wire held at unit weighting potential."""
    w = cfg.problem.weighting
    e = cfg.electrode(w.electrode)
    idx = w.index
    if idx is None:
        idx = (e.lattice.count - 1) // 2  # central wire
    if not 0 <= idx < e.lattice.count:
        raise ValueError(
            f"problem.weighting.index {idx} out of range for electrode "
            f"{e.name!r} with {e.lattice.count} wires"
        )
    return (e.name, int(idx))
