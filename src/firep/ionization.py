"""Post-drift ionization: Gaussian electron groups arriving at the response plane.

A field response starts from a single electron launched at a *logical plane*
(``response.y_start``).  Real ionization arrives there as a cloud: a point-like
deposit that has drifted a distance ``L`` spreads by diffusion into a
three-dimensional Gaussian and loses electrons to electronegative impurities.
This module represents that cloud and nothing else -- it is the input to the
field response, not part of it.

Geometry
--------
``x`` is transverse (across the wire pitch), ``y`` is the drift direction and
``z`` runs along the wires.  A group therefore has three widths:

* ``sigma_x`` -- transverse, across the pitch.  This is what smears the signal
  over neighbouring wires.
* ``sigma_z`` -- transverse, along the wire.  In a 2D model the wires are
  infinite in ``z``, so this integrates out and does not affect any waveform.
  It is carried anyway, because it is physically part of the cloud and a 3D
  model would need it.
* ``sigma_t`` -- longitudinal, expressed as an arrival-time width;
  ``sigma_y = v sigma_t`` is the same thing as a length.

Diffusion and absorption
------------------------
For a drift time ``t = L / v``,

.. math::

    \\sigma_L = \\sqrt{2 D_L t}, \\qquad \\sigma_T = \\sqrt{2 D_T t},
    \\qquad N = N_0 \\, e^{-t/\\tau},

with ``D_L``, ``D_T`` the longitudinal and transverse diffusion coefficients and
``tau`` the electron lifetime.  The defaults are 6.63 cm^2/s (Li et al.,
Phys. Rev. D **94** (2016) 082004, at 500 V/cm), 13.0 cm^2/s, and 3 ms.  All are
field- and purity-dependent and all are configurable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CM2_PER_S_TO_MM2_PER_US = 1e2 / 1e6  # cm^2/s -> mm^2/us


@dataclass
class ArgonProperties:
    """Transport properties of the liquid argon between deposit and readout."""

    drift_speed: float = 1.6281  # mm/us, at the nominal drift field
    d_long: float = 6.63  # cm^2/s
    d_tran: float = 13.0  # cm^2/s
    lifetime: float = 3000.0  # us, electron lifetime (inf to disable absorption)

    def drift_time(self, distance: float | np.ndarray) -> float | np.ndarray:
        return np.asarray(distance) / self.drift_speed

    def spread(self, distance: float | np.ndarray):
        """``(sigma_long_mm, sigma_tran_mm, sigma_time_us)`` after drifting."""
        t = self.drift_time(distance)
        s_l = np.sqrt(2.0 * self.d_long * CM2_PER_S_TO_MM2_PER_US * t)
        s_t = np.sqrt(2.0 * self.d_tran * CM2_PER_S_TO_MM2_PER_US * t)
        return s_l, s_t, s_l / self.drift_speed

    def survival(self, distance: float | np.ndarray):
        """Fraction of electrons surviving attachment over the drift."""
        if not np.isfinite(self.lifetime) or self.lifetime <= 0:
            return np.ones_like(np.asarray(distance, dtype=float))
        return np.exp(-self.drift_time(distance) / self.lifetime)


@dataclass
class Groups:
    """A set of 3D Gaussian electron groups at the response plane.

    Stored column-wise so that a track is one object, not thousands.  All arrays
    have the same length; ``n`` is a number of electrons, not a charge.
    """

    n: np.ndarray  # electrons
    x: np.ndarray  # mm, transverse centre
    z: np.ndarray  # mm, along-wire centre
    t: np.ndarray  # us, arrival-time centre at the response plane
    sigma_x: np.ndarray  # mm
    sigma_z: np.ndarray  # mm
    sigma_t: np.ndarray  # us
    t_offset: float = 0.0  # us subtracted from every t, for reporting

    def __post_init__(self):
        arrays = [self.n, self.x, self.z, self.t,
                  self.sigma_x, self.sigma_z, self.sigma_t]
        lengths = {len(np.atleast_1d(a)) for a in arrays}
        if len(lengths) != 1:
            raise ValueError(f"Groups: mismatched array lengths {lengths}")
        for name in ("n", "x", "z", "t", "sigma_x", "sigma_z", "sigma_t"):
            setattr(self, name, np.atleast_1d(np.asarray(getattr(self, name), float)))

    def __len__(self) -> int:
        return len(self.n)

    @property
    def total(self) -> float:
        return float(self.n.sum())

    def describe(self) -> str:
        return "\n".join([
            f"groups: {len(self)}   electrons: {self.total:,.0f}",
            f"  x      [{self.x.min():8.3f}, {self.x.max():8.3f}] mm",
            f"  t      [{self.t.min():8.3f}, {self.t.max():8.3f}] us "
            f"(offset {self.t_offset:.3f} us removed)",
            f"  sigma_x  {self.sigma_x.min():.4f} .. {self.sigma_x.max():.4f} mm",
            f"  sigma_t  {self.sigma_t.min():.4f} .. {self.sigma_t.max():.4f} us "
            f"(= {self.sigma_t.min() * 1.6281:.4f} .. "
            f"{self.sigma_t.max() * 1.6281:.4f} mm)",
        ])


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------


def from_points(x: np.ndarray, dy: np.ndarray, z: np.ndarray, n: np.ndarray,
                drift: float, props: ArgonProperties,
                recentre: bool = True) -> Groups:
    """Drift point-like deposits to the response plane.

    ``dy`` is the extra distance beyond ``drift`` from the response plane, so a
    deposit at ``dy > 0`` is further away: it drifts longer, arrives later,
    diffuses more and loses more charge.
    """
    x = np.atleast_1d(np.asarray(x, float))
    dy = np.atleast_1d(np.asarray(dy, float))
    z = np.atleast_1d(np.asarray(z, float))
    n = np.atleast_1d(np.asarray(n, float))
    distance = drift + dy
    if np.any(distance <= 0):
        raise ValueError("every deposit must be upstream of the response plane")

    s_l, s_t, s_time = props.spread(distance)
    survive = props.survival(distance)
    t = props.drift_time(distance)
    offset = float(np.mean(t)) if recentre else 0.0
    return Groups(n=n * survive, x=x, z=z, t=t - offset,
                  sigma_x=s_t, sigma_z=s_t, sigma_t=s_time, t_offset=offset)


def line_source(drift: float, props: ArgonProperties, length: float = 60.0,
                angle: float = 45.0, x_centre: float = 0.0, z_centre: float = 0.0,
                per_mm: float = 5000.0, step: float = 0.1) -> Groups:
    """An ideal straight ionization track in the drift (x, y) plane.

    ``angle`` is measured from the transverse axis: 0 degrees puts the track
    parallel to the wire plane (all charge arrives at once, spread across
    wires), 90 degrees puts it along the drift (all charge on one wire, spread
    in time).  ``per_mm`` electrons per mm are sampled every ``step`` mm, so
    each sample carries ``per_mm * step`` electrons before absorption.
    """
    if length <= 0 or step <= 0:
        raise ValueError("line_source needs a positive length and step")
    nsteps = max(int(round(length / step)), 1)
    s = (np.arange(nsteps) + 0.5) * (length / nsteps) - 0.5 * length
    th = math.radians(angle)
    x = x_centre + s * math.cos(th)
    dy = s * math.sin(th)
    z = np.full(nsteps, z_centre)
    n = np.full(nsteps, per_mm * (length / nsteps))
    return from_points(x, dy, z, n, drift, props)


# --------------------------------------------------------------------------
# depositing groups onto a grid
# --------------------------------------------------------------------------


def _erf(x: np.ndarray) -> np.ndarray:
    """Vectorised error function.

    ``math.erf`` is scalar-only and SciPy is not a dependency, but torch already
    is -- and ``torch.erf`` is vectorised and fast.
    """
    import torch

    return torch.erf(torch.as_tensor(np.asarray(x, dtype=np.float64))).numpy()


def bin_fractions(centres: np.ndarray, sigma: np.ndarray, edges: np.ndarray,
                  n_sigma: float = 3.0) -> np.ndarray:
    """Fraction of each truncated Gaussian in each bin; shape ``(ngroups, nbins)``.

    The Gaussian is cut at ``n_sigma`` and the surviving weight renormalised to
    one, so truncation loses no charge -- it only reshapes the tails.
    """
    lo, hi = edges[:-1], edges[1:]
    s = np.maximum(np.atleast_1d(sigma).astype(float), 1e-12)[:, None]
    c = np.atleast_1d(centres).astype(float)[:, None]
    cut = n_sigma * s
    clo = np.clip(lo[None, :], c - cut, c + cut)
    chi = np.clip(hi[None, :], c - cut, c + cut)
    root2 = math.sqrt(2.0)
    frac = 0.5 * (_erf((chi - c) / (s * root2)) - _erf((clo - c) / (s * root2)))
    norm = _erf(n_sigma / root2)  # total weight kept by the truncation
    return frac / norm


def arrival_grid(groups: Groups, x_edges: np.ndarray, t_edges: np.ndarray,
                 n_sigma: float = 3.0, chunk: int = 256) -> np.ndarray:
    """Electrons arriving in each (transverse, time) bin; shape ``(nx, nt)``.

    This is the object that is convolved with the field response: every group
    has been reduced to a density on one common grid, so the cost of the
    response convolution no longer depends on how many groups there were.
    """
    out = np.zeros((len(x_edges) - 1, len(t_edges) - 1))
    for i in range(0, len(groups), chunk):
        sl = slice(i, i + chunk)
        fx = bin_fractions(groups.x[sl], groups.sigma_x[sl], x_edges, n_sigma)
        ft = bin_fractions(groups.t[sl], groups.sigma_t[sl], t_edges, n_sigma)
        out += np.einsum("gi,gj,g->ij", fx, ft, groups.n[sl], optimize=True)
    return out
