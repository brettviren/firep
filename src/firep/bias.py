"""Electrode bias voltages from the grid-transparency criterion.

Geometry convention
-------------------
``y`` increases toward the cathode; electrons drift toward *decreasing* ``y``.
Therefore the drift field points along **+y** (electrons feel ``-eE``), and the
potential *decreases* with increasing ``y``: the cathode is the most negative
electrode.

Transparency
------------
For an infinite grid of wires of radius ``r`` and pitch ``p`` separating a
region of (upward) field ``E_above`` from one of field ``E_below``, Bunemann,
Cranshaw and Harvey (Can. J. Res. A **27** (1949) 191) give the condition for
*no* field line to terminate on the grid -- i.e. full transparency -- as

.. math::

    \\frac{E_\\mathrm{below}}{E_\\mathrm{above}} \\;\\ge\\;
        \\frac{1 + \\rho}{1 - \\rho}, \\qquad \\rho = \\frac{2\\pi r}{p}.

The last (collection) plane must instead be *opaque*: every field line arriving
from above has to end on a wire.  That is guaranteed when the field immediately
below the collection plane is reversed, which here means holding the collection
wires *above* the ground-plane potential.  The reversal magnitude is set by
``bias.collection_reverse_ratio`` as a multiple of the field in the last
induction gap.

Wire potential vs. plane-averaged potential
-------------------------------------------
The fields in the criterion above are the **transverse averages** ``<E_y>``.
Averaging Laplace's equation across one period kills the ``d^2/dx^2`` term, so
``<V>`` is piecewise linear in ``y`` and ``<E_y>`` is piecewise constant, with a
kink at each wire plane.

The catch is that a wire's own potential is *not* the plane-averaged potential
at its height.  A row of line charges of strength ``gamma`` (in volts, i.e.
``lambda / 2 pi eps0``) contributes ``gamma * ln(2)/2`` to the plane average but
``gamma * (ln(p / 2 pi r) + ln(2)/2)`` at its own surface, so

.. math::

    V_\\mathrm{wire} = \\langle V \\rangle(y_i)
        + \\gamma_i \\, \\Lambda_i, \\qquad
    \\Lambda_i = \\ln\\frac{p_i}{2\\pi r_i},

and Gauss's law across the row fixes the charge from the field it deflects:

.. math::

    \\gamma_i = \\frac{p_i}{2\\pi}
        \\left( E_\\mathrm{above} - E_\\mathrm{below} \\right).

Ignoring ``Lambda`` is not a small error.  For the default model the collection
row carries ``gamma = 144 V`` and ``Lambda = 2.36``, so its bias is **340 V**
away from its plane-averaged potential -- enough to invert the ordering of the
gap fields and destroy transparency.  Both directions of this relation are
implemented here: ``mode: transparency`` goes forward (pick the fields, get the
voltages) and ``mode: explicit`` goes backward (given the voltages, solve the
linear system for the fields).

Anchoring
---------
The transparency cascade multiplies the field up by ``(1+rho)/(1-rho)`` at each
plane, so the cathode voltage, the drift field and the wire biases cannot all be
chosen independently.  ``bias.anchor`` selects what is held fixed:

``cathode_uniform`` (default)
    The cathode potential is the one that *would* give the nominal
    ``bias.drift_field`` uniformly across the whole cathode-to-ground gap if the
    wires were not there -- a literal reading of "a nominal 500 V/cm field would
    exist if not for the wires".  The realised drift field in the cathode-to-
    first-plane region then comes out somewhat higher; it is reported as
    ``drift_field_actual``.

``drift_region``
    The cathode-to-first-plane field is exactly ``bias.drift_field`` and the
    cathode potential is whatever that requires.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import Config, ConfigError

V_PER_CM_TO_V_PER_MM = 0.1
TWO_PI = 2.0 * math.pi


@dataclass
class Gap:
    """One field region between two conductors, named from top to bottom.

    ``field`` is the transverse-averaged ``E_y``, signed so that positive points
    at the cathode (and so drifts electrons downward).
    """

    name: str
    y_hi: float
    y_lo: float
    field: float  # V/cm

    @property
    def length(self) -> float:
        return self.y_hi - self.y_lo


@dataclass
class BiasSolution:
    cathode: float  # V
    ground: float  # V
    electrodes: dict[str, float] = field(default_factory=dict)  # name -> wire V
    plane_average: dict[str, float] = field(default_factory=dict)  # name -> <V>(y_i)
    gamma: dict[str, float] = field(default_factory=dict)  # name -> lambda/2 pi eps0 [V]
    lam: dict[str, float] = field(default_factory=dict)  # name -> ln(p / 2 pi r)
    gaps: list[Gap] = field(default_factory=list)
    ratios: dict[str, float] = field(default_factory=dict)  # required E ratio per plane
    drift_field_nominal: float = 0.0  # V/cm
    drift_field_actual: float = 0.0  # V/cm, in the cathode-to-first-plane gap

    @property
    def span(self) -> float:
        """Full potential span of the problem, in volts."""
        vals = [self.cathode, self.ground, *self.electrodes.values()]
        return max(vals) - min(vals)

    def offset(self, name: str) -> float:
        """``V_wire - <V>``, the wire self-potential."""
        return self.electrodes[name] - self.plane_average[name]

    def to_dict(self) -> dict:
        d = {k: v for k, v in vars(self).items() if k != "gaps"}
        d["gaps"] = [vars(g) for g in self.gaps]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "BiasSolution":
        data = dict(data)
        gaps = [Gap(**g) for g in data.pop("gaps", [])]
        return cls(gaps=gaps, **data)


def transparency_ratio(radius: float, pitch: float, margin: float = 1.0) -> float:
    """``margin * (1 + rho) / (1 - rho)`` with ``rho = 2 pi r / p``."""
    rho = TWO_PI * radius / pitch
    if rho >= 1.0:
        raise ConfigError(
            f"wire radius {radius} at pitch {pitch} gives rho = {rho:.3f} >= 1; "
            "the wires nearly touch and the transparency criterion breaks down"
        )
    return margin * (1.0 + rho) / (1.0 - rho)


def self_potential(radius: float, pitch: float) -> float:
    """``Lambda = ln(p / 2 pi r)``, the wire potential above the plane average."""
    lam = math.log(pitch / (TWO_PI * radius))
    if lam <= 0:
        raise ConfigError(
            f"wire radius {radius} is too large for pitch {pitch}: the wires "
            "overlap once their images are included"
        )
    return lam


# --------------------------------------------------------------------------


def _layout(cfg: Config):
    """Plane list (cathode side first), gap lengths, pitches, radii, Lambdas."""
    planes = cfg.electrodes_by_y
    n = len(planes)
    ylo, yhi = cfg.domain.bounds.y
    lengths = [yhi - planes[0].plane]
    lengths += [planes[i - 1].plane - planes[i].plane for i in range(1, n)]
    lengths.append(planes[-1].plane - ylo)
    if min(lengths) <= 0:
        raise ConfigError(f"non-positive gap length among {lengths}")
    lam = [self_potential(e.shape.radius, e.lattice.pitch) for e in planes]
    return planes, lengths, lam


def _assemble(cfg: Config, fields: np.ndarray, cathode_v: float, ground_v: float,
              ratios: dict[str, float]) -> BiasSolution:
    """Turn a set of plane-averaged gap fields (V/mm) into a full solution."""
    planes, lengths, lam = _layout(cfg)
    n = len(planes)
    ylo, yhi = cfg.domain.bounds.y

    avg, gamma, wire = {}, {}, {}
    running = cathode_v
    for i, e in enumerate(planes):
        running += fields[i] * lengths[i]  # marching downward raises <V>
        avg[e.name] = running
        gamma[e.name] = e.lattice.pitch * (fields[i] - fields[i + 1]) / TWO_PI
        wire[e.name] = running + gamma[e.name] * lam[i]

    tops = [("cathode", yhi)] + [(e.name, e.plane) for e in planes] + [("ground", ylo)]
    gaps = [
        Gap(name=f"{a[0]}-{b[0]}", y_hi=a[1], y_lo=b[1],
            field=fields[j] / V_PER_CM_TO_V_PER_MM)
        for j, (a, b) in enumerate(zip(tops, tops[1:]))
    ]
    return BiasSolution(
        cathode=cathode_v,
        ground=ground_v,
        electrodes={e.name: wire[e.name] for e in cfg.electrodes},
        plane_average={e.name: avg[e.name] for e in cfg.electrodes},
        gamma={e.name: gamma[e.name] for e in cfg.electrodes},
        lam={e.name: lam[planes.index(e)] for e in cfg.electrodes},
        gaps=gaps,
        ratios=ratios,
        drift_field_nominal=cfg.bias.drift_field,
        drift_field_actual=fields[0] / V_PER_CM_TO_V_PER_MM,
    )


def solve(cfg: Config) -> BiasSolution:
    """Compute cathode and electrode potentials for ``cfg``."""
    cfg.require_solvable()
    if cfg.bias.mode == "explicit":
        return _explicit(cfg)
    return _transparency(cfg)


def _transparency(cfg: Config) -> BiasSolution:
    """Forward: choose the gap fields from the cascade, then get the voltages."""
    planes, lengths, lam = _layout(cfg)
    n = len(planes)
    ylo, yhi = cfg.domain.bounds.y
    ground_v = cfg.domain.boundaries.ylo.potential
    if ground_v == "auto":
        raise ConfigError("domain.boundaries.ylo.potential must be a number (the reference)")
    ground_v = float(ground_v)

    ratios, gains = {}, [1.0]
    for i in range(n - 1):
        e = planes[i]
        k = transparency_ratio(e.shape.radius, e.lattice.pitch, cfg.bias.margin)
        ratios[e.name] = k
        gains.append(gains[-1] * k)
    rev = cfg.bias.collection_reverse_ratio

    # <V> connects the two faces directly, so the anchor arithmetic involves
    # only the gap fields and lengths -- the wire self-potentials cancel.
    s = sum(gains[i] * lengths[i] for i in range(n)) - rev * gains[n - 1] * lengths[n]
    if s <= 0:
        raise ConfigError(
            "bias.collection_reverse_ratio is too large: the reversed gap below the "
            "collection plane would swallow the whole cathode-to-ground potential "
            f"drop (S = {s:.4g} mm)"
        )

    f_nom = cfg.bias.drift_field * V_PER_CM_TO_V_PER_MM  # V/mm
    if cfg.bias.anchor == "cathode_uniform":
        drop = f_nom * (yhi - ylo)  # V_ground - V_cathode
        e0 = drop / s
    else:  # "drift_region"
        e0 = f_nom
        drop = e0 * s
    cathode_v = ground_v - drop

    fields = np.array([g * e0 for g in gains] + [-rev * gains[n - 1] * e0])
    return _assemble(cfg, fields, cathode_v, ground_v, ratios)


def _explicit(cfg: Config) -> BiasSolution:
    """Inverse: given the wire voltages, solve for the plane-averaged fields.

    The unknowns are the ``n+1`` gap fields.  Each wire contributes

        <V>(y_i) + gamma_i Lambda_i = V_i
        => sum_{j<=i} d_j E_j + (Lambda_i p_i / 2 pi)(E_i - E_{i+1}) = V_i - V_cath

    and the two faces contribute ``sum_j d_j E_j = V_ground - V_cathode``.
    """
    yhi_face, ylo_face = cfg.domain.boundaries.yhi, cfg.domain.boundaries.ylo
    for face, name in ((yhi_face, "yhi"), (ylo_face, "ylo")):
        if face.potential == "auto":
            raise ConfigError(
                f"bias.mode is 'explicit' but domain.boundaries.{name}.potential is 'auto'"
            )
    cathode_v, ground_v = float(yhi_face.potential), float(ylo_face.potential)

    planes, lengths, lam = _layout(cfg)
    n = len(planes)
    a = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    for i, e in enumerate(planes):
        if e.bias == "auto":
            raise ConfigError(
                f"bias.mode is 'explicit' but electrode {e.name!r} has bias 'auto'"
            )
        a[i, : i + 1] += lengths[: i + 1]
        w = lam[i] * e.lattice.pitch / TWO_PI
        a[i, i] += w
        a[i, i + 1] -= w
        b[i] = float(e.bias) - cathode_v
    a[n, :] = lengths
    b[n] = ground_v - cathode_v

    try:
        fields = np.linalg.solve(a, b)
    except np.linalg.LinAlgError as err:  # pragma: no cover - degenerate geometry
        raise ConfigError(f"cannot solve for the gap fields: {err}") from err
    return _assemble(cfg, fields, cathode_v, ground_v, {})


# --------------------------------------------------------------------------


def report(cfg: Config, sol: BiasSolution) -> str:
    """Human-readable summary table."""
    lines = [f"problem: {cfg.problem.kind}   bias mode: {cfg.bias.mode}"]
    if cfg.bias.mode == "transparency":
        lines.append(
            f"anchor: {cfg.bias.anchor}   margin: {cfg.bias.margin}   "
            f"collection_reverse_ratio: {cfg.bias.collection_reverse_ratio}"
        )
    lines += ["", "conductor potentials [V]"]
    lines.append(f"  {'cathode':<10} y = {cfg.domain.bounds.y[1]:8.3f} mm  {sol.cathode:12.2f}")
    for e in cfg.electrodes_by_y:
        lines.append(
            f"  {e.name:<10} y = {e.plane:8.3f} mm  {sol.electrodes[e.name]:12.2f}"
            f"   ({e.role}, {e.lattice.count} x {e.kind}, "
            f"pitch {e.lattice.pitch} mm, r {e.shape.radius} mm)"
        )
    lines.append(f"  {'ground':<10} y = {cfg.domain.bounds.y[0]:8.3f} mm  {sol.ground:12.2f}")

    lines += ["", "wire self-potential: V_wire = <V> + gamma * Lambda"]
    lines.append(f"  {'plane':<10}{'<V> [V]':>12}{'gamma [V]':>12}{'Lambda':>10}"
                 f"{'offset [V]':>12}")
    for e in cfg.electrodes_by_y:
        lines.append(
            f"  {e.name:<10}{sol.plane_average[e.name]:12.2f}{sol.gamma[e.name]:12.3f}"
            f"{sol.lam[e.name]:10.4f}{sol.offset(e.name):12.2f}"
        )

    lines += ["", "transverse-averaged gap fields [V/cm]  "
                  "(positive = points at the cathode, drifts e- down)"]
    for g in sol.gaps:
        lines.append(f"  {g.name:<18}{g.length:8.3f} mm  {g.field:11.2f}")

    fields_ = [g.field for g in sol.gaps]
    lines += ["", "transparency: required E_below/E_above, and what was achieved"]
    for i, e in enumerate(cfg.electrodes_by_y[:-1]):
        need = sol.ratios.get(
            e.name, transparency_ratio(e.shape.radius, e.lattice.pitch, cfg.bias.margin)
        )
        got = fields_[i + 1] / fields_[i] if fields_[i] else float("nan")
        ok = "ok" if got >= need - 1e-9 else "FAIL"
        lines.append(f"  {e.name:<10} need >= {need:7.4f}   got {got:7.4f}   {ok}")
    last = cfg.electrodes_by_y[-1]
    below = fields_[-1]
    ok = "ok (reversed, fully opaque)" if below < 0 else "FAIL (field still points up)"
    lines.append(f"  {last.name:<10} collection: field below = {below:8.2f} V/cm  {ok}")

    lines += ["", f"drift field: nominal {sol.drift_field_nominal:.2f} V/cm, "
                  f"realised {sol.drift_field_actual:.2f} V/cm in the cathode gap"]
    return "\n".join(lines)
