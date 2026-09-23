"""Field response: drift trajectories through the drift field, turned into
induced current on each electrode by the Shockley-Ramo theorem.

The combined model
------------------
:class:`CombinedModel` holds one *drift* solution (the real potential, with the
electrodes at their bias voltages) and one *weighting* solution per sensing
plane (every conductor grounded except the sensing wire, which is at unit
potential).  Neither alone is a field response; together with a drift velocity
they are.

Shockley-Ramo
-------------
The charge induced on electrode *k* by a carrier of charge ``q`` at position
``r`` is ``Q_k = -q phi_k(r)``, with ``phi_k`` the weighting potential.  The
current is its time derivative,

.. math::

    i_k(t) = \\frac{dQ_k}{dt} = -q\\,\\frac{d\\phi_k}{dt}
           = q\\, \\mathbf{v}\\cdot\\mathbf{E}_{w,k},
      \\qquad \\mathbf{E}_{w,k} = -\\nabla\\phi_k .

Both forms are computed here.  The **charge form** is the one reported: it is
exactly conservative, needs no gradient of the weighting potential, and its time
integral is fixed by the endpoints alone, so an electron that starts far away
(``phi = 0``) and lands on sensing wire *k* (``phi = 1``) must induce exactly
``-q = +e`` there and exactly zero on every other electrode, whatever the path
did in between.  The **field form** is computed independently as a cross-check;
:func:`response` reports how far apart they end up.

Translational symmetry
----------------------
The drift field is periodic over one pitch, so a trajectory computed in the
central cell also describes an electron in any other cell.  The response of the
wire ``n`` pitches away is therefore the same trajectory read against the
weighting potential shifted by ``n`` pitches -- which is why one 21-wire
weighting solve yields all 21 wire responses.

Drift velocity
--------------
Electrons move along ``-E`` at a speed set by the field.  The default is the
Walkowiak parameterisation (NIM A **449** (2000) 288), fitted over
0.5-12.6 kV/cm, which gives 1.628 mm/us at 500 V/cm and 87.3 K.  Fields at a
wire surface exceed that range, so :func:`response` reports the largest field
any trajectory saw.  Diffusion is deliberately absent: a field response is a
single-electron, drift-only quantity, and diffusion belongs downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .config import Config
from .geometry import Geometry
from .siren import FieldModel
from .train import DTYPES, resolve_device

MM_PER_CM = 10.0
E_CHARGE_C = 1.602176634e-19  # coulomb
# 1 e/us in amperes
E_PER_US_IN_A = E_CHARGE_C / 1e-6

# Walkowiak NIM A 449 (2000) 288, table 1.  E in kV/cm, T in K, v in mm/us.
WALKOWIAK = dict(T0=90.371, P1=-0.01481, P2=-0.0075, P3=0.141, P4=12.4,
                 P5=1.627, P6=0.317)
WALKOWIAK_E_RANGE = (0.5, 12.6)  # kV/cm, the fitted range


# --------------------------------------------------------------------------
# drift velocity
# --------------------------------------------------------------------------


def walkowiak_speed(e_kv_per_cm: np.ndarray, temperature: float) -> np.ndarray:
    """Electron drift speed in mm/us for a field in kV/cm."""
    w = WALKOWIAK
    dt = temperature - w["T0"]
    e = np.asarray(e_kv_per_cm, dtype=float)
    safe = np.maximum(e, 1e-12)
    body = w["P3"] * safe * np.log1p(w["P4"] / safe) + w["P5"] * safe ** w["P6"]
    v = (w["P1"] * dt + 1.0) * body + w["P2"] * dt
    # The P2 offset is a fit term, not a zero-field limit; force v(0) = 0.
    return np.where(e > 1e-9, np.maximum(v, 0.0), 0.0)


def constant_mobility_speed(e_kv_per_cm: np.ndarray, mobility: float) -> np.ndarray:
    """``v = mu E`` with ``mu`` in cm^2/(V s); returns mm/us."""
    # mu [cm^2/V/s] * E [kV/cm] = mu*1e3 [cm/s] -> *1e-1 mm/us... explicitly:
    # v[cm/s] = mu * E[V/cm] = mu * 1e3 * E[kV/cm]
    # v[mm/us] = v[cm/s] * 10 [mm/cm] / 1e6 [us/s]
    v_cm_s = mobility * 1e3 * np.asarray(e_kv_per_cm, dtype=float)
    return v_cm_s * 10.0 / 1e6


def drift_speed(e_v_per_mm: np.ndarray, model: str, temperature: float,
                mobility: float) -> np.ndarray:
    """Speed in mm/us for a field magnitude in V/mm."""
    e_kv_cm = np.asarray(e_v_per_mm, dtype=float) * MM_PER_CM / 1e3
    if model == "walkowiak":
        return walkowiak_speed(e_kv_cm, temperature)
    if model == "constant":
        return constant_mobility_speed(e_kv_cm, mobility)
    raise ValueError(f"unknown drift velocity model {model!r}")


# --------------------------------------------------------------------------
# the combined model
# --------------------------------------------------------------------------


@dataclass
class WeightingSolution:
    """One weighting solve, plus where its sensing wire is."""

    plane: str
    cfg: Config
    geom: Geometry
    model: FieldModel
    unit: float
    sense_x: float
    sense_y: float
    pitch: float


class CombinedModel:
    """Drift solution + one weighting solution per sensing plane.

    This is the "combined model": every quantity a field response needs, behind
    one object.  Potentials and fields come from the SIRENs by automatic
    differentiation; velocity and current come from the physics above.
    """

    def __init__(self, drift_cfg: Config, drift_geom: Geometry, drift_model: FieldModel,
                 weighting: list[WeightingSolution], velocity: str = "walkowiak",
                 temperature: float = 87.3, mobility: float = 320.0):
        self.cfg = drift_cfg
        self.geom = drift_geom
        self.model = drift_model
        self.weighting = {w.plane: w for w in weighting}
        self.velocity = velocity
        self.temperature = temperature
        self.mobility = mobility
        self.device = resolve_device(drift_cfg.train.device)
        self.dtype = DTYPES[drift_cfg.train.precision]
        self._check_consistency()

    # -- consistency --------------------------------------------------------

    def _check_consistency(self) -> None:
        """A weighting solve only pairs with a drift solve of the same lattice."""
        for w in self.weighting.values():
            if w.plane not in {e.name for e in self.cfg.electrodes}:
                raise ValueError(
                    f"weighting solution for plane {w.plane!r} has no matching "
                    f"electrode in the drift model"
                )
            de = self.cfg.electrode(w.plane)
            if not math.isclose(de.lattice.pitch, w.pitch, rel_tol=1e-9):
                raise ValueError(
                    f"plane {w.plane!r}: drift pitch {de.lattice.pitch} mm but "
                    f"weighting pitch {w.pitch} mm"
                )
            if not math.isclose(de.plane, w.sense_y, rel_tol=0, abs_tol=1e-9):
                raise ValueError(
                    f"plane {w.plane!r}: drift plane at y={de.plane} mm but "
                    f"weighting sensing wire at y={w.sense_y} mm"
                )
            if not math.isclose(de.shape.radius, w.geom.cr[0], rel_tol=1e-9):
                raise ValueError(
                    f"plane {w.plane!r}: wire radius differs between the drift "
                    f"({de.shape.radius} mm) and weighting ({w.geom.cr[0]} mm) models"
                )

    # -- field evaluation ---------------------------------------------------

    def _t(self, a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(a), device=self.device, dtype=self.dtype)

    def drift_field(self, xy: np.ndarray, chunk: int = 16384) -> np.ndarray:
        """``-grad V`` of the drift solution, V/mm, shape (n, 2).

        The transverse coordinate is folded into the periodic cell first, which
        is exact: the drift solution has that period.
        """
        pts = np.column_stack((self.geom.wrap_x(xy[:, 0]), xy[:, 1]))
        out = []
        for i in range(0, len(pts), chunk):
            blk = self._t(pts[i : i + chunk]).requires_grad_(True)
            v = self.model(blk)
            (g,) = torch.autograd.grad(v.sum(), blk)
            out.append(-g.detach().cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, 2))

    def weighting_potential(self, plane: str, xy: np.ndarray,
                            chunk: int = 16384) -> np.ndarray:
        """Dimensionless weighting potential of ``plane``'s sensing wire.

        ``xy`` are *unwrapped* physical positions: the weighting potential is
        periodic only over the whole 21-pitch domain, not over one pitch.
        """
        w = self.weighting[plane]
        out = []
        with torch.no_grad():
            for i in range(0, len(xy), chunk):
                out.append(w.model(self._t(xy[i : i + chunk]))[:, 0].cpu().numpy())
        return np.concatenate(out) / w.unit if out else np.empty(0)

    def weighting_field(self, plane: str, xy: np.ndarray,
                        chunk: int = 16384) -> np.ndarray:
        """``-grad phi`` of the weighting potential, 1/mm, shape (n, 2)."""
        w = self.weighting[plane]
        out = []
        for i in range(0, len(xy), chunk):
            blk = self._t(xy[i : i + chunk]).requires_grad_(True)
            v = w.model(blk)
            (g,) = torch.autograd.grad(v.sum(), blk)
            out.append(-g.detach().cpu().numpy() / w.unit)
        return np.concatenate(out) if out else np.empty((0, 2))

    def speed(self, e_v_per_mm: np.ndarray) -> np.ndarray:
        return drift_speed(e_v_per_mm, self.velocity, self.temperature, self.mobility)


# --------------------------------------------------------------------------
# trajectories
# --------------------------------------------------------------------------


@dataclass
class Trajectory:
    t: np.ndarray  # us
    xy: np.ndarray  # (nstep, nimpact, 2), unwrapped physical mm
    alive: np.ndarray  # (nstep, nimpact) bool
    landed: list[str]
    e_max: float = 0.0  # largest |E| seen, V/mm
    vel: np.ndarray | None = None  # (nstep, nimpact, 2) mm/us, from the field


def trajectories(cm: CombinedModel, x0: np.ndarray, y_start: float,
                 tick: float = 0.1, max_time: float = 200.0,
                 substeps: int = 4, max_steps: int = 200000) -> Trajectory:
    """Drift every impact position downward through the field, together.

    Integration is RK2 (midpoint) in *time* with a step capped both by a
    fraction of a tick -- so the later resampling onto ticks is well resolved --
    and by the time to cross a fraction of the distance to the nearest
    conductor, which is what keeps the last micron before a wire accurate.
    """
    n = len(x0)
    pos = np.column_stack((np.asarray(x0, dtype=float), np.full(n, float(y_start))))
    alive = np.ones(n, dtype=bool)
    landed = ["" for _ in range(n)]
    t = 0.0
    ts, xys, alives = [t], [pos.copy()], [alive.copy()]
    e_max = 0.0
    dt_cap = tick / max(1, substeps)

    def velocity(p):
        e = cm.drift_field(p)
        mag = np.hypot(e[:, 0], e[:, 1])
        sp = cm.speed(mag)
        with np.errstate(invalid="ignore", divide="ignore"):
            unit = np.where(mag[:, None] > 0, e / np.maximum(mag, 1e-30)[:, None], 0.0)
        return -sp[:, None] * unit, mag  # electrons move along -E

    for _ in range(max_steps):
        idx = np.flatnonzero(alive)
        if not len(idx) or t >= max_time:
            break
        p = pos[idx]
        v1, mag = velocity(p)
        e_max = max(e_max, float(mag.max()) if mag.size else 0.0)
        speed = np.hypot(v1[:, 0], v1[:, 1])
        gap = cm.geom.min_gap_distance(np.column_stack((cm.geom.wrap_x(p[:, 0]), p[:, 1])))
        with np.errstate(divide="ignore", invalid="ignore"):
            dt_gap = np.where(speed > 0, 0.3 * np.maximum(gap, 1e-6) / speed, dt_cap)
        dt = float(min(dt_cap, np.nanmin(dt_gap)))
        dt = max(dt, 1e-9)

        v2, _ = velocity(p + 0.5 * dt * v1)
        new = p + dt * v2
        pos[idx] = new
        t += dt

        wrapped = np.column_stack((cm.geom.wrap_x(new[:, 0]), new[:, 1]))
        d = cm.geom.min_gap_distance(wrapped)
        for j, k in enumerate(idx):
            if d[j] <= 1e-3:
                landed[k] = _nearest_plane(cm.geom, wrapped[j])
                alive[k] = False
            elif new[j, 1] <= cm.geom.ylo:
                landed[k], alive[k] = "ground", False
            elif new[j, 1] >= cm.geom.yhi:
                landed[k], alive[k] = "cathode", False

        ts.append(t)
        xys.append(pos.copy())
        alives.append(alive.copy())

    for k in np.flatnonzero(alive):
        landed[k] = "stalled"
    xy = np.array(xys)
    # Velocity from the field itself, not a finite difference of the path: the
    # Ramo field-form cross-check should be independent of the integrator.
    flat = xy.reshape(-1, 2)
    e = cm.drift_field(flat)
    mag = np.hypot(e[:, 0], e[:, 1])
    sp = cm.speed(mag)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.where(mag[:, None] > 0, e / np.maximum(mag, 1e-30)[:, None], 0.0)
    vel = (-sp[:, None] * unit).reshape(xy.shape)
    return Trajectory(np.array(ts), xy, np.array(alives), landed, e_max, vel)


def _nearest_plane(geom: Geometry, p: np.ndarray) -> str:
    dx = p[0] - geom.cx
    dx = dx - geom.width * np.round(dx / geom.width)
    d = np.hypot(dx, p[1] - geom.cy) - geom.cr
    return geom.conductors[int(np.argmin(d))].electrode


# --------------------------------------------------------------------------
# the response itself
# --------------------------------------------------------------------------


@dataclass
class Response:
    time: np.ndarray  # (nt,) us, bin centres
    impact: np.ndarray  # (ni,) mm, starting transverse offset
    offsets: np.ndarray  # (nw,) integer wire offsets from the sensing wire
    planes: list[str]
    current: np.ndarray  # (np, nw, ni, nt) e/us, from the charge form
    current_field: np.ndarray  # (np, nw, ni, nt) e/us, from q v.E_w
    integrated: np.ndarray  # (np, nw, ni) e, total induced charge
    landed: list[str]
    e_max: float  # V/mm
    phi_start: np.ndarray | None = None  # (nplane, nimpact) phi at the launch point
    trajectory: dict = field(default_factory=dict)
    agreement: float = 0.0  # max |charge form - field form| / peak

    @property
    def tick(self) -> float:
        return float(self.time[1] - self.time[0]) if len(self.time) > 1 else 0.0


def response(cm: CombinedModel, impacts: int = 11, y_start: float | str = "auto",
             tick: float = 0.1, max_time: float = 200.0, n_wires: int | None = None,
             half_pitch: bool = True, substeps: int = 4) -> Response:
    """Compute the field response of every plane to a drifting electron.

    ``impacts`` starting positions are spread over half a pitch (the geometry is
    symmetric about the wire) or a whole pitch when ``half_pitch`` is False.
    """
    planes = [e.name for e in cm.cfg.electrodes_by_y if e.name in cm.weighting]
    if not planes:
        raise ValueError("no weighting solutions supplied for any electrode")
    pitch = cm.cfg.electrode(planes[0]).lattice.pitch

    if y_start == "auto":
        # Launch just below the cathode.  The cathode is grounded in a weighting
        # solve, so every phi vanishes there and the Ramo integral is exact.
        # Launching lower truncates the response; `phi_start` measures by how
        # much, because the u-plane weighting potential reaches a long way up.
        top = max(e.plane for e in cm.cfg.electrodes)
        y_start = cm.geom.yhi - 0.01 * (cm.geom.yhi - top)
    y_start = float(y_start)

    span = 0.5 * pitch if half_pitch else pitch
    x0 = np.linspace(0.0, span, impacts) if half_pitch else (
        np.linspace(-0.5 * pitch, 0.5 * pitch, impacts))

    traj = trajectories(cm, x0, y_start, tick=tick, max_time=max_time,
                        substeps=substeps)

    if n_wires is None:
        n_wires = min(w.cfg.electrode(w.plane).lattice.count
                      for w in cm.weighting.values())
    half = (n_wires - 1) // 2
    offsets = np.arange(-half, half + 1)

    # Uniform tick grid covering the whole drift.
    nt = int(np.ceil(traj.t[-1] / tick))
    edges = np.arange(nt + 1) * tick
    centres = 0.5 * (edges[:-1] + edges[1:])

    ni, nw, npl = len(x0), len(offsets), len(planes)
    current = np.zeros((npl, nw, ni, nt))
    current_field = np.zeros((npl, nw, ni, nt))
    integrated = np.zeros((npl, nw, ni))
    phi_start = np.zeros((npl, ni))

    # q = -e for an electron; we work in units of e, so q = -1.
    q = -1.0
    peak = 0.0
    for ip, plane in enumerate(planes):
        for ii in range(ni):
            live = traj.alive[:, ii]
            # Keep the first dead sample: that is where the charge lands.
            last = int(np.argmax(~live)) if (~live).any() else len(live) - 1
            sl = slice(0, last + 1)
            tt = traj.t[sl]
            path = traj.xy[sl, ii, :]
            vel = traj.vel[sl, ii, :]
            ns = len(tt)

            # All wire offsets in one batch: the response of the wire `off`
            # pitches away is this path read against the potential shifted by
            # -off pitches.
            shifted = np.repeat(path[None, :, :], nw, axis=0).reshape(-1, 2)
            shifted[:, 0] -= np.repeat(offsets * pitch, ns)
            phi = cm.weighting_potential(plane, shifted).reshape(nw, ns)
            ew = cm.weighting_field(plane, shifted).reshape(nw, ns, 2)

            phi_start[ip, ii] = abs(phi[np.argmin(np.abs(offsets)), 0])

            # Charge form: Q(t) = -q phi(t).  Exact and conservative.
            qcum = -q * phi
            for iw in range(nw):
                qi = np.interp(edges, tt, qcum[iw], left=qcum[iw, 0],
                               right=qcum[iw, -1])
                current[ip, iw, ii, :] = np.diff(qi) / tick
                integrated[ip, iw, ii] = qcum[iw, -1] - qcum[iw, 0]

            # Field form: i = q v . E_w, integrated independently.
            inst = q * (vel[None, :, 0] * ew[:, :, 0] + vel[None, :, 1] * ew[:, :, 1])
            dtt = np.diff(tt)
            icum = np.concatenate(
                (np.zeros((nw, 1)),
                 np.cumsum(0.5 * (inst[:, 1:] + inst[:, :-1]) * dtt[None, :], axis=1)),
                axis=1,
            )
            for iw in range(nw):
                gi = np.interp(edges, tt, icum[iw], left=0.0, right=icum[iw, -1])
                current_field[ip, iw, ii, :] = np.diff(gi) / tick
            peak = max(peak, float(np.abs(current[ip, :, ii, :]).max()))

    diff = np.abs(current - current_field).max()
    return Response(
        time=centres, impact=x0, offsets=offsets, planes=planes,
        current=current, current_field=current_field, integrated=integrated,
        landed=traj.landed, e_max=traj.e_max, phi_start=phi_start,
        agreement=float(diff / peak) if peak > 0 else float("nan"),
        trajectory={
            "t": traj.t,
            "x": traj.xy[:, :, 0],
            "y": traj.xy[:, :, 1],
            "alive": traj.alive,
        },
    )


def report(cm: CombinedModel, resp: Response) -> str:
    """Summary plus the conservation checks that Ramo's theorem guarantees."""
    lines = [
        f"velocity model: {cm.velocity}"
        + (f"  T = {cm.temperature} K" if cm.velocity == "walkowiak"
           else f"  mu = {cm.mobility} cm^2/(V s)"),
    ]
    v500 = drift_speed(np.array([50.0]), cm.velocity, cm.temperature, cm.mobility)[0]
    lines.append(f"  drift speed at 500 V/cm: {v500:.4f} mm/us")
    e_kv = resp.e_max * MM_PER_CM / 1e3
    beyond = (cm.velocity == "walkowiak" and e_kv > WALKOWIAK_E_RANGE[1])
    lines.append(f"  largest |E| on any trajectory: {e_kv:.2f} kV/cm"
                 + (f"  (beyond the {WALKOWIAK_E_RANGE[1]} kV/cm fit range, "
                    "extrapolated)" if beyond else ""))
    lines.append("")
    lines.append(f"impacts: {len(resp.impact)} from {resp.impact[0]:.3f} to "
                 f"{resp.impact[-1]:.3f} mm    wires: {len(resp.offsets)}    "
                 f"tick: {resp.tick:.4f} us    samples: {len(resp.time)}")
    ends = {}
    for name in resp.landed:
        ends[name] = ends.get(name, 0) + 1
    lines.append("electrons ended on: "
                 + ", ".join(f"{k} x{v}" for k, v in sorted(ends.items())))
    lines.append("")
    lines.append("total induced charge [e] per plane and wire offset")
    lines.append("  (Ramo: exactly +1 on the wire that collects, exactly 0 elsewhere)")
    head = "  plane  " + "".join(f"{o:>9d}" for o in resp.offsets if abs(o) <= 3)
    lines.append(head)
    keep = [i for i, o in enumerate(resp.offsets) if abs(o) <= 3]
    for ip, plane in enumerate(resp.planes):
        row = "".join(f"{resp.integrated[ip, i, 0]:9.5f}" for i in keep)
        lines.append(f"  {plane:<7}" + row)
    worst_ind = 0.0
    for ip, plane in enumerate(resp.planes):
        role = cm.cfg.electrode(plane).role
        if role == "collection":
            continue
        worst_ind = max(worst_ind, float(np.abs(resp.integrated[ip]).max()))
    lines.append("")
    if resp.phi_start is not None:
        lines.append("weighting potential at the launch point (the truncation error):")
        for ip, plane in enumerate(resp.planes):
            lines.append(f"  {plane:<7} phi(start) = {resp.phi_start[ip].max():.3e}")
        lines.append("")
    lines.append(f"largest |induced charge| on an induction plane: {worst_ind:.2e} e "
                 "(want 0: the signal is bipolar)")
    lines.append(f"charge form vs field form (q v.E_w) agree to "
                 f"{resp.agreement:.2e} of the peak current")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_weighting(path: str, device: str | None = None) -> WeightingSolution:
    """Load a weighting checkpoint and locate its sensing wire."""
    from . import train as train_mod

    cfg, geom, model, sol = train_mod.load(path, device)
    if cfg.problem.kind != "weighting":
        raise ValueError(f"{path}: problem.kind is {cfg.problem.kind!r}, not 'weighting'")
    hot = [c for c in geom.conductors if c.potential != 0.0]
    if len(hot) != 1:
        raise ValueError(f"{path}: expected exactly one energised wire, found {len(hot)}")
    c = hot[0]
    return WeightingSolution(
        plane=c.electrode, cfg=cfg, geom=geom, model=model,
        unit=c.potential, sense_x=c.x, sense_y=c.y,
        pitch=cfg.electrode(c.electrode).lattice.pitch,
    )


def load_combined(drift_path: str, weight_paths: dict[str, str] | list[str],
                  device: str | None = None, velocity: str = "walkowiak",
                  temperature: float = 87.3, mobility: float = 320.0) -> CombinedModel:
    """Build a :class:`CombinedModel` from checkpoint paths.

    ``weight_paths`` may be a list (the plane of each is read from the
    checkpoint) or a ``{plane: path}`` mapping, in which case the declared plane
    is checked against the checkpoint.
    """
    from . import train as train_mod

    cfg, geom, model, sol = train_mod.load(drift_path, device)
    if cfg.problem.kind != "drift":
        raise ValueError(f"{drift_path}: problem.kind is {cfg.problem.kind!r}, not 'drift'")

    items = weight_paths.items() if isinstance(weight_paths, dict) else (
        (None, p) for p in weight_paths)
    weighting = []
    for declared, path in items:
        w = load_weighting(path, device)
        if declared is not None and declared != w.plane:
            raise ValueError(
                f"{path}: declared plane {declared!r} but the checkpoint senses "
                f"wire {w.plane!r}"
            )
        weighting.append(w)
    seen = [w.plane for w in weighting]
    if len(set(seen)) != len(seen):
        raise ValueError(f"more than one weighting solution per plane: {seen}")
    return CombinedModel(cfg, geom, model, weighting, velocity=velocity,
                         temperature=temperature, mobility=mobility)


def save_npz(path: str, resp: Response, cm: CombinedModel) -> None:
    import json

    meta = {
        "planes": resp.planes,
        "landed": resp.landed,
        "velocity": cm.velocity,
        "temperature_K": cm.temperature,
        "mobility_cm2_per_Vs": cm.mobility,
        "e_max_V_per_mm": resp.e_max,
        "agreement": resp.agreement,
        "pitch_mm": cm.cfg.electrode(resp.planes[0]).lattice.pitch,
        "units": {
            "time": "us", "impact": "mm", "current": "e/us",
            "integrated": "e", "trajectory": "mm",
        },
        "current_in_amperes": E_PER_US_IN_A,
        "config": {"drift": cm.cfg.name},
    }
    np.savez_compressed(
        path,
        metadata=np.array(json.dumps(meta, indent=1)),
        time=resp.time, impact=resp.impact, offsets=resp.offsets,
        current=resp.current, current_field=resp.current_field,
        integrated=resp.integrated,
        traj_t=resp.trajectory["t"], traj_x=resp.trajectory["x"],
        traj_y=resp.trajectory["y"], traj_alive=resp.trajectory["alive"],
    )


def load_npz(path: str):
    import json

    blob = np.load(path, allow_pickle=False)
    meta = json.loads(str(blob["metadata"]))
    arrays = {k: blob[k] for k in blob.files if k != "metadata"}
    return arrays, meta


def impact_table(arrays: dict, meta: dict | None = None, per_side: int = 5,
                 centred: bool = True):
    """Re-index a tabulated response as a single function of impact offset.

    :func:`response` produces ``current[plane, wire, impact, tick]`` with
    ``impact`` covering half a pitch and ``wire`` running over the lattice
    offsets.  The symmetries of Eq.~(response-symmetry) --- periodicity by one
    pitch and a mirror about each wire --- collapse those two indices into one:
    the current induced on a single wire by an electron launched a signed
    distance ``u`` away,

    .. math::  \rho_p(u, t) = r_p(u,\, 0;\, t),

    with ``u`` running over every wire region the offsets reach.

    ``per_side`` launch positions are kept on each side of every wire centre,
    so the grid step is ``pitch / (2 * per_side)``.  With ``centred`` (the
    default) the positions are the *centres* of those bins, half a step from
    each edge: the grid then avoids both stagnation lines of the drift field,
    the axis through a wire and the saddle midway between two, where the
    trajectory is unstable and the sample is decided by rounding.  With
    ``centred=False`` the positions are the bin edges and both lines are
    sampled exactly.

    Returns a dict with ``impact`` (mm), ``time`` (us), ``table``
    ``(plane, impact, tick)`` and ``planes``.
    """
    cur = np.asarray(arrays["current"], dtype=float)
    imp = np.asarray(arrays["impact"], dtype=float)
    off = np.asarray(arrays["offsets"], dtype=int)
    time = np.asarray(arrays["time"], dtype=float)
    meta = meta or {}
    pitch = float(meta.get("pitch_mm", 0.0)) or float(2.0 * imp.max())
    planes = list(meta.get("planes", [str(i) for i in range(cur.shape[0])]))

    if per_side < 1:
        raise ValueError("per_side must be >= 1")
    step = 0.5 * pitch / per_side
    half = int(np.abs(off).max())
    shift = 0.5 if centred else 0.0

    # The half-pitch launch positions we need must already be tabulated.
    want = (np.arange(per_side + (0 if centred else 1)) + shift) * step
    col = []
    for w in want:
        j = int(np.argmin(np.abs(imp - w)))
        if abs(imp[j] - w) > 1e-6 * max(pitch, 1.0):
            need = 2 * per_side * (2 if centred else 1) + 1
            raise ValueError(
                f"impact {w:g} mm is not in the tabulation "
                f"{np.array2string(imp, precision=3)}; recompute the response "
                f"with impacts = {need} over a half pitch")
        col.append(j)
    col = np.array(col)
    row = {int(k): i for i, k in enumerate(off)}

    if centred:
        n_reg = 2 * half + 1
        grid = (np.arange(-n_reg * per_side, n_reg * per_side) + 0.5) * step
    else:
        grid = np.arange(-half - 0.5, half + 0.5 + 1e-9, 1.0 / (2 * per_side)) * pitch

    table = np.zeros((cur.shape[0], len(grid), cur.shape[-1]))
    for m, u in enumerate(grid):
        v = abs(u)
        n = int(np.floor(v / pitch + 0.5))
        if n > half:            # the node grid's endpoint sits exactly on a
            n = half            # cell edge; take the decomposition we have
        rem = v - n * pitch
        k, val = (-n, rem) if rem >= -1e-9 else (n, -rem)
        i = int(round(abs(val) / step - shift))
        table[:, m, :] = cur[:, row[k], col[i], :]

    return {"impact": grid, "time": time, "table": table, "planes": planes,
            "pitch": pitch, "per_side": per_side, "centred": centred}
