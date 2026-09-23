"""Physics cross-checks on a solved field.

These do not look at the loss at all.  They ask whether the solution obeys
relations it was never trained on:

``gauss``
    Gauss's law across each electrode plane.  The transverse average of ``E_y``
    just above and just below a plane differs by the enclosed line charge, and
    that same line charge is what the analytic monopole coefficient of the
    baseline represents.  Two independent routes to the same number.

``drift``
    Integrate electron trajectories through the solved field and see which
    conductor each one lands on.  This is the transparency condition measured
    rather than assumed: with a correct bias, no electron may end on an
    induction wire and none may reach the ground plane.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import Config
from .geometry import Geometry
from .siren import FieldModel
from .train import DTYPES, resolve_device

MM_PER_CM = 10.0


def _evaluator(cfg: Config, model: FieldModel):
    device = resolve_device(cfg.train.device)
    dtype = DTYPES[cfg.train.precision]

    def field(pts: np.ndarray) -> np.ndarray:
        """-grad V at ``pts``, in V/mm, shape (n, 2)."""
        out = []
        for start in range(0, len(pts), 16384):
            block = torch.as_tensor(
                pts[start : start + 16384], device=device, dtype=dtype
            ).requires_grad_(True)
            v = model(block)
            (g,) = torch.autograd.grad(v.sum(), block)
            out.append(-g.detach().cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, 2))

    return field


# --------------------------------------------------------------------------


def gauss_check(cfg: Config, geom: Geometry, model: FieldModel,
                n_x: int = 2048, offset: float = 1.0) -> list[dict]:
    """Line charge per plane, from the field jump and from the baseline.

    ``offset`` is how far above/below each plane (in mm) the transverse average
    of ``E_y`` is taken; it must be far enough that the near-wire multipoles have
    decayed (they die as ``exp(-2 pi dy / W)``) and close enough to stay inside
    the gap.
    """
    x = geom.xlo + (np.arange(n_x) + 0.5) * geom.width / n_x
    field = _evaluator(cfg, model)
    out = []
    planes = cfg.electrodes_by_y
    for i, e in enumerate(planes):
        gap_up = (planes[i - 1].plane if i else geom.yhi) - e.plane
        gap_dn = e.plane - (planes[i + 1].plane if i + 1 < len(planes) else geom.ylo)
        dy = min(offset, 0.4 * gap_up, 0.4 * gap_dn)
        above = field(np.column_stack((x, np.full(n_x, e.plane + dy))))[:, 1].mean()
        below = field(np.column_stack((x, np.full(n_x, e.plane - dy))))[:, 1].mean()
        # lambda / eps0 = W * (E_above - E_below), and the monopole coefficient
        # of a line charge is c = lambda / (2 pi eps0).
        c_gauss = geom.width * (above - below) / (2.0 * np.pi)
        rec = {
            "electrode": e.name,
            "offset_mm": dy,
            "e_above": above * MM_PER_CM,  # V/cm
            "e_below": below * MM_PER_CM,
            "coeff_gauss": c_gauss,
        }
        if model.baseline.coeff is not None:
            # The first len(conductors) coefficients are the monopoles; the
            # dipole columns carry no net charge and do not enter Gauss's law.
            nc = len(geom.conductors)
            idx = [j for j, c in enumerate(geom.conductors) if c.electrode == e.name]
            rec["coeff_model"] = float(model.baseline.coeff.detach()[:nc][idx].sum())
        out.append(rec)
    return out


def gap_field_check(cfg: Config, geom: Geometry, model: FieldModel,
                    n_x: int = 512) -> list[dict]:
    """Transverse-averaged ``E_y`` at each gap midpoint vs the design value."""
    from . import bias as bias_mod

    sol = bias_mod.solve(cfg)
    x = geom.xlo + (np.arange(n_x) + 0.5) * geom.width / n_x
    field = _evaluator(cfg, model)
    out = []
    for g in sol.gaps:
        y = 0.5 * (g.y_hi + g.y_lo)
        got = field(np.column_stack((x, np.full(n_x, y))))[:, 1].mean() * MM_PER_CM
        out.append({"gap": g.name, "y": y, "expected": g.field, "measured": float(got)})
    return out


# --------------------------------------------------------------------------


def drift_paths(cfg: Config, geom: Geometry, model: FieldModel, n: int = 64,
                y_start: float | None = None, max_steps: int = 20000,
                step_max: float = 0.5, store: bool = False):
    """Trace electrons from near the cathode and record where they end up.

    Electrons move along ``-E``.  The step is a fixed fraction of the distance
    to the nearest conductor surface (capped by ``step_max``), so paths take
    fine steps exactly where the field curves hardest.  Integration is RK2
    (midpoint) on the unit drift direction.
    """
    if y_start is None:
        y_start = geom.yhi - 0.02 * geom.height
    field = _evaluator(cfg, model)

    pos = np.column_stack(
        (geom.xlo + (np.arange(n) + 0.5) * geom.width / n, np.full(n, y_start))
    )
    alive = np.ones(n, dtype=bool)
    landed = np.full(n, "", dtype=object)
    tracks = [[p.copy()] for p in pos] if store else None

    def direction(p):
        e = field(p)
        mag = np.hypot(e[:, 0], e[:, 1])[:, None]
        return -e / np.maximum(mag, 1e-30)

    for _ in range(max_steps):
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        p = pos[idx]
        gap = geom.min_gap_distance(p)
        h = np.clip(0.5 * gap, 1e-4, step_max)[:, None]
        mid = p + 0.5 * h * direction(p)
        mid[:, 0] = geom.wrap_x(mid[:, 0])
        new = p + h * direction(mid)
        new[:, 0] = geom.wrap_x(new[:, 0])
        pos[idx] = new
        if store:
            for j, k in enumerate(idx):
                tracks[k].append(new[j].copy())

        # Landed on a conductor?
        d = geom.min_gap_distance(new)
        hit = np.flatnonzero(d <= 1e-3)
        for k in idx[hit]:
            landed[k] = _nearest_label(geom, pos[k])
            alive[k] = False
        # Reached a face?
        for k, yv in zip(idx, new[:, 1]):
            if alive[k] and yv <= geom.ylo:
                landed[k], alive[k] = "ground", False
            elif alive[k] and yv >= geom.yhi:
                landed[k], alive[k] = "cathode", False

    for k in np.flatnonzero(alive):
        landed[k] = "stalled"
    return landed, (tracks if store else None), pos


def _nearest_label(geom: Geometry, p: np.ndarray) -> str:
    dx = p[0] - geom.cx
    dx = dx - geom.width * np.round(dx / geom.width)
    d = np.hypot(dx, p[1] - geom.cy) - geom.cr
    return geom.conductors[int(np.argmin(d))].electrode


def transparency_report(cfg: Config, geom: Geometry, landed) -> str:
    names, counts = np.unique(np.array(landed, dtype=str), return_counts=True)
    total = len(landed)
    lines = [f"traced {total} electrons from y = {geom.yhi:.1f} mm downward", ""]
    for name, count in sorted(zip(names, counts), key=lambda t: -t[1]):
        lines.append(f"  {name:<10} {count:6d}   {100.0 * count / total:6.2f} %")
    lines.append("")
    collection = {e.name for e in cfg.electrodes if e.role == "collection"}
    induction = {e.name for e in cfg.electrodes if e.name not in collection}
    caught = sum(c for n, c in zip(names, counts) if n in induction)
    got = sum(c for n, c in zip(names, counts) if n in collection)
    lost = sum(c for n, c in zip(names, counts) if n in ("ground", "cathode", "stalled"))
    lines.append(f"transparency: {caught} / {total} electrons stopped on an induction "
                 f"plane ({100.0 * caught / total:.2f} %) -- want 0")
    lines.append(f"collection:   {got} / {total} reached a collection electrode "
                 f"({100.0 * got / total:.2f} %) -- want 100")
    if lost:
        lines.append(f"escaped/stalled: {lost}")
    verdict = "PASS" if caught == 0 and got == total else "FAIL"
    lines.append(f"verdict: {verdict}")
    return "\n".join(lines)
