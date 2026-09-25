"""PINN training: fit the SIREN so that the Laplace residual and the Dirichlet
boundary conditions are simultaneously small.

Loss
----
All three terms are dimensionless.  The residual is scaled by ``L^2 / V_ref``
with ``L`` the smallest electrode pitch and ``V_ref`` the potential span, so a
value of 1 means "the potential is wrong by ``V_ref`` over a pitch".  The
boundary terms are scaled by ``V_ref``.  This puts the terms on a common footing
before the ``w_*`` weights are applied.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from . import bias as bias_mod
from . import config as config_mod
from .config import Config
from .geometry import Geometry, Sampler, TorchSampler
from .siren import FieldModel, laplacian, numpy_to_tensor

DTYPES = {"float32": torch.float32, "float64": torch.float64}


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


@dataclass
class Losses:
    total: float = 0.0
    pde: float = 0.0
    face: float = 0.0
    surface: float = 0.0
    rms_face_v: float = 0.0  # volts
    rms_surface_v: float = 0.0  # volts
    rms_residual: float = 0.0  # V/mm^2


@dataclass
class History:
    steps: list[int] = field(default_factory=list)
    records: list[Losses] = field(default_factory=list)

    def append(self, step: int, rec: Losses) -> None:
        self.steps.append(step)
        self.records.append(rec)

    def as_arrays(self) -> dict[str, np.ndarray]:
        out = {"step": np.array(self.steps)}
        if self.records:
            for key in vars(self.records[0]):
                out[key] = np.array([getattr(r, key) for r in self.records])
        return out


class Trainer:
    def __init__(self, cfg: Config, geom: Geometry, model: FieldModel, device, dtype):
        self.cfg = cfg
        self.geom = geom
        self.model = model
        self.device = device
        self.dtype = dtype
        if cfg.sampling.backend == "torch":
            self.sampler = TorchSampler(geom, cfg.sampling.seed, cfg.sampling.near_factor,
                                        device=device, dtype=dtype)
        else:
            self.sampler = Sampler(geom, cfg.sampling.seed, cfg.sampling.near_factor)
        if cfg.train.laplacian not in ("explicit", "autograd"):
            raise ValueError(f"train.laplacian {cfg.train.laplacian!r} unknown")
        self.v_ref = geom.potential_span
        self.l_ref = min(e.lattice.pitch for e in cfg.electrodes)
        self.res_scale = self.l_ref**2 / self.v_ref

        # Boundary targets that never change.
        self._face_targets = {
            "ylo": geom.v_ground,
            "yhi": geom.v_cathode,
        }

    # -- batches ------------------------------------------------------------

    def _t(self, a) -> torch.Tensor:
        if isinstance(a, torch.Tensor):
            return a.to(device=self.device, dtype=self.dtype)
        return numpy_to_tensor(a, self.device, self.dtype)

    def draw(self):
        s, cfg = self.sampler, self.cfg.sampling
        if isinstance(s, TorchSampler):
            col = torch.cat((s.bulk(cfg.n_bulk), s.near(cfg.n_near)))
            faces, face_vals = [], []
            for key, y in (("ylo", self.geom.ylo), ("yhi", self.geom.yhi)):
                pts = s.face(cfg.n_face, y)
                faces.append(pts)
                face_vals.append(torch.full((len(pts),), float(self._face_targets[key]),
                                            device=self.device, dtype=self.dtype))
            surf, surf_vals = s.surfaces(cfg.n_surface)
            return col, torch.cat(faces), torch.cat(face_vals), surf, surf_vals
        col = np.vstack((s.bulk(cfg.n_bulk), s.near(cfg.n_near)))
        faces, face_vals = [], []
        for key, y in (("ylo", self.geom.ylo), ("yhi", self.geom.yhi)):
            pts = s.face(cfg.n_face, y)
            faces.append(pts)
            face_vals.append(np.full(len(pts), self._face_targets[key]))
        surf, surf_vals = s.surfaces(cfg.n_surface)
        return (
            self._t(col),
            self._t(np.vstack(faces)),
            self._t(np.concatenate(face_vals)),
            self._t(surf),
            self._t(surf_vals),
        )

    @torch.no_grad()
    def baseline_bc_residual(self) -> tuple[float, float]:
        """RMS error of the analytic baseline alone on the faces and conductors.

        This is what the network is there to remove, so it sets the scale the
        network should work at.
        """
        _, face_xy, face_v, surf_xy, surf_v = self.draw()
        face = float((self.model.baseline(face_xy)[:, 0] - face_v).pow(2).mean().sqrt())
        if surf_xy.numel():
            surf = float(
                (self.model.baseline(surf_xy)[:, 0] - surf_v).pow(2).mean().sqrt()
            )
        else:
            surf = 0.0
        return face, surf

    # -- loss ---------------------------------------------------------------

    def losses(self, batch) -> tuple[torch.Tensor, Losses]:
        col, face_xy, face_v, surf_xy, surf_v = batch
        t = self.cfg.train

        if t.laplacian == "explicit":
            lap = self.model.network_laplacian(col)
        else:
            lap = laplacian(self.model.network_part, col)
        pde = (lap * self.res_scale).pow(2).mean()

        face_err = (self.model(face_xy)[:, 0] - face_v) / self.v_ref
        face = face_err.pow(2).mean()

        if surf_xy.numel():
            surf_err = (self.model(surf_xy)[:, 0] - surf_v) / self.v_ref
            surface = surf_err.pow(2).mean()
        else:
            surf_err = torch.zeros(1, device=self.device, dtype=self.dtype)
            surface = torch.zeros((), device=self.device, dtype=self.dtype)

        total = t.w_pde * pde + t.w_face * face + t.w_surface * surface
        with torch.no_grad():
            rec = Losses(
                total=float(total),
                pde=float(pde),
                face=float(face),
                surface=float(surface),
                rms_face_v=float(face.sqrt()) * self.v_ref,
                rms_surface_v=float(surface.sqrt()) * self.v_ref,
                rms_residual=float(lap.pow(2).mean().sqrt()),
            )
        return total, rec


def build(cfg: Config, sol: bias_mod.BiasSolution | None = None):
    """Create the geometry and an untrained model on the configured device."""
    cfg.require_solvable()
    if sol is None:
        sol = bias_mod.solve(cfg)
    dtype = DTYPES[cfg.train.precision]
    torch.set_default_dtype(dtype)
    device = resolve_device(cfg.train.device)
    torch.manual_seed(cfg.train.seed)
    geom = Geometry(cfg, sol)
    model = FieldModel(cfg, geom).to(device=device, dtype=dtype)
    return geom, model, device, dtype, sol


def train(cfg: Config, sol: bias_mod.BiasSolution | None = None, on_log=None):
    """Run the full training schedule.  Returns ``(model, geom, history, sol)``."""
    geom, model, device, dtype, sol = build(cfg, sol)
    trainer = Trainer(cfg, geom, model, device, dtype)
    history = History()

    # Least-squares warm start of the analytic basis coefficients.
    if cfg.model.baseline.mode == "wires":
        xy, vals = trainer.sampler.surfaces(max(64, cfg.sampling.n_surface))
        rms = model.baseline.warm_start(trainer._t(xy), trainer._t(vals))
        if on_log:
            on_log(f"warm start: conductor-surface RMS error {rms:.4g} V")

    # The network only has to supply what the analytic baseline could not, so
    # give it an output scale of that size.  Leaving it at the full potential
    # span forces the head weights down to ~1e-4 of their natural magnitude,
    # where Adam's step is larger than the weights themselves.  Size it on the
    # *worst* boundary -- in a weighting solve the conductors are fitted almost
    # exactly and everything left over sits on the faces.
    if model.auto_scale:
        face_rms, surf_rms = trainer.baseline_bc_residual()
        scale = 10.0 * max(face_rms, surf_rms)
        scale = min(max(scale, 1e-4 * geom.potential_span), geom.potential_span)
        model.output_scale.fill_(scale)
        if on_log:
            on_log(
                f"baseline boundary residual: faces {face_rms:.4g} V, "
                f"conductors {surf_rms:.4g} V -> network output scale {scale:.4g} V"
            )

    t = cfg.train
    opt = torch.optim.Adam(model.parameters(), lr=t.lr, eps=t.adam_eps)
    gamma = (t.lr_final / t.lr) ** (1.0 / max(t.steps, 1))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

    batch = trainer.draw()
    t0 = time.time()
    for step in range(1, t.steps + 1):
        if cfg.sampling.resample_every and step % cfg.sampling.resample_every == 0:
            batch = trainer.draw()
        opt.zero_grad(set_to_none=True)
        total, rec = trainer.losses(batch)
        total.backward()
        opt.step()
        sched.step()
        if step % t.log_every == 0 or step == 1 or step == t.steps:
            history.append(step, rec)
            if on_log:
                on_log(
                    f"step {step:7d}/{t.steps}  loss {rec.total:.4e}  "
                    f"pde {rec.pde:.3e}  face {rec.face:.3e}  surf {rec.surface:.3e}  "
                    f"| BC rms: face {rec.rms_face_v:8.3f} V  wire {rec.rms_surface_v:8.3f} V"
                    f"  lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:6.1f}s"
                )

    if t.lbfgs_steps > 0:
        batch = trainer.draw()
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            max_iter=t.lbfgs_steps,
            history_size=50,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
        )
        state = {"n": 0}

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            total, rec = trainer.losses(batch)
            total.backward()
            state["n"] += 1
            if on_log and state["n"] % max(1, t.log_every // 10) == 0:
                on_log(
                    f"lbfgs {state['n']:5d}  loss {rec.total:.4e}  "
                    f"| BC rms: face {rec.rms_face_v:8.3f} V  wire {rec.rms_surface_v:8.3f} V"
                )
            return total

        lbfgs.step(closure)
        _, rec = trainer.losses(batch)
        history.append(t.steps + state["n"], rec)

    return model, geom, history, sol


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def save(path: str, cfg: Config, model: FieldModel, sol: bias_mod.BiasSolution,
         history: History | None = None) -> None:
    torch.save(
        {
            "firep_version": 1,
            "config": config_mod.to_dict(cfg),
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "bias": sol.to_dict(),
            "history": history.as_arrays() if history else {},
        },
        path,
    )


def load(path: str, device: str | None = None):
    """Rebuild ``(cfg, geom, model, sol)`` from a checkpoint."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = config_mod.from_dict(blob["config"])
    if device is not None:
        cfg.train.device = device
    # Trust the stored solution over a re-solve, so that a checkpoint keeps
    # meaning what it meant even if the bias solver later changes.
    stored = blob.get("bias") or {}
    try:
        sol = bias_mod.BiasSolution.from_dict(stored) if stored else bias_mod.solve(cfg)
    except TypeError:
        sol = bias_mod.solve(cfg)
    geom, model, dev, dtype, sol = build(cfg, sol)
    model.load_state_dict({k: v.to(device=dev, dtype=dtype) for k, v in blob["state_dict"].items()})
    model.eval()
    return cfg, geom, model, sol
