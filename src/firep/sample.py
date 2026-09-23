"""Evaluate a trained model onto a regular grid and write NumPy files."""

from __future__ import annotations

import json

import numpy as np
import torch

from . import config as config_mod
from .config import Config
from .geometry import Geometry, Sampler
from .siren import FieldModel, laplacian
from .train import DTYPES, resolve_device

MM_PER_CM = 10.0


def evaluate_grid(
    cfg: Config,
    geom: Geometry,
    model: FieldModel,
    nx: int,
    ny: int,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    chunk: int = 65536,
    residual: bool = False,
) -> dict[str, np.ndarray]:
    """Sample potential, field and (optionally) PDE residual on a grid.

    Returns arrays with ``x`` of shape ``(nx,)``, ``y`` of shape ``(ny,)`` and
    every field of shape ``(ny, nx)`` -- i.e. indexed ``[iy, ix]``, the layout
    ``matplotlib.pcolormesh`` and ``imshow`` want.
    """
    device = resolve_device(cfg.train.device)
    dtype = DTYPES[cfg.train.precision]
    xlo, xhi = xlim if xlim else (geom.xlo, geom.xhi)
    ylo, yhi = ylim if ylim else (geom.ylo, geom.yhi)
    x = np.linspace(xlo, xhi, nx)
    y = np.linspace(ylo, yhi, ny)
    xx, yy = np.meshgrid(x, y)  # (ny, nx)
    pts = np.column_stack((xx.ravel(), yy.ravel()))

    vs, gs, rs = [], [], []
    for start in range(0, len(pts), chunk):
        block = torch.as_tensor(pts[start : start + chunk], device=device, dtype=dtype)
        block.requires_grad_(True)
        v = model(block)
        (grad,) = torch.autograd.grad(v.sum(), block, create_graph=False)
        vs.append(v.detach().cpu().numpy()[:, 0])
        gs.append(grad.detach().cpu().numpy())
        if residual:
            lap = laplacian(model.network_part, block.detach().clone(), create_graph=False)
            rs.append(lap.detach().cpu().numpy()[:, 0])

    v = np.concatenate(vs).reshape(ny, nx)
    g = np.concatenate(gs).reshape(ny, nx, 2)
    # E = -grad V, in V/mm; report V/cm.
    ex = -g[:, :, 0] * MM_PER_CM
    ey = -g[:, :, 1] * MM_PER_CM
    mask = geom.inside(pts).reshape(ny, nx)

    out = {
        "x": x,
        "y": y,
        "potential": v,
        "ex": ex,
        "ey": ey,
        "emag": np.hypot(ex, ey),
        "inside": mask,
    }
    if residual:
        out["residual"] = np.concatenate(rs).reshape(ny, nx)
    return out


def metadata(cfg: Config, geom: Geometry) -> dict:
    return {
        "name": cfg.name,
        "problem": cfg.problem.kind,
        "config": config_mod.to_dict(cfg),
        "conductors": [
            {
                "electrode": c.electrode,
                "index": c.index,
                "x": c.x,
                "y": c.y,
                "radius": c.radius,
                "potential": c.potential,
            }
            for c in geom.conductors
        ],
        "faces": {"cathode": geom.v_cathode, "ground": geom.v_ground},
        "units": {
            "x": "mm",
            "y": "mm",
            "potential": "V",
            "ex": "V/cm",
            "ey": "V/cm",
            "residual": "V/mm^2",
        },
    }


def save_npz(path: str, arrays: dict[str, np.ndarray], meta: dict) -> None:
    np.savez_compressed(path, metadata=np.array(json.dumps(meta, indent=1)), **arrays)


def load_npz(path: str) -> tuple[dict[str, np.ndarray], dict]:
    blob = np.load(path, allow_pickle=False)
    arrays = {k: blob[k] for k in blob.files if k != "metadata"}
    meta = json.loads(str(blob["metadata"])) if "metadata" in blob.files else {}
    return arrays, meta


def save_npy_set(stem: str, arrays: dict[str, np.ndarray], meta: dict) -> list[str]:
    """Write each array as its own ``.npy`` plus a ``.json`` sidecar."""
    written = []
    for key, arr in arrays.items():
        path = f"{stem}.{key}.npy"
        np.save(path, arr)
        written.append(path)
    path = f"{stem}.json"
    with open(path, "w") as fp:
        json.dump(meta, fp, indent=1)
    written.append(path)
    return written


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------


def diagnostics(cfg: Config, geom: Geometry, model: FieldModel, n: int = 200000) -> dict:
    """Fresh-sample check of how well the PDE and the BCs are actually satisfied."""
    device = resolve_device(cfg.train.device)
    dtype = DTYPES[cfg.train.precision]
    sampler = Sampler(geom, cfg.sampling.seed + 1, cfg.sampling.near_factor)

    def to_t(a):
        return torch.as_tensor(a, device=device, dtype=dtype)

    col = np.vstack((sampler.bulk(n // 2), sampler.near(n // 2)))
    laps = []
    for start in range(0, len(col), 16384):
        block = to_t(col[start : start + 16384])
        laps.append(
            laplacian(model.network_part, block, create_graph=False)
            .detach()
            .cpu()
            .numpy()[:, 0]
        )
    lap = np.concatenate(laps)

    out: dict = {
        "n_collocation": len(col),
        "residual_rms": float(np.sqrt(np.mean(lap**2))),
        "residual_max": float(np.max(np.abs(lap))),
    }
    # Dimensionless: residual times pitch^2 over the potential span.
    scale = min(e.lattice.pitch for e in cfg.electrodes) ** 2 / geom.potential_span
    out["residual_rms_scaled"] = out["residual_rms"] * scale

    with torch.no_grad():
        for key, yv, target in (
            ("ground", geom.ylo, geom.v_ground),
            ("cathode", geom.yhi, geom.v_cathode),
        ):
            pts = to_t(sampler.face(20000, yv))
            err = model(pts)[:, 0].cpu().numpy() - target
            out[f"face_{key}_rms_V"] = float(np.sqrt(np.mean(err**2)))
            out[f"face_{key}_max_V"] = float(np.max(np.abs(err)))

        per = {}
        for c in geom.conductors:
            th = np.linspace(0, 2 * np.pi, 512, endpoint=False)
            pts = np.column_stack(
                (c.x + c.radius * np.cos(th), c.y + c.radius * np.sin(th))
            )
            err = model(to_t(pts))[:, 0].cpu().numpy() - c.potential
            per[c.label] = {
                "rms_V": float(np.sqrt(np.mean(err**2))),
                "max_V": float(np.max(np.abs(err))),
                "mean_V": float(np.mean(err)),
            }
        out["conductors"] = per
        if per:
            out["surface_rms_V"] = float(
                np.sqrt(np.mean([v["rms_V"] ** 2 for v in per.values()]))
            )
            out["surface_max_V"] = float(max(v["max_V"] for v in per.values()))
    return out
