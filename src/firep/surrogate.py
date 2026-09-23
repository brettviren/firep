"""Learning the field response itself, as a conditioned neural field.

The models of :mod:`firep.siren` represent *potentials* and the response is then
computed from them by tracing and Ramo.  A different use of the same machinery
is to represent the response directly,

.. math::

    \\mathcal{R}_\\theta : (x_0,\\; t,\\; \\mu) \\;\\longmapsto\\;
      r_p(x_0, x_w;\\, t \\mid \\mu),

where ``mu`` is whatever the response is to be conditioned on.  Three choices of
``mu`` cover the generalisations discussed in the technical note:

``()``
    nothing: a plain surrogate for the tabulated response.  Useful because it is
    continuous in ``x_0`` and ``t`` -- no impact binning, no tick -- and
    differentiable with respect to both.

``(sigma,)``
    the Gaussian-smeared response.  Instead of convolving a tabulated response
    with a diffusion kernel at simulation time, the network is trained on the
    already-smeared response over a range of widths and interpolates in
    ``sigma``.  The convolution moves from run time to training time.

``(sigma_x, sigma_t, a)``
    the full post-drift response: transverse and longitudinal widths and an
    attenuation factor, all of which a drift distance determines.  A model
    conditioned this way needs no logical response plane at all: it is evaluated
    with the widths appropriate to wherever the charge actually is.

The output is a whole waveform per wire, not a scalar, so the network maps
``(x_0, mu)`` to ``n_wires x n_time`` values through a shared trunk and a linear
head -- which keeps the wires consistent with one another by construction and
makes the Ramo sum rule a property of the output layer rather than a hope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .siren import SineLayer


@dataclass
class SurrogateSpec:
    """What the surrogate represents and how big it is."""

    n_wires: int
    n_time: int
    conditions: tuple = ()  # names of the conditioning variables
    hidden: int = 128
    layers: int = 3
    omega0: float = 12.0
    omega_hidden: float = 12.0
    x_scale: float = 2.5  # half a pitch: x_0 is scaled onto [-1, 1]
    cond_lo: tuple = ()
    cond_hi: tuple = ()

    @property
    def n_cond(self) -> int:
        return len(self.conditions)


class ResponseSurrogate(nn.Module):
    """``(x_0, mu) -> r[wire, time]``, a SIREN trunk with a linear read-out.

    Time is *not* an input.  Making the network emit the whole waveform at once
    costs one wide output layer and buys two things: the waveform is consistent
    across wires by construction, and a single forward pass replaces one per
    sample, which matters when the surrogate is used inside a simulation loop.
    """

    def __init__(self, spec: SurrogateSpec):
        super().__init__()
        self.spec = spec
        # Build in float32 explicitly rather than inheriting whatever global
        # default dtype happens to be set; a surrogate is fitted to tabulated
        # data, not differentiated twice, so it has no need of float64.
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            self._build(spec)
        finally:
            torch.set_default_dtype(prev)

    def _build(self, spec: SurrogateSpec) -> None:
        n_in = 1 + spec.n_cond
        mods = [SineLayer(n_in, spec.hidden, spec.omega0, first=True)]
        for _ in range(spec.layers - 1):
            mods.append(SineLayer(spec.hidden, spec.hidden, spec.omega_hidden))
        self.body = nn.Sequential(*mods)
        self.head = nn.Linear(spec.hidden, spec.n_wires * spec.n_time)
        with torch.no_grad():
            bound = math.sqrt(6.0 / spec.hidden) / spec.omega_hidden
            self.head.weight.uniform_(-bound, bound)
            self.head.bias.zero_()
        lo = torch.as_tensor(spec.cond_lo or [0.0] * spec.n_cond, dtype=torch.float32)
        hi = torch.as_tensor(spec.cond_hi or [1.0] * spec.n_cond, dtype=torch.float32)
        self.register_buffer("cond_lo", lo)
        self.register_buffer("cond_hi", hi)
        self.register_buffer("out_scale", torch.tensor(1.0))

    # -- input scaling ------------------------------------------------------

    @property
    def dtype(self) -> torch.dtype:
        return self.head.weight.dtype

    def encode(self, x0: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        if self.spec.n_cond and cond is None:
            raise ValueError(f"this surrogate needs {self.spec.conditions}")
        x0 = x0.to(self.dtype)
        feats = [x0[:, None] / self.spec.x_scale]
        if self.spec.n_cond:
            span = torch.clamp(self.cond_hi - self.cond_lo, min=1e-12)
            feats.append(2.0 * (cond.to(self.dtype) - self.cond_lo) / span - 1.0)
        return torch.cat(feats, dim=1)

    def forward(self, x0: torch.Tensor, cond: torch.Tensor | None = None):
        h = self.body(self.encode(x0, cond))
        out = self.head(h) * self.out_scale
        return out.view(-1, self.spec.n_wires, self.spec.n_time)


# --------------------------------------------------------------------------
# training data
# --------------------------------------------------------------------------


def smear(current: np.ndarray, tick: float, sigma_t: float) -> np.ndarray:
    """Convolve the last axis with a Gaussian of width ``sigma_t`` (in time)."""
    if sigma_t <= 0:
        return current
    half = max(int(math.ceil(4.0 * sigma_t / tick)), 1)
    tt = (np.arange(-half, half + 1)) * tick
    g = np.exp(-0.5 * (tt / sigma_t) ** 2)
    g /= g.sum()
    n = current.shape[-1]
    pad = [(0, 0)] * (current.ndim - 1) + [(half, half)]
    padded = np.pad(current, pad, mode="edge")
    out = np.apply_along_axis(lambda v: np.convolve(v, g, mode="valid"), -1, padded)
    return out[..., :n]


def training_set(arrays: dict, conditions: tuple = (), n_sigma: int = 12,
                 sigma_range: tuple = (0.0, 2.0), plane: int = 0,
                 rng: np.random.Generator | None = None):
    """Build ``(x0, cond, target)`` from a tabulated response.

    With no conditions the target is the response as tabulated.  With
    ``("sigma_t",)`` the response is smeared by a range of widths, which is the
    training set for the "learn the smeared response" generalisation.
    """
    rng = rng or np.random.default_rng(0)
    cur = np.asarray(arrays["current"])[plane]  # (wire, impact, time)
    impact = np.asarray(arrays["impact"], float)
    tick = float(arrays["time"][1] - arrays["time"][0])

    if not conditions:
        x0 = impact.copy()
        target = np.transpose(cur, (1, 0, 2))  # (impact, wire, time)
        return x0, np.zeros((len(x0), 0)), target

    if conditions != ("sigma_t",):
        raise ValueError(f"training_set does not know how to vary {conditions}")
    sig = np.linspace(sigma_range[0], sigma_range[1], n_sigma)
    xs, cs, ts = [], [], []
    for s in sig:
        sm = smear(cur, tick, float(s))
        xs.append(impact)
        cs.append(np.full((len(impact), 1), s))
        ts.append(np.transpose(sm, (1, 0, 2)))
    return (np.concatenate(xs), np.concatenate(cs, axis=0),
            np.concatenate(ts, axis=0))


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------


@dataclass
class FitResult:
    history: list = field(default_factory=list)
    rel_rms: float = 0.0  # relative to the peak of the target
    charge_error: float = 0.0  # worst |integral error| in electrons


def fit(spec: SurrogateSpec, x0: np.ndarray, cond: np.ndarray, target: np.ndarray,
        steps: int = 4000, lr: float = 2e-3, batch: int | None = None,
        seed: int = 0, on_log=None) -> tuple[ResponseSurrogate, FitResult]:
    """Least-squares fit of the surrogate to a tabulated response."""
    torch.manual_seed(seed)
    if cond.shape[1]:
        spec.cond_lo = tuple(cond.min(axis=0).tolist())
        spec.cond_hi = tuple(cond.max(axis=0).tolist())
    model = ResponseSurrogate(spec)
    scale = float(np.abs(target).max())
    with torch.no_grad():
        model.out_scale.fill_(scale)

    X = torch.as_tensor(x0, dtype=model.dtype)
    C = torch.as_tensor(cond, dtype=model.dtype)
    Y = torch.as_tensor(target, dtype=model.dtype)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(
        opt, gamma=(1e-2) ** (1.0 / max(steps, 1)))
    res = FitResult()
    for step in range(1, steps + 1):
        if batch and batch < len(X):
            idx = torch.randint(0, len(X), (batch,))
            xb, cb, yb = X[idx], C[idx], Y[idx]
        else:
            xb, cb, yb = X, C, Y
        opt.zero_grad(set_to_none=True)
        pred = model(xb, cb if spec.n_cond else None)
        loss = ((pred - yb) / scale).pow(2).mean()
        loss.backward()
        opt.step()
        sched.step()
        if step % max(1, steps // 20) == 0 or step == 1:
            res.history.append((step, float(loss.detach())))
            if on_log:
                on_log(f"step {step:6d}/{steps}  loss {float(loss):.4e}")

    with torch.no_grad():
        pred = model(X, C if spec.n_cond else None)
        err = (pred - Y).numpy()
        res.rel_rms = float(np.sqrt((err ** 2).mean()) / scale)
        res.charge_error = float(np.abs(err.sum(axis=-1)).max())
    return model, res
