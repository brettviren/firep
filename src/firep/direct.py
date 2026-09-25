"""Learning the field response directly: the transport formulation.

This implements Sec. 15.4 of the technical note.  Instead of tracing each
electron through the drift field and applying Shockley-Ramo along its path
(:mod:`firep.response`), the induced charge is treated as a *field* over launch
point and elapsed time,

.. math::

    \\mathcal{Q}_k(\\mathbf r, t) = -q\\,\\varphi_k\\big(X_t(\\mathbf r)\\big),

with :math:`X_t` the drift flow.  Because the drift field is static it obeys the
backward (Koopman) transport equation

.. math::

    \\partial_t \\mathcal{Q}_k = \\mathbf v\\cdot\\nabla\\mathcal{Q}_k,
    \\qquad \\mathcal{Q}_k(\\mathbf r, 0) = -q\\,\\varphi_k(\\mathbf r),

and a SIREN is trained on its residual alone.  No response is ever a target:
the drift and weighting solutions enter only as the coefficients ``v`` and the
initial data.  The traced response therefore stays an independent check, and so
does the Ramo sum rule, which is the long-time limit of the field and is not in
the loss.

The domain
----------
The drift field is periodic over one pitch and mirror-symmetric about every
wire, so ``v_x = 0`` on the lines through a wire (``x = 0``) and midway between
two (``x = p/2``).  No trajectory crosses either line, so the half cell
``0 <= x <= p/2`` is invariant under the flow: the equation can be solved there
with no condition on its sides.  The separatrix that divides two collection
wires is the side ``x = p/2`` itself, so the step of Sec. 15.4 never appears
*inside* the domain.

What would be one field per plane over the whole 21-pitch weighting period is
unfolded onto that half cell as one output per (plane, wire offset): output
``(P, m)`` is the charge induced on wire ``m`` of plane ``P`` by an electron
launched at ``(x, y)`` in the central half cell.  All outputs share one trunk,
because all of them are functions of the same flow.

Above a plane ``y_R`` a few pitches over the topmost electrode the drift is a
uniform translation at speed ``v0``, so there

.. math::

    \\mathcal{Q}(x, y, t) = -q\\varphi(x, y - v_0 t) \\quad (t < t_R),
    \\qquad \\mathcal{Q}(x, y_R, t - t_R) \\quad (t \\ge t_R),

with ``t_R = (y - y_R)/v0``.  The network is needed only below ``y_R``.

The ansatz
----------
.. math::

    \\mathcal{Q}_\\theta = -q\\varphi(\\mathbf r)
        + (1 - e^{-t/\\tau})\\, D(\\mathbf r)\\, \\sigma N_\\theta(\\mathbf r, t),

with ``D = tanh(d/ell)`` and ``d`` the distance to the nearest conductor, makes
the initial condition and the conductor condition exact.  The transport
operator ``L = d/dt - v.grad`` is a single directional derivative in
``(x, y, t)``, so ``L N`` for every output comes from one forward-mode JVP.

Units: mm, us, and charge in units of ``e`` with ``q = -1``, so ``Q = +phi``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.autograd import forward_ad as fwAD

from .config import DirectCfg
from .response import CombinedModel, Response, drift_speed
from .siren import Siren
from .train import resolve_device

Q_ELECTRON = -1.0  # q in units of e


# --------------------------------------------------------------------------
# geometry of the half cell
# --------------------------------------------------------------------------


@dataclass
class HalfCell:
    """The invariant half cell and the conductors that bound it."""

    pitch: float
    ylo: float
    ytop: float  # y_R
    cx: np.ndarray  # conductor centres (the cell's own wires and their images)
    cy: np.ndarray
    cr: np.ndarray
    names: list[str]
    # Surfaces where charge *arrives* (v . n < 0).  Only there does the
    # conductor condition Q = -q phi hold; a wire that repels electrons -- every
    # transparent grid and induction wire -- is an inflow boundary of the
    # backward equation and takes no condition at all.
    absorbing: np.ndarray | None = None
    inward: dict = field(default_factory=dict)  # electrode -> fraction of surface absorbing

    @property
    def half(self) -> float:
        return 0.5 * self.pitch

    def distance(self, xy: np.ndarray, absorbing_only: bool = False
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Distance to the nearest (absorbing) conductor surface and its gradient."""
        sel = (self.absorbing if absorbing_only and self.absorbing is not None
               else np.ones(len(self.cx), dtype=bool))
        cx, cy, cr = self.cx[sel], self.cy[sel], self.cr[sel]
        dx = xy[:, 0:1] - cx[None, :]
        dy = xy[:, 1:2] - cy[None, :]
        rho = np.hypot(dx, dy)
        d = rho - cr[None, :]
        j = np.argmin(d, axis=1)
        k = np.arange(len(xy))
        r = np.maximum(rho[k, j], 1e-12)
        grad = np.column_stack((dx[k, j] / r, dy[k, j] / r))
        return d[k, j], grad


def half_cell(cm: CombinedModel, spec: DirectCfg) -> HalfCell:
    planes = cm.cfg.electrodes_by_y
    pitch = cm.cfg.electrode(planes[0].name).lattice.pitch
    for e in planes:
        if not math.isclose(e.lattice.pitch, pitch, rel_tol=1e-9):
            raise ValueError("the transport solver assumes one common pitch")
        if e.lattice.count != 1 or abs(e.lattice.offset) > 1e-12:
            raise ValueError("the drift model must have one wire per plane at x = 0")
    top = max(e.plane for e in planes)
    bottom = min(e.plane for e in planes)
    ytop = top + spec.top_pitches * pitch if spec.y_top == "auto" else float(spec.y_top)
    ylo = bottom - spec.below if spec.y_bottom == "auto" else float(spec.y_bottom)
    cx, cy, cr, names = [], [], [], []
    for e in planes:
        for img in (-1, 0, 1):
            cx.append(img * pitch)
            cy.append(e.plane)
            cr.append(e.shape.radius)
            names.append(e.name)
    cell = HalfCell(pitch, ylo, ytop, np.array(cx), np.array(cy), np.array(cr), names)
    # Which surfaces absorb?  Measured on the solved field, not assumed.
    th = np.linspace(0.0, 2.0 * math.pi, 128, endpoint=False)
    for e in planes:
        a = e.shape.radius * (1.0 + 1e-3)
        pts = np.column_stack((a * np.cos(th), e.plane + a * np.sin(th)))
        v = velocity(cm, pts)
        vn = v[:, 0] * np.cos(th) + v[:, 1] * np.sin(th)
        cell.inward[e.name] = float(np.mean(vn < 0))
    cell.absorbing = np.array([cell.inward[n] > 0.5 for n in names])
    return cell


# --------------------------------------------------------------------------
# the coefficients of the transport equation
# --------------------------------------------------------------------------


def velocity(cm: CombinedModel, xy: np.ndarray) -> np.ndarray:
    """Electron drift velocity, mm/us, from the drift solution."""
    e = cm.drift_field(xy)
    mag = np.hypot(e[:, 0], e[:, 1])
    sp = cm.speed(mag)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.where(mag[:, None] > 0, e / np.maximum(mag, 1e-30)[:, None], 0.0)
    return -sp[:, None] * unit


def weighting(cm: CombinedModel, planes: list[str], offsets: np.ndarray,
              xy: np.ndarray, pitch: float, grad: bool = True):
    """``phi`` (and ``grad phi``) of every (plane, wire offset) output.

    Output ``(P, m)`` is the weighting potential of the wire ``m`` pitches from
    the sensing wire, i.e. the sensing wire's potential read at ``x - m p``.
    Returns ``phi`` of shape ``(n, nout)`` and ``grad`` of shape ``(n, nout, 2)``.
    """
    n, nw = len(xy), len(offsets)
    phis, grads = [], []
    for plane in planes:
        shifted = np.repeat(xy[None, :, :], nw, axis=0).reshape(-1, 2).copy()
        shifted[:, 0] -= np.repeat(offsets * pitch, n)
        phis.append(cm.weighting_potential(plane, shifted).reshape(nw, n).T)
        if grad:
            g = -cm.weighting_field(plane, shifted).reshape(nw, n, 2)
            grads.append(np.transpose(g, (1, 0, 2)))
    phi = np.concatenate(phis, axis=1)
    return (phi, np.concatenate(grads, axis=1)) if grad else (phi, None)


@dataclass
class Pool:
    """Collocation points with every coefficient evaluated once."""

    xy: np.ndarray  # (n, 2)
    v: np.ndarray  # (n, 2) mm/us
    phi: np.ndarray  # (n, nout)
    lphi: np.ndarray  # (n, nout)  -v . grad phi
    dist: np.ndarray  # (n,)  D = tanh(d / ell)
    ldist: np.ndarray  # (n,)  -v . grad D


def sample_points(cell: HalfCell, n: int, near_fraction: float, near_factor: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Uniform points plus log-radial half annuli around the cell's own wires."""
    n_near = int(n * near_fraction)
    out = []
    got = 0
    while got < n - n_near:
        m = 2 * (n - n_near - got) + 16
        p = np.column_stack((rng.uniform(0.0, cell.half, m),
                             rng.uniform(cell.ylo, cell.ytop, m)))
        p = p[cell.distance(p)[0] > 0]
        out.append(p)
        got += len(p)
    own = np.flatnonzero(np.abs(cell.cx) < 1e-12)
    per = max(1, n_near // len(own))
    for j in own:
        rho = cell.cr[j] * np.exp(rng.uniform(0.0, math.log(near_factor), per))
        th = rng.uniform(-0.5 * math.pi, 0.5 * math.pi, per)
        p = np.column_stack((rho * np.cos(th), cell.cy[j] + rho * np.sin(th)))
        keep = ((p[:, 0] <= cell.half) & (p[:, 1] >= cell.ylo) & (p[:, 1] <= cell.ytop))
        p = p[keep]
        out.append(p[cell.distance(p)[0] > 0])
    pts = np.concatenate(out)
    return pts[rng.permutation(len(pts))][:n]


def build_pool(cm: CombinedModel, cell: HalfCell, planes: list[str], offsets: np.ndarray,
               spec: DirectCfg, rng: np.random.Generator) -> Pool:
    xy = sample_points(cell, spec.pool, spec.near_fraction, spec.near_factor, rng)
    v = velocity(cm, xy)
    phi, g = weighting(cm, planes, offsets, xy, cell.pitch)
    lphi = -(v[:, None, 0] * g[:, :, 0] + v[:, None, 1] * g[:, :, 1])
    d, gd = cell.distance(xy, absorbing_only=True)
    dist = np.tanh(d / spec.ell)
    sech2 = 1.0 - dist ** 2
    ldist = -(sech2 / spec.ell) * (v[:, 0] * gd[:, 0] + v[:, 1] * gd[:, 1])
    return Pool(xy, v, phi, lphi, dist, ldist)


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


class ArrivalModel(nn.Module):
    """The time to arrival ``T(r)`` on an absorbing conductor, learned from
    ``v . grad T = -1`` with ``T = 0`` on the absorbing surfaces.

    ``T`` grows without bound near the stagnation lines, so the network carries
    the bounded ``U = tanh(T / Tc)`` instead, which obeys
    ``v . grad U = -(1 - U^2) / Tc`` and vanishes on the absorbing surfaces by
    construction.  Along a characteristic ``s = T(r) - t`` is constant, so the
    kink in ``Q`` at arrival lies on ``s = 0``: giving the transport network
    ``s`` as an input aligns its hardest feature with one coordinate.  This is
    physics only -- a second transport equation, with no target -- and it is
    the method of characteristics written as a learnable encoding.
    """

    def __init__(self, cell: HalfCell, spec: DirectCfg):
        super().__init__()
        self.tc = float(spec.duration)
        self.ell = float(spec.ell)
        ab = cell.absorbing if cell.absorbing is not None else np.ones(len(cell.cx), bool)
        self.register_buffer("ax", torch.tensor(cell.cx[ab], dtype=torch.float32))
        self.register_buffer("ay", torch.tensor(cell.cy[ab], dtype=torch.float32))
        self.register_buffer("ar", torch.tensor(cell.cr[ab], dtype=torch.float32))
        own = np.flatnonzero(np.abs(cell.cx) < 1e-12)
        self.register_buffer("wx", torch.tensor(cell.cx[own], dtype=torch.float32))
        self.register_buffer("wy", torch.tensor(cell.cy[own], dtype=torch.float32))
        self.register_buffer("wr", torch.tensor(cell.cr[own], dtype=torch.float32))
        self.rho_max = float(math.hypot(cell.half, cell.ytop - cell.ylo))
        yc, ys = 0.5 * (cell.ylo + cell.ytop), 0.5 * (cell.ytop - cell.ylo)
        self.register_buffer("shift", torch.tensor([0.5 * cell.half, yc]))
        self.register_buffer("scale", torch.tensor([0.5 * cell.half, ys]))
        self.net = Siren(2 + len(own), spec.arrival_hidden, spec.arrival_layers,
                         spec.arrival_omega0, 30.0, out_features=1, final_init_scale=1.0)

    def features(self, xy: torch.Tensor) -> torch.Tensor:
        dx = xy[:, 0:1] - self.wx[None, :]
        dy = xy[:, 1:2] - self.wy[None, :]
        rho = torch.sqrt(dx * dx + dy * dy)
        span = torch.log(self.rho_max / self.wr)
        lrho = 2.0 * torch.log(rho / self.wr[None, :]) / span[None, :] - 1.0
        return torch.cat(((xy - self.shift) / self.scale, lrho), dim=1)

    def u(self, xy: torch.Tensor) -> torch.Tensor:
        dx = xy[:, 0:1] - self.ax[None, :]
        dy = xy[:, 1:2] - self.ay[None, :]
        d = torch.sqrt(dx * dx + dy * dy) - self.ar[None, :]
        gate = torch.tanh(torch.clamp(d.min(dim=1).values, min=0.0) / self.ell)
        return torch.tanh(gate * nn.functional.softplus(self.net(self.features(xy))[:, 0]))

    def time(self, xy: torch.Tensor) -> torch.Tensor:
        u = torch.clamp(self.u(xy), max=math.tanh(3.0))
        return self.tc * torch.atanh(u)


def train_arrival(arr: ArrivalModel, pool: "Pool", spec: DirectCfg, v0: float,
                  pitch: float, device: torch.device, log=None) -> dict:
    """Minimise ``(v . grad U + (1 - U^2)/Tc)``, per unit path, over the pool."""
    xy = torch.as_tensor(pool.xy, dtype=torch.float32, device=device)
    v = torch.as_tensor(pool.v, dtype=torch.float32, device=device)
    vmag = torch.linalg.vector_norm(v, dim=1)
    pscale = pitch / torch.clamp(vmag, min=spec.v_floor * v0)
    opt = torch.optim.Adam(arr.parameters(), lr=spec.lr)
    gamma = (spec.lr_final / spec.lr) ** (1.0 / max(1, spec.arrival_steps))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    gen = torch.Generator(device=device)
    gen.manual_seed(spec.seed + 1)
    hist = {"step": [], "loss": []}
    t0 = time.time()
    for step in range(1, spec.arrival_steps + 1):
        idx = torch.randint(0, len(xy), (spec.batch,), device=device, generator=gen)
        with fwAD.dual_level():
            out = arr.u(fwAD.make_dual(xy[idx], v[idx]))
            u, vgu = fwAD.unpack_dual(out)
        res = pscale[idx] * (vgu + (1.0 - u * u) / arr.tc)
        loss = (res ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if spec.clip > 0:
            torch.nn.utils.clip_grad_norm_(arr.parameters(), spec.clip)
        opt.step()
        sched.step()
        if step % spec.log_every == 0 or step == spec.arrival_steps:
            hist["step"].append(step)
            hist["loss"].append(float(loss))
            if log:
                log(f"arrival step {step:6d}  residual {float(loss):.4e}  {time.time() - t0:6.1f}s")
    for p_ in arr.parameters():
        p_.requires_grad_(False)
    return hist


class DirectModel(nn.Module):
    """``Q_theta(x, y, t)`` for every (plane, wire offset), per the ansatz."""

    def __init__(self, cell: HalfCell, planes: list[str], offsets: np.ndarray,
                 spec: DirectCfg, v0: float):
        super().__init__()
        self.planes = list(planes)
        self.offsets = np.asarray(offsets, dtype=int)
        self.nout = len(planes) * len(offsets)
        self.pitch = cell.pitch
        self.ylo, self.ytop = cell.ylo, cell.ytop
        self.duration = float(spec.duration)
        self.tau = float(spec.tau)
        self.sigma = float(spec.sigma)
        self.v0 = float(v0)
        # Inputs: the three affine coordinates, the log-distance to each of the
        # cell's own wires, and a log-time.  Both logs resolve what happens
        # within tens of microns of a wire in the first fraction of a
        # microsecond, which is where every sharp feature of the response is
        # made.  All are torch functions of (x, y, t), so the transport
        # derivative through them comes out of the same JVP.
        own = np.flatnonzero(np.abs(cell.cx) < 1e-12)
        self.register_buffer("wx", torch.tensor(cell.cx[own], dtype=torch.float32))
        self.register_buffer("wy", torch.tensor(cell.cy[own], dtype=torch.float32))
        self.register_buffer("wr", torch.tensor(cell.cr[own], dtype=torch.float32))
        self.log_features = bool(spec.log_features)
        self.t_log = float(spec.t_log)
        self.rho_max = float(math.hypot(cell.half, cell.ytop - cell.ylo))
        n_in = 3 + (len(own) + 1 if self.log_features else 0)
        self.arrival = ArrivalModel(cell, spec) if spec.arrival else None
        self.arrival_log = float(spec.arrival_log)
        if self.arrival is not None:
            n_in += 2  # s = (T(r) - t) / duration, and relu(s): the kink at arrival
            if self.arrival_log > 0:
                n_in += 1  # ln(1 + relu(T - t) / tau_a): the wire's ln(rho) in time
        self.net = Siren(n_in, spec.hidden, spec.layers, spec.omega0, spec.omega_hidden,
                         out_features=self.nout, final_init_scale=spec.final_init_scale)
        # Affine map to the network's [-1, 1]^3 inputs, and its Jacobian.
        yc = 0.5 * (self.ylo + self.ytop)
        ys = 0.5 * (self.ytop - self.ylo)
        self.register_buffer("shift", torch.tensor([0.5 * cell.half, yc, 0.5 * self.duration]))
        self.register_buffer("scale", torch.tensor([0.5 * cell.half, ys, 0.5 * self.duration]))

    def index(self, plane: str, offset: int) -> int:
        return self.planes.index(plane) * len(self.offsets) + int(
            np.flatnonzero(self.offsets == offset)[0])

    def inputs(self, xyt: torch.Tensor) -> torch.Tensor:
        feats = [(xyt - self.shift) / self.scale]
        if self.log_features:
            dx = xyt[:, 0:1] - self.wx[None, :]
            dy = xyt[:, 1:2] - self.wy[None, :]
            rho = torch.sqrt(dx * dx + dy * dy)
            span = torch.log(self.rho_max / self.wr)
            feats.append(2.0 * torch.log(rho / self.wr[None, :]) / span[None, :] - 1.0)
            feats.append(2.0 * torch.log1p(xyt[:, 2:3] / self.t_log)
                         / math.log1p(self.duration / self.t_log) - 1.0)
        if self.arrival is not None:
            togo = self.arrival.time(xyt[:, :2])[:, None] - xyt[:, 2:3]
            sgo = togo / self.duration
            feats += [sgo, torch.relu(sgo)]
            if self.arrival_log > 0:
                # Near an absorbing wire phi ~ ln(rho), and the electron closes
                # the last stretch at speed v, so Q ~ ln(1 + v s / a) in the
                # time-to-go s: the wire's logarithm carried into time, with
                # scale a/v.  Handing it over analytically is the response's
                # counterpart of the psi columns in B_c.
                la = math.log1p(self.duration / self.arrival_log)
                feats.append(2.0 * torch.log1p(torch.relu(togo) / self.arrival_log) / la - 1.0)
        return torch.cat(feats, dim=1)

    def gate(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e = torch.exp(-t / self.tau)
        return 1.0 - e, e / self.tau

    def forward(self, xyt: torch.Tensor, phi: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        """``Q`` at ``xyt`` given the precomputed ``phi`` and ``D`` there."""
        g, _ = self.gate(xyt[:, 2:3])
        return -Q_ELECTRON * phi + g * dist[:, None] * self.sigma * self.net(self.inputs(xyt))

    def residual(self, xyt: torch.Tensor, v: torch.Tensor, lphi: torch.Tensor,
                 dist: torch.Tensor, ldist: torch.Tensor):
        """``L Q = dQ/dt - v . grad Q`` for every output, and ``N`` itself.

        ``L`` is a directional derivative along ``(-v_x, -v_y, 1)`` in
        ``(x, y, t)``: one forward-mode JVP gives it for every output.
        """
        tangent = torch.cat((-v, torch.ones_like(v[:, :1])), dim=1)
        with fwAD.dual_level():
            out = self.net(self.inputs(fwAD.make_dual(xyt, tangent)))
            n, ln = fwAD.unpack_dual(out)
        g, dg = self.gate(xyt[:, 2:3])
        dd = dist[:, None]
        lq = (-Q_ELECTRON * lphi
              + self.sigma * (dg * dd * n + g * ldist[:, None] * n + g * dd * ln))
        return lq, n


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


@dataclass
class DirectHistory:
    step: list = field(default_factory=list)
    loss: list = field(default_factory=list)
    weighted: list = field(default_factory=list)
    min_weight: list = field(default_factory=list)
    lr: list = field(default_factory=list)
    wall: list = field(default_factory=list)
    arrival: dict | None = None


def causal_weights(bin_loss: torch.Tensor, eps: float) -> torch.Tensor:
    """``w_i = exp(-eps sum_{j<i} L_j)``, detached (Wang, Sankaran, Perdikaris)."""
    cum = torch.cumsum(bin_loss.detach(), 0) - bin_loss.detach()
    return torch.exp(-eps * cum)


def train(model: DirectModel, pool: Pool, spec: DirectCfg, device: torch.device,
          log=None) -> DirectHistory:
    """Minimise the transport residual, Eq. (response-loss) of the note."""
    dt = torch.float32
    tens = {k: torch.as_tensor(getattr(pool, k), dtype=dt, device=device)
            for k in ("xy", "v", "lphi", "dist", "ldist")}
    npool = len(pool.xy)
    # Each pool point's learned arrival time, for sampling where the bandwidth
    # is: the kink (and, with arrival_log, the logarithm) sits at t = T(r).
    t_arr = None
    if model.arrival is not None and spec.arrival_time_fraction > 0:
        with torch.no_grad():
            t_arr = torch.cat([model.arrival.time(tens["xy"][i:i + 65536])
                               for i in range(0, npool, 65536)])
    # The error in Q accumulates along a characteristic as the integral of the
    # residual over time, i.e. of residual/|v| over path length.  Normalising
    # per unit *path* ("electrons per pitch of path") therefore weights each
    # point by how much its error matters downstream; per unit *time* over-
    # weights the fast near-wire region.  |v| is floored near the saddle.
    if spec.normalise == "path":
        vmag = torch.linalg.vector_norm(tens["v"], dim=1)
        pscale = model.pitch / torch.clamp(vmag, min=spec.v_floor * model.v0)
    else:
        pscale = torch.full((npool,), model.pitch / model.v0, device=device, dtype=dt)
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    opt = torch.optim.Adam(params, lr=spec.lr)
    gamma = (spec.lr_final / spec.lr) ** (1.0 / max(1, spec.steps))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    gen = torch.Generator(device=device)
    gen.manual_seed(spec.seed)
    nb = spec.causal_bins
    hist = DirectHistory()
    t0 = time.time()
    for step in range(1, spec.steps + 1):
        idx = torch.randint(0, npool, (spec.batch,), device=device, generator=gen)
        # Stratified in time so every causal bin is populated every step.
        tb = torch.randint(0, nb, (spec.batch,), device=device, generator=gen)
        tt = (tb + torch.rand(spec.batch, device=device, generator=gen)) * (model.duration / nb)
        # A fraction of the times log-uniform, so the fast fronts near the
        # wires at small t are sampled at all.
        nlog = int(spec.log_time_fraction * spec.batch)
        if nlog:
            u = torch.rand(nlog, device=device, generator=gen)
            tt[:nlog] = spec.t_log * ((1.0 + model.duration / spec.t_log) ** u - 1.0)
            tb[:nlog] = torch.clamp((tt[:nlog] / (model.duration / nb)).long(), 0, nb - 1)
        # And a fraction just before and after each point's own arrival, with
        # the offset log-uniform down to t_log: the feature there has a
        # bandwidth of order 1/(offset), and points spaced more coarsely than
        # that leave the network free to ring between them.
        narr = int(spec.arrival_time_fraction * spec.batch) if t_arr is not None else 0
        if narr:
            sl = slice(nlog, nlog + narr)
            u = torch.rand(narr, device=device, generator=gen)
            off = spec.t_log * ((1.0 + 1.0 / spec.t_log) ** u - 1.0)  # t_log .. ~1 us
            sign = torch.where(torch.rand(narr, device=device, generator=gen) < 0.7, -1.0, 1.0)
            tt[sl] = torch.clamp(t_arr[idx[sl]] + sign * off, 0.0, model.duration)
            tb[sl] = torch.clamp((tt[sl] / (model.duration / nb)).long(), 0, nb - 1)
        xyt = torch.cat((tens["xy"][idx], tt[:, None]), dim=1)
        lq, _ = model.residual(xyt, tens["v"][idx], tens["lphi"][idx],
                               tens["dist"][idx], tens["ldist"][idx])
        per = ((pscale[idx, None] * lq) ** 2).mean(dim=1)
        bins = torch.zeros(nb, device=device, dtype=dt).index_add_(0, tb, per)
        counts = torch.bincount(tb, minlength=nb).clamp_min(1).to(dt)
        bin_loss = bins / counts
        w = causal_weights(bin_loss, spec.causal_eps) if spec.causal_eps > 0 else torch.ones_like(bin_loss)
        loss = (w * bin_loss).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if spec.clip > 0:
            torch.nn.utils.clip_grad_norm_(params, spec.clip)
        opt.step()
        sched.step()
        if step % spec.log_every == 0 or step == 1 or step == spec.steps:
            hist.step.append(step)
            hist.loss.append(float(bin_loss.mean()))
            hist.weighted.append(float(loss))
            hist.min_weight.append(float(w.min()))
            hist.lr.append(float(sched.get_last_lr()[0]))
            hist.wall.append(time.time() - t0)
            if log:
                log(f"step {step:7d}  residual {hist.loss[-1]:.4e}  weighted {hist.weighted[-1]:.4e}"
                    f"  min causal weight {hist.min_weight[-1]:.3f}  lr {hist.lr[-1]:.2e}"
                    f"  {hist.wall[-1]:7.1f}s")
    return hist


# --------------------------------------------------------------------------
# evaluation: the response, on the same grid as firep.response
# --------------------------------------------------------------------------


def launch_charge(model: DirectModel, cm: CombinedModel, x0: np.ndarray, y_start: float,
                  edges: np.ndarray, device: torch.device, chunk: int = 8192):
    """``Q[out, impact, edge]`` for electrons launched at ``(x0, y_start)``.

    Above ``y_R`` the translation of the weighting potential; below it the
    learned field.  Also returns ``dQ/dt`` at the same times from the network.
    """
    t_r = max(0.0, (y_start - model.ytop) / model.v0)
    ni, ne = len(x0), len(edges)
    q = np.zeros((model.nout, ni, ne))
    dq = np.zeros((model.nout, ni, ne))
    for i, x in enumerate(x0):
        above = edges < t_r
        if above.any():
            pts = np.column_stack((np.full(above.sum(), x), y_start - model.v0 * edges[above]))
            phi, g = weighting(cm, model.planes, model.offsets, pts, model.pitch)
            q[:, i, above] = (-Q_ELECTRON * phi).T
            # dQ/dt = -q grad phi . v with v = (0, -v0)
            dq[:, i, above] = (-Q_ELECTRON * g[:, :, 1] * (-model.v0)).T
        below = ~above
        if below.any():
            tl = np.clip(edges[below] - t_r, 0.0, model.duration)
            y_learn = min(y_start, model.ytop)  # launches below y_R start in the learned region
            pt = np.column_stack((np.full(below.sum(), x), np.full(below.sum(), y_learn)))
            phi, _ = weighting(cm, model.planes, model.offsets, pt[:1], model.pitch, grad=False)
            cell = model._cell
            d, _ = cell.distance(pt[:1], absorbing_only=True)
            dist = float(np.tanh(d[0] / model._ell))
            for a in range(0, len(tl), chunk):
                tb = torch.as_tensor(tl[a:a + chunk], dtype=torch.float32, device=device)
                xyt = torch.stack((torch.full_like(tb, x), torch.full_like(tb, y_learn), tb), 1)
                xyt.requires_grad_(True)
                ph = torch.as_tensor(phi, dtype=torch.float32, device=device).expand(len(tb), -1)
                dd = torch.full((len(tb),), dist, dtype=torch.float32, device=device)
                tangent = torch.zeros_like(xyt)
                tangent[:, 2] = 1.0
                with fwAD.dual_level():
                    out = model(fwAD.make_dual(xyt.detach(), tangent), ph, dd)
                    val, der = fwAD.unpack_dual(out)
                sl = np.flatnonzero(below)[a:a + chunk]
                q[:, i, sl] = val.detach().cpu().numpy().T
                dq[:, i, sl] = der.detach().cpu().numpy().T
    return q, dq, t_r


def response(model: DirectModel, cm: CombinedModel, x0: np.ndarray, y_start: float,
             tick: float, device: torch.device) -> Response:
    """The learned field response, as a :class:`firep.response.Response`.

    ``current`` is the charge form, ``(Q(t_{i+1}) - Q(t_i)) / tick``, exactly as
    :func:`firep.response.response` bins the traced one; ``current_field`` is
    the network's own ``dQ/dt`` at the bin centres, the analogue of the field
    form.
    """
    t_r = max(0.0, (y_start - model.ytop) / model.v0)
    nt = int(math.ceil((t_r + model.duration) / tick))
    edges = np.arange(nt + 1) * tick
    centres = 0.5 * (edges[:-1] + edges[1:])
    q, _, _ = launch_charge(model, cm, x0, y_start, edges, device)
    _, dq, _ = launch_charge(model, cm, x0, y_start, centres, device)
    npl, nw, ni = len(model.planes), len(model.offsets), len(x0)
    q = q.reshape(npl, nw, ni, nt + 1)
    dq = dq.reshape(npl, nw, ni, nt)
    current = np.diff(q, axis=-1) / tick
    integrated = q[..., -1] - q[..., 0]
    phi_start = np.abs(q[:, int(np.flatnonzero(model.offsets == 0)[0]), :, 0])
    peak = float(np.abs(current).max()) or 1.0
    # The launch is a straight line down to y_R; below that the learned field
    # carries no trajectory, so none is recorded.
    tr = np.array([0.0, t_r])
    traj_y = np.column_stack([np.array([y_start, model.ytop])] * ni)
    traj_x = np.tile(np.asarray(x0, float), (2, 1))
    return Response(
        time=centres, impact=np.asarray(x0, float), offsets=model.offsets.copy(),
        planes=list(model.planes), current=current, current_field=dq,
        integrated=integrated, landed=["learned"] * ni, e_max=0.0,
        phi_start=phi_start,
        agreement=float(np.abs(current - dq).max() / peak),
        trajectory={"t": tr, "x": traj_x, "y": traj_y,
                    "alive": np.ones((2, ni), dtype=bool)},
    )


# --------------------------------------------------------------------------
# the whole thing, and checkpoints
# --------------------------------------------------------------------------


def drift_speed_at(cm: CombinedModel, y: float) -> float:
    """Speed of the uniform translation above ``y_R``."""
    v = velocity(cm, np.array([[0.25 * cm.cfg.electrode(cm.cfg.electrodes[0].name).lattice.pitch, y]]))
    return float(np.hypot(*v[0]))


def build(cm: CombinedModel, spec: DirectCfg, planes: list[str] | None = None,
          n_wires: int | None = None):
    planes = planes or [e.name for e in cm.cfg.electrodes_by_y if e.name in cm.weighting]
    if n_wires is None:
        n_wires = min(w.cfg.electrode(w.plane).lattice.count for w in cm.weighting.values())
    half = (n_wires - 1) // 2
    offsets = np.arange(-half, half + 1)
    cell = half_cell(cm, spec)
    v0 = drift_speed_at(cm, cell.ytop)
    model = DirectModel(cell, planes, offsets, spec, v0)
    model._cell = cell
    model._ell = spec.ell
    return model, cell


def fit(cm: CombinedModel, spec: DirectCfg, log=None):
    """Build the pool, train, and return ``(model, cell, pool, history)``."""
    device = resolve_device(spec.device)
    torch.manual_seed(spec.seed)
    rng = np.random.default_rng(spec.seed)
    model, cell = build(cm, spec)
    t0 = time.time()
    pool = build_pool(cm, cell, model.planes, model.offsets, spec, rng)
    if log:
        log(f"half cell x in [0, {cell.half:g}] mm, y in [{cell.ylo:g}, {cell.ytop:g}] mm, "
            f"t in [0, {spec.duration:g}] us;  v0 = {model.v0:.4f} mm/us")
        log("absorbing fraction of each wire surface: " + ", ".join(
            f"{k} {v:.2f}" for k, v in cell.inward.items())
            + "  (the conductor condition is imposed only where it is > 0.5)")
        log(f"outputs: {len(model.planes)} planes x {len(model.offsets)} wires = {model.nout}; "
            f"pool {len(pool.xy)} points, coefficients in {time.time() - t0:.1f}s")
    model.to(device)
    arrival_hist = None
    if model.arrival is not None:
        arrival_hist = train_arrival(model.arrival, pool, spec, model.v0, cell.pitch,
                                     device, log=log)
    hist = train(model, pool, spec, device, log=log)
    hist.arrival = arrival_hist
    return model, cell, pool, hist


def save(path: str, model: DirectModel, spec: DirectCfg, hist: DirectHistory,
         drift_path: str, weight_paths: list[str]) -> None:
    from .config import _plain

    torch.save({
        "state_dict": model.state_dict(),
        "spec": _plain(spec),
        "planes": model.planes,
        "offsets": model.offsets.tolist(),
        "history": vars(hist),
        "drift": drift_path,
        "weights": list(weight_paths),
    }, path)


# Options added after checkpoints existed, and the value that is "off".
LEGACY = {"arrival": False, "arrival_log": 0.0, "arrival_time_fraction": 0.0,
          "normalise": "path"}


def load(path: str, cm: CombinedModel, device: str | None = None):
    from .config import _build

    blob = torch.load(path, map_location="cpu", weights_only=False)
    # A checkpoint written before an option existed was trained without it:
    # fill such keys with the value that reproduces that behaviour, not with
    # today's default (which may change the network's inputs).
    stored = dict(LEGACY)
    stored.update(blob["spec"])
    spec = _build(DirectCfg, stored)
    if device is not None:
        spec.device = device
    model, cell = build(cm, spec, planes=blob["planes"], n_wires=len(blob["offsets"]))
    model.load_state_dict(blob["state_dict"])
    if model.arrival is not None:
        for p_ in model.arrival.parameters():
            p_.requires_grad_(False)
    dev = resolve_device(spec.device)
    model.to(dev).eval()
    return model, cell, spec, blob, dev
