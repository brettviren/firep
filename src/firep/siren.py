"""SIREN (Sitzmann et al., arXiv:2006.09661) plus the harmonic ansatz that
makes a LArTPC field problem tractable for one.

Three things are built in so the network never has to learn them:

1. **Exact periodicity.**  The transverse coordinate reaches the network only
   through ``sin``/``cos`` harmonics of the domain period, so *every* function
   the model can represent is periodic.  The side boundary condition is then a
   property of the architecture, not a penalty term competing with the others.

2. **Exact harmonicity of the large-scale solution.**  The potential is written
   ``V = B(x, y) + s N(x, y)`` with ``B`` an analytic harmonic ansatz.  Because
   ``B`` is harmonic for any value of its parameters, ``lap V = s lap N``: the
   PDE residual never has to fight the baseline.

3. **The near-wire logarithm.**  ``B`` contains, for each conductor, the exact
   potential of an infinite periodic row of line charges,

   .. math::
       \\psi_i(x,y) = -\\tfrac12 \\ln\\!\\left[
           \\cosh\\frac{2\\pi (y-y_i)}{W} - \\cos\\frac{2\\pi (x-x_i)}{W}\\right],

   with ``W`` the domain period.  This carries the ``ln r`` behaviour at the
   wire surface -- the one feature a smooth network is worst at -- with a single
   learned coefficient per conductor (physically, its line-charge density).
   Each ``psi_i`` is made to vanish on both faces by subtracting the harmonic
   function that matches its own values there.  That subtraction has a closed
   form: the ``q = 0`` term is a linear ramp in ``y`` and the ``q >= 1`` terms
   are hyperbolic ramps times ``cos``/``sin`` of ``q`` transverse harmonics.
   Only ``q = 0`` matters when ``exp(-2 pi dy / P) << 1``, which holds for a
   drift solve and not for a weighting solve, where ``P`` is many pitches.

What is left for the SIREN is a bounded, smooth correction: the image charges
in the two planes and the multipole corrections from the finite wire radius.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .config import Config
from .geometry import Geometry

LN2 = math.log(2.0)


# --------------------------------------------------------------------------
# SIREN proper
# --------------------------------------------------------------------------


class SineLayer(nn.Module):
    """``sin(omega * (W x + b))`` with the initialisation from the SIREN paper.

    First layer: ``W ~ U(-1/n, 1/n)``.  Hidden layers: ``W ~ U(-c, c)`` with
    ``c = sqrt(6/n)/omega``, which keeps the pre-activations unit-normal so that
    the sine stays in its linear regime at depth.
    """

    def __init__(self, in_features: int, out_features: int, omega: float, first: bool = False):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / omega
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(x))


class Siren(nn.Module):
    """SIREN MLP with a linear read-out."""

    def __init__(
        self,
        in_features: int,
        hidden: int,
        layers: int,
        omega0: float,
        omega_hidden: float,
        out_features: int = 1,
        final_init_scale: float = 1.0,
    ):
        super().__init__()
        mods: list[nn.Module] = [SineLayer(in_features, hidden, omega0, first=True)]
        for _ in range(layers - 1):
            mods.append(SineLayer(hidden, hidden, omega_hidden))
        self.body = nn.Sequential(*mods)
        self.head = nn.Linear(hidden, out_features)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden) / omega_hidden * final_init_scale
            self.head.weight.uniform_(-bound, bound)
            self.head.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


# --------------------------------------------------------------------------
# input encoding
# --------------------------------------------------------------------------


class PeriodicEncoding(nn.Module):
    """Map physical ``(x, y)`` in mm to network features.

    ``x`` becomes ``2*x_harmonics`` sine/cosine features of the domain period;
    ``y`` becomes a single feature affinely scaled to roughly ``[-1, 1]``.
    """

    def __init__(self, period: float, x_harmonics: int, y_center: float, y_scale: float):
        super().__init__()
        if x_harmonics < 1:
            raise ValueError("model.encoding.x_harmonics must be >= 1")
        self.register_buffer("modes", torch.arange(1, x_harmonics + 1, dtype=torch.get_default_dtype()))
        self.period = float(period)
        self.y_center = float(y_center)
        self.y_scale = float(y_scale)
        self.out_features = 2 * x_harmonics + 1

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        phase = (2.0 * math.pi / self.period) * xy[:, 0:1] * self.modes[None, :]
        yhat = (xy[:, 1:2] - self.y_center) / self.y_scale
        return torch.cat((torch.sin(phase), torch.cos(phase), yhat), dim=1)


# --------------------------------------------------------------------------
# harmonic baseline
# --------------------------------------------------------------------------


class HarmonicBaseline(nn.Module):
    """Analytic, exactly-harmonic part of the potential (see module docstring).

    With ``multipole = 0`` each conductor contributes one basis function, the
    periodic line-charge row ``psi_i`` (a monopole).  With ``multipole = 1`` it
    also contributes the two dipole rows ``d psi_i / d x_i`` and
    ``d psi_i / d y_i``, which behave as ``cos(theta)/rho`` and
    ``sin(theta)/rho`` at the wire surface.  A derivative of a harmonic function
    with respect to the *source* position is still harmonic, so this costs
    nothing in exactness -- and it is exactly the leading correction a wire
    needs when its surroundings are not axially symmetric, which in a wire plane
    they never are.

    Every column is made to vanish on both faces by subtracting the harmonic
    interpolant of its own face values, so the basis leaves the cathode and
    ground conditions untouched.  ``wall_harmonics`` caps how many transverse
    harmonics that interpolant keeps; terms below ``1e-10`` are dropped, so a
    drift solve pays for one and a weighting solve for a few tens.
    """

    def __init__(self, geom: Geometry, mode: str, multipole: int = 1,
                 wall_harmonics: int = 64):
        super().__init__()
        self.mode = mode
        self.multipole = int(multipole)
        if self.multipole not in (0, 1):
            raise ValueError("model.baseline.multipole must be 0 or 1")
        self.ylo, self.yhi = float(geom.ylo), float(geom.yhi)
        self.period = float(geom.width)
        self.v_ground = float(geom.v_ground)
        self.v_cathode = float(geom.v_cathode)
        self.k = 2.0 * math.pi / self.period

        dt = torch.get_default_dtype()
        cx = torch.as_tensor(geom.cx, dtype=dt)
        cy = torch.as_tensor(geom.cy, dtype=dt)
        cr = torch.as_tensor(geom.cr, dtype=dt)
        self.register_buffer("cx", cx)
        self.register_buffer("cy", cy)
        self.register_buffer("cr", cr)

        # Asymptotic value of each column at each face, for the de-trending.
        # monopole: -(|k dy| - ln2)/2 ; dx-dipole: 0 ; dy-dipole: +-k r /2.
        k = self.k
        lo = [-0.5 * ((k * (self.ylo - cy)).abs() - LN2)]
        hi = [-0.5 * ((k * (self.yhi - cy)).abs() - LN2)]
        if self.multipole >= 1:
            zero = torch.zeros_like(cx)
            lo += [zero, -0.5 * k * cr]
            hi += [zero, +0.5 * k * cr]
        self.register_buffer("a_lo", torch.cat(lo))
        self.register_buffer("a_hi", torch.cat(hi))

        # The de-trending above is only the q = 0 term of the correct wall
        # correction.  Writing the stable form of psi as a Fourier series in x,
        #
        #     psi = -(|u| - ln2)/2 + sum_q (1/q) e^{-q|u|} cos(q v),
        #
        # every column has closed-form Fourier coefficients on each face, and
        # they are negligible only when e^{-2 pi dy / P} << 1.  That holds for a
        # drift solve (P = one pitch) and emphatically not for a weighting solve
        # (P = 21 pitches), where the ground plane sits well inside a period of
        # the nearest row.  Keep the q >= 1 terms too.
        a_lo_q = torch.exp(-(k * (self.ylo - cy)).abs())   # per conductor
        a_hi_q = torch.exp(-(k * (self.yhi - cy)).abs())
        qmax = max(int(wall_harmonics), 0)
        qs, clo, chi, slo, shi = [], [], [], [], []
        for q in range(1, qmax + 1):
            plo, phi = a_lo_q ** q, a_hi_q ** q
            if float(torch.max(plo).item()) < 1e-10 and \
               float(torch.max(phi).item()) < 1e-10:
                break
            cl = [plo / q]                       # monopole: cos harmonics
            ch = [phi / q]
            sl = [torch.zeros_like(plo)]         # monopole: no sin harmonics
            sh = [torch.zeros_like(phi)]
            if self.multipole >= 1:
                sl += [cr * k * plo]             # x-dipole: sin harmonics
                sh += [cr * k * phi]
                cl += [torch.zeros_like(plo)]
                ch += [torch.zeros_like(phi)]
                s_lo = torch.sign(torch.as_tensor(self.ylo, dtype=dt) - cy)
                s_hi = torch.sign(torch.as_tensor(self.yhi, dtype=dt) - cy)
                cl += [cr * k * s_lo * plo]      # y-dipole: cos harmonics
                ch += [cr * k * s_hi * phi]
                sl += [torch.zeros_like(plo)]
                sh += [torch.zeros_like(phi)]
            order = [0, 1, 2] if self.multipole >= 1 else [0]
            qs.append(q)
            clo.append(torch.cat([cl[i] for i in order]))
            chi.append(torch.cat([ch[i] for i in order]))
            slo.append(torch.cat([sl[i] for i in order]))
            shi.append(torch.cat([sh[i] for i in order]))
        self.n_wall = len(qs)
        if qs:
            q = torch.tensor(qs, dtype=dt)
            clo_, chi_ = torch.stack(clo), torch.stack(chi)      # (nq, ncols)
            slo_, shi_ = torch.stack(slo), torch.stack(shi)
            # cos(q k (x - x_i)) splits into cos/sin of q k x times cos/sin of
            # q k x_i, so the whole correction becomes four matrix products
            # with these (nq, ncols) mixing matrices -- see _wall_correction.
            reps = 1 + 2 * self.multipole
            phase = q[:, None] * self.k * cx.repeat(reps)[None, :]
            cc, ss = torch.cos(phase), torch.sin(phase)
            self.register_buffer("wall_q", q)
            self.register_buffer("wall_m1", clo_ * cc - slo_ * ss)
            self.register_buffer("wall_m2", clo_ * ss + slo_ * cc)
            self.register_buffer("wall_m3", chi_ * cc - shi_ * ss)
            self.register_buffer("wall_m4", chi_ * ss + shi_ * cc)

        ncols = cx.numel() * (1 + 2 * self.multipole)
        self.ncols = ncols
        if mode == "wires":
            self.coeff = nn.Parameter(torch.zeros(ncols))
        else:
            self.register_parameter("coeff", None)

    # -- pieces -------------------------------------------------------------

    def ramp(self, y: torch.Tensor) -> torch.Tensor:
        """Normalised 0..1 ramp from the ground face to the cathode face."""
        return (y - self.ylo) / (self.yhi - self.ylo)

    def linear_part(self, xy: torch.Tensor) -> torch.Tensor:
        t = self.ramp(xy[:, 1:2])
        return self.v_ground + (self.v_cathode - self.v_ground) * t

    def basis(self, xy: torch.Tensor) -> torch.Tensor:
        """De-trended harmonic basis; shape ``(npoints, ncols)``.

        Written in a form stable for ``|y - y_i| >> W`` by factoring
        ``cosh u - cos v = (e^{|u|}/2) (1 - 2 e^{-|u|} cos v + e^{-2|u|})``.
        """
        k = self.k
        dy = k * (xy[:, 1:2] - self.cy[None, :])
        a = dy.abs()
        v = k * (xy[:, 0:1] - self.cx[None, :])
        em = torch.exp(-a)
        # |1 - e^{-|u|} e^{iv}|^2, strictly positive outside the conductor cores.
        core = torch.clamp(1.0 - 2.0 * em * torch.cos(v) + em * em, min=1e-30)

        cols = [-0.5 * (a - LN2 + torch.log(core))]
        if self.multipole >= 1:
            # Scaled by the wire radius so every column is O(1) at the surface.
            rr = self.cr[None, :]
            cols.append(rr * k * em * torch.sin(v) / core)
            cols.append(rr * 0.5 * k * torch.sign(dy) * (1.0 - em * em) / core)
        out = torch.cat(cols, dim=1)

        t = self.ramp(xy[:, 1:2])
        out = out - (self.a_lo[None, :] + (self.a_hi - self.a_lo)[None, :] * t)
        if self.n_wall:
            out = out - self._wall_correction(xy)
        return out

    def _wall_correction(self, xy: torch.Tensor) -> torch.Tensor:
        """The ``q >= 1`` part of the interpolant that zeroes a column on the faces.

        For each transverse harmonic ``q`` the harmonic function matching given
        values on the two faces is a pair of hyperbolic ramps,
        ``sinh(q k (y_hi - y)) / sinh(q k H)`` and its mirror, times ``cos`` or
        ``sin`` of ``q k (x - x_i)``.  Subtracting it leaves the column harmonic
        and periodic, and now zero on both faces for any period ``P``.

        Expanding the phase about the origin,

            cos(q k (x - x_i)) = cos(q k x) cos(q k x_i) + sin(q k x) sin(q k x_i)

        separates point from column, so the sum over ``q`` is four matrix
        products rather than a loop: the per-point factors are ``(n, nq)`` and
        the per-column mixing matrices ``(nq, ncols)`` were built once.
        """
        k, y = self.k, xy[:, 1:2]
        h = self.yhi - self.ylo
        q = self.wall_q[None, :]                     # (1, nq)
        c = q * (k * h)
        den = -torch.expm1(-2.0 * c)
        a1 = q * (k * (self.yhi - y))                # 1 on the ground face
        a2 = q * (k * (y - self.ylo))                # 1 on the cathode face
        glo = (torch.exp(a1 - c) - torch.exp(-a1 - c)) / den
        ghi = (torch.exp(a2 - c) - torch.exp(-a2 - c)) / den
        phase = q * (k * xy[:, 0:1])
        ca, sa = torch.cos(phase), torch.sin(phase)  # (n, nq)
        return ((glo * ca) @ self.wall_m1 + (glo * sa) @ self.wall_m2
                + (ghi * ca) @ self.wall_m3 + (ghi * sa) @ self.wall_m4)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return torch.zeros_like(xy[:, 0:1])
        out = self.linear_part(xy)
        if self.mode == "wires":
            out = out + self.basis(xy) @ self.coeff[:, None]
        return out

    @torch.no_grad()
    def warm_start(self, xy: torch.Tensor, target: torch.Tensor) -> float:
        """Least-squares fit of the basis coefficients to surface potentials.

        ``xy`` are points on conductor surfaces and ``target`` the potential each
        should have.  Returns the residual RMS in volts.  This alone gets the
        wire boundary conditions close, so Adam starts from a physically
        sensible field rather than from noise.
        """
        if self.mode != "wires":
            return float("nan")
        a = self.basis(xy).double()
        b = (target[:, None] - self.linear_part(xy)).double()
        sol = torch.linalg.lstsq(a, b, driver="gelsd").solution
        self.coeff.copy_(sol[:, 0].to(self.coeff.dtype))
        return float((a @ sol - b).pow(2).mean().sqrt())


# --------------------------------------------------------------------------
# the full model
# --------------------------------------------------------------------------


class FieldModel(nn.Module):
    """``V(x, y) = baseline(x, y) + output_scale * SIREN(encode(x, y))``."""

    def __init__(self, cfg: Config, geom: Geometry):
        super().__init__()
        mcfg = cfg.model
        y_center = mcfg.encoding.y_center
        if y_center == "auto":
            y_center = 0.5 * (geom.ylo + geom.yhi)
        y_scale = mcfg.encoding.y_scale
        if y_scale == "auto":
            y_scale = 0.5 * (geom.yhi - geom.ylo)
        out_scale = mcfg.output_scale
        if out_scale == "auto":
            out_scale = geom.potential_span

        self.encoding = PeriodicEncoding(
            period=geom.width,
            x_harmonics=mcfg.encoding.x_harmonics,
            y_center=float(y_center),
            y_scale=float(y_scale),
        )
        self.net = Siren(
            in_features=self.encoding.out_features,
            hidden=mcfg.hidden,
            layers=mcfg.layers,
            omega0=mcfg.omega0,
            omega_hidden=mcfg.omega_hidden,
            final_init_scale=mcfg.final_init_scale,
        )
        self.baseline = HarmonicBaseline(
            geom, mcfg.baseline.mode, mcfg.baseline.multipole,
            mcfg.baseline.wall_harmonics,
        )
        # A buffer, not a plain float, so that an adaptive rescale (see
        # firep.train) travels with the checkpoint.
        self.register_buffer("output_scale", torch.tensor(float(out_scale)))
        self.auto_scale = mcfg.output_scale == "auto"

    def network_part(self, xy: torch.Tensor) -> torch.Tensor:
        """The part of ``V`` whose Laplacian is not identically zero."""
        return self.output_scale * self.net(self.encoding(xy))

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.baseline(xy) + self.network_part(xy)


# --------------------------------------------------------------------------
# differential operators
# --------------------------------------------------------------------------


def laplacian(fn, xy: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """``lap f`` at ``xy`` by automatic differentiation, in 1/mm^2 * [f]."""
    xy = xy.detach().clone().requires_grad_(True)
    val = fn(xy)
    grad = torch.autograd.grad(val.sum(), xy, create_graph=True)[0]
    lap = torch.zeros_like(val)
    for i in range(xy.shape[1]):
        gi = torch.autograd.grad(
            grad[:, i].sum(), xy, create_graph=create_graph, retain_graph=True
        )[0]
        lap = lap + gi[:, i : i + 1]
    return lap


def gradient(fn, xy: torch.Tensor) -> torch.Tensor:
    """``grad f`` at ``xy``; for the potential this is ``-E``."""
    xy = xy.detach().clone().requires_grad_(True)
    val = fn(xy)
    return torch.autograd.grad(val.sum(), xy, create_graph=False)[0]


def numpy_to_tensor(a: np.ndarray, device, dtype) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a), device=device, dtype=dtype)
