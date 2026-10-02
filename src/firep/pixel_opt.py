"""A differentiable model of a pixel anode with a focusing grid, for design
optimisation.

The design: rounded-square pads of half-width ``a`` (corner radius ``0.4 a``)
lying flush on a charged PCB at ``y0``, and a thin grid at ``y0 + h`` with round
holes of radius ``r`` centred over the pads, biased ``dV`` below the pads.  The
drift field above the grid is held at ``E0``: the cathode is at
``-dV - E0 (top - y0 - h)``.  The weighting solve has FR4 under the pads.

The fields are the spectral single layer of :mod:`firep.pixel`, rebuilt so
that every quantity depends smoothly and differentiably on
``theta = (a, r, h, dV)``:

* Every density is either fixed or an exact scaling about its pad's (hole's)
  centre, ``f((x - c)/s)`` with ``s = a`` or ``r``.  Differentiating a sampled
  singular density with respect to where its edge sits is ill-posed, but the
  scaling identity

      dS/ds = (2/s) S - (i/s) (k_x FFT(xi_x g) + k_z FFT(xi_z g)),

  with ``xi`` the offset from the centre, gives the exact derivative of its
  spectrum from FFTs of regular sampled functions.
* The grid height enters the slab Green's function analytically.
* The voltages enter the fit's targets and offset linearly.

The derivatives of the fit's design matrix with respect to ``(a, r, h)`` are
propagated forward (three tangent columns), and the fit is linearised about
the current point, so that everything downstream -- the least-squares fit,
the series coefficients, the drift paths, the induced currents, the diffusion
smearing and the objective -- is an ordinary torch graph whose gradient is
exact at that point.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from . import pixel as X

CORNER = 0.4  # corner radius / half-width of the rounded-square pad
TRUNCATE_SLIDING = True  # cut the sensitivity chain of paths sliding on the board


# --------------------------------------------------------------------------
# slab Green's function with a PCB, differentiable in the source height
# --------------------------------------------------------------------------


def green(kappa, y, yp, top, kind, pcb, y0):
    """``g_kappa(y, y_p)`` (rows: points, columns: modes) above a PCB surface
    ``y0`` of relative permittivity ``pcb`` (0: a charged, zero-slope board).
    ``y`` and ``yp`` may be tensors carrying gradients; ``y, yp >= y0``."""
    y = y.reshape(-1, 1)
    yp = torch.as_tensor(yp, dtype=y.dtype, device=y.device).reshape(-1, 1) if not torch.is_tensor(yp) \
        else yp.reshape(-1, 1).to(y.dtype)
    ylo, yhi = torch.minimum(y, yp), torch.maximum(y, yp)
    k = kappa.reshape(1, -1)
    safe = torch.where(k > 0, k, torch.ones_like(k))
    t = torch.tanh(safe * y0) if kind == "drift" else 1.0 / torch.tanh(safe * y0)
    rho = pcb * t
    e = torch.exp(-safe * (yhi - ylo))
    low = (1 + rho) + (1 - rho) * torch.exp(-2 * safe * (ylo - y0))
    b = torch.exp(-2.0 * safe * (top - yhi))
    c = math.exp(0) * torch.exp(-2.0 * safe * (top - y0))
    if kind == "drift":
        g = e * low * (1 - b) / ((1 + rho) + (1 - rho) * c) / safe
        g0 = 2.0 * (top - yhi)
    else:
        g = e * low * (1 + b) / ((1 + rho) - (1 - rho) * c) / safe
        g0 = 2.0 * (y0 / pcb + (ylo - y0))
    return torch.where(k > 0, g, g0.expand_as(g))


def rounded_square_sdf_t(x, z, half, radius):
    q = torch.stack((x.abs(), z.abs()), -1) - (half - radius)
    return torch.linalg.vector_norm(q.clamp_min(0), dim=-1) + q.max(-1).values.clamp_max(0) - radius


def d4_t(f):
    """Average a centred square grid over the 8 elements of D4 (torch)."""
    acc = torch.zeros_like(f)
    for g in (f, f.T):
        for r in range(4):
            acc = acc + torch.rot90(g, r, (0, 1))
    return acc / 8.0


# --------------------------------------------------------------------------
# the domain
# --------------------------------------------------------------------------


@dataclass
class Domain:
    """One solve: a drift cell (``ncell = 1``, charged board, cathode above)
    or an ``N x N`` weighting domain (FR4 below, zero-slope top)."""

    kind: str
    ncell: int
    grid: int
    modes: int
    pitch: float = 4.4
    y0: float = 10.0
    top: float = 149.9
    pcb: float = 0.0
    device: str = "cuda"
    dtype: torch.dtype = torch.float64
    arr: X.PadArray = field(init=False)

    def __post_init__(self):
        self.arr = X.PadArray(pitch=self.pitch, ncell=self.ncell, y_pad=self.y0, top=self.top, pcb=self.pcb)
        self.L = self.arr.width
        self.n = int(self.grid) | 1
        self.h = self.L / self.n
        dev, dt = self.device, self.dtype
        n = self.n
        xs = (torch.arange(n, device=dev, dtype=dt) - n // 2) * self.h
        self.X, self.Z = torch.meshgrid(xs, xs, indexing="ij")
        # the offset of every node from its nearest pad (and hole) centre
        c = torch.round(self.X / self.pitch) * self.pitch
        d = torch.round(self.Z / self.pitch) * self.pitch
        self.XI, self.ZE = self.X - c, self.Z - d
        f = torch.fft.fftfreq(n, d=1.0 / n, device=dev).to(dt)
        self.kx = (2 * math.pi / self.L) * f[:, None].expand(n, n)
        self.kz = (2 * math.pi / self.L) * f[None, :].expand(n, n)
        self.kfull = torch.sqrt(self.kx ** 2 + self.kz ** 2).reshape(-1)
        m = torch.arange(self.modes + 1, device=dev, dtype=dt)
        self.m = m
        self.k = 2 * math.pi / self.L
        self.kappa = (self.k * torch.sqrt(m[:, None] ** 2 + m[None, :] ** 2)).reshape(-1)
        w = torch.full((self.modes + 1,), 2.0, device=dev, dtype=dt)
        w[0] = 1.0
        self.wmn = w[:, None] * w[None, :]
        self.centres = self.arr.pad_centres()
        self.classes = self.arr.classes()

    def G(self, yq, yp):
        return green(self.kfull, torch.as_tensor(yq, device=self.device, dtype=self.dtype).reshape(1), yp,
                     self.top, self.kind, self.pcb, self.y0)[0].reshape(self.n, self.n)


def _spectrum(dom, g):
    """Unshifted spectrum of a centred grid density, normalised so that
    ``ifft2(S * G)`` is its potential."""
    return torch.fft.fft2(torch.fft.ifftshift(g)) * (dom.h ** 2 / dom.L ** 2) * dom.n ** 2


def _scaled_spectrum(dom, g, s):
    """Spectrum of a density that is an exact scaling about the nearest centre
    by ``s``, and its derivative with respect to ``s`` (the scaling identity)."""
    S = _spectrum(dom, g)
    Sx, Sz = _spectrum(dom, dom.XI * g), _spectrum(dom, dom.ZE * g)
    dS = (2.0 / s) * S - (1j / s) * (dom.kx * Sx + dom.kz * Sz)
    return S, dS


def _scaled_spectrum_tapered(dom, g, f_xigradE, s):
    """As :func:`_scaled_spectrum` for ``g = f((x - c)/s) E(x)`` with a fixed
    smooth taper ``E``: the identity gains the regular term
    ``FFT(f (xi . grad E)) / s``."""
    S, dS = _scaled_spectrum(dom, g, s)
    return S, dS + _spectrum(dom, f_xigradE) / s


def taper(XI, ZE, P):
    """A fixed taper, 1 at the cell centre and 0 on its sides, flat over most
    of the cell: E = (1 - (2 xi/P)^16)(1 - (2 zeta/P)^16); and xi . grad E."""
    u, v = 2 * XI / P, 2 * ZE / P
    inside = (u.abs() < 1) & (v.abs() < 1)
    ex, ez = 1 - u ** 16, 1 - v ** 16
    E = torch.where(inside, ex * ez, torch.zeros_like(u))
    xgE = torch.where(inside, -16 * u ** 16 * ez - 16 * v ** 16 * ex, torch.zeros_like(u))
    return E, xgE


def _place(dom, cls, local):
    """Put a density built on a window about pad ``cls[0]`` onto the grid and
    symmetrise; ``local`` is a function of the window's (xi, zeta)."""
    n, h = dom.n, dom.h
    i = cls[0]
    ci = n // 2 + int(round(dom.centres[i, 0] / h))
    cj = n // 2 + int(round(dom.centres[i, 1] / h))
    hw = int(math.ceil(0.5 * dom.pitch / h)) + 1
    loc = (torch.arange(-hw, hw + 1, device=dom.device, dtype=dom.dtype)) * h
    XI, ZE = torch.meshgrid(loc, loc, indexing="ij")
    val = local(XI, ZE)
    dens = torch.zeros((n, n), device=dom.device, dtype=dom.dtype)
    ix = (ci + torch.arange(-hw, hw + 1, device=dom.device)) % n
    jx = (cj + torch.arange(-hw, hw + 1, device=dom.device)) % n
    dens.index_put_((ix[:, None].expand_as(val), jx[None, :].expand_as(val)), val, accumulate=True)
    raw = float(dens.abs().max())
    return d4_t(dens), raw


# --------------------------------------------------------------------------
# basis terms
# --------------------------------------------------------------------------


def pad_terms(ring):
    d = 4 if ring <= 1 else 2
    sc = [None, 0.05, 0.25] if ring <= 1 else [None, 0.25]
    return [(p, q, s) for s in sc for p in range(d + 1) for q in range(d + 1 - p)]


def grid_terms(ring):
    """("tile", p, q): the cell's monomial outside the hole; ("rim", p, q, s):
    a band at the hole's rim with the 1/sqrt edge factor."""
    dt, dr = (4, 2) if ring <= 1 else (2, 0)
    out = [("tile", p, q) for p in range(dt + 1) for q in range(dt + 1 - p)]
    for s in ((0.04, 0.15) if ring <= 1 else (0.1,)):
        out += [("rim", p, q, s) for p in range(dr + 1) for q in range(dr + 1 - p)]
    return out


# --------------------------------------------------------------------------
# the linearised solve
# --------------------------------------------------------------------------


@dataclass
class Linear:
    """A solve's design at theta0 and its tangents: rows = fit nodes, columns
    = basis densities.  ``D`` (rows, cols), ``dD`` (3, rows, cols) for
    (a, r, h); ``A`` (cols, M+1, M+1) series coefficients and ``dA``; which
    sheet each column is on (0 pads, 1 grid); the fit's row targets are
    assembled by the caller; ``q_pad`` the charge per column on the sensing
    pad (for its capacitance)."""

    D: torch.Tensor
    dD: torch.Tensor
    A: torch.Tensor
    dA: torch.Tensor
    sheet: torch.Tensor
    row_kind: np.ndarray  # 0: a pad node (owner in row_owner), 1: a grid node
    row_owner: np.ndarray
    col_centre: torch.Tensor  # columns on the sensing pad's class, for its charge
    q0: torch.Tensor  # integral of each column's density over the grid
    seconds: float


def reference_nodes(count, rng):
    """Fixed reference points on the unit rounded square (half 1, corner 0.4),
    denser near the edge, and fixed (t, phi) for the grid bars."""
    u = rng.uniform(-1, 1, size=(8 * count, 2))
    d = -X.rounded_square_sdf(u[:, 0], u[:, 1], 1.0, CORNER)
    keep = (d > 0) & (rng.uniform(size=len(u)) < 0.15 + np.exp(-d / 0.05))
    u = u[keep][:count]
    t = np.sqrt(rng.uniform(size=count)) ** 2  # denser at the rim
    t = rng.uniform(size=count) ** 2
    phi = rng.uniform(0, 2 * np.pi, size=count)
    return u, t, phi


def _wedge_select(c, pts_rel):
    """Scale-invariant selection of points belonging to the symmetry wedge
    ``0 <= z <= x``, for a pad centred at ``c``, points given relative to it."""
    cx, cz = c
    ok = np.ones(len(pts_rel), bool)
    if cz < 0 or cz > cx + 1e-9:
        return ~ok
    if abs(cz) < 1e-9:
        ok &= pts_rel[:, 1] >= 0
    if abs(cz - cx) < 1e-9:
        ok &= pts_rel[:, 1] <= pts_rel[:, 0]
    return ok


def build(dom: Domain, theta0: np.ndarray, nodes, pad_basis=pad_terms, grid_basis=grid_terms) -> Linear:
    """The design matrix at ``theta0 = (a, r, h, dV)`` and its derivatives in
    (a, r, h), by FFTs on the full grid (exact on the sheets)."""
    t0 = time.perf_counter()
    a0, r0, h0 = float(theta0[0]), float(theta0[1]), float(theta0[2])
    y_pad, y_grid = dom.y0, dom.y0 + h0
    dev, dt = dom.device, dom.dtype
    n, h, P = dom.n, dom.h, dom.pitch
    u, tt, phi = nodes
    # fit nodes, and their positions' derivatives in a (pads) and r (grid bars)
    pad_rows, grid_rows = [], []
    for i, c in enumerate(dom.centres):
        sel = _wedge_select(c, u)
        for uu in u[sel]:
            pad_rows.append((i, c[0] + a0 * uu[0], c[1] + a0 * uu[1], uu[0], uu[1]))
        dirs = np.column_stack((np.cos(phi), np.sin(phi)))
        rmax = 0.5 * P / np.maximum(np.abs(dirs[:, 0]), np.abs(dirs[:, 1]))
        rho = r0 + (rmax - r0) * tt
        rel = dirs * rho[:, None]
        sel = _wedge_select(c, rel)
        for dd, t_, rm in zip(dirs[sel], tt[sel], rmax[sel]):
            drho_dr = 1.0 - t_
            rr = r0 + (rm - r0) * t_
            grid_rows.append((i, c[0] + rr * dd[0], c[1] + rr * dd[1], drho_dr * dd[0], drho_dr * dd[1]))
    PR = torch.tensor(pad_rows, device=dev, dtype=dt)
    GR = torch.tensor(grid_rows, device=dev, dtype=dt)
    # evaluation planes: pads (y0) and grid (y0 + h), and dG/dh by a small step
    eps = 1e-5
    yh = lambda hh: dom.y0 + hh
    Gs = {}
    for q, yq in (("p", lambda hh: y_pad), ("g", yh)):
        for p, ypf in (("p", lambda hh: y_pad), ("g", yh)):
            G0 = dom.G(yq(h0), ypf(h0))
            Gp, Gm = dom.G(yq(h0 + eps), ypf(h0 + eps)), dom.G(yq(h0 - eps), ypf(h0 - eps))
            Gs[q, p] = (G0, (Gp - Gm) / (2 * eps))
    ikx, ikz = 1j * dom.kx, 1j * dom.kz

    def sample(field, xz):
        """Bilinear, periodic, at physical positions (rows, 2)."""
        fx = xz[:, 0] / h  # the unshifted inverse FFT has the origin at index 0
        fz = xz[:, 1] / h
        i0, j0 = torch.floor(fx), torch.floor(fz)
        wx, wz = fx - i0, fz - j0
        i0, j0 = i0.long() % n, j0.long() % n
        i1, j1 = (i0 + 1) % n, (j0 + 1) % n
        return ((1 - wx) * (1 - wz) * field[i0, j0] + wx * (1 - wz) * field[i1, j0]
                + (1 - wx) * wz * field[i0, j1] + wx * wz * field[i1, j1])

    def plane_values(S, dS_s, s_idx, source):
        """Values at the pad rows and the grid rows, and their derivatives in
        (a, r, h), of a column with spectrum S on sheet ``source``."""
        out, dout = [], []
        for q, R in (("p", PR), ("g", GR)):
            G0, dG = Gs[q, source]
            spec = torch.fft.ifft2
            phi0 = spec(S * G0).real
            xz = R[:, 1:3]
            v = sample(phi0, xz)
            gx, gz = sample(spec(ikx * S * G0).real, xz), sample(spec(ikz * S * G0).real, xz)
            dv = torch.zeros((3, len(R)), device=dev, dtype=dt)
            if dS_s is not None:  # the density's own scaling
                dv[s_idx] += sample(spec(dS_s * G0).real, xz)
            # the nodes move: pads with a, grid bars with r
            move = 0 if q == "p" else 1
            dv[move] += gx * R[:, 3] + gz * R[:, 4]
            dv[2] += sample(spec(S * dG).real, xz)
            out.append(v)
            dout.append(dv)
        return torch.cat(out), torch.cat(dout, 1)

    def coeffs(S):
        M = dom.modes
        return S[: M + 1, : M + 1].real / (n * n) * dom.wmn

    cols, dcols, As, dAs, sheet, centre_cols, q0 = [], [], [], [], [], [], []
    cc = dom.centres / P
    for ic, cls in enumerate(dom.classes):
        ring = int(round(np.abs(cc[cls[0]]).max()))
        # pads: exact scalings about the pad centre by a
        for (p, q, s) in pad_basis(ring):
            def loc(XI, ZE, p=p, q=q, s=s):
                uu, vv = XI / a0, ZE / a0
                d = -rounded_square_sdf_t(uu, vv, 1.0, CORNER)
                base = torch.where(d > 0, 1.0 / torch.sqrt(d.clamp_min(0.5 * h / a0)), torch.zeros_like(d))
                if s is not None:
                    base = base * torch.exp(-d.clamp_min(0) / s)
                return base * uu ** p * vv ** q
            g, raw = _place(dom, cls, loc)
            if float(g.abs().max()) < 1e-9 * raw:
                continue
            S, dSa = _scaled_spectrum(dom, g, a0)
            v, dv = plane_values(S, dSa, 0, "p")
            cols.append(v)
            dcols.append(dv)
            As.append(coeffs(S))
            dA = torch.zeros((3, dom.modes + 1, dom.modes + 1), device=dev, dtype=dt)
            dA[0] = coeffs(dSa)
            dAs.append(dA)
            sheet.append(0)
            centre_cols.append(ic == 0)
            q0.append(float(g.sum()) * h * h)
        # the grid: a fixed tile minus a disk scaled by r, and rim bands scaled by r
        for term in grid_basis(ring):
            if term[0] == "tile":
                _, p, q = term

                def tile(XI, ZE, p=p, q=q):
                    inside = (XI.abs() < 0.5 * P) & (ZE.abs() < 0.5 * P)
                    return torch.where(inside, (XI / P) ** p * (ZE / P) ** q, torch.zeros_like(XI))

                def disk(XI, ZE, p=p, q=q):
                    uu, vv = XI / r0, ZE / r0
                    return torch.where(uu * uu + vv * vv < 1.0, uu ** p * vv ** q, torch.zeros_like(uu))
                gt, raw = _place(dom, cls, tile)
                gd, _ = _place(dom, cls, disk)
                fac = (r0 / P) ** (p + q)
                g = gt - fac * gd
                if float(g.abs().max()) < 1e-9 * max(raw, 1e-30):
                    continue
                St = _spectrum(dom, gt)
                Sd, dSd = _scaled_spectrum(dom, gd, r0)
                S = St - fac * Sd
                dSr = -fac * dSd - ((p + q) / r0) * fac * Sd
            else:
                _, p, q, s = term

                def rim_f(XI, ZE, p=p, q=q, s=s):
                    uu, vv = XI / r0, ZE / r0
                    rr = torch.sqrt(uu * uu + vv * vv)
                    d = rr - 1.0
                    base = torch.where(d > 0, 1.0 / torch.sqrt(d.clamp_min(0.5 * h / r0)), torch.zeros_like(d))
                    return base * torch.exp(-d.clamp_min(0) / s) * uu ** p * vv ** q

                def rim(XI, ZE):
                    return rim_f(XI, ZE) * taper(XI, ZE, P)[0]

                def rim_x(XI, ZE):
                    return rim_f(XI, ZE) * taper(XI, ZE, P)[1]
                g, raw = _place(dom, cls, rim)
                if float(g.abs().max()) < 1e-9 * raw:
                    continue
                gx_, _ = _place(dom, cls, rim_x)
                S, dSr = _scaled_spectrum_tapered(dom, g, gx_, r0)
            v, dv = plane_values(S, dSr, 1, "g")
            cols.append(v)
            dcols.append(dv)
            As.append(coeffs(S))
            dA = torch.zeros((3, dom.modes + 1, dom.modes + 1), device=dev, dtype=dt)
            dA[1] = coeffs(dSr)
            dAs.append(dA)
            sheet.append(1)
            centre_cols.append(False)
            q0.append(float(g.sum()) * h * h)
    D = torch.stack(cols, 1)
    dD = torch.stack(dcols, 2)
    row_kind = np.concatenate((np.zeros(len(PR), int), np.ones(len(GR), int)))
    row_owner = np.concatenate((PR[:, 0].cpu().numpy().astype(int), GR[:, 0].cpu().numpy().astype(int)))
    if dom.device.startswith("cuda"):
        torch.cuda.synchronize()
    return Linear(D=D, dD=dD, A=torch.stack(As), dA=torch.stack(dAs, 0),
                  sheet=torch.tensor(sheet, device=dev), row_kind=row_kind, row_owner=row_owner,
                  col_centre=torch.tensor(centre_cols, device=dev),
                  q0=torch.tensor(q0, device=dev, dtype=dt), seconds=time.perf_counter() - t0)


def solve(lin: Linear, theta, theta0, targets, offset, ridge=1e-10):
    """The fit at ``theta`` (a tensor), linearised about ``theta0``:
    coefficients ``c`` of the columns, by ridge-regularised normal equations
    (differentiable).  ``targets`` (rows,) and ``offset`` are tensors."""
    dth = (theta[:3] - torch.as_tensor(theta0[:3], device=theta.device, dtype=theta.dtype))
    D = lin.D + torch.einsum("i,irc->rc", dth, lin.dD)
    scale = lin.D.abs().max(0).values.clamp_min(1e-30)
    Ds = D / scale
    b = targets - offset
    N = Ds.T @ Ds
    N = N + ridge * torch.trace(N) / N.shape[0] * torch.eye(N.shape[0], device=N.device, dtype=N.dtype)
    c = torch.linalg.solve(N, Ds.T @ b) / scale
    r = D @ c - b
    return c, r


def coefficients(lin: Linear, c, theta, theta0):
    """Series coefficients per sheet, ``(2, M+1, M+1)``, at ``theta``."""
    dth = (theta[:3] - torch.as_tensor(theta0[:3], device=theta.device, dtype=theta.dtype))
    A = lin.A + torch.einsum("i,jimn->jmn", dth, lin.dA)
    out = []
    for s in (0, 1):
        sel = lin.sheet == s
        out.append((c[sel][:, None, None] * A[sel]).sum(0))
    return torch.stack(out)


class Field:
    """A fitted potential as a differentiable function of position."""

    def __init__(self, dom, A, y_sheets, offset):
        self.dom, self.A, self.y_sheets, self.offset = dom, A, y_sheets, offset

    def __call__(self, p, grad=False):
        dom = self.dom
        k, m = dom.k, dom.m
        nm = m.numel()
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        cx, cz = torch.cos(k * x[:, None] * m), torch.cos(k * z[:, None] * m)
        v = torch.zeros(len(p), device=p.device, dtype=p.dtype) + self.offset
        g = []
        if grad:
            sx, sz = -k * m * torch.sin(k * x[:, None] * m), -k * m * torch.sin(k * z[:, None] * m)
            gx = torch.zeros_like(v)
            gy = torch.zeros_like(v)
            gz = torch.zeros_like(v)
        for ip, yp in enumerate(self.y_sheets):
            A = self.A[ip]
            G = green(dom.kappa, y, yp, dom.top, dom.kind, dom.pcb, dom.y0).reshape(-1, nm, nm)
            AG = A[None] * G
            v = v + torch.einsum("pm,pmn,pn->p", cx, AG, cz)
            if grad:
                hh = 1e-4
                dG = (green(dom.kappa, y + hh, yp, dom.top, dom.kind, dom.pcb, dom.y0)
                      - green(dom.kappa, y - hh, yp, dom.top, dom.kind, dom.pcb, dom.y0)).reshape(-1, nm, nm) / (2 * hh)
                gx = gx + torch.einsum("pm,pmn,pn->p", sx, AG, cz)
                gz = gz + torch.einsum("pm,pmn,pn->p", cx, AG, sz)
                gy = gy + torch.einsum("pm,pmn,pn->p", cx, A[None] * dG, cz)
        return (v, torch.stack((gx, gy, gz), 1)) if grad else v


# --------------------------------------------------------------------------
# drift, induction, diffusion, objective
# --------------------------------------------------------------------------


def trace(drift: Field, starts, a, y0, dt, nsteps, block=40, temperature=87.0, pitch=4.4):
    """Differentiable RK4 in time; paths stop on a pad (inside the rounded
    square of half-width ``a`` at the board ``y0``) and slide on the charged
    board between pads.  Returns positions ``(nsteps + 1, paths, 3)`` and a
    landed flag per step."""
    def vel(p):
        w0 = torch.remainder(p[:, 0] + 0.5 * pitch, pitch) - 0.5 * pitch
        w2 = torch.remainder(p[:, 2] + 0.5 * pitch, pitch) - 0.5 * pitch
        # an RK stage may probe just below the board, where the field is not defined
        w = torch.stack((w0, torch.clamp(p[:, 1], min=y0), w2), 1)
        _, g = drift(w, grad=True)
        e = -g
        mag = torch.linalg.vector_norm(e, dim=1).clamp_min(1e-9)
        return -(X.bnl_speed(mag, temperature) / mag)[:, None] * e

    def on_pad(p):
        w0 = torch.remainder(p[:, 0] + 0.5 * pitch, pitch) - 0.5 * pitch
        w2 = torch.remainder(p[:, 2] + 0.5 * pitch, pitch) - 0.5 * pitch
        return rounded_square_sdf_t(w0, w2, a, CORNER * a) < 0

    def steps(p, landed):
        out, flags = [], []
        for _ in range(block):
            k1 = vel(p)
            k2 = vel(p + 0.5 * dt * k1)
            k3 = vel(p + 0.5 * dt * k2)
            k4 = vel(p + dt * k3)
            new = p + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            below = new[:, 1] <= y0
            hit = below & on_pad(new) & ~landed
            # land: interpolate to the board crossing (differentiable), then stay
            # (the division only where a path lands: a sliding path has no height change)
            den = torch.where(hit, p[:, 1] - new[:, 1], torch.ones_like(p[:, 1]))
            fr = torch.where(hit & (den > 1e-9), (p[:, 1] - y0) / den.clamp_min(1e-9), torch.zeros_like(den))
            fr = fr.clamp(0, 1)
            land_pt = p + fr[:, None] * (new - p)
            slid = torch.stack((new[:, 0], torch.maximum(new[:, 1], torch.full_like(new[:, 1], y0)), new[:, 2]), 1)
            nxt = torch.where(hit[:, None], land_pt, slid)
            nxt = torch.where(landed[:, None], p, nxt)
            # Sliding on the charged board near its stagnation points is
            # unstable: neighbouring paths separate exponentially, and so do
            # their parameter sensitivities.  Truncate the chain there: a sliding
            # path's position is detached, though it still feels the parameters
            # through the field it moves in.
            if TRUNCATE_SLIDING:
                sliding = (nxt[:, 1] <= y0 + 1e-12) & ~(landed | hit)
                nxt = torch.where(sliding[:, None], nxt.detach(), nxt)
            landed = landed | hit
            p = nxt
            out.append(p)
            flags.append(landed)
        return torch.stack(out), torch.stack(flags)

    p = starts
    landed = torch.zeros(len(p), dtype=torch.bool, device=p.device)
    P, F = [p[None]], [landed[None]]
    for _ in range(0, nsteps, block):
        if p.requires_grad or any(t.requires_grad for t in (drift.A,)):
            outs, flags = checkpoint(steps, p, landed, use_reentrant=False)
        else:
            outs, flags = steps(p, landed)
        p, landed = outs[-1], flags[-1]
        P.append(outs)
        F.append(flags)
    return torch.cat(P), torch.cat(F)


def induced(weight: Field, P, F, block=64, shift=None):
    """Charge on the sensing pad along each path: ``phi_w`` before landing, 1
    after (landing on the sensing pad; the launches are over it).  With
    ``shift`` (a pad-lattice vector), the charge the same paths induce on the
    neighbour at ``-shift``: by translation, the sensing pad's weighting
    potential at the shifted positions, and exactly 0 after landing (the
    landing pad is at 0 V in that neighbour's weighting solve)."""
    T, Np, _ = P.shape
    flat = P.reshape(-1, 3)
    if shift is not None:
        flat = flat + torch.as_tensor(shift, device=P.device, dtype=P.dtype)

    def ev(pts):
        return weight(pts)
    vals = []
    step = block * Np
    for s in range(0, len(flat), step):
        chunk = flat[s:s + step]
        vals.append(checkpoint(ev, chunk, use_reentrant=False) if chunk.requires_grad or weight.A.requires_grad
                    else ev(chunk))
    Q = torch.cat(vals).reshape(T, Np)
    return torch.where(F, torch.ones_like(Q) if shift is None else torch.zeros_like(Q), Q)


def smear(I, dt, sigma, edge=False):
    """Convolve a current (time along the last axis) with a Gaussian of rms
    ``sigma`` (longitudinal diffusion).  ``edge`` pads with the end values (for
    a charge, which does not start or end at zero) instead of zeros."""
    if sigma <= 0:
        return I
    half = int(math.ceil(4 * sigma / dt))
    t = torch.arange(-half, half + 1, device=I.device, dtype=I.dtype) * dt
    ker = torch.exp(-0.5 * (t / sigma) ** 2)
    ker = ker / ker.sum()
    pad = torch.nn.functional.pad(I.reshape(1, 1, -1), (half, half), mode="replicate" if edge else "constant")
    return torch.nn.functional.conv1d(pad, ker.flip(0).reshape(1, 1, -1)).reshape(-1)


def launches(step):
    """Launch points on one eighth of the pitch square (cell centres of a
    ``step`` grid), with weights standing for the whole square."""
    nn = int(round(2.2 / step))
    c = (np.arange(nn) + 0.5) * step
    # a launch on the cell side would sit on the stagnation point above a grid
    # bar's midline, and never move
    assert c.max() < 2.2 - 1e-9, "the launch grid must not reach the cell side"
    pts, w = [], []
    for i, x in enumerate(c):
        for j, z in enumerate(c[: i + 1]):
            pts.append((x, z))
            w.append(4.0 if j == i else 8.0)
    w = np.array(w)
    return np.array(pts), w / w.sum()


@dataclass
class Design:
    """The optimisation problem: resolution, physics and constraints."""

    E0: float = 50.05  # V/mm, the drift field held above the grid
    top: float = 149.9
    y0: float = 10.0
    pitch: float = 4.4
    eps_fr4: float = 4.4 / 1.505
    D_L: float = 4.0e-4  # mm^2/us (4 cm^2/s), longitudinal diffusion
    drift_time: float = 86.0  # us, for the diffusion width
    v_drift: float = 1.60  # mm/us at 500 V/cm
    E_max: float = 1000.0  # V/mm, the largest grid-to-pad field allowed
    launch_step: float = 0.2  # mm
    above: float = 6.0  # mm, launch height above the grid
    dt: float = 0.01  # us
    t_max: float = 20.0  # us
    drift_res: tuple = (881, 60)
    weight_res: tuple = (1101, 110)
    weight_cells: int = 5
    nodes: int = 1500
    device: str = "cuda"
    # "width": the rms duration alone; "combined": width/width_ref + X/X_ref;
    # "neighbour": X alone -- X the peak unwanted charge on an edge neighbour
    objective: str = "width"
    window: float = 1.5  # us: the neighbour's swing is measured within this of the collection peak
    width_ref: float = 1.0
    x_ref: float = 1.0

    @property
    def sigma_t(self):
        return math.sqrt(2 * self.D_L * self.drift_time) / self.v_drift


class Problem:
    """Build, evaluate and differentiate the objective at a design point."""

    def __init__(self, cfg: Design, seed=1):
        self.cfg = cfg
        dev = cfg.device
        self.drift_dom = Domain("drift", 1, cfg.drift_res[0], cfg.drift_res[1], cfg.pitch, cfg.y0, cfg.top,
                                0.0, dev)
        self.weight_dom = Domain("weighting", cfg.weight_cells, cfg.weight_res[0], cfg.weight_res[1],
                                 cfg.pitch, cfg.y0, cfg.top, cfg.eps_fr4, dev)
        rng = np.random.default_rng(seed)
        self.nodes_drift = reference_nodes(cfg.nodes, rng)
        self.nodes_weight = reference_nodes(cfg.nodes, np.random.default_rng(seed + 1))
        self.pts, self.w = launches(cfg.launch_step)
        self.timing = {}

    def evaluate(self, theta_np, grad=True, full=False):
        """Objective (and gradient) at ``theta = (a, r, h, dV)``."""
        cfg = self.cfg
        dev = cfg.device
        dt_ = torch.float64
        tm = {}
        sync = (lambda: torch.cuda.synchronize()) if dev.startswith("cuda") else (lambda: None)
        if dev.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        th0 = np.asarray(theta_np, float)
        lin_d = build(self.drift_dom, th0, self.nodes_drift)
        lin_w = build(self.weight_dom, th0, self.nodes_weight)
        sync()
        tm["build"] = time.perf_counter() - t0
        theta = torch.tensor(th0, device=dev, dtype=dt_, requires_grad=grad)
        a, r, h, dV = theta[0], theta[1], theta[2], theta[3]
        t1 = time.perf_counter()
        # drift: pads at 0, the grid at -dV, the cathode holding E0 above the grid
        y_grid = cfg.y0 + h
        V_cath = -dV - cfg.E0 * (cfg.top - y_grid)
        tgt = torch.where(torch.as_tensor(lin_d.row_kind == 1, device=dev), -dV, torch.zeros((), device=dev, dtype=dt_))
        c_d, res_d = solve(lin_d, theta, th0, tgt, V_cath)
        A_d = coefficients(lin_d, c_d, theta, th0)
        drift = Field(self.drift_dom, A_d, (cfg.y0, y_grid), V_cath)
        centre = len(self.weight_dom.centres) // 2
        tw = torch.as_tensor(((lin_w.row_kind == 0) & (lin_w.row_owner == centre)).astype(float), device=dev,
                             dtype=dt_)
        c_w, res_w = solve(lin_w, theta, th0, tw, torch.zeros((), device=dev, dtype=dt_))
        A_w = coefficients(lin_w, c_w, theta, th0)
        weight = Field(self.weight_dom, A_w, (cfg.y0, y_grid), torch.zeros((), device=dev, dtype=dt_))
        sync()
        tm["fit"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        xs = torch.as_tensor(self.pts[:, 0], device=dev, dtype=dt_)
        zs = torch.as_tensor(self.pts[:, 1], device=dev, dtype=dt_)
        starts = torch.stack((xs, (y_grid + cfg.above).expand(len(xs)), zs), 1)
        nsteps = int(round(cfg.t_max / cfg.dt))
        P, F = trace(drift, starts, a, cfg.y0, cfg.dt, nsteps, pitch=cfg.pitch)
        Q = induced(weight, P, F)
        I = (Q[1:] - Q[:-1]) / cfg.dt  # (T, paths)
        w = torch.as_tensor(self.w, device=dev, dtype=dt_)
        Iavg = (I * w[None]).sum(1)
        Is = smear(Iavg, cfg.dt, cfg.sigma_t)
        t = (torch.arange(len(Is), device=dev, dtype=dt_) + 0.5) * cfg.dt
        tot = Is.sum()
        tbar = (t * Is).sum() / tot
        width = torch.sqrt(((t - tbar) ** 2 * Is).sum() / tot)
        # the unwanted signal on the neighbours: the charge the same electrons
        # induce on the edge and the diagonal neighbour, averaged over the
        # launches, smeared by diffusion; its peak, per electron collected
        # The swing that matters is the one around the collection time: the
        # peak-to-peak excursion within +-window of the collecting pad's peak
        # (away from it the neighbour's charge changes slowly, and the readout's
        # threshold has long since settled).
        P_ = cfg.pitch
        ipk = int(torch.argmax(Is.detach()))
        hw = int(round(cfg.window / cfg.dt))
        lo_, hi_ = max(ipk - hw, 0), min(ipk + hw + 1, len(Is))
        xt = {}
        for key, sh in (("edge", (P_, 0.0, 0.0)), ("diag", (P_, 0.0, P_))):
            Qn = induced(weight, P, F, shift=sh)
            Qavg = (Qn * w[None]).sum(1)
            Qs = smear(Qavg, cfg.dt, cfg.sigma_t, edge=True)
            seg = Qs[lo_:hi_ + 1]
            xt[key] = (seg.max() - seg.min(), Qs)
        X_e = xt["edge"][0]
        # constraint: the grid-to-pad field
        over = torch.relu(dV / h - cfg.E_max) / cfg.E_max
        if cfg.objective == "width":
            J = width
        elif cfg.objective == "combined":
            J = width / cfg.width_ref + X_e / cfg.x_ref
        elif cfg.objective == "neighbour":
            J = X_e / cfg.x_ref
        else:
            raise ValueError(cfg.objective)
        J = J + 10.0 * over ** 2
        sync()
        tm["trace"] = time.perf_counter() - t2
        g = None
        if grad:
            t3 = time.perf_counter()
            (g,) = torch.autograd.grad(J, theta)
            g = g.cpu().numpy()
            sync()
            tm["backward"] = time.perf_counter() - t3
        tm["total"] = time.perf_counter() - t0
        if dev.startswith("cuda"):
            tm["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        tm["cols_drift"], tm["cols_weight"] = lin_d.D.shape[1], lin_w.D.shape[1]
        tm["rows_drift"], tm["rows_weight"] = lin_d.D.shape[0], lin_w.D.shape[0]
        self.timing = tm
        out = dict(J=float(J), width=float(width), grad=g, timing=tm, x_edge=float(X_e),
                   x_diag=float(xt["diag"][0]),
                   fit_drift=float(res_d[lin_d.row_kind == 1].pow(2).mean().sqrt() / max(float(dV), 1.0)),
                   fit_weight=float(res_w.pow(2).mean().sqrt()), landed=float(F[-1].float().mean()))
        if full:
            with torch.no_grad():
                # the sensing pad's charge at 1 V: 2 eps_LAr times its sheet density
                eps = 8.854e-15 * 1.505  # F/mm
                cw = c_w.detach()
                cap_pf = float(2 * eps * (cw[lin_w.col_centre] * lin_w.q0[lin_w.col_centre]).sum()) * 1e12
                # did any path cross the grid plane outside a hole?
                Pn = P.detach()
                yg = float(y_grid)
                cr = (Pn[:-1, :, 1] - yg) * (Pn[1:, :, 1] - yg) <= 0
                wx = torch.remainder(Pn[1:, :, 0] + 0.5 * cfg.pitch, cfg.pitch) - 0.5 * cfg.pitch
                wz = torch.remainder(Pn[1:, :, 2] + 0.5 * cfg.pitch, cfg.pitch) - 0.5 * cfg.pitch
                caught = (cr & (wx * wx + wz * wz > float(r) ** 2)).any(0)
                out["grid_caught"] = float((caught.double().cpu().numpy() * self.w).sum())
                out.update(Qe=xt["edge"][1].detach().cpu().numpy(), Qd=xt["diag"][1].detach().cpu().numpy())
                out.update(t=t.cpu().numpy(), I=I.detach().cpu().numpy(), Iavg=Iavg.detach().cpu().numpy(),
                           Is=Is.detach().cpu().numpy(), P=P.detach().cpu().numpy(), F=F.cpu().numpy(),
                           peak=float(Is.max()), cap_pf=cap_pf)
        return out
