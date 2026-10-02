"""3D pixel anodes: the spectral single layer (firep.pixel)."""

import math

import numpy as np
import pytest
import torch

from firep import pixel as X

DT = torch.float64


def small(ncell=1, kind="drift", terms=((0, 0), (2, 0)), grid=221, modes=30, nodes=None):
    arr = X.PadArray(pitch=4.4, ncell=ncell, shape=X.PadShape("rounded_square", 1.9, 0.4),
                     y_pad=(10.0, 10.1), top=40.0)
    return arr, X.SpectralSingleLayer(arr, kind, modes, grid, list(terms), nodes=nodes)


@pytest.mark.parametrize("kind", ["drift", "weighting"])
def test_slab_green_meets_its_face_conditions(kind):
    kap = torch.tensor([0.0, 0.3, 2.0, 15.0], dtype=DT)
    top, yp, h = 40.0, 10.0, 1e-6
    g = lambda y: X.slab_green(kap, torch.tensor([y], dtype=DT), yp, top, kind)[0]
    fixed, free = (top, 0.0) if kind == "drift" else (0.0, top)
    assert torch.allclose(g(fixed), torch.zeros(4, dtype=DT), atol=1e-12)
    slope = (g(free + h) - g(free - h)) / (2 * h) if free > 0 else (g(h) - g(0.0)) / h
    assert torch.allclose(slope, torch.zeros(4, dtype=DT), atol=1e-5)
    # (d^2/dy^2 - kappa^2) g = -2 delta: the slope jumps by -2 across the sheet
    jump = (g(yp + 2 * h) - g(yp + h)) / h - (g(yp - h) - g(yp - 2 * h)) / h
    assert torch.allclose(jump, torch.full((4,), -2.0, dtype=DT), atol=1e-4)


@pytest.mark.parametrize("kind", ["drift", "weighting"])
def test_slab_green_with_a_pcb(kind):
    kap = torch.tensor([0.0, 0.3, 2.0, 15.0], dtype=DT)
    top, y0, yp, h = 40.0, 10.0, 10.1, 1e-6
    g = lambda y, pcb: X.slab_green(kap, torch.tensor([y], dtype=DT), yp, top, kind, pcb, y0)[0]
    free = lambda y: X.slab_green(kap, torch.tensor([y], dtype=DT), yp, top, kind)[0]
    # a PCB with the medium's own permittivity is free space, above and below
    for y in (3.0, 10.0, 10.05, 25.0):
        assert torch.allclose(g(y, 1.0), free(y), rtol=1e-10, atol=1e-12)
    # V and eps dV/dy are continuous at the surface
    r = 2.9
    assert torch.allclose(g(y0 - h, r), g(y0 + h, r), atol=1e-5)
    below = (g(y0 - h, r) - g(y0 - 2 * h, r)) / h
    above = (g(y0 + 2 * h, r) - g(y0 + h, r)) / h
    assert torch.allclose(r * below, above, atol=1e-4)
    # the far face is untouched
    fixed = top if kind == "drift" else 0.0
    if kind == "drift":
        assert torch.allclose(g(fixed, r), torch.zeros(4, dtype=DT), atol=1e-12)
        # a charged-up insulator: zero slope on its surface, above it
        slope = (g(y0 + 2 * h, 0.0) - g(y0 + h, 0.0)) / h
        assert torch.allclose(slope[1:], torch.zeros(3, dtype=DT), atol=1e-4)


def test_d4_symmetrise_is_a_projection():
    f = np.random.default_rng(0).normal(size=(11, 11))
    s = X.d4_symmetrise(f)
    assert np.allclose(s, s.T) and np.allclose(s, np.rot90(s))
    assert np.allclose(X.d4_symmetrise(s), s)


def test_odd_terms_survive_on_off_centre_pads():
    """Regression: placing an odd local term on every pad of an orbit made it
    cancel under the 180-degree rotation, so the fit could not tilt the charge
    on a neighbour of the sensing pad."""
    arr, ssl = small(ncell=3, kind="weighting", terms=((0, 0), (1, 0)), grid=331, modes=40)
    labels = [(cls, p, q) for cls, p, q, _y in ssl.labels]
    # the (1, 0) term exists for both off-centre classes (on each of the two
    # sheets) but not for the centre pad, where the symmetry removes it
    odd = [lab for lab in labels if lab[1] == 1]
    assert len(odd) == 2 * 2
    assert all(len(cls) > 1 for cls, _, _ in odd)


def test_truncated_series_matches_the_full_grid_off_the_sheets():
    arr, ssl = small(ncell=3, kind="weighting", terms=((0, 0), (1, 0), (2, 0)), grid=331, modes=60)
    coef = torch.linspace(0.3, 1.0, len(ssl.labels), dtype=DT)
    y = 11.5
    lay, xs = ssl.layers(coef, [y])
    i = len(xs) // 2
    pts = torch.tensor([[xs[i], y, xs[i]], [xs[i + 60], y, xs[i + 17]], [xs[i + 100], y, xs[i]]], dtype=DT)
    ref = torch.stack([lay[0, i, i], lay[0, i + 60, i + 17], lay[0, i + 100, i]])
    assert torch.allclose(ssl.evaluate(pts, coef), ref, rtol=1e-9, atol=1e-12)


def test_drift_cell_fit_and_field():
    arr = X.PadArray(pitch=4.4, ncell=1, shape=X.PadShape("rounded_square", 1.9, 0.4),
                     y_pad=(10.0, 10.1), top=40.0)
    ii, jj, own, _ = X.pad_nodes(arr, 441, 3000, np.random.default_rng(1))
    terms = [(0, 0), (2, 0), (0, 0, 0.1), (2, 0, 0.1)]
    ssl = X.SpectralSingleLayer(arr, "drift", 44, 441, terms, nodes=(ii, jj))
    coef, rms, mx = X.fit_on_nodes(ssl, np.zeros(len(own)), -1000.0)
    assert rms < 0.5  # volts, on a 1000 V drop over 30 mm
    f = X.SheetPotential.from_fit(ssl, coef, -1000.0)
    v, g = f.evaluate(torch.tensor([[0.0, 35.0, 0.0], [1.3, 35.0, 0.7]], dtype=DT), grad=True)
    # far above, a uniform field pointing at the cathode: -dV/dy = 1000 V over (top - pad)
    assert abs(float(-g[0, 1]) - 1000.0 / (40.0 - 10.05)) < 0.5
    assert abs(float(g[0, 1] - g[1, 1])) < 1e-6
    # saved and restored, it evaluates the same
    f2 = X.SheetPotential.from_state(f.state())
    assert torch.allclose(f2.evaluate(torch.tensor([[0.4, 12.0, 0.2]], dtype=DT)),
                          f.evaluate(torch.tensor([[0.4, 12.0, 0.2]], dtype=DT)))


def test_bnl_speed_matches_pochoir():
    v = X.bnl_speed(torch.tensor(50.0565, dtype=DT), 87.0)
    assert float(v) == pytest.approx(1.60204, abs=1e-5)


def test_d4_encoding_has_the_symmetry_and_zero_side_slope():
    enc = X.D4Encoding(4.4, 4, 12.0, 3.0).double()
    p = torch.tensor([[0.3, 11.0, 1.1]], dtype=DT)
    images = [(p[0, 0], p[0, 2]), (-p[0, 0], p[0, 2]), (p[0, 2], p[0, 0]), (-p[0, 2], -p[0, 0])]
    f0 = enc(p)
    for x, z in images:
        assert torch.allclose(enc(torch.tensor([[x, 11.0, z]], dtype=DT)), f0)
    side = torch.tensor([[2.2, 11.0, 0.7]], dtype=DT, requires_grad=True)
    (jac,) = torch.autograd.grad(enc(side).sum(), side)
    assert abs(float(jac[0, 0])) < 1e-9


def test_hemisphere_on_a_charged_board_fits_its_dome():
    """Smooth sheets inside a dome on a zero-slope board: with its image the
    dome is a whole sphere, so interior sources fit its surface closely."""
    arr = X.PadArray(pitch=4.4, ncell=1, shape=X.PadShape("hemisphere", 1.0, 0.0),
                     y_pad=10.0, top=40.0, pcb=0.0)
    ssl = X.SpectralSingleLayer(arr, "drift", 30, 221, [(p, q) for p in range(5) for q in range(5 - p)])
    xyz, own = X.dome_nodes(arr, 800, np.random.default_rng(1))
    D = ssl.evaluate(torch.as_tensor(xyz, dtype=DT))
    coef, rms, mx = X.fit_design(D, np.zeros(len(own)), -1000.0)
    assert rms < 1e-2  # volts, of a 1000 V drop
    f = X.SheetPotential.from_fit(ssl, coef, -1000.0)
    inside = arr.shape.inside(torch.tensor([0.0, 0.9]), torch.tensor([10.5, 10.1]),
                              torch.tensor([0.0, 0.0]), 10.0, 11.0)
    assert inside.tolist() == [True, True]
    assert not bool(arr.shape.inside(torch.tensor([0.9]), torch.tensor([10.9]), torch.tensor([0.0]), 10.0, 11.0))
    # far above, the uniform field
    _, g = f.evaluate(torch.tensor([[0.0, 35.0, 0.0]], dtype=DT), grad=True)
    assert 30.0 < float(-g[0, 1]) < 45.0


def test_quad_pad_is_four_ganged_dots():
    sh = X.PadShape("quad", 0.55, 1.1)
    assert sh.extent == pytest.approx(1.65)
    assert sh.sdf(np.array([1.1]), np.array([1.1]))[0] == pytest.approx(-0.55)
    assert sh.sdf(np.array([0.0]), np.array([0.0]))[0] == pytest.approx(math.hypot(1.1, 1.1) - 0.55)
    ins = sh.inside(torch.tensor([1.1, -1.1, 0.0]), torch.tensor([10.05] * 3), torch.tensor([-1.1, 1.1, 0.0]),
                    10.0, 10.1)
    assert ins.tolist() == [True, True, False]
    # the four dots are one electrode: a drift cell fits them all at once
    arr = X.PadArray(pitch=4.4, ncell=1, shape=sh, y_pad=(10.0, 10.1), top=40.0, pcb=0.0)
    ii, jj, own, _ = X.pad_nodes(arr, 441, 3000, np.random.default_rng(1))
    ssl = X.SpectralSingleLayer(arr, "drift", 44, 441, [(0, 0), (2, 0), (1, 1), (0, 0, 0.1)], nodes=(ii, jj))
    coef, rms, mx = X.fit_on_nodes(ssl, np.zeros(len(own)), -1000.0)
    assert rms < 2.0


def test_grid_electrode_above_the_pads():
    """A grid sheet with holes over the pads, fitted with its own voltage on
    its own plane, and a tracer that stops on it."""
    pad = X.PadShape("rounded_square", 1.75, 0.7)
    arr = X.PadArray(pitch=4.4, ncell=1, shape=pad, y_pad=(10.0, 10.1), top=40.0, pcb=0.0)
    sheets = [X.Sheet(10.0, pad), X.Sheet(10.1, pad),
              X.Sheet(13.3, X.PadShape("hole", 2.2, 1.9), "edge", tile=True)]
    ip, jp, _, _ = X.pad_nodes(arr, 441, 2500, np.random.default_rng(1))
    ig, jg = X.grid_nodes(arr, 1.9, 441, 2500, np.random.default_rng(2))
    nodes = {0: (ip, jp), 1: (ip, jp), 2: (ig, jg)}
    terms = [(0, 0), (2, 0), (1, 1), (0, 0, 0.1), (2, 0, 0.1)]
    ssl = X.SpectralSingleLayer(arr, "drift", 44, 441, terms, nodes=nodes, sheets=sheets)
    target = np.concatenate((np.zeros(2 * len(ip)), np.full(len(ig), -300.0)))
    coef, rms, mx = X.fit_rows(ssl, target, -1000.0)
    assert rms < 3.0  # volts, coarse test resolution
    f = X.SheetPotential.from_fit(ssl, coef, -1000.0)
    # on the grid, away from the hole, the potential is the grid's
    v = f.evaluate(torch.tensor([[2.1, 13.3, 2.1], [0.0, 13.3, 2.15]], dtype=DT))
    assert torch.allclose(v, torch.full((2,), -300.0, dtype=DT), atol=10.0)
    # a launch over a grid bar ends on the grid; one over the hole reaches the pad
    pa = X.trace(f, torch.tensor([[2.15, 20.0, 2.15], [0.0, 20.0, 0.0]], dtype=DT), tick=0.05, t_max=20.0,
                 substeps=4, landing="conductor", slide=True, grid=(13.3, 1.9))
    assert torch.isnan(pa.pad[0]).all() and torch.isfinite(pa.arrived[0])
    assert torch.allclose(pa.pad[1], torch.zeros(2, dtype=DT)) and float(pa.xyz[1, -1, 1]) < 10.2


def test_scaling_identity_gives_the_spectrum_derivative():
    """dS/ds for a pad density scaled about its centre: the identity against a
    finite difference of the sampled spectrum at low wave numbers."""
    from firep import pixel_opt as O
    dom = O.Domain("drift", 1, 441, 20, device="cpu")

    def spec(a):
        def loc(XI, ZE):
            uu, vv = XI / a, ZE / a
            d = -O.rounded_square_sdf_t(uu, vv, 1.0, O.CORNER)
            return torch.where(d > 0, 1.0 / torch.sqrt(d.clamp_min(0.5 * dom.h / a)), torch.zeros_like(d)) * uu ** 2
        g, _ = O._place(dom, dom.classes[0], loc)
        return O._scaled_spectrum(dom, g, a)

    # a sampled density jitters as its edge crosses grid nodes, so the finite
    # difference needs a step of several nodes (h = 0.01 mm here)
    a0, da = 1.6, 0.02
    S, dS = spec(a0)
    fd = (spec(a0 + da)[0] - spec(a0 - da)[0]) / (2 * da)
    low = slice(0, 6)
    ref = fd[low, low].real
    assert torch.allclose(dS[low, low].real, ref, rtol=0.03, atol=0.03 * float(ref.abs().max()))


def test_optimiser_green_matches_pixel_green():
    from firep import pixel_opt as O
    kap = torch.tensor([0.0, 0.5, 3.0], dtype=DT)
    y = torch.tensor([10.0, 11.0, 30.0], dtype=DT)
    for kind, pcb in (("drift", 0.0), ("weighting", 2.9)):
        a = O.green(kap, y, 10.4, 40.0, kind, pcb, 10.0)
        b = X.slab_green(kap, y, 10.4, 40.0, kind, pcb, 10.0)
        assert torch.allclose(a, b, rtol=1e-10, atol=1e-12)
