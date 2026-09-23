import numpy as np
import pytest
import torch

from firep import bias, config as C
from firep.geometry import Geometry, Sampler
from firep.siren import FieldModel, laplacian

torch.set_default_dtype(torch.float64)


def make(sets=()):
    cfg = C.load_with_overrides(None, ("train.precision=float64",) + tuple(sets))
    geom = Geometry(cfg, bias.solve(cfg))
    return cfg, geom, FieldModel(cfg, geom).double()


def interior_points(geom, cfg, n=3000, seed=3):
    s = Sampler(geom, seed, cfg.sampling.near_factor)
    return torch.tensor(np.vstack((s.bulk(n), s.near(n))))


def test_laplacian_operator_on_a_known_harmonic_function():
    xy = torch.tensor(np.random.default_rng(0).uniform(-1, 1, (500, 2)))
    harmonic = lambda p: (p[:, 0:1] ** 2 - p[:, 1:2] ** 2) + 3 * p[:, 0:1] * p[:, 1:2]
    assert laplacian(harmonic, xy, create_graph=False).abs().max() < 1e-9
    quad = lambda p: p[:, 0:1] ** 2 + p[:, 1:2] ** 2
    assert laplacian(quad, xy, create_graph=False).mean() == pytest.approx(4.0)


@pytest.mark.parametrize("multipole", [0, 1])
def test_baseline_is_harmonic(multipole):
    cfg, geom, m = make((f"model.baseline.multipole={multipole}",))
    with torch.no_grad():
        m.baseline.coeff.normal_(0, 100.0)
    xy = interior_points(geom, cfg)
    lap = laplacian(m.baseline, xy, create_graph=False)
    # Compare against the scale of a 100 V variation over one pitch.
    assert float(lap.abs().max()) < 1e-6 * 100.0 / 25.0


@pytest.mark.parametrize("multipole", [0, 1])
def test_baseline_leaves_the_faces_alone(multipole):
    cfg, geom, m = make((f"model.baseline.multipole={multipole}",))
    with torch.no_grad():
        m.baseline.coeff.normal_(0, 100.0)
    for y, target in ((geom.ylo, geom.v_ground), (geom.yhi, geom.v_cathode)):
        p = torch.tensor(
            np.column_stack((np.linspace(geom.xlo, geom.xhi, 401), np.full(401, y)))
        )
        with torch.no_grad():
            assert float((m.baseline(p)[:, 0] - target).abs().max()) < 1e-2


def test_model_is_exactly_periodic():
    cfg, geom, m = make()
    with torch.no_grad():
        m.baseline.coeff.normal_(0, 100.0)
    a = torch.tensor(
        np.column_stack((np.linspace(-2.5, 2.5, 71), np.linspace(-9.0, 99.0, 71)))
    )
    b = a.clone()
    b[:, 0] += geom.width
    assert float((m(a) - m(b)).abs().max()) < 1e-9


def test_warm_start_fits_the_conductor_potentials():
    cfg, geom, m = make()
    s = Sampler(geom, 1, cfg.sampling.near_factor)
    xy, vals = s.surfaces(256)
    xy_t, vals_t = torch.tensor(xy), torch.tensor(vals)
    before = float((m.baseline(xy_t)[:, 0] - vals_t).abs().max())
    rms = m.baseline.warm_start(xy_t, vals_t)
    after = float((m.baseline(xy_t)[:, 0] - vals_t).abs().max())
    assert after < before / 100
    assert rms < 1.0  # volts, out of a ~6400 V span


def test_dipoles_beat_monopoles_at_the_wire_surface():
    _, _, m0 = make(("model.baseline.multipole=0",))
    cfg, geom, m1 = make(("model.baseline.multipole=1",))
    s = Sampler(geom, 1, cfg.sampling.near_factor)
    xy, vals = s.surfaces(256)
    xy_t, vals_t = torch.tensor(xy), torch.tensor(vals)
    r0 = m0.baseline.warm_start(xy_t, vals_t)
    r1 = m1.baseline.warm_start(xy_t, vals_t)
    assert r1 < r0


def test_baseline_modes_none_and_linear():
    _, geom, m = make(("model.baseline.mode=none",))
    p = torch.tensor([[0.0, 50.0]])
    assert float(m.baseline(p)) == 0.0
    _, geom, m = make(("model.baseline.mode=linear",))
    assert m.baseline.coeff is None
    assert float(m.baseline(torch.tensor([[0.0, geom.yhi]]))) == pytest.approx(geom.v_cathode)
    assert float(m.baseline(torch.tensor([[0.0, geom.ylo]]))) == pytest.approx(geom.v_ground)


# --------------------------------------------------------------------------
# the basis must vanish on both faces, at any domain period
# --------------------------------------------------------------------------


def _faces(cfg_path, sets=()):
    from firep import bias as bias_mod
    from firep import config as C
    from firep.geometry import Geometry

    cfg = C.load_with_overrides(cfg_path, tuple(sets))
    sol = bias_mod.solve(cfg)
    geom = Geometry(cfg, sol)
    model = FieldModel(cfg, geom)
    x = np.linspace(geom.xlo, geom.xhi, 1501)
    out = []
    for yf in (geom.ylo, geom.yhi):
        pts = torch.tensor(np.column_stack((x, np.full_like(x, yf))),
                           dtype=torch.get_default_dtype())
        with torch.no_grad():
            out.append(float(model.baseline.basis(pts).abs().max()))
    return model, out


@pytest.mark.parametrize("path,worst", [
    ("examples/drift-2d-3plane.yaml", 1e-6),
    ("examples/weighting-2d-21wire.yaml", 1e-6),
])
def test_every_basis_column_vanishes_on_both_faces(path, worst):
    """The face conditions are carried by the ramp alone, for any period.

    The q = 0 de-trending is enough only when exp(-2 pi dy / P) << 1.  A
    weighting domain is 21 pitches wide and it is not, which is what the
    higher wall harmonics are for.
    """
    _, faces = _faces(path)
    assert max(faces) < worst


def test_wall_harmonics_are_what_fixes_the_wide_domain():
    """Turning them off must reproduce the old, much larger face residual."""
    _, good = _faces("examples/weighting-2d-21wire.yaml")
    _, bad = _faces("examples/weighting-2d-21wire.yaml",
                    ["model.baseline.wall_harmonics=0"])
    assert max(bad) > 0.1          # a single column leaves O(1) on the ground
    assert max(good) < 1e-6
    # ...and a drift solve needs almost none of them
    m, _ = _faces("examples/drift-2d-3plane.yaml")
    assert m.baseline.n_wall <= 2


def test_wall_correction_stays_harmonic():
    """Subtracting it must not spoil what the whole ansatz is built on."""
    from firep import bias as bias_mod
    from firep import config as C
    from firep.geometry import Geometry
    from firep.siren import laplacian

    torch.set_default_dtype(torch.float64)
    try:
        cfg = C.load("examples/weighting-2d-21wire.yaml")
        sol = bias_mod.solve(cfg)
        geom = Geometry(cfg, sol)
        model = FieldModel(cfg, geom).double()
        assert model.baseline.n_wall > 10
        rng = np.random.default_rng(3)
        pts = np.column_stack((rng.uniform(geom.xlo, geom.xhi, 512),
                               rng.uniform(geom.ylo + 1.0, geom.yhi - 1.0, 512)))
        pts = pts[~geom.inside(pts, 0.5)]
        t = torch.tensor(pts, dtype=torch.float64, requires_grad=True)
        lap = laplacian(model.baseline, t, create_graph=False)
        assert float(lap.abs().max()) < 1e-6
    finally:
        torch.set_default_dtype(torch.float32)
