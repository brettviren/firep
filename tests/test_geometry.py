import numpy as np
import pytest

from firep import bias, config as C
from firep.geometry import Geometry, Sampler, lattice_positions


def make(sets=()):
    cfg = C.load_with_overrides(None, sets)
    return cfg, Geometry(cfg, bias.solve(cfg))


def test_lattice_positions_centre_on_offset():
    assert lattice_positions(5.0, 1, 0.0) == pytest.approx([0.0])
    assert lattice_positions(5.0, 2, 0.0) == pytest.approx([-2.5, 2.5])
    assert lattice_positions(5.0, 21, 0.0)[[0, 10, 20]] == pytest.approx([-50, 0, 50])


def test_default_geometry_has_three_conductors():
    cfg, g = make()
    assert len(g.conductors) == 3
    assert [c.electrode for c in g.conductors] == ["u", "v", "w"]
    assert all(c.x == 0.0 for c in g.conductors)


def test_inside_test():
    cfg, g = make()
    r = 0.075
    pts = np.array([[0.0, 0.0], [0.0, r * 0.5], [0.0, r * 2], [2.0, 50.0]])
    assert g.inside(pts).tolist() == [True, True, False, False]


def test_distance_wraps_across_the_periodic_edge():
    # Put the collection wire near the right-hand edge, at x = +2.4 mm.
    cfg, g = make(("electrodes.2.lattice.offset=2.4",))
    # A point just inside the *left* edge is 0.15 mm from that wire's image in
    # the neighbouring period, not 4.85 mm from the wire itself.
    d = g.min_gap_distance(np.array([[-2.45, 0.0]]))
    assert d[0] == pytest.approx(0.15 - 0.075, abs=1e-9)
    # Without wrapping the naive answer would have been much larger.
    assert np.hypot(-2.45 - 2.4, 0.0) - 0.075 > 4.0


def test_sampled_points_avoid_conductors():
    cfg, g = make()
    s = Sampler(g, 0, cfg.sampling.near_factor)
    for pts in (s.bulk(5000), s.near(5000)):
        assert len(pts) > 0
        assert not g.inside(pts).any()
        assert (pts[:, 1] > g.ylo).all() and (pts[:, 1] < g.yhi).all()


def test_near_sampling_concentrates_on_the_wires():
    cfg, g = make()
    s = Sampler(g, 0, 20.0)
    near, bulk = s.near(20000), s.bulk(20000)
    # Half the log-radial points should be within sqrt(20) ~ 4.5 radii.
    assert np.median(g.min_gap_distance(near)) < 0.4
    assert np.median(g.min_gap_distance(bulk)) > 1.0


def test_surface_points_sit_on_the_surface_with_the_right_potential():
    cfg, g = make()
    s = Sampler(g, 0, cfg.sampling.near_factor)
    xy, vals = s.surfaces(64)
    assert len(xy) == 3 * 64
    assert np.abs(g.min_gap_distance(xy)).max() < 1e-9
    assert set(np.round(vals, 6)) == {round(c.potential, 6) for c in g.conductors}


def test_weighting_problem_grounds_everything_but_one_wire():
    cfg, g = make(
        ("problem.kind=weighting", "domain.bounds.x=[-52.5, 52.5]")
        + tuple(f"electrodes.{i}.lattice.count=21" for i in range(3))
    )
    assert len(g.conductors) == 63
    assert g.v_cathode == 0.0 and g.v_ground == 0.0
    hot = [c for c in g.conductors if c.potential != 0.0]
    assert len(hot) == 1
    assert hot[0].electrode == "w" and hot[0].x == pytest.approx(0.0)


def test_weighting_index_selects_a_specific_wire():
    cfg, g = make(
        ("problem.kind=weighting", "problem.weighting.electrode=u",
         "problem.weighting.index=0", "domain.bounds.x=[-52.5, 52.5]")
        + tuple(f"electrodes.{i}.lattice.count=21" for i in range(3))
    )
    hot = [c for c in g.conductors if c.potential != 0.0]
    assert hot[0].electrode == "u" and hot[0].x == pytest.approx(-50.0)


def test_torch_sampler_draws_valid_points():
    """TorchSampler: the same distributions as Sampler, on the device."""
    import torch

    from firep import bias
    from firep import config as C
    from firep.geometry import Geometry, TorchSampler

    cfg = C.from_dict({})
    geom = Geometry(cfg, bias.solve(cfg))
    s = TorchSampler(geom, 7, near_factor=20.0, device="cpu", dtype=torch.float64)
    bulk = s.bulk(4000).numpy()
    near = s.near(4000).numpy()
    assert len(bulk) == 4000
    for pts in (bulk, near):
        assert (geom.min_gap_distance(pts) >= 1e-3 - 1e-12).all()
        assert (pts[:, 0] >= geom.xlo).all() and (pts[:, 0] <= geom.xhi).all()
        assert (pts[:, 1] > geom.ylo).all() and (pts[:, 1] < geom.yhi).all()
    # near points lie in the log-radial annuli
    assert (geom.min_gap_distance(near) <= geom.cr.max() * 20.0).all()
    xy, v = s.surfaces(16)
    d = geom.min_gap_distance(xy.numpy())
    assert abs(d).max() < 1e-9
    assert set(np.round(v.numpy(), 6)) == set(np.round(geom.cv, 6))
