"""Electrode shapes as signed distance functions.

Every shape must satisfy three properties, which between them are what the
sampler, the boundary loss and the drift integrator rely on: points drawn on the
surface have zero signed distance, the interior is negative and the exterior
positive, and the distance is a true distance (unit gradient) away from the
medial axis.
"""

import math

import numpy as np
import pytest

from firep import shapes as S

ALL = [
    ("wire2", S.Wire(ndim=2, radius=0.075)),
    ("wire3", S.Wire(ndim=3, radius=0.075, angle=60.0)),
    ("strip2", S.Strip(ndim=2, width=5.0, thickness=0.035)),
    ("strip3", S.Strip(ndim=3, width=5.0, thickness=0.035, angle=-30.0)),
    ("hole", S.Hole(ndim=3, radius=2.5, thickness=0.5, sheet_extent=20.0)),
    ("pad", S.Pad(ndim=3, width=3.0, length=3.0, thickness=0.035)),
]


@pytest.mark.parametrize("name,shape", ALL, ids=[n for n, _ in ALL])
def test_surface_samples_have_zero_signed_distance(name, shape):
    assert shape.check_surface(4000) < 1e-9


@pytest.mark.parametrize("name,shape", ALL, ids=[n for n, _ in ALL])
def test_signed_distance_has_unit_gradient_outside(name, shape):
    """A true distance function; the drift step size relies on it."""
    rng = np.random.default_rng(1)
    pts = shape.sample_surface(40, rng)
    # step a little way along the outward normal direction, estimated by hand
    h = 1e-5
    out = 0.0
    for p in pts:
        probe = p + rng.normal(0, 0.4, shape.ndim)
        if shape.sdf(probe[None])[0] < 0.05:
            continue
        g = np.array([
            (shape.sdf(probe + e * h)[0] - shape.sdf(probe - e * h)[0]) / (2 * h)
            for e in np.eye(shape.ndim)
        ])
        out = max(out, abs(np.linalg.norm(g) - 1.0))
    assert out < 1e-4


def test_wire_reduces_to_a_circle_in_2d():
    w = S.Wire(ndim=2, centre=(1.0, 2.0), radius=0.5)
    assert w.sdf(np.array([[1.0, 2.0]]))[0] == pytest.approx(-0.5)
    assert w.sdf(np.array([[1.5, 2.0]]))[0] == pytest.approx(0.0)
    assert w.sdf(np.array([[3.0, 2.0]]))[0] == pytest.approx(1.5)


@pytest.mark.parametrize("angle", [0.0, 30.0, 60.0, -60.0, 90.0])
def test_wires_and_strips_are_invariant_along_their_own_axis(angle):
    """The property that keeps the analytic basis harmonic in 3D."""
    a = math.radians(angle)
    axis = np.array([math.sin(a), 0.0, math.cos(a)])
    p = np.array([[0.9, 0.4, -1.3]])
    for shape in (S.Wire(ndim=3, radius=0.075, angle=angle),
                  S.Strip(ndim=3, width=4.0, thickness=0.05, angle=angle)):
        assert shape.sdf(p)[0] == pytest.approx(shape.sdf(p + 17.0 * axis)[0],
                                                abs=1e-9)


def test_hole_conductor_is_the_sheet_not_the_aperture():
    h = S.Hole(ndim=3, radius=2.0, thickness=0.4)
    assert h.sdf(np.array([[0.0, 0.0, 0.0]]))[0] > 0  # the aperture is not metal
    assert h.sdf(np.array([[6.0, 0.0, 0.0]]))[0] < 0  # the sheet is
    assert h.sdf(np.array([[6.0, 5.0, 0.0]]))[0] == pytest.approx(4.8)
    # the aperture is round
    for th in np.linspace(0, 2 * np.pi, 9):
        p = np.array([[2.0 * np.cos(th), 0.0, 2.0 * np.sin(th)]])
        assert h.sdf(p)[0] == pytest.approx(0.0, abs=1e-12)


def test_hole_is_intrinsically_3d():
    with pytest.raises(ValueError, match="intrinsically 3D"):
        S.Hole(ndim=2)
    with pytest.raises(ValueError, match="intrinsically 3D"):
        S.Pad(ndim=2)


def test_pad_is_bounded_in_every_direction():
    p = S.Pad(ndim=3, width=3.0, length=4.0, thickness=0.1)
    lo, hi = p.bounds()
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert hi - lo == pytest.approx([3.0, 0.1, 4.0])
    # ...unlike a wire, which is not
    lo, hi = S.Wire(ndim=3).bounds()
    assert not np.all(np.isfinite(lo))


def test_strip_width_and_thickness_are_the_right_way_round():
    s = S.Strip(ndim=2, width=6.0, thickness=0.2)
    assert s.sdf(np.array([[2.9, 0.0]]))[0] < 0  # inside across the width
    assert s.sdf(np.array([[3.1, 0.0]]))[0] > 0
    assert s.sdf(np.array([[0.0, 0.09]]))[0] < 0  # inside through the thickness
    assert s.sdf(np.array([[0.0, 0.11]]))[0] > 0


def test_make_from_config_dispatches_and_validates():
    from firep import config as C

    cfg = C.from_dict({})
    e = cfg.electrode("w")
    sh = S.make("wire", 2, (0.0, 0.0), e.shape)
    assert isinstance(sh, S.Wire) and sh.radius == e.shape.radius
    with pytest.raises(ValueError, match="unknown electrode kind"):
        S.make("sphere", 2, (0.0, 0.0), e.shape)
    with pytest.raises(ValueError, match="needs shape.width"):
        S.make("strip", 2, (0.0, 0.0), e.shape)


def test_dune_vd_hole_lattice_aligns_with_every_strip_view():
    """Every strip gap must bisect a row of holes -- the constraint that fixes
    the lattice: 5.1 = 2d and 7.65 = 3d, so d = 2.55 mm."""
    from firep import config as C

    cfg = C.load("examples/geom-dune-vd.yaml")
    holes = S.lattice_shapes(cfg.electrode("holes2"), 3)
    pts = np.array([[h.centre[0], h.centre[2]] for h in holes])

    # equilateral packing
    d = cfg.electrode("holes2").lattice.pitch
    a = 2 * d / math.sqrt(3)
    assert cfg.electrode("holes2").lattice.pitch_z == pytest.approx(a, abs=1e-5)
    assert cfg.electrode("holes2").lattice.stagger

    for name in ("grid", "u", "v", "w"):
        e = cfg.electrode(name)
        th = math.radians(e.lattice.angle)
        across = pts @ np.array([math.cos(th), -math.sin(th)])
        rows = np.unique(np.round(across, 4))  # 0.1 um: merge round-off
        rows = rows[np.abs(rows) < 12.0]
        assert np.min(np.diff(np.sort(rows))) == pytest.approx(d, abs=1e-3)

        from firep.geometry import lattice_positions

        centres = lattice_positions(e.lattice.pitch, e.lattice.count,
                                    e.lattice.offset)
        gaps = 0.5 * (centres[:-1] + centres[1:])
        gaps = gaps[np.abs(gaps) < 11.0]
        assert len(gaps)
        for g in gaps:
            assert np.min(np.abs(rows - g)) < 1e-3, (name, g)


def test_dune_vd_holes_are_inside_a_strip_or_split_between_two():
    """No hole straddles a single strip edge asymmetrically."""
    from firep import config as C
    from firep.geometry import lattice_positions

    cfg = C.load("examples/geom-dune-vd.yaml")
    holes = S.lattice_shapes(cfg.electrode("holes2"), 3)
    pts = np.array([[h.centre[0], h.centre[2]] for h in holes])
    r = cfg.electrode("holes2").shape.radius
    for name in ("w", "u"):
        e = cfg.electrode(name)
        th = math.radians(e.lattice.angle)
        across = pts @ np.array([math.cos(th), -math.sin(th)])
        centres = lattice_positions(e.lattice.pitch, e.lattice.count,
                                    e.lattice.offset)
        half = 0.5 * e.shape.width
        for x in np.unique(np.round(across, 4)):
            if abs(x) > 10.0:
                continue
            dist = np.min(np.abs(centres - x))
            on_gap = np.min(np.abs(0.5 * (centres[:-1] + centres[1:]) - x))
            # either comfortably inside a strip, or centred on a gap
            assert dist + r < half + 1e-3 or on_gap < 1e-3, (name, x)


def test_example_geometries_describe_all_three_detectors():
    """The configs in examples/ are valid and build the shapes they claim."""
    from firep import config as C

    for path, kinds in (("examples/geom-microboone-3d.yaml", {"wire"}),
                        ("examples/geom-dune-vd.yaml", {"strip", "hole"}),
                        ("examples/geom-pixels.yaml", {"pad"})):
        cfg = C.load(path)
        assert cfg.domain.dimension == 3
        assert {e.kind for e in cfg.electrodes} == kinds
        for e in cfg.electrodes:
            sh = S.make(e.kind, 3, (0.0, e.plane, 0.0), e.shape, e.lattice.angle)
            assert sh.check_surface(500) < 1e-9
        # ...and the solver says clearly that it cannot solve them yet
        with pytest.raises(C.ConfigError):
            cfg.require_solvable()
