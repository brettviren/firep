import pytest

from firep import config as C


def test_defaults_load():
    cfg = C.from_dict({})
    assert cfg.domain.dimension == 2
    assert [e.name for e in cfg.electrodes] == ["u", "v", "w"]
    assert [e.plane for e in cfg.electrodes_by_y] == [10.0, 5.0, 0.0]
    assert cfg.electrode("w").role == "collection"


def test_yaml_roundtrip():
    cfg = C.from_dict({})
    again = C.from_dict(C.to_dict(cfg))
    assert C.to_yaml(again) == C.to_yaml(cfg)


def test_unknown_key_rejected():
    with pytest.raises(C.ConfigError, match="unknown key"):
        C.from_dict({"model": {"hiden": 32}})


def test_partial_override_keeps_defaults():
    cfg = C.from_dict({"model": {"hidden": 64}})
    assert cfg.model.hidden == 64
    assert cfg.model.layers == C.Model().layers


def test_set_path_overrides():
    cfg = C.load_with_overrides(None, ("train.steps=7", "electrodes.0.plane=12.5"))
    assert cfg.train.steps == 7
    assert cfg.electrode("u").plane == 12.5


def test_lattice_must_tile_the_period():
    with pytest.raises(C.ConfigError, match="tile the transverse period"):
        C.load_with_overrides(None, ("electrodes.0.lattice.count=3",))


def test_twentyone_wire_setup_is_valid():
    cfg = C.load_with_overrides(
        None,
        ("domain.bounds.x=[-52.5, 52.5]",)
        + tuple(f"electrodes.{i}.lattice.count=21" for i in range(3)),
    )
    assert cfg.domain.width == 105.0


def test_3d_and_other_kinds_are_describable():
    """The schema describes more than the solver implements, on purpose."""
    cfg = C.load_with_overrides(
        None, ("domain.dimension=3", "domain.bounds.z=[-10.0, 10.0]"))
    assert cfg.domain.dimension == 3
    cfg = C.load_with_overrides(
        None, ("electrodes.0.kind=strip", "electrodes.0.shape.width=5.0"))
    assert cfg.electrode("u").kind == "strip"


def test_3d_needs_z_bounds():
    with pytest.raises(C.ConfigError, match="bounds.z is missing"):
        C.load_with_overrides(None, ("domain.dimension=3",))


def test_solver_refuses_what_it_cannot_do():
    """...but the solver says so clearly, in one place."""
    cfg = C.load_with_overrides(
        None, ("domain.dimension=3", "domain.bounds.z=[-10.0, 10.0]"))
    with pytest.raises(C.ConfigError, match="solver is 2D"):
        cfg.require_solvable()

    cfg = C.load_with_overrides(
        None, ("electrodes.0.kind=strip", "electrodes.0.shape.width=5.0"))
    with pytest.raises(C.ConfigError, match="round wires only"):
        cfg.require_solvable()

    cfg = C.load_with_overrides(None, ("electrodes.0.lattice.angle=60.0",))
    with pytest.raises(C.ConfigError, match="stereo angle"):
        cfg.require_solvable()

    assert C.from_dict({}).require_solvable() is None


def test_electrode_shape_requirements():
    with pytest.raises(C.ConfigError, match="needs shape.width"):
        C.load_with_overrides(None, ("electrodes.0.kind=pad",))


def test_electrode_must_be_inside_the_domain():
    with pytest.raises(C.ConfigError, match="outside the domain"):
        C.load_with_overrides(None, ("electrodes.0.plane=200.0",))


def test_explicit_bias_requires_numbers():
    with pytest.raises(C.ConfigError):
        cfg = C.load_with_overrides(None, ("bias.mode=explicit",))
        from firep import bias

        bias.solve(cfg)
