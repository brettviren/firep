import math

import pytest

from firep import bias, config as C


def test_transparency_ratio_matches_bunemann():
    # rho = 2 pi r / p with r = 0.075 mm, p = 5 mm
    rho = 2 * math.pi * 0.075 / 5.0
    assert bias.transparency_ratio(0.075, 5.0) == pytest.approx((1 + rho) / (1 - rho))
    assert bias.transparency_ratio(0.075, 5.0) == pytest.approx(1.208114, abs=1e-5)


def test_default_solution_is_self_consistent():
    cfg = C.from_dict({})
    sol = bias.solve(cfg)

    # The literal reading of the spec: remove the wires and a uniform
    # 500 V/cm spans the whole 110 mm cathode-to-ground gap.
    assert sol.ground == 0.0
    assert sol.cathode == pytest.approx(-500.0 * 0.1 * 110.0)

    # Ordering: the cathode is the most negative, and the potential rises
    # monotonically down through the planes toward the collection wire.
    pots = [sol.electrodes[e.name] for e in cfg.electrodes_by_y]
    assert sol.cathode < pots[0] < pots[1] < pots[2]

    # The collection wire sits above the ground plane potential, so the field
    # in the gap below it is reversed: electrons cannot get through.
    assert pots[2] > sol.ground
    assert sol.gaps[-1].field < 0


def test_plane_average_potential_connects_the_two_faces():
    # <V> is piecewise linear in y, so marching the gap fields from the cathode
    # to the ground plane must land exactly on the ground potential.
    cfg = C.from_dict({})
    sol = bias.solve(cfg)
    v = sol.cathode
    for g in sol.gaps:
        v += g.field * 0.1 * g.length  # V/cm -> V/mm
    assert v == pytest.approx(sol.ground, abs=1e-9)


def test_wire_self_potential_offset():
    # V_wire = <V> + gamma * Lambda, with Lambda = ln(p / 2 pi r) and
    # gamma = p (E_above - E_below) / 2 pi from Gauss's law.
    cfg = C.from_dict({})
    sol = bias.solve(cfg)
    lam = math.log(5.0 / (2 * math.pi * 0.075))
    for i, e in enumerate(cfg.electrodes_by_y):
        gamma = 5.0 * (sol.gaps[i].field - sol.gaps[i + 1].field) * 0.1 / (2 * math.pi)
        assert sol.lam[e.name] == pytest.approx(lam)
        assert sol.gamma[e.name] == pytest.approx(gamma)
        assert sol.offset(e.name) == pytest.approx(gamma * lam)

    # The correction is large, not a refinement: the collection wire sits
    # 340 V above its own plane-averaged potential.
    assert sol.offset("w") == pytest.approx(340.0, abs=1.0)


def test_explicit_mode_inverts_the_transparency_solution():
    # Feed the derived wire voltages back in as explicit biases; the solver
    # must recover the same plane-averaged gap fields.
    fwd = bias.solve(C.from_dict({}))
    sets = ("bias.mode=explicit", f"domain.boundaries.yhi.potential={fwd.cathode}") + tuple(
        f"electrodes.{i}.bias={fwd.electrodes[n]}" for i, n in enumerate("uvw")
    )
    inv = bias.solve(C.load_with_overrides(None, sets))
    for a, b in zip(fwd.gaps, inv.gaps):
        assert b.field == pytest.approx(a.field, abs=1e-6)
    for name in "uvw":
        assert inv.plane_average[name] == pytest.approx(fwd.plane_average[name], abs=1e-6)
        assert inv.gamma[name] == pytest.approx(fwd.gamma[name], abs=1e-6)


def test_checkpoint_roundtrip_of_the_solution():
    sol = bias.solve(C.from_dict({}))
    again = bias.BiasSolution.from_dict(sol.to_dict())
    assert again.electrodes == sol.electrodes
    assert [g.field for g in again.gaps] == [g.field for g in sol.gaps]
    assert again.offset("w") == pytest.approx(sol.offset("w"))


def test_transparency_is_actually_satisfied():
    cfg = C.from_dict({})
    sol = bias.solve(cfg)
    fields = [g.field for g in sol.gaps]
    for i, e in enumerate(cfg.electrodes_by_y[:-1]):
        assert fields[i + 1] / fields[i] >= sol.ratios[e.name] - 1e-9


def test_margin_increases_the_gap_fields():
    a = bias.solve(C.from_dict({}))
    b = bias.solve(C.from_dict({"bias": {"margin": 1.2}}))
    assert b.gaps[1].field / b.gaps[0].field > a.gaps[1].field / a.gaps[0].field


def test_drift_region_anchor_gives_the_nominal_field():
    cfg = C.from_dict({"bias": {"anchor": "drift_region"}})
    sol = bias.solve(cfg)
    assert sol.gaps[0].field == pytest.approx(500.0)
    assert sol.drift_field_actual == pytest.approx(500.0)


def test_cathode_uniform_anchor_overshoots_the_nominal_field():
    # Documented consequence: the transparency cascade needs more volts per mm
    # in the wire region, so the drift region gets less than the naive share.
    sol = bias.solve(C.from_dict({}))
    assert sol.drift_field_actual > sol.drift_field_nominal


def test_explicit_mode_passes_potentials_through():
    cfg = C.load_with_overrides(
        None,
        (
            "bias.mode=explicit",
            "domain.boundaries.yhi.potential=-5000.0",
            "electrodes.0.bias=-100.0",
            "electrodes.1.bias=0.0",
            "electrodes.2.bias=500.0",
        ),
    )
    sol = bias.solve(cfg)
    assert sol.cathode == -5000.0
    # Reconstructed through the linear solve, so exact only to round-off.
    for name, want in (("u", -100.0), ("v", 0.0), ("w", 500.0)):
        assert sol.electrodes[name] == pytest.approx(want, abs=1e-9)
    # The naive reading would put 500 V over 5 mm -> 1000 V/cm in the v-w gap.
    # The wire self-potentials make the real plane-averaged field different,
    # and every gap must still march <V> from the cathode to the ground plane.
    assert sol.gaps[2].field != pytest.approx(1000.0, rel=1e-3)
    v = sol.cathode
    for g in sol.gaps:
        v += g.field * 0.1 * g.length
    assert v == pytest.approx(sol.ground, abs=1e-9)


def test_too_much_reversal_is_rejected():
    with pytest.raises(C.ConfigError, match="collection_reverse_ratio"):
        bias.solve(C.from_dict({"bias": {"collection_reverse_ratio": 50.0}}))


def test_report_runs():
    cfg = C.from_dict({})
    text = bias.report(cfg, bias.solve(cfg))
    assert "transparency" in text and "FAIL" not in text
