"""Field response: drift velocity, Ramo bookkeeping, and the combined model.

Models here are built from the analytic warm start alone (no training), which
is accurate enough to exercise every code path quickly.
"""

import numpy as np
import pytest
import torch

from firep import bias, config as C, response as R
from firep.geometry import Geometry, Sampler
from firep.siren import FieldModel


# --------------------------------------------------------------------------
# drift velocity
# --------------------------------------------------------------------------


def test_walkowiak_reproduces_the_known_lartpc_value():
    # 500 V/cm at 87.3 K is the textbook ~1.6 mm/us.
    v = R.walkowiak_speed(np.array([0.5]), 87.3)[0]
    assert v == pytest.approx(1.6281, abs=1e-3)


def test_walkowiak_is_monotonic_and_sublinear():
    e = np.array([0.5, 1.0, 2.0, 5.0, 12.0])
    v = R.walkowiak_speed(e, 87.3)
    assert np.all(np.diff(v) > 0)
    # Saturating: doubling the field less than doubles the speed.
    assert v[1] < 2 * v[0]


def test_walkowiak_vanishes_at_zero_field():
    assert R.walkowiak_speed(np.array([0.0]), 87.3)[0] == 0.0


def test_walkowiak_temperature_dependence_has_the_right_sign():
    # Colder argon drifts faster.
    cold = R.walkowiak_speed(np.array([0.5]), 86.0)[0]
    warm = R.walkowiak_speed(np.array([0.5]), 89.0)[0]
    assert cold > warm


def test_constant_mobility_units():
    # mu = 320 cm^2/(V s) at 500 V/cm -> 1.6e5 cm/s = 1.6 mm/us
    v = R.constant_mobility_speed(np.array([0.5]), 320.0)[0]
    assert v == pytest.approx(1.6, rel=1e-9)


def test_drift_speed_dispatch_and_units():
    # 50 V/mm == 500 V/cm == 0.5 kV/cm
    assert R.drift_speed(np.array([50.0]), "constant", 87.3, 320.0)[0] == pytest.approx(1.6)
    with pytest.raises(ValueError, match="unknown drift velocity model"):
        R.drift_speed(np.array([50.0]), "nope", 87.3, 320.0)


# --------------------------------------------------------------------------
# the combined model
# --------------------------------------------------------------------------


def _warm(sets):
    cfg = C.load_with_overrides(None, ("train.precision=float64",) + tuple(sets))
    sol = bias.solve(cfg)
    geom = Geometry(cfg, sol)
    torch.set_default_dtype(torch.float64)
    model = FieldModel(cfg, geom).double()
    s = Sampler(geom, 11, cfg.sampling.near_factor)
    xy, vals = s.surfaces(64)
    model.baseline.warm_start(torch.tensor(xy), torch.tensor(vals))
    with torch.no_grad():
        model.output_scale.zero_()  # analytic part only
    return cfg, geom, model, sol


def _combined(n_wires=5):
    half = n_wires * 5.0 / 2
    dcfg, dgeom, dmodel, dsol = _warm(())
    weighting = []
    for plane in ("u", "v", "w"):
        sets = (f"domain.bounds.x=[{-half},{half}]", "problem.kind=weighting",
                f"problem.weighting.electrode={plane}") + tuple(
                f"electrodes.{i}.lattice.count={n_wires}" for i in range(3))
        wcfg, wgeom, wmodel, _ = _warm(sets)
        hot = [c for c in wgeom.conductors if c.potential != 0.0][0]
        weighting.append(R.WeightingSolution(
            plane=plane, cfg=wcfg, geom=wgeom, model=wmodel, unit=hot.potential,
            sense_x=hot.x, sense_y=hot.y, pitch=5.0))
    return R.CombinedModel(dcfg, dgeom, dmodel, weighting)


def test_combined_model_rejects_a_mismatched_pitch():
    cm = _combined()
    bad = cm.weighting["u"]
    bad.pitch = 4.0
    with pytest.raises(ValueError, match="drift pitch"):
        R.CombinedModel(cm.cfg, cm.geom, cm.model, [bad])


def test_combined_model_rejects_an_unknown_plane():
    cm = _combined()
    w = cm.weighting["u"]
    w.plane = "zzz"
    with pytest.raises(ValueError, match="no matching electrode"):
        R.CombinedModel(cm.cfg, cm.geom, cm.model, [w])


def test_weighting_potential_is_one_on_its_own_wire_and_zero_on_the_others():
    cm = _combined()
    w = cm.weighting["w"]
    r = cm.cfg.electrode("w").shape.radius
    on = cm.weighting_potential("w", np.array([[0.0, w.sense_y + r]]))[0]
    assert on == pytest.approx(1.0, abs=0.05)
    # A neighbouring w wire, one pitch away, is grounded.
    off = cm.weighting_potential("w", np.array([[5.0, w.sense_y + r]]))[0]
    assert abs(off) < 0.05
    # And a u wire is grounded in the w weighting solve.
    other = cm.weighting_potential("w", np.array([[0.0, 10.0 + r]]))[0]
    assert abs(other) < 0.05


def test_drift_field_is_periodic_over_one_pitch():
    cm = _combined()
    a = cm.drift_field(np.array([[1.3, 20.0], [-2.0, 3.0]]))
    b = cm.drift_field(np.array([[1.3 + 5.0, 20.0], [-2.0 + 15.0, 3.0]]))
    assert np.allclose(a, b, atol=1e-6)


# --------------------------------------------------------------------------
# Ramo
# --------------------------------------------------------------------------


def _response(**kw):
    cm = _combined()
    kw.setdefault("impacts", 2)
    # Launch from just below the cathode: every weighting potential vanishes on
    # that face, so Ramo's integral is not truncated and the conservation
    # checks below mean what they say.
    kw.setdefault("y_start", "auto")
    kw.setdefault("tick", 0.5)
    kw.setdefault("n_wires", 3)
    return cm, R.response(cm, **kw)


def test_electrons_reach_the_collection_plane():
    cm, resp = _response()
    assert set(resp.landed) == {"w"}


def test_integrated_charge_equals_the_weighting_potential_difference():
    """Pure bookkeeping: Ramo's integral depends only on the endpoints.

    This is independent of how good the field solutions are -- it checks that
    the code integrates what it claims to integrate.
    """
    cm, resp = _response()
    pitch = 5.0
    for ip, plane in enumerate(resp.planes):
        for ii in range(len(resp.impact)):
            x = resp.trajectory["x"][:, ii]
            y = resp.trajectory["y"][:, ii]
            live = resp.trajectory["alive"][:, ii]
            last = int(np.argmax(~live)) if (~live).any() else len(live) - 1
            for iw, off in enumerate(resp.offsets):
                ends = np.array([[x[0] - off * pitch, y[0]],
                                 [x[last] - off * pitch, y[last]]])
                phi = cm.weighting_potential(plane, ends)
                assert resp.integrated[ip, iw, ii] == pytest.approx(
                    phi[1] - phi[0], abs=1e-9)


def test_collection_wire_collects_one_electron():
    cm, resp = _response()
    iw = int(np.argmin(np.abs(resp.offsets)))
    ip = resp.planes.index("w")
    assert resp.integrated[ip, iw, 0] == pytest.approx(1.0, abs=0.02)


def test_induction_planes_see_no_net_charge():
    """The induction signal is bipolar: charge in, charge out, nothing left."""
    cm, resp = _response()
    for plane in ("u", "v"):
        ip = resp.planes.index(plane)
        assert np.abs(resp.integrated[ip]).max() < 0.02


def test_launch_truncation_explains_any_induction_residual():
    """Launch low and the residual appears, equal to phi at the launch point.

    Ramo's integral is phi(end) - phi(start); starting inside the reach of an
    induction plane's weighting potential truncates it by exactly phi(start).
    """
    cm, resp = _response(impacts=1, y_start=14.0, tick=2.0, n_wires=1)
    ip = resp.planes.index("u")
    assert resp.phi_start[ip, 0] > 0.1  # y=14 mm is well inside u's reach
    # -phi(start), up to phi(end): the u weighting potential does not quite
    # vanish at the collection wire where the electron stops.
    assert resp.integrated[ip, 0, 0] == pytest.approx(-resp.phi_start[ip, 0], abs=1e-3)


def test_current_integrates_back_to_the_induced_charge():
    cm, resp = _response()
    tot = resp.current.sum(axis=-1) * resp.tick
    assert np.allclose(tot, resp.integrated, atol=1e-9)


def test_charge_and_field_forms_agree():
    """i = dQ/dt and i = q v.E_w are independent routes to the same current."""
    cm, resp = _response()
    assert resp.agreement < 0.15


def test_response_shapes_and_metadata():
    cm, resp = _response(impacts=3, n_wires=5)
    npl, nw, ni = len(resp.planes), len(resp.offsets), len(resp.impact)
    assert resp.current.shape == (npl, nw, ni, len(resp.time))
    assert resp.integrated.shape == (npl, nw, ni)
    assert list(resp.offsets) == [-2, -1, 0, 1, 2]
    assert resp.impact[0] == 0.0 and resp.impact[-1] == pytest.approx(2.5)
    assert resp.tick == pytest.approx(0.5)


def test_full_pitch_impacts_are_symmetric():
    cm, resp = _response(impacts=5, half_pitch=False)
    assert resp.impact[0] == pytest.approx(-2.5)
    assert resp.impact[-1] == pytest.approx(2.5)
    ip = resp.planes.index("w")
    iw = int(np.argmin(np.abs(resp.offsets)))
    # Mirror-image impacts must induce the same charge on the central wire.
    # Indices 1 and 3 (-1.25 and +1.25 mm); the +-2.5 mm seeds sit exactly on
    # the saddle between two wires and fall either way.
    assert resp.integrated[ip, iw, 1] == pytest.approx(
        resp.integrated[ip, iw, 3], abs=1e-3)


def test_auto_launch_height_sits_just_below_the_cathode():
    cm = _combined()
    resp = R.response(cm, impacts=1, tick=2.0, n_wires=1, y_start="auto")
    assert resp.trajectory["y"][0, 0] == pytest.approx(99.1, abs=0.5)
    # Every weighting potential vanishes at the cathode, so the launch is clean.
    assert resp.phi_start.max() < 1e-2


def test_report_mentions_the_conservation_check():
    cm, resp = _response()
    text = R.report(cm, resp)
    assert "Ramo" in text and "induced charge" in text and "mm/us" in text


def test_no_weighting_solutions_is_an_error():
    cm = _combined()
    cm.weighting = {}
    with pytest.raises(ValueError, match="no weighting solutions"):
        R.response(cm)


# --------------------------------------------------------------------------
# re-indexing a response as a function of impact offset
# --------------------------------------------------------------------------


def _synthetic_table(pitch=5.0, half=4, sub=20, nt=3):
    """A tabulation whose answer is known: r(x0, k p) = f(x0 - k p).

    ``f`` is even, so the mirror symmetry the re-indexing relies on holds
    exactly and any error is in the index algebra rather than the physics.
    ``sub`` launch positions per pitch, so ``sub // 2 + 1`` over the half pitch.
    """
    imp = np.arange(sub // 2 + 1) * (pitch / sub)
    off = np.arange(-half, half + 1)
    t = np.arange(nt) * 0.1

    def f(u):
        return np.exp(-(u ** 2) / 50.0)

    cur = np.zeros((1, len(off), len(imp), nt))
    for iw, k in enumerate(off):
        for ii, x0 in enumerate(imp):
            cur[0, iw, ii, :] = f(x0 - k * pitch) * (1.0 + t)
    arrays = {"current": cur, "impact": imp, "offsets": off, "time": t}
    return arrays, {"pitch_mm": pitch, "planes": ["p"]}, f, t


@pytest.mark.parametrize("centred", [True, False])
def test_impact_table_reproduces_a_known_response(centred):
    arrays, meta, f, t = _synthetic_table()
    tab = R.impact_table(arrays, meta, per_side=5, centred=centred)
    u = tab["impact"]
    want = f(u)[:, None] * (1.0 + t)[None, :]
    assert tab["table"][0] == pytest.approx(want, abs=1e-12)


def test_impact_table_grid_covers_every_wire_region():
    arrays, meta, _, _ = _synthetic_table(pitch=5.0, half=10)
    u = R.impact_table(arrays, meta, per_side=5)["impact"]
    # 21 wire regions x 10 bin centres, none of them on a cell edge
    assert len(u) == 210
    assert u[0] == pytest.approx(-52.25)
    assert u[-1] == pytest.approx(52.25)
    assert np.diff(u) == pytest.approx(0.5)


def test_bin_centred_grid_avoids_both_stagnation_lines():
    """No sample sits on a wire axis or on the saddle midway between two."""
    arrays, meta, _, _ = _synthetic_table(pitch=5.0, half=10)
    pitch, half_step = 5.0, 0.25
    u = R.impact_table(arrays, meta, per_side=5, centred=True)["impact"]
    to_wire = np.abs((u + 0.5 * pitch) % pitch - 0.5 * pitch)
    assert to_wire.min() == pytest.approx(half_step)          # never on a wire
    assert np.abs(to_wire - 0.5 * pitch).min() == pytest.approx(half_step)
    # ...whereas the node grid sits on both
    v = R.impact_table(arrays, meta, per_side=5, centred=False)["impact"]
    to_wire = np.abs((v + 0.5 * pitch) % pitch - 0.5 * pitch)
    assert to_wire.min() == pytest.approx(0.0, abs=1e-9)
    assert np.abs(to_wire - 0.5 * pitch).min() == pytest.approx(0.0, abs=1e-9)


def test_impact_table_step_follows_per_side():
    arrays, meta, _, _ = _synthetic_table(pitch=5.0, half=2, sub=20)
    for per_side, step in ((1, 2.5), (5, 0.5)):
        u = R.impact_table(arrays, meta, per_side=per_side)["impact"]
        assert np.diff(u) == pytest.approx(step)


def test_impact_table_refuses_a_grid_it_was_not_given():
    arrays, meta, _, _ = _synthetic_table(sub=20)  # impacts every 0.25 mm
    with pytest.raises(ValueError, match="not in the tabulation"):
        R.impact_table(arrays, meta, per_side=7)


def test_impact_table_keeps_the_ramo_sum_rule():
    """The collecting wire still integrates to one electron in its own region.

    On the bin-centred grid there is no ambiguous row: every sample is either
    cleanly inside a collection cell or cleanly outside it.
    """
    cm, resp = _response(impacts=3)   # 0, 1.25, 2.5 mm
    arrays = {"current": resp.current, "impact": resp.impact,
              "offsets": resp.offsets, "time": resp.time}
    meta = {"pitch_mm": 5.0, "planes": resp.planes}
    tab = R.impact_table(arrays, meta, per_side=1)
    tick = float(resp.time[1] - resp.time[0])
    q = tab["table"].sum(axis=-1) * tick
    u = tab["impact"]
    ip = resp.planes.index("w")
    inside = np.abs(u) < 2.5
    assert np.abs(q[ip, inside]).min() > 0.95
    assert np.abs(q[ip, ~inside]).max() < 0.05
