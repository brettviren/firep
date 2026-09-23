"""Ionization groups, front-end electronics, and their composition with the
field response.

The composition is checked against a *synthetic* response whose answer is known
exactly, so the bookkeeping is tested independently of any solved field.
"""

import numpy as np
import pytest

from firep import electronics as E
from firep import ionization as I
from firep import signal as S

PITCH = 5.0


# --------------------------------------------------------------------------
# ionization
# --------------------------------------------------------------------------


def test_argon_transport_over_one_metre():
    p = I.ArgonProperties()
    t = p.drift_time(1000.0)
    assert t == pytest.approx(1000.0 / p.drift_speed)
    s_l, s_t, s_time = p.spread(1000.0)
    # sigma = sqrt(2 D t); 6.63 cm^2/s over 614 us is ~0.9 mm.
    assert s_l == pytest.approx(0.9025, abs=1e-3)
    assert s_t == pytest.approx(1.2637, abs=1e-3)
    assert s_time == pytest.approx(s_l / p.drift_speed)
    assert s_t > s_l  # transverse diffusion is the larger of the two
    assert p.survival(1000.0) == pytest.approx(np.exp(-t / p.lifetime))


def test_diffusion_grows_as_sqrt_distance():
    p = I.ArgonProperties()
    a, _, _ = p.spread(1000.0)
    b, _, _ = p.spread(4000.0)
    assert b / a == pytest.approx(2.0, rel=1e-9)


def test_absorption_can_be_disabled():
    p = I.ArgonProperties(lifetime=float("inf"))
    assert p.survival(1000.0) == pytest.approx(1.0)


def test_line_source_charge_and_extent():
    p = I.ArgonProperties()
    g = I.line_source(1000.0, p, length=60.0, angle=45.0, per_mm=5000.0, step=0.1)
    assert len(g) == 600
    # 5000 e/mm over 60 mm, less absorption over ~1 m.
    assert g.total == pytest.approx(5000.0 * 60.0 * p.survival(1000.0), rel=2e-3)
    half = 0.5 * 60.0 * np.cos(np.radians(45.0))
    assert g.x.max() == pytest.approx(half, abs=0.1)
    assert g.x.min() == pytest.approx(-half, abs=0.1)
    # Recentred on the track centre's arrival.
    assert abs(g.t.mean()) < 1e-6
    assert g.t_offset == pytest.approx(p.drift_time(1000.0), rel=1e-3)


def test_line_source_angles():
    p = I.ArgonProperties(lifetime=float("inf"))
    flat = I.line_source(1000.0, p, length=20.0, angle=0.0)
    assert np.ptp(flat.t) == pytest.approx(0.0, abs=1e-9)  # arrives all at once
    assert np.ptp(flat.x) == pytest.approx(20.0, abs=0.1)
    steep = I.line_source(1000.0, p, length=20.0, angle=90.0)
    assert np.ptp(steep.x) == pytest.approx(0.0, abs=1e-9)  # all on one wire
    assert np.ptp(steep.t) > 0


def test_far_deposits_arrive_later_and_dimmer():
    p = I.ArgonProperties()
    g = I.from_points([0.0, 0.0], [-100.0, +100.0], [0.0, 0.0], [1000.0, 1000.0],
                      1000.0, p, recentre=False)
    assert g.t[1] > g.t[0]  # further away -> later
    assert g.n[1] < g.n[0]  # further away -> more absorbed
    assert g.sigma_x[1] > g.sigma_x[0]  # further away -> more diffuse


def test_truncated_gaussian_bins_sum_to_one():
    edges = np.linspace(-20, 20, 4001)
    f = I.bin_fractions(np.array([0.0, 3.0]), np.array([1.0, 2.0]), edges, n_sigma=3.0)
    assert np.allclose(f.sum(axis=1), 1.0, atol=1e-9)


def test_truncation_is_respected():
    edges = np.linspace(-20, 20, 4001)
    centres = 0.5 * (edges[:-1] + edges[1:])
    f = I.bin_fractions(np.array([0.0]), np.array([1.0]), edges, n_sigma=3.0)[0]
    assert f[np.abs(centres) > 3.01].sum() == pytest.approx(0.0, abs=1e-12)
    assert f[np.abs(centres) < 2.99].sum() > 0.99


def test_arrival_grid_conserves_charge():
    p = I.ArgonProperties()
    g = I.line_source(1000.0, p, length=40.0, angle=30.0)
    xe = np.linspace(-60, 60, 961)
    te = np.linspace(-40, 40, 801)
    a = I.arrival_grid(g, xe, te)
    assert a.sum() == pytest.approx(g.total, rel=1e-9)


def test_groups_reject_ragged_input():
    with pytest.raises(ValueError, match="mismatched array lengths"):
        I.Groups(n=[1, 2], x=[0], z=[0], t=[0], sigma_x=[1], sigma_z=[1], sigma_t=[1])


# --------------------------------------------------------------------------
# electronics
# --------------------------------------------------------------------------


def _impulse(dt=0.01, n=4000, at=10, charge_fc=1.0):
    t = np.arange(n) * dt
    cur = np.zeros(n)
    cur[at] = charge_fc / E.E_IN_FC / dt  # e/us delivering `charge_fc` in one bin
    return t, cur, at


def test_shaper_peak_gain_and_peaking_time():
    """A 1 fC impulse must peak at exactly the quoted gain, at t = tau."""
    e = E.Electronics(gain=14.0, mode="shaper", filter="rc2", tau=2.0)
    t, cur, at = _impulse()
    v = e.shape(cur, 0.01)
    assert v.max() == pytest.approx(14.0, rel=1e-3)
    assert t[v.argmax()] - t[at] == pytest.approx(2.0, abs=0.02)


def test_uboone_shape_is_unipolar_and_peaks_at_tau():
    """The MicroBooNE cold-electronics response, as used by default."""
    e = E.Electronics(gain=14.0, mode="shaper", filter="uboone", tau=2.0)
    t, cur, at = _impulse(dt=0.005, n=6000)
    v = e.shape(cur, 0.005)
    assert v.max() == pytest.approx(14.0, rel=1e-3)
    assert t[v.argmax()] - t[at] == pytest.approx(2.0, abs=0.01)
    assert v.min() > -0.01 * v.max()  # unipolar
    assert abs(v[-1]) < 1e-6 * v.max()  # returns to baseline


def test_uboone_raw_shape_properties():
    x = np.linspace(0, 12, 200001)
    y = E.uboone_shape(x)
    assert E.uboone_shape(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-4)
    # One real pole and two complex pairs: a single maximum, no ringing.
    assert x[y.argmax()] == pytest.approx(E.UBOONE_PEAK_X, abs=1e-3)
    assert y.min() > -1e-3 * y.max()
    assert abs(E.uboone_shape(np.array([10.0]))[0]) < 1e-9 * y.max()


def test_uboone_is_narrower_than_a_generic_shaper():
    """It rises faster than CR-RC^2 at the same peaking time."""
    dt = 0.005
    ub = E.shaping_kernel("uboone", 2.0, dt, normalise="peak")
    rc = E.shaping_kernel("rc2", 2.0, dt, order=2, normalise="peak")
    t = np.arange(len(ub)) * dt

    def fwhm(k):
        idx = np.flatnonzero(k > 0.5 * k.max())
        return (idx[-1] - idx[0]) * dt

    assert fwhm(ub) < fwhm(rc[:len(ub)])


def test_shaper_gain_is_linear_in_charge():
    e = E.Electronics()
    t, c1, _ = _impulse(charge_fc=1.0)
    _, c3, _ = _impulse(charge_fc=3.0)
    assert e.shape(c3, 0.01).max() == pytest.approx(3 * e.shape(c1, 0.01).max(), rel=1e-9)


def test_shaper_returns_to_baseline():
    e = E.Electronics()
    t, cur, _ = _impulse()
    v = e.shape(cur, 0.01)
    assert abs(v[-1]) < 1e-6 * v.max()


def test_integrator_response_to_an_impulse_is_a_step():
    e = E.Electronics(mode="integrator", filter="rc", tau=2.0)
    t, cur, _ = _impulse()
    v = e.shape(cur, 0.01)
    # Steps up to G*Q and stays: no feedback resistor.
    assert v[-1] == pytest.approx(14.0, rel=1e-3)
    assert v[-1] == pytest.approx(v.max(), rel=1e-6)


def test_integrator_with_decay_comes_back_down():
    e = E.Electronics(mode="integrator", filter="rc", tau=2.0, decay=20.0)
    t, cur, _ = _impulse()
    v = e.shape(cur, 0.01)
    assert v.max() > 0
    assert v[-1] < 0.2 * v.max()


def test_induction_like_current_gives_a_bipolar_shaped_signal():
    """Zero net charge in, zero net area out, and both signs present."""
    e = E.Electronics()
    dt, n = 0.01, 4000
    t = np.arange(n) * dt
    cur = np.zeros(n)
    cur[500:1000] = 1.0 / E.E_IN_FC / (500 * dt)  # +1 fC
    cur[1000:1500] = -1.0 / E.E_IN_FC / (500 * dt)  # -1 fC
    v = e.shape(cur, dt)
    assert v.max() > 0 and v.min() < 0
    assert abs(v.sum() * dt) < 1e-6 * abs(v).max() * len(v) * dt


def test_adc_quantisation_and_range():
    e = E.Electronics(adc_bits=14, adc_range=1000.0)
    assert e.lsb == pytest.approx(2000.0 / 16384)
    assert e.count_limits == (-8192, 8191)
    t = np.arange(10) * 0.5
    volts = np.array([0.0, e.lsb, 2 * e.lsb, -3 * e.lsb, 1e6, -1e6, 0, 0, 0, 0])
    counts, ts = e.digitise(volts, t)
    assert list(counts[:4]) == [0, 1, 2, -3]
    assert counts[4] == 8191 and counts[5] == -8192  # saturation
    assert ts[1] - ts[0] == pytest.approx(0.5)


def test_adc_sampling_rate():
    e = E.Electronics(adc_rate=2.0)
    t = np.arange(1000) * 0.1  # 100 us at 0.1 us
    counts, ts = e.digitise(np.zeros(1000), t)
    assert ts[1] - ts[0] == pytest.approx(0.5)
    assert len(ts) == pytest.approx(200, abs=1)


def test_electronics_rejects_nonsense():
    with pytest.raises(ValueError):
        E.Electronics(mode="magic")
    with pytest.raises(ValueError):
        E.Electronics(adc_range=-1)
    with pytest.raises(ValueError, match="unknown filter kind"):
        E.shaping_kernel("nope", 2.0, 0.1)


# --------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------


def _synthetic_response(ni=3, offsets=(-1, 0, 1), nt=40, tick=0.1):
    """A response in which one electron induces exactly 1 e on its own wire."""
    npl, noff = 1, len(offsets)
    cur = np.zeros((npl, noff, ni, nt))
    i0 = list(offsets).index(0)
    cur[0, i0, :, 1] = 1.0 / tick  # integral = 1 e, on the wire it lands on
    arrays = {
        "time": np.arange(nt) * tick,
        "impact": np.linspace(0.0, 0.5 * PITCH, ni),
        "offsets": np.array(offsets),
        "current": cur,
    }
    return arrays, {"planes": ["w"], "pitch_mm": PITCH}


def _point(x, n=1000.0, t=0.0, sx=1e-3, st=1e-3):
    return I.Groups(n=[n], x=[x], z=[0.0], t=[t], sigma_x=[sx], sigma_z=[sx],
                    sigma_t=[st])


def test_composition_puts_charge_on_the_right_wire():
    arrays, meta = _synthetic_response(ni=6)
    e = E.Electronics()
    for x, want in ((0.0, 0), (5.0, 1), (-10.0, -2), (11.5, 2), (-1.5, 0)):
        frame = S.simulate(arrays, meta, _point(x), e)
        tick = frame.time[1] - frame.time[0]
        q = frame.current[0].sum(axis=-1) * tick
        assert frame.wires[int(np.argmax(q))] == want, (x, frame.wires[np.argmax(q)])
        assert q.max() == pytest.approx(1000.0, rel=1e-6)


def test_charge_exactly_on_a_cell_boundary_splits_evenly():
    """A point midway between two wires belongs to neither, so it is shared.

    Without this the half-pitch cells would be lopsided about their own wire and
    the composition would lose its mirror symmetry.
    """
    arrays, meta = _synthetic_response(ni=6)
    frame = S.simulate(arrays, meta, _point(0.5 * PITCH), E.Electronics())
    tick = frame.time[1] - frame.time[0]
    q = frame.current[0].sum(axis=-1) * tick
    share = {int(w): v for w, v in zip(frame.wires, q) if v > 1e-6}
    assert share == pytest.approx({0: 500.0, 1: 500.0}, rel=1e-6)
    assert sum(share.values()) == pytest.approx(1000.0, rel=1e-9)


def test_composition_conserves_total_charge():
    arrays, meta = _synthetic_response()
    p = I.ArgonProperties()
    groups = I.line_source(1000.0, p, length=30.0, angle=45.0, step=0.5)
    frame = S.simulate(arrays, meta, groups, E.Electronics())
    tick = frame.time[1] - frame.time[0]
    assert frame.current[0].sum() * tick == pytest.approx(groups.total, rel=1e-6)


def test_composition_is_mirror_symmetric():
    arrays, meta = _synthetic_response(ni=6)
    e = E.Electronics()
    a = S.simulate(arrays, meta, _point(+1.0, sx=0.8), e)
    b = S.simulate(arrays, meta, _point(-1.0, sx=0.8), e)
    tick = a.time[1] - a.time[0]
    qa = a.current[0].sum(axis=-1) * tick
    qb = b.current[0].sum(axis=-1) * tick
    # Same charge pattern, reflected about wire 0.
    fa = {int(w): v for w, v in zip(a.wires, qa) if v > 1e-6}
    fb = {int(w): v for w, v in zip(b.wires, qb) if v > 1e-6}
    assert set(fa) == {-w for w in fb}
    for w, v in fa.items():
        assert v == pytest.approx(fb[-w], rel=1e-6)


def test_composition_delays_by_the_arrival_time():
    arrays, meta = _synthetic_response()
    e = E.Electronics()
    a = S.simulate(arrays, meta, _point(0.0, t=0.0), e)
    b = S.simulate(arrays, meta, _point(0.0, t=5.0), e)
    ia = int(np.argmax(np.abs(a.current[0]).sum(axis=0)))
    ib = int(np.argmax(np.abs(b.current[0]).sum(axis=0)))
    assert b.time[ib] - a.time[ia] == pytest.approx(5.0, abs=0.15)


def test_transverse_spread_shares_charge_between_wires():
    arrays, meta = _synthetic_response(ni=6)
    frame = S.simulate(arrays, meta, _point(0.0, sx=2.0), E.Electronics())
    tick = frame.time[1] - frame.time[0]
    q = frame.current[0].sum(axis=-1) * tick
    busy = q[q > 1e-3]
    assert len(busy) >= 3  # sigma = 2 mm on a 5 mm pitch reaches the neighbours
    assert q.sum() == pytest.approx(1000.0, rel=1e-6)


def test_simulate_rejects_a_full_pitch_response():
    arrays, meta = _synthetic_response()
    arrays["impact"] = np.linspace(-0.5 * PITCH, 0.5 * PITCH, 3)
    with pytest.raises(ValueError, match="half-pitch impact tabulation"):
        S.simulate(arrays, meta, _point(0.0), E.Electronics())


def test_trim_response_removes_leading_silence():
    nt, tick = 100, 0.1
    cur = np.zeros((1, 1, 1, nt))
    cur[0, 0, 0, 60:70] = 1.0
    trimmed, delay = S.trim_response(cur, tick, pad=0)
    assert trimmed.shape[-1] == 10
    assert delay == pytest.approx(6.0)


def test_frame_roundtrip(tmp_path):
    arrays, meta = _synthetic_response()
    frame = S.simulate(arrays, meta, _point(0.0), E.Electronics())
    path = str(tmp_path / "s.npz")
    S.save_npz(path, frame)
    back, bmeta = S.load_npz(path)
    assert bmeta["planes"] == ["w"]
    assert np.allclose(back["adc"], frame.adc)
    assert "describe" not in bmeta
    assert frame.describe()
