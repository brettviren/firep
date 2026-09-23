"""Learning the response directly, and conditioned on a diffusion width."""

import numpy as np
import pytest
import torch

from firep import surrogate as S


def _toy(n_impact=7, n_wires=5, n_time=64, tick=0.1):
    """A response-shaped array with a smooth, known dependence on impact."""
    t = np.arange(n_time) * tick
    imp = np.linspace(0.0, 2.5, n_impact)
    cur = np.zeros((n_wires, n_impact, n_time))
    for j, x in enumerate(imp):
        for w in range(n_wires):
            off = w - n_wires // 2
            peak = 3.0 + 0.8 * x + 0.5 * abs(off)
            amp = np.exp(-0.5 * (off + 0.4 * x) ** 2)
            cur[w, j] = amp * np.exp(-0.5 * ((t - peak) / 0.4) ** 2)
    return {"current": cur[None], "impact": imp,
            "offsets": np.arange(n_wires) - n_wires // 2,
            "time": t}, tick


def test_smear_conserves_area_and_widens():
    _, tick = _toy()
    t = np.arange(400) * tick
    y = np.exp(-0.5 * ((t - 20.0) / 0.3) ** 2)
    out = S.smear(y[None, None], tick, 1.0)[0, 0]
    assert out.sum() == pytest.approx(y.sum(), rel=1e-3)
    assert out.max() < y.max()  # spread out
    w0 = (y > 0.5 * y.max()).sum()
    w1 = (out > 0.5 * out.max()).sum()
    assert w1 > w0


def test_smear_with_zero_width_is_a_no_op():
    _, tick = _toy()
    y = np.random.default_rng(0).normal(size=(2, 3, 50))
    assert np.array_equal(S.smear(y, tick, 0.0), y)


def test_training_set_shapes():
    arrays, _ = _toy()
    x0, c, y = S.training_set(arrays, ())
    assert x0.shape == (7,) and c.shape == (7, 0) and y.shape == (7, 5, 64)
    x0, c, y = S.training_set(arrays, ("sigma_t",), n_sigma=4)
    assert x0.shape == (28,) and c.shape == (28, 1) and y.shape == (28, 5, 64)
    with pytest.raises(ValueError, match="does not know how to vary"):
        S.training_set(arrays, ("banana",))


def test_surrogate_learns_the_tabulated_response():
    arrays, tick = _toy()
    x0, c, y = S.training_set(arrays, ())
    spec = S.SurrogateSpec(n_wires=y.shape[1], n_time=y.shape[2])
    model, res = S.fit(spec, x0, c, y, steps=1500, seed=0)
    assert res.rel_rms < 5e-3
    with torch.no_grad():
        pred = model(torch.tensor(x0, dtype=torch.float32)).numpy()
    assert pred.shape == y.shape


def test_surrogate_is_continuous_between_tabulated_impacts():
    """The point of a surrogate: no impact binning."""
    arrays, _ = _toy()
    x0, c, y = S.training_set(arrays, ())
    spec = S.SurrogateSpec(n_wires=y.shape[1], n_time=y.shape[2])
    model, _ = S.fit(spec, x0, c, y, steps=1500, seed=0)
    mid = 0.5 * (x0[2] + x0[3])
    with torch.no_grad():
        a = model(torch.tensor([x0[2]], dtype=torch.float32)).numpy()[0]
        m = model(torch.tensor([mid], dtype=torch.float32)).numpy()[0]
        b = model(torch.tensor([x0[3]], dtype=torch.float32)).numpy()[0]
    # the interpolant lies between its neighbours, not outside them
    assert np.abs(m).max() <= 1.2 * max(np.abs(a).max(), np.abs(b).max())
    assert not np.allclose(m, a) and not np.allclose(m, b)


def test_conditioned_surrogate_interpolates_in_sigma():
    arrays, tick = _toy()
    held = 0.55
    sig = [s for s in np.linspace(0.0, 1.2, 13) if abs(s - held) > 1e-9]
    cur = arrays["current"][0]
    imp = arrays["impact"]
    xs, cs, ts = [], [], []
    for s in sig:
        ts.append(np.transpose(S.smear(cur, tick, float(s)), (1, 0, 2)))
        xs.append(imp)
        cs.append(np.full((len(imp), 1), s))
    spec = S.SurrogateSpec(n_wires=cur.shape[0], n_time=cur.shape[-1],
                           conditions=("sigma_t",), hidden=128)
    model, res = S.fit(spec, np.concatenate(xs), np.concatenate(cs),
                       np.concatenate(ts), steps=2500, seed=0)
    truth = np.transpose(S.smear(cur, tick, held), (1, 0, 2))
    with torch.no_grad():
        pred = model(torch.tensor(imp, dtype=torch.float32),
                     torch.full((len(imp), 1), held)).numpy()
    rel = np.sqrt(((pred - truth) ** 2).mean()) / np.abs(truth).max()
    assert rel < 0.05, rel  # a width never seen in training


def test_conditioned_surrogate_needs_its_condition():
    arrays, _ = _toy()
    x0, c, y = S.training_set(arrays, ("sigma_t",), n_sigma=3)
    spec = S.SurrogateSpec(n_wires=y.shape[1], n_time=y.shape[2],
                           conditions=("sigma_t",))
    model, _ = S.fit(spec, x0, c, y, steps=50)
    with pytest.raises(ValueError, match="needs"):
        model(torch.tensor(x0, dtype=torch.float32), None)
