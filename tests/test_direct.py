"""The transport formulation of the field response (firep.direct)."""

import numpy as np
import pytest
import torch

from firep import direct
from firep.config import DirectCfg


def small_model(log_features=True):
    cell = direct.HalfCell(
        pitch=4.0, ylo=-1.0, ytop=10.0,
        cx=np.array([-4.0, 0.0, 4.0, -4.0, 0.0, 4.0]),
        cy=np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0]),
        cr=np.full(6, 0.1), names=["w"] * 3 + ["u"] * 3)
    spec = DirectCfg(hidden=16, layers=2, duration=5.0, log_features=log_features,
                     final_init_scale=1.0, device="cpu")
    torch.manual_seed(0)
    model = direct.DirectModel(cell, ["w"], np.array([-1, 0, 1]), spec, v0=1.5)
    return model.double(), cell


# Analytic stand-ins for the coefficients, so the residual can be checked
# against a finite difference of Q along the transport direction.
def phi(xy):
    return torch.stack([torch.sin(xy[:, 0] + k) + 0.1 * xy[:, 1] ** 2 for k in range(3)], 1)


def dist(xy):
    return torch.tanh(xy[:, 0] ** 2 + 0.3 * xy[:, 1])


def vel(xy):
    return torch.stack((0.2 * torch.cos(xy[:, 1]), -1.5 - 0.1 * xy[:, 0]), 1)


def test_initial_condition_is_exact():
    model, _ = small_model()
    xy = torch.rand(50, 2, dtype=torch.float64) * torch.tensor([2.0, 9.0]) + torch.tensor([0.0, 0.5])
    xyt = torch.cat((xy, torch.zeros(50, 1, dtype=torch.float64)), 1)
    q = model(xyt, phi(xy), dist(xy))
    # q = -1, so Q = +phi at t = 0 whatever the network does.
    assert torch.allclose(q, phi(xy), atol=0, rtol=0)


@pytest.mark.parametrize("log_features", [True, False])
def test_residual_is_the_transport_derivative(log_features):
    model, _ = small_model(log_features)
    n = 40
    xy = torch.rand(n, 2, dtype=torch.float64) * torch.tensor([2.0, 9.0]) + torch.tensor([0.2, 0.5])
    t = torch.rand(n, 1, dtype=torch.float64) * 4.0 + 0.3
    xyt = torch.cat((xy, t), 1)
    v = vel(xy)

    def Q(p):
        return model(p, phi(p[:, :2]), dist(p[:, :2]))

    # -v . grad phi and -v . grad D by autograd on the analytic stand-ins
    xr = xy.clone().requires_grad_(True)
    gphi = torch.stack([torch.autograd.grad(phi(xr)[:, k].sum(), xr, retain_graph=True)[0]
                        for k in range(3)], 1)
    lphi = -(v[:, None, :] * gphi).sum(-1)
    gd = torch.autograd.grad(dist(xr).sum(), xr)[0]
    ldist = -(v * gd).sum(-1)

    lq, _ = model.residual(xyt, v, lphi.detach(), dist(xy), ldist.detach())
    h = 1e-5
    step = torch.cat((-v, torch.ones(n, 1, dtype=torch.float64)), 1)
    fd = (Q(xyt + h * step) - Q(xyt - h * step)) / (2 * h)
    assert torch.allclose(lq, fd, atol=1e-6, rtol=1e-5)


def test_causal_weights():
    w = direct.causal_weights(torch.tensor([1.0, 2.0, 3.0]), eps=0.5)
    assert torch.allclose(w, torch.exp(-0.5 * torch.tensor([0.0, 1.0, 3.0])))


def test_half_cell_distance():
    _, cell = small_model()
    d, g = cell.distance(np.array([[0.0, 1.0], [0.5, 5.0]]))
    assert d == pytest.approx([0.9, 0.4])
    assert g == pytest.approx(np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_old_checkpoint_keeps_its_inputs(tmp_path):
    """A spec saved before an option existed loads with that option off."""
    from firep.config import _plain

    model, cell = small_model()
    spec = DirectCfg(hidden=16, layers=2, duration=5.0, arrival_log=0.0, device="cpu")
    old = _plain(spec)
    for k in ("arrival_log", "arrival_time_fraction", "normalise"):
        old.pop(k)
    stored = dict(direct.LEGACY)
    stored.update(old)
    from firep.config import _build

    back = _build(DirectCfg, stored)
    assert back.arrival_log == 0.0 and back.arrival_time_fraction == 0.0
    assert back.normalise == "path"
