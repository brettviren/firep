"""Physics cross-checks, run against the analytic baseline alone.

The least-squares warm start already satisfies every boundary condition to
~1e-5 of the potential span, so it is a valid solution to check against without
paying for a training run.
"""

import numpy as np
import pytest
import torch

from firep import bias, config as C, physics
from firep.geometry import Geometry, Sampler
from firep.siren import FieldModel


def warm_started(sets=()):
    """Config, geometry and a model whose network output is switched off."""
    cfg = C.load_with_overrides(None, ("train.precision=float64",) + tuple(sets))
    sol = bias.solve(cfg)
    geom = Geometry(cfg, sol)
    torch.set_default_dtype(torch.float64)
    model = FieldModel(cfg, geom).double()
    s = Sampler(geom, 5, cfg.sampling.near_factor)
    xy, vals = s.surfaces(256)
    rms = model.baseline.warm_start(torch.tensor(xy), torch.tensor(vals))
    with torch.no_grad():
        model.output_scale.zero_()  # analytic part only
    return cfg, geom, model, sol, rms


def test_warm_start_alone_is_an_accurate_solution():
    cfg, geom, model, sol, rms = warm_started()
    assert rms < 0.05  # volts, out of a ~6700 V span


def test_gap_fields_match_the_design():
    """The solved transverse-averaged E_y must reproduce the design gap fields.

    This is the check that caught the missing wire self-potential: with the
    naive bias (wire potential treated as the plane-averaged potential) the
    v-w gap came out 43 % low.
    """
    cfg, geom, model, sol, _ = warm_started()
    for rec in physics.gap_field_check(cfg, geom, model, n_x=1024):
        assert rec["measured"] == pytest.approx(rec["expected"], rel=2e-3), rec["gap"]


def test_gauss_law_matches_the_fitted_line_charge():
    cfg, geom, model, sol, _ = warm_started()
    for rec in physics.gauss_check(cfg, geom, model, n_x=1024):
        assert rec["coeff_model"] == pytest.approx(rec["coeff_gauss"], rel=1e-3)
        # ... and both agree with the design value from the bias solver, to the
        # accuracy of the design formula itself: it treats each row as an
        # isolated line charge, and neglects the coupling to the neighbouring
        # rows one period away, which is O(exp(-2 pi)) = 0.2 %.
        assert rec["coeff_gauss"] == pytest.approx(sol.gamma[rec["electrode"]], rel=1e-2)


def test_averaged_field_is_constant_within_a_gap():
    """<E_y> is piecewise constant, a consequence of Laplace the model never saw."""
    cfg, geom, model, sol, _ = warm_started()
    field = physics._evaluator(cfg, model)
    x = geom.xlo + (np.arange(1024) + 0.5) * geom.width / 1024
    for y0, y1 in ((20.0, 80.0), (6.0, 9.0), (1.0, 4.0), (-8.0, -2.0)):
        vals = [
            field(np.column_stack((x, np.full(len(x), y))))[:, 1].mean()
            for y in np.linspace(y0, y1, 5)
        ]
        assert np.ptp(vals) / abs(np.mean(vals)) < 2e-3


def test_drift_paths_confirm_transparency():
    cfg, geom, model, sol, _ = warm_started()
    landed, _, _ = physics.drift_paths(cfg, geom, model, n=24, max_steps=8000)
    counts = dict(zip(*np.unique(np.array(landed, dtype=str), return_counts=True)))
    assert counts.get("w", 0) == 24, counts  # all collected, none lost
    assert "u" not in counts and "v" not in counts
    assert "ground" not in counts and "stalled" not in counts
    assert "PASS" in physics.transparency_report(cfg, geom, landed)


def test_drift_paths_detect_a_leaky_collection_plane():
    """Leave the field below the collection plane pointing up and electrons escape."""
    cfg, geom, model, sol, _ = warm_started(
        ("bias.collection_reverse_ratio=-0.5",)
    )
    assert sol.gaps[-1].field > 0  # not reversed: the plane is not opaque
    landed, _, _ = physics.drift_paths(cfg, geom, model, n=16, max_steps=8000)
    counts = dict(zip(*np.unique(np.array(landed, dtype=str), return_counts=True)))
    assert counts.get("ground", 0) > 0, counts
    assert "FAIL" in physics.transparency_report(cfg, geom, landed)


def test_drift_paths_detect_an_opaque_induction_plane():
    """Under-bias the v plane and it starts catching charge meant for w."""
    cfg, geom, model, sol, _ = warm_started(("bias.mode=explicit",
        "domain.boundaries.yhi.potential=-5500.0",
        "electrodes.0.bias=53.67", "electrodes.1.bias=100.0",
        "electrodes.2.bias=1244.59"))
    landed, _, _ = physics.drift_paths(cfg, geom, model, n=16, max_steps=8000)
    counts = dict(zip(*np.unique(np.array(landed, dtype=str), return_counts=True)))
    assert counts.get("w", 0) < 16, counts
    assert "FAIL" in physics.transparency_report(cfg, geom, landed)
