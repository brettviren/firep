"""End-to-end CLI smoke tests.  Training is cut to a handful of steps."""

import numpy as np
import pytest
from click.testing import CliRunner

from firep.cli import main

TINY = [
    "--set", "train.steps=3",
    "--set", "train.log_every=1",
    "--set", "sampling.n_bulk=64",
    "--set", "sampling.n_near=64",
    "--set", "sampling.n_face=32",
    "--set", "sampling.n_surface=32",
    "--set", "model.hidden=16",
    "--set", "model.layers=2",
]


@pytest.fixture
def run():
    return CliRunner()


def check(result):
    if result.exit_code != 0:
        raise AssertionError(f"exit {result.exit_code}\n{result.output}\n{result.exception}")
    return result.output


def test_config_and_bias_and_geometry(run, tmp_path):
    out = check(run.invoke(main, ["config"]))
    assert "lartpc-2d-wires" in out and "collection" in out

    out = check(run.invoke(main, ["bias"]))
    assert "transparency" in out and "FAIL" not in out

    path = tmp_path / "g.png"
    out = check(run.invoke(main, ["geometry", "-o", str(path)]))
    assert "conductors: 3" in out and path.exists()


def test_config_write_then_read_back(run, tmp_path):
    path = tmp_path / "c.yaml"
    check(run.invoke(main, ["config", "-o", str(path)]))
    out = check(run.invoke(main, ["bias", "-c", str(path)]))
    assert "cathode" in out


def test_bad_override_is_reported(run):
    result = run.invoke(main, ["bias", "--set", "model.nonesuch=1"])
    assert result.exit_code != 0
    assert "unknown key" in result.output


def test_train_sample_plot_roundtrip(run, tmp_path):
    ckpt = tmp_path / "m.pt"
    check(run.invoke(main, ["train", *TINY, "-o", str(ckpt)]))
    assert ckpt.exists()

    out = check(run.invoke(main, ["check", str(ckpt), "-n", "500"]))
    assert "Laplace residual" in out

    npz = tmp_path / "s.npz"
    check(run.invoke(main, ["sample", str(ckpt), "-o", str(npz),
                            "--nx", "21", "--ny", "41", "--residual"]))
    blob = np.load(npz)
    assert blob["potential"].shape == (41, 21)
    assert set(blob.files) >= {"x", "y", "potential", "ex", "ey", "emag",
                               "inside", "residual", "metadata"}

    for name, extra in (("p.png", []), ("p.pdf", ["--contours", "5"]),
                        ("e.png", ["-f", "emag", "--log"]),
                        ("d.png", ["--drift", "3"])):
        path = tmp_path / name
        check(run.invoke(main, ["plot", str(npz), "-o", str(path), *extra]))
        assert path.stat().st_size > 0

    path = tmp_path / "l.png"
    check(run.invoke(main, ["losses", str(ckpt), "-o", str(path)]))
    assert path.exists()


def test_sample_npy_format(run, tmp_path):
    ckpt = tmp_path / "m.pt"
    check(run.invoke(main, ["train", *TINY, "-o", str(ckpt), "-q"]))
    check(run.invoke(main, ["sample", str(ckpt), "-o", str(tmp_path / "s.npz"),
                            "--nx", "11", "--ny", "11", "--format", "npy"]))
    assert np.load(tmp_path / "s.potential.npy").shape == (11, 11)
    assert (tmp_path / "s.json").exists()


def test_plot_rejects_a_missing_field(run, tmp_path):
    ckpt = tmp_path / "m.pt"
    check(run.invoke(main, ["train", *TINY, "-o", str(ckpt), "-q"]))
    npz = tmp_path / "s.npz"
    check(run.invoke(main, ["sample", str(ckpt), "-o", str(npz), "--nx", "11", "--ny", "11"]))
    result = run.invoke(main, ["plot", str(npz), "-f", "nope", "-o", str(tmp_path / "x.png")])
    assert result.exit_code != 0


def test_weighting_solve_runs(run, tmp_path):
    ckpt = tmp_path / "w.pt"
    check(run.invoke(main, [
        "train", *TINY, "-o", str(ckpt),
        "--set", "problem.kind=weighting",
        "--set", "domain.bounds.x=[-17.5, 17.5]",
        *sum([["--set", f"electrodes.{i}.lattice.count=7"] for i in range(3)], []),
    ]))
    assert ckpt.exists()


def test_run_subcommand(run, tmp_path):
    prefix = str(tmp_path / "out")
    check(run.invoke(main, ["run", *TINY, "--prefix", prefix,
                            "--set", "grid.nx=21", "--set", "grid.ny=41"]))
    for suffix in (".pt", ".npz", ".png", ".losses.png"):
        assert (tmp_path / f"out{suffix}").exists()


def test_response_table_reindexes_and_plots(run, tmp_path):
    """`firep response-table` turns a response into rho_p(u, t) and draws it."""
    import json

    src = tmp_path / "response.npz"
    pitch, half, sub, nt = 5.0, 3, 20, 4
    imp = np.arange(sub // 2 + 1) * (pitch / sub)
    off = np.arange(-half, half + 1)
    t = np.arange(nt) * 0.1
    cur = np.zeros((2, len(off), len(imp), nt))
    for iw, k in enumerate(off):
        for ii, x0 in enumerate(imp):
            cur[:, iw, ii, :] = np.exp(-((x0 - k * pitch) ** 2) / 50.0)
    np.savez_compressed(
        src, current=cur, impact=imp, offsets=off, time=t,
        metadata=json.dumps({"planes": ["a", "b"], "pitch_mm": pitch}))

    out = tmp_path / "table.npz"
    png = tmp_path / "table.png"
    res = run.invoke(main, ["response-table", str(src), "-o", str(out),
                            "--plot", str(png), "--tmin", "0.0"])
    text = check(res)
    assert "(plane, impact, tick)" in text
    blob = np.load(out, allow_pickle=False)
    assert blob["table"].shape == (2, 2 * (2 * half + 1) * 5, nt)
    assert blob["impact"][0] == pytest.approx(-(half + 0.5) * pitch + 0.25)
    assert png.exists()


def test_response_table_rejects_an_ungridded_request(run, tmp_path):
    import json

    src = tmp_path / "response.npz"
    imp = np.array([0.0, 2.5])
    np.savez_compressed(
        src, current=np.zeros((1, 1, 2, 2)), impact=imp,
        offsets=np.array([0]), time=np.array([0.0, 0.1]),
        metadata=json.dumps({"planes": ["a"], "pitch_mm": 5.0}))
    res = run.invoke(main, ["response-table", str(src), "-o",
                            str(tmp_path / "t.npz")])
    assert res.exit_code != 0
    assert "not in the tabulation" in res.output
