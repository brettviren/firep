"""Timing records and the table the technical note builds from them."""

import json
import os
import subprocess
import sys

import pytest

from firep import perf

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "scripts")
sys.path.insert(0, SCRIPTS)
import perf_table  # noqa: E402


@pytest.fixture
def log(tmp_path, monkeypatch):
    p = tmp_path / "perf.jsonl"
    monkeypatch.setenv(perf.ENV, str(p))
    return p


def test_a_step_records_wall_cpu_and_memory(log):
    with perf.step("demo", "train", steps=7) as d:
        d["params"] = 3
        sum(range(200000))
    (rec,) = perf.load(str(log))
    assert rec["name"] == "demo" and rec["kind"] == "train"
    assert rec["wall_s"] >= 0.0 and rec["cpu_s"] >= 0.0
    assert rec["rss_mb"] > 0.0
    assert rec["detail"] == {"steps": 7, "params": 3}


def test_nothing_is_written_without_the_environment_variable(tmp_path, monkeypatch):
    monkeypatch.delenv(perf.ENV, raising=False)
    with perf.step("demo", "train"):
        pass
    assert perf.path() is None


def test_a_failing_step_is_still_recorded(log):
    with pytest.raises(ZeroDivisionError):
        with perf.step("boom", "train"):
            1 / 0
    assert len(perf.load(str(log))) == 1


def test_load_skips_malformed_lines(log):
    log.write_text('{"name": "ok", "kind": "x"}\nnot json\n\n')
    assert [r["name"] for r in perf.load(str(log))] == ["ok"]


def test_subprocess_wrapper_records_a_child(log):
    rc = subprocess.call([sys.executable, "-m", "firep.perf", "--name", "child",
                          "--kind", "doc", "--", sys.executable, "-c",
                          "sum(range(100000))"])
    assert rc == 0
    (rec,) = perf.load(str(log))
    assert rec["name"] == "child" and rec["detail"]["returncode"] == 0


def test_host_record_describes_the_machine():
    h = perf.host()
    assert h["kind"] == "host"
    assert h["detail"]["cores"] and h["detail"]["python"]


# -- the table --------------------------------------------------------------


def _records():
    return [
        perf.host(),
        {"name": "train lartpc-2d-3plane-drift", "kind": "train", "wall_s": 120.0,
         "cpu_s": 600.0, "rss_mb": 1000.0, "threads": 8, "started": 1.0,
         "detail": {"steps": 20000, "conductors": 3, "problem": "drift"}},
        {"name": "train lartpc-2d-21wire-weighting (u)", "kind": "train",
         "wall_s": 100.0, "cpu_s": 500.0, "rss_mb": 2000.0, "threads": 8,
         "started": 2.0,
         "detail": {"steps": 6000, "conductors": 63, "problem": "weighting"}},
        {"name": "training, 4 concurrent solves", "kind": "stage",
         "wall_s": 130.0, "cpu_s": 0.0, "rss_mb": 0.0, "threads": 0,
         "started": 0.0, "detail": {}},
    ]


def test_table_shortens_labels_and_totals_correctly():
    table, macros = perf_table.build(_records())
    assert "train drift" in table
    assert "train weighting (u)" in table
    assert "lartpc" not in table          # the config name buys width, not meaning
    assert r"\begin{tabular}" in table and r"\end{tabular}" in table
    assert macros["valPerfCores"]
    # 120 + 100 sequential against a 130 s concurrent stage
    assert macros["valPerfSpeedup"] == "1.7"


def test_table_marks_a_warm_start_as_having_no_gradient_steps():
    recs = _records()
    recs[1]["detail"]["steps"] = 0
    table, _ = perf_table.build(recs)
    assert "warm start drift" in table and "no gradient steps" in table


def test_stub_defines_every_macro_the_note_may_use(tmp_path):
    out = tmp_path / "perf.tex"
    perf_table.stub(str(out))
    text = out.read_text()
    assert r"\newcommand{\perfTable}" in text
    for name in perf_table.STUB_MACROS:
        assert f"\\newcommand{{\\{name}}}" in text


def test_main_falls_back_to_a_stub_when_the_log_is_missing(tmp_path):
    out = tmp_path / "perf.tex"
    assert perf_table.main(str(tmp_path / "nope.jsonl"), str(out)) == 0
    assert "perfTable" in out.read_text()


def test_main_writes_a_real_table_and_never_leaves_a_macro_undefined(tmp_path):
    src = tmp_path / "perf.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in _records()))
    out = tmp_path / "perf.tex"
    assert perf_table.main(str(src), str(out)) == 0
    text = out.read_text()
    assert "tabular" in text
    for name in perf_table.STUB_MACROS:
        assert f"\\newcommand{{\\{name}}}" in text
