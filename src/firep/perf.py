"""Wall-clock, CPU and memory accounting for the steps of the pipeline.

Every long-running step writes one JSON object to the file named by the
``FIREP_PERF`` environment variable (JSON Lines, append mode, one short line per
record, so concurrent writers interleave safely).  Nothing is recorded when the
variable is unset, so the instrumentation costs nothing in ordinary use.

``python -m firep.perf --name X --kind Y -- cmd ...`` records a subprocess the
same way, which is how the figure build and the LaTeX runs are measured.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys
import time
from contextlib import contextmanager

ENV = "FIREP_PERF"


def _rss_mb(who=resource.RUSAGE_SELF) -> float:
    """Peak resident set size in MiB (Linux reports ru_maxrss in KiB)."""
    return resource.getrusage(who).ru_maxrss / 1024.0


def _cpu_s(who=resource.RUSAGE_SELF) -> float:
    r = resource.getrusage(who)
    return r.ru_utime + r.ru_stime


def path() -> str | None:
    return os.environ.get(ENV) or None


def write(record: dict) -> None:
    p = path()
    if not p:
        return
    os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
    with open(p, "a") as fp:
        fp.write(json.dumps(record, sort_keys=True) + "\n")


def _threads() -> int:
    try:
        import torch

        return int(torch.get_num_threads())
    except Exception:
        return 0


@contextmanager
def step(name: str, kind: str, **detail):
    """Time a block and record it.  ``detail`` may be extended inside the block."""
    t0, c0, r0 = time.perf_counter(), _cpu_s(), _rss_mb()
    try:
        yield detail
    finally:
        wall = time.perf_counter() - t0
        cpu = _cpu_s() - c0
        write({
            "name": name,
            "kind": kind,
            "wall_s": round(wall, 3),
            "cpu_s": round(cpu, 3),
            "rss_mb": round(max(_rss_mb(), r0), 1),
            "threads": _threads(),
            "detail": {k: v for k, v in detail.items() if v is not None},
            "started": round(t0, 3),
        })


def host() -> dict:
    """One record describing the machine, so the numbers can be read later."""
    model = ""
    try:
        with open("/proc/cpuinfo") as fp:
            for line in fp:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        import torch

        tver = torch.__version__
    except Exception:
        tver = ""
    return {
        "name": "host", "kind": "host", "wall_s": 0.0, "cpu_s": 0.0,
        "rss_mb": 0.0, "threads": _threads(),
        "detail": {
            "cpu": model or platform.processor(),
            "cores": os.cpu_count(),
            "python": platform.python_version(),
            "torch": tver,
        },
    }


def load(p: str) -> list[dict]:
    """Read a JSONL perf file, skipping anything malformed."""
    out = []
    with open(p) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _main(argv: list[str]) -> int:
    name, kind = "subprocess", "other"
    args = list(argv)
    while args and args[0].startswith("--"):
        flag = args.pop(0)
        if flag == "--":
            break
        val = args.pop(0) if args else ""
        if flag == "--name":
            name = val
        elif flag == "--kind":
            kind = val
    if args and args[0] == "--":
        args.pop(0)
    if not args:
        print("usage: python -m firep.perf --name N --kind K -- cmd ...",
              file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    c0, r0 = _cpu_s(resource.RUSAGE_CHILDREN), _rss_mb(resource.RUSAGE_CHILDREN)
    rc = subprocess.call(args)
    write({
        "name": name, "kind": kind,
        "wall_s": round(time.perf_counter() - t0, 3),
        "cpu_s": round(_cpu_s(resource.RUSAGE_CHILDREN) - c0, 3),
        "rss_mb": round(max(_rss_mb(resource.RUSAGE_CHILDREN), r0), 1),
        "threads": 0, "detail": {"argv": args[0], "returncode": rc},
        "started": round(t0, 3),
    })
    return rc


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
