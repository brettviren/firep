#!/usr/bin/env python3
"""Keep every k-th event of a WCT depo or frame file, renumbered.

    subset.py depos  IN.npz OUT.npz --stride K [--first 0]
    subset.py frames IN.npz OUT.npz --stride K [--first 0] [--event0 100]

Depo files hold depo_{data,info}_<i> with i = 0, 1, ...; frame files hold
{frame,channels,tickinfo}_<tag>_<event> with event = event0 + i (WCT's
first_frame_number, 100 for PDHD).  Kept event j of the output is input
index first + j * stride, and gets index j (depos) or event0 + j (frames),
so a subset of depos simulated by WCT lines up with the same subset of
frames cut from a full-sample simulation.
"""
import argparse
import re

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("kind", choices=["depos", "frames"])
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--stride", type=int, required=True)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--event0", type=int, default=100)
    a = ap.parse_args()

    f = np.load(a.input)
    out = {}
    if a.kind == "depos":
        pat = re.compile(r"^(depo_(?:data|info))_(\d+)$")
        base = 0
    else:
        pat = re.compile(r"^((?:frame|channels|tickinfo)_.+)_(\d+)$")
        base = a.event0
    for key in f.files:
        m = pat.match(key)
        if not m:
            raise ValueError(f"unexpected array {key} in {a.input}")
        i = int(m.group(2)) - base
        if i < a.first or (i - a.first) % a.stride:
            continue
        j = (i - a.first) // a.stride
        out[f"{m.group(1)}_{base + j}"] = f[key]
    if not out:
        raise ValueError("nothing selected")
    np.savez_compressed(a.output, **out)


if __name__ == "__main__":
    main()
