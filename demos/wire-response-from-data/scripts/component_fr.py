#!/usr/bin/env python3
"""Write one learner-2 basis component as a WCT field-response file.

    component_fr.py FIREP_FINE.npz OUT.json.bz2 --bin K --offset D [--y-start 100]
    component_fr.py FIREP_FINE.npz OUT.json.bz2 --weights W.json      # a composition

A component is a field response whose paths are all zero except those of
impact bin K (pitchpos n p - K p/10 for every wire region n and plane), which
hold firep's current from electrons started at offset D (pitch fraction, a
multiple of the fine impact step) from the path position.  WCT's simulation is
linear in the field response, so the ADC of any composition sum_{K,D} w_{K,D}
C_{K,D} is the same sum of the components' simulations.

With --weights (a JSON list of [K, D, w]) the file holds that composition
instead: the learned response.

Layout and units follow firep_to_wct.py (21 wires x 6 paths, WCT e/ns,
tstart 0 at launch, 100 ns period, 1000 samples).
"""
import argparse
import bz2
import json

import numpy as np

from firep_to_wct import current_at, FIREP_TO_WCT


def component_currents(f, k, d, nticks=1000):
    """(plane, n, nt) currents on the central wire for bin k, offset d [pitch]."""
    impact = f["impact"]
    offsets = f["offsets"]
    cur = f["current"]
    pitch = 2.0 * impact[-1]
    xk = 0.1 * k * pitch
    out = np.zeros((cur.shape[0], 21, nticks))
    nt = min(cur.shape[-1], nticks)
    for jn, n in enumerate(range(-10, 11)):
        c = current_at(cur, offsets, impact, n * pitch - xk + d * pitch)
        if c is not None:
            out[:, jn, :nt] = c[:, :nt] * FIREP_TO_WCT
    return out


def write(path, f, planes_cur, meta):
    """planes_cur[k] = (plane, n, nt) currents for bin k (or None)."""
    impact = f["impact"]
    pitch = 2.0 * impact[-1]
    nticks = next(c for c in planes_cur if c is not None).shape[-1]
    names = [str(p) for p in json.loads(str(f["metadata"]))["planes"]]
    out_planes = []
    for ip, name in enumerate(names):
        paths = []
        for jn, n in enumerate(range(-10, 11)):
            for kk in range(6):          # pitchpos (n - 1/2) p + kk p/10
                k = 5 - kk               # bin index: x = k p/10
                c = planes_cur[k]
                cur = np.zeros(nticks) if c is None else c[ip, jn]
                paths.append({"PathResponse": {
                    "current": {"array": {"shape": [nticks], "elements": cur.tolist()}},
                    "pitchpos": round(n * pitch - 0.1 * k * pitch, 6),
                    "wirepos": 0.0}})
        out_planes.append({"PlaneResponse": {
            "paths": paths, "planeid": "uvw".index(name),
            "location": float(meta["locations"][name]), "pitch": pitch}})
    out_planes.sort(key=lambda p: p["PlaneResponse"]["planeid"])
    fr = {"FieldResponse": {"planes": out_planes, "axis": [1.0, 0.0, 0.0],
                            "origin": meta["y_start"], "tstart": 0.0,
                            "period": float(f["time"][1] - f["time"][0]) * 1000.0,
                            "speed": meta["speed"] / 1000.0}}
    with bz2.open(path, "wt") as fp:
        json.dump(fr, fp)


def main():
    import yaml
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("fine")
    ap.add_argument("output")
    ap.add_argument("--config", default="firep/bad-apa-drift.yaml")
    ap.add_argument("--bin", type=int)
    ap.add_argument("--offset", type=float, help="pitch fraction")
    ap.add_argument("--weights", help="JSON list of [bin, offset, weight]")
    ap.add_argument("--y-start", type=float, default=100.0)
    ap.add_argument("--speed", type=float, default=1.628, help="informational [mm/us]")
    a = ap.parse_args()
    f = np.load(a.fine)
    cfg = yaml.safe_load(open(a.config))
    meta = {"locations": {e["name"]: e["plane"] for e in cfg["electrodes"]},
            "y_start": a.y_start, "speed": a.speed}
    planes_cur = [None] * 6
    if a.weights:
        for k, d, w in json.load(open(a.weights)):
            c = w * component_currents(f, int(k), float(d))
            planes_cur[int(k)] = c if planes_cur[int(k)] is None else planes_cur[int(k)] + c
    else:
        planes_cur[a.bin] = component_currents(f, a.bin, a.offset)
    write(a.output, f, planes_cur, meta)


if __name__ == "__main__":
    main()
