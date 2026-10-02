#!/usr/bin/env python3
"""Write a firep field response as a WCT FieldResponse file (.json.bz2).

    firep_to_wct.py CONFIG.yaml RESPONSE.npz OUTPUT.json.bz2 [--y-start 100]
                    [--nticks 1000] [--speed MM_PER_US] [--bin-average]

RESPONSE.npz is ``firep response`` output on the Garfield impact grid
(``response.impacts: 6``, 0 to 1/2 pitch) with ``--y-start`` at the WCT
response plane (100 mm for PDHD), and one weighting solve per plane with 21
wires.

Layout, matching the Garfield-derived files: per plane, for each wire region
n = -10..10, six paths at pitchpos (n - 1/2) p + k p/10, k = 0..5, holding the
current on the central wire.  By mirror symmetry the path at pitchpos
n p - x is firep's current on wire offset n from an electron at impact x.

With --bin-average, RESPONSE.npz holds a finer regular impact grid (e.g.
``--impacts 51``, p/100 steps) and each of the six output paths is the
trapezoid average of the current over its bin, pitchpos +- p/20, instead of
the value at one point.  Positions outside the tabulated half pitch map back
by translation and mirror symmetry: the current on wire o from an electron at
relative position r depends only on |r - o p|.

Units: WCT currents are e/ns of electron charge, firep's are e/us of induced
charge, so wct = firep / -1000.  Time starts at launch (tstart 0), sampled at
firep's tick.  Plane locations come from the firep drift CONFIG.  ``speed``
is informational in WCT simulation; by default it is the Walkowiak speed at
the config's drift field.
"""
import argparse
import bz2
import json

import numpy as np

FIREP_TO_WCT = -1.0e-3


def current_at(current, offsets, impact, r):
    """Current (plane, t) on the central wire from an electron at relative
    position r [mm], from a table over impacts 0..p/2 and wire offsets."""
    pitch = 2.0 * impact[-1]
    s = abs(r)
    o = int(round(s / pitch))
    x = s - o * pitch                     # in [-p/2, p/2]
    if x >= 0:
        wire, xi = -o, x                  # electron at x from wire 0, wire at -o
    else:
        wire, xi = o, -x                  # mirror image
    i = int(round(xi / (impact[1] - impact[0])))
    if abs(impact[i] - xi) > 1e-6 or wire not in offsets:
        return None
    iw = int(np.flatnonzero(offsets == wire)[0])
    if i == 0 and -wire in offsets and wire != 0:
        # the x = 0 path is launched nudged off the symmetry line (firep
        # response.impact_nudge); average the mirror wires to restore symmetry
        return 0.5 * (current[:, iw, 0] + current[:, int(np.flatnonzero(offsets == -wire)[0]), 0])
    return current[:, iw, i]


def bin_average(current, offsets, impact, nout=6):
    """Resample a fine table to nout impacts, each the bin average."""
    pitch = 2.0 * impact[-1]
    h = impact[1] - impact[0]
    half = pitch / (2 * (nout - 1)) / 2    # half bin width, p/20 for nout = 6
    m = int(round(half / h))
    if abs(m * h - half) > 1e-6 or m < 1:
        raise ValueError(f"fine step {h} mm does not divide the half bin {half} mm")
    w = np.ones(2 * m + 1)
    w[0] = w[-1] = 0.5
    w /= w.sum()
    coarse = np.linspace(0.0, pitch / 2, nout)
    out = np.zeros(current.shape[:1] + (len(offsets), nout) + current.shape[-1:])
    for iw, o in enumerate(offsets):
        for k, xk in enumerate(coarse):
            acc, wsum = 0.0, 0.0
            for j, wj in zip(range(-m, m + 1), w):
                # electron at xk + j h from wire 0, seen by wire o
                c = current_at(current, offsets, impact, xk + j * h - o * pitch)
                if c is not None:
                    acc = acc + wj * c
                    wsum += wj
            out[:, iw, k] = acc / wsum
    return coarse, out


def build(arrays, meta, nticks=None, speed=None, bin_avg=False):
    planes = [str(p) for p in meta["planes"]]
    impact = np.asarray(arrays["impact"], float)          # mm, 0 .. p/2
    offsets = np.asarray(arrays["offsets"], int)
    current = np.asarray(arrays["current"], float)        # (plane, wire, impact, t)
    time = np.asarray(arrays["time"], float)              # us, bin centres
    tick = float(time[1] - time[0])
    pitch = 2.0 * impact[-1]
    if not np.allclose(impact, np.linspace(0.0, pitch / 2, len(impact))):
        raise ValueError("impacts must be a regular grid from 0 to half a pitch")
    if bin_avg:
        impact, current = bin_average(current, offsets, impact)
    elif len(impact) != 6:
        raise ValueError(f"{len(impact)} impacts: use --bin-average or 6 impacts")
    nimp = len(impact)
    nt = current.shape[-1]
    nticks = nticks or nt
    if nticks < nt:
        raise ValueError(f"nticks {nticks} would truncate the {nt}-sample response")

    locations = meta["locations"]
    if set(planes) - set(locations):
        raise ValueError(f"config lacks planes {set(planes) - set(locations)}")
    out_planes = []
    for ip, name in enumerate(planes):
        paths = []
        for n in range(offsets.min(), offsets.max() + 1):
            iw = int(np.flatnonzero(offsets == n)[0])
            for k in range(nimp):
                i = nimp - 1 - k                          # impact x = p/2 - k p/10
                cur = np.zeros(nticks)
                c = current[ip, iw, i]
                if i == 0 and n != 0 and -n in offsets:   # nudged x = 0: symmetrise
                    c = 0.5 * (c + current[ip, int(np.flatnonzero(offsets == -n)[0]), 0])
                cur[:nt] = c * FIREP_TO_WCT
                paths.append({"PathResponse": {
                    "current": {"array": {"shape": [nticks],
                                          "elements": cur.tolist()}},
                    "pitchpos": round(n * pitch - impact[i], 6),
                    "wirepos": 0.0}})
        out_planes.append({"PlaneResponse": {
            "paths": paths,
            "planeid": "uvw".index(name),
            "location": float(locations[name]),
            "pitch": pitch}})
    out_planes.sort(key=lambda p: p["PlaneResponse"]["planeid"])
    return {"FieldResponse": {
        "planes": out_planes,
        "axis": [1.0, 0.0, 0.0],
        "origin": float(meta["y_start"]),
        "tstart": 0.0,
        "period": tick * 1000.0,                          # ns
        "speed": float(speed) / 1000.0,                   # mm/ns
    }}


def main():
    import yaml
    from firep.response import walkowiak_speed

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", help="firep drift config the response was made with")
    ap.add_argument("response")
    ap.add_argument("output")
    ap.add_argument("--y-start", type=float, default=100.0,
                    help="launch height used for the response [mm]")
    ap.add_argument("--nticks", type=int, default=1000)
    ap.add_argument("--speed", type=float, default=None, help="[mm/us]")
    ap.add_argument("--bin-average", action="store_true",
                    help="average a fine impact grid over each path's bin")
    a = ap.parse_args()

    f = np.load(a.response, allow_pickle=False)
    meta = json.loads(str(f["metadata"]))
    cfg = yaml.safe_load(open(a.config))
    meta["locations"] = {e["name"]: e["plane"] for e in cfg["electrodes"]}
    meta["y_start"] = a.y_start
    arrays = {k: f[k] for k in ("impact", "offsets", "current", "time")}
    speed = a.speed
    if speed is None:
        e_kv_cm = cfg["bias"]["drift_field"] / 1000.0
        speed = float(walkowiak_speed(np.array([e_kv_cm]), meta["temperature_K"])[0])
    fr = build(arrays, meta, a.nticks, speed, a.bin_average)
    with bz2.open(a.output, "wt") as fp:
        json.dump(fr, fp)


if __name__ == "__main__":
    main()
