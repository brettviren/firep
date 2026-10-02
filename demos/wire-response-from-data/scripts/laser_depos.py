#!/usr/bin/env python3
"""Idealized laser shots -> WCT depo file (.npz).

    laser_depos.py OUTPUT.npz LASER.csv [LASER.csv ...] --box x0,y0,z0,x1,y1,z1 [options]

Each shot is a straight line from a laser port along a firing direction.  The
line is clipped to an axis-aligned box (the active volume of the target anode)
and ionization is placed every ``--step`` along it, ``--dqdx`` electrons per
unit length, with zero initial extent (all diffusion comes from the drift in
WCT).  Light crosses the detector in tens of ns, far below a tick, but its
travel time is kept for completeness.

The output follows the WCT depo-file convention read by ``DepoFileSource``:
per depo set ``i``, ``depo_data_i`` is float32 (N, 7) holding (t, q, x, y, z,
L, T) and ``depo_info_i`` is int32 (N, 4) holding (id, pdg, gen, child).
Charge is negative (electrons).  ``DepoFileSource`` rejects any other
arrays in the file, so ``track_info_i`` (each clipped shot) goes to a
separate file given by ``--tracks``.
Units are WCT system units: mm, ns.

Shots come from the PDHD laser CSV files (``laser.py``), optionally moved
(``--shift``, ``--flip``) onto the anode under study.
"""
import argparse
import json
import numpy as np

import laser

MM = 1.0
NS = 1.0
CLIGHT = 299.792458 * MM / NS   # mm/ns

track_info_dtype = np.dtype([("shot", "i4"), ("pmin", "3f4"), ("pmax", "3f4"),
                             ("tmin", "f4"), ("tmax", "f4"),
                             ("step", "f4"), ("dqdx", "f4")])


def clip_to_box(origin, direction, lo, hi):
    """Parameter interval [s0, s1] (s >= 0) of origin + s*direction inside the
    box [lo, hi], or None if the ray misses it (slab method)."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    o = np.asarray(origin, float)
    s0, s1 = 0.0, np.inf
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if not lo[k] <= o[k] <= hi[k]:
                return None
            continue
        a, b = (lo[k] - o[k]) / d[k], (hi[k] - o[k]) / d[k]
        s0, s1 = max(s0, min(a, b)), min(s1, max(a, b))
    if s1 <= s0:
        return None
    return s0, s1, d


def shot_depos(origin, direction, time, lo, hi, step, dqdx, speed):
    """Depo columns (t, q, x, y, z, L, T) for one shot, or None."""
    clip = clip_to_box(origin, direction, lo, hi)
    if clip is None:
        return None
    s0, s1, d = clip
    n = max(1, int(round((s1 - s0) / step)))
    # depo at the centre of each of n equal segments carrying its charge
    ds = (s1 - s0) / n
    s = s0 + ds * (np.arange(n) + 0.5)
    pts = np.asarray(origin, float)[None, :] + s[:, None] * d[None, :]
    t = time + s / speed
    q = np.full(n, -dqdx * ds)
    z = np.zeros(n)
    return np.column_stack([t, q, pts, z, z]), (s0, s1, d, ds)


def make_depos(shots, lo, hi, step, dqdx, speed):
    """Group shots by event into WCT depo sets."""
    out = {}
    events = sorted({sh.get("event", 0) for sh in shots})
    for iset, ev in enumerate(events):
        datas, tinfo = [], []
        for ishot, sh in enumerate(shots):
            if sh.get("event", 0) != ev:
                continue
            got = shot_depos(sh["origin"], sh["direction"], sh.get("time", 0.0),
                             lo, hi, step, dqdx, speed)
            if got is None:
                continue
            data, (s0, s1, d, ds) = got
            o = np.asarray(sh["origin"], float)
            tinfo.append((ishot, o + s0 * d, o + s1 * d,
                          data[0, 0], data[-1, 0], ds, dqdx))
            datas.append(data)
        if not datas:
            continue
        data = np.vstack(datas)
        data = data[np.argsort(data[:, 0], kind="stable")]
        info = np.zeros((len(data), 4), dtype="i4")
        info[:, 0] = np.arange(len(data))
        out[f"depo_data_{iset}"] = data.astype("f4")
        out[f"depo_info_{iset}"] = info
        out[f"track_info_{iset}"] = np.array(tinfo, dtype=track_info_dtype)
    return out


def floats(text, n):
    v = [float(x) for x in text.split(",")]
    if len(v) != n:
        raise argparse.ArgumentTypeError(f"want {n} comma-separated numbers")
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("output")
    ap.add_argument("laserfiles", nargs="+", help="laser CSV files")
    ap.add_argument("--box", required=True, type=lambda t: floats(t, 6),
                    help="active volume x0,y0,z0,x1,y1,z1 [mm]")
    ap.add_argument("--shift", default="0,0,0", type=lambda t: floats(t, 3),
                    help="translate every laser port by dx,dy,dz [mm]")
    ap.add_argument("--flip", default="1,1,1", type=lambda t: floats(t, 3),
                    help="multiply direction components, e.g. 1,1,-1 mirrors in z")
    ap.add_argument("--stride", type=int, default=1, help="keep every n-th shot")
    ap.add_argument("--per-event", type=int, default=1,
                    help="shots per depo set")
    ap.add_argument("--step", type=float, default=1.0, help="depo spacing [mm]")
    ap.add_argument("--dqdx", type=float, default=5000.0,
                    help="ionization electrons per mm (MIP-like default)")
    ap.add_argument("--index", type=float, default=1.0,
                    help="refractive index for the light travel time")
    ap.add_argument("--tracks", help="write the per-shot track table (.npz) here")
    ap.add_argument("--summary", help="write a JSON summary here")
    a = ap.parse_args()

    lo, hi = np.array(a.box[:3]), np.array(a.box[3:])
    shots = laser.shots(a.laserfiles, a.shift, a.flip, a.stride, a.per_event)
    out = make_depos(shots, lo, hi, a.step * MM, a.dqdx / MM, CLIGHT / a.index)
    tracks = {k: out.pop(k) for k in list(out) if k.startswith("track_info_")}
    np.savez_compressed(a.output, **out)
    if a.tracks:
        np.savez_compressed(a.tracks, **tracks)
    if a.summary:
        nsets = sum(1 for k in out if k.startswith("depo_data_"))
        ntracks = sum(len(v) for v in tracks.values())
        ndepos = sum(len(v) for k, v in out.items() if k.startswith("depo_data_"))
        with open(a.summary, "w") as fp:
            json.dump(dict(nshots=len(shots), nsets=nsets, ntracks=ntracks,
                           ndepos=ndepos), fp, indent=2)


if __name__ == "__main__":
    main()
