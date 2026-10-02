#!/usr/bin/env python3
"""Windows around laser tracks, and cropping frames to them (learner 2).

    l2_crop.py windows REF.npz OUT.npz [--anode 0] [--thr 0.5] [--pre 60] [--post 100]
    l2_crop.py crop FRAMES.npz WINDOWS.npz OUT.npz [--anode 0] [--scale S] [--shift T]

``windows``: for every event and every channel of the anode where the
baseline-subtracted REF frame exceeds ``thr``, a tick window from ``pre``
before its first active tick to ``post`` after its last.  The windows are
wide enough for the time shifts learner 2 profiles.

``crop``: each event's frame (baseline subtracted if integer ADC, scaled by
``scale``, optionally moved later by ``shift`` ticks via FFT) cut to the
windows and flattened per plane into e<ev>_p<plane>, float32.  Data and basis
components cropped with the same windows line up sample by sample.
"""
import argparse

import numpy as np

SPLIT = (800, 1600)


def plane_of(rel):
    return np.where(rel < SPLIT[0], 0, np.where(rel < SPLIT[1], 1, 2))


def events(f, anode):
    return sorted(int(k.rsplit("_", 1)[1]) for k in f.files
                  if k.startswith(f"frame_orig{anode}_"))


def load(f, anode, ev):
    x = f[f"frame_orig{anode}_{ev}"]
    ch = f[f"channels_orig{anode}_{ev}"]
    order = np.argsort(ch)
    x, ch = x[order], ch[order]
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(float)
        x -= np.median(x, axis=1, keepdims=True)
    return ch - ch.min(), np.asarray(x, float)


def shift_ticks(x, s):
    if not s:
        return x
    n = x.shape[1]
    k = np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(x, axis=1) * np.exp(-2j * np.pi * k * s), n, axis=1)


def windows(a):
    f = np.load(a.ref)
    out = {}
    for ev in events(f, a.anode):
        rel, x = load(f, a.anode, ev)
        act = np.abs(x) > a.thr
        rows = np.flatnonzero(act.any(1))
        t0 = np.array([np.argmax(act[r]) for r in rows]) - a.pre
        t1 = np.array([x.shape[1] - np.argmax(act[r][::-1]) for r in rows]) + a.post
        out[f"e{ev}_ch"] = rel[rows]
        out[f"e{ev}_t0"] = np.clip(t0, 0, x.shape[1])
        out[f"e{ev}_t1"] = np.clip(t1, 0, x.shape[1])
    np.savez_compressed(a.out, **out)


def crop_event(rel, x, ch, t0, t1):
    """Flattened per-plane vectors for one event."""
    row = {c: i for i, c in enumerate(rel)}
    pl = plane_of(ch)
    parts = [[], [], []]
    for c, a0, a1, p in zip(ch, t0, t1, pl):
        parts[p].append(x[row[c], a0:a1])
    return [np.concatenate(v).astype(np.float32) if v else np.zeros(0, np.float32)
            for v in parts]


def crop(a):
    f = np.load(a.frames)
    w = np.load(a.windows)
    out = {}
    for ev in events(f, a.anode):
        rel, x = load(f, a.anode, ev)
        x = shift_ticks(x * a.scale, a.shift)
        for ip, v in enumerate(crop_event(rel, x, w[f"e{ev}_ch"], w[f"e{ev}_t0"],
                                          w[f"e{ev}_t1"])):
            out[f"e{ev}_p{ip}"] = v
    np.savez(a.out, **out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("windows")
    w.add_argument("ref")
    w.add_argument("out")
    w.add_argument("--anode", type=int, default=0)
    w.add_argument("--thr", type=float, default=0.5)
    w.add_argument("--pre", type=int, default=60)
    w.add_argument("--post", type=int, default=100)
    c = sub.add_parser("crop")
    c.add_argument("frames")
    c.add_argument("windows")
    c.add_argument("out")
    c.add_argument("--anode", type=int, default=0)
    c.add_argument("--scale", type=float, default=1.0)
    c.add_argument("--shift", type=float, default=0.0)
    a = ap.parse_args()
    windows(a) if a.cmd == "windows" else crop(a)


if __name__ == "__main__":
    main()
