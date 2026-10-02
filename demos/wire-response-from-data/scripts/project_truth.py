#!/usr/bin/env python3
"""Project a WCT field response onto physical compositions of firep paths.

    project_truth.py TARGET.json.bz2 FIREP_FINE.npz OUT.json.bz2 OUTDIR --tag TAG
                     [--spread 0.1] [--shift 2.0,5.0,0.1] [--stretch 0.96,1.04,0.01]
                     [--shaping 2.2]

A "composition" at impact bin k is a distribution of electron starting
positions around the bin, w_k(d) >= 0 for d within +-spread pitch.  If a
tabulated response is physical in firep's field, then for every plane and
every wire the path at bin k equals sum_d w_k(d) R(u + d), with the same
weights for all planes and wires, R the firep response and u the path's
pitchpos.  Each such component obeys the Ramo sum rules, so a composition
cannot violate them.

The data only see the response through the front end, so the fit and its
residuals use target and basis both convolved with the WCT cold-electronics
impulse response (``--shaping`` Tp in us; 0 for raw currents, where sharp
collection spikes and tiny timing differences dominate).  The projection
written out is unshaped, sum_d w_k(d) R.

For each bin the weights are fitted by non-negative least squares jointly
over 3 planes x 21 wires x time; one global time map, firep time = s (t - dt),
is scanned to align firep's drift speed and time origin with the target.

Writes OUT.json.bz2 (the projection: TARGET's layout, each path replaced by
its composition fit) and OUTDIR/proj-TAG.{pdf,tex}: per-bin weights, the
residual per plane and bin, and central-wire paths before and after.
"""
import argparse
import bz2
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from firep_to_wct import current_at, FIREP_TO_WCT
from firep.electronics import uboone_shape

PLANES = "UVW"
LETTERS = "ABCDEF"


def load_target(path):
    d = json.load(bz2.open(path, "rt"))
    fr = d["FieldResponse"]
    planes = []
    for p in fr["planes"]:
        p = p["PlaneResponse"]
        pos = np.array([q["PathResponse"]["pitchpos"] for q in p["paths"]])
        cur = np.array([q["PathResponse"]["current"]["array"]["elements"] for q in p["paths"]])
        planes.append((p["planeid"], p["pitch"], pos, cur))
    return d, fr["period"] / 1000.0, planes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("target")
    ap.add_argument("fine")
    ap.add_argument("output")
    ap.add_argument("outdir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--spread", type=float, default=0.1, help="pitch fraction")
    ap.add_argument("--shift", default="2.0,5.0,0.1", help="dt lo,hi,step [us]")
    ap.add_argument("--stretch", default="0.96,1.04,0.01", help="s lo,hi,step")
    ap.add_argument("--shaping", type=float, default=2.2,
                    help="cold-electronics Tp [us] applied before fitting; 0 = raw")
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    doc, T, planes = load_target(a.target)
    pitch = planes[0][1]
    nt = planes[0][3].shape[1]
    t = np.arange(nt) * T

    f = np.load(a.fine)
    impact = f["impact"]
    offsets = f["offsets"]
    fcur = f["current"] * FIREP_TO_WCT               # (plane, wire, impact, tf)
    tf = f["time"]
    h = impact[1] - impact[0]
    m = int(round(a.spread * pitch / h))
    deltas = np.arange(-m, m + 1) * h                 # starting-position offsets

    # For bin k, target paths: every plane, wire region n (pitchpos n p - x_k)
    xk = np.arange(6) * 0.1 * pitch
    nlist = np.arange(-10, 11)

    def target_index(ip, n, k):
        pos = planes[ip][2]
        return int(np.argmin(np.abs(pos - (n * pitch - xk[k]))))

    # basis currents at firep time: (k, delta, plane, n, tf)
    raw = np.zeros((6, len(deltas), 3, len(nlist), len(tf)))
    ok = np.ones((6, len(deltas)), bool)
    for k in range(6):
        for idl, dl in enumerate(deltas):
            for jn, n in enumerate(nlist):
                c = current_at(fcur, offsets, impact, n * pitch - xk[k] + dl)
                if c is None:
                    ok[k, idl] = False
                    break
                raw[k, idl, :, jn] = c
    tgt = np.zeros((6, 3, len(nlist), nt))
    for k in range(6):
        for ip in range(3):
            for jn, n in enumerate(nlist):
                tgt[k, ip, jn] = planes[ip][3][target_index(ip, n, k)]

    if a.shaping > 0:
        kt = np.arange(int(10 * a.shaping / T) + 1) * T
        kern = uboone_shape(kt / a.shaping)
        kern /= kern.sum()
        L = 1 << int(np.ceil(np.log2(nt + len(kern))))
        K = np.fft.rfft(kern, L)
        shape = lambda x: np.fft.irfft(np.fft.rfft(x, L, axis=-1) * K, L, axis=-1)[..., :nt]
    else:
        shape = lambda x: x
    tgt_s = shape(tgt)

    def fit_all(dt, s):
        tq = s * (t - dt)
        res, W, fits = 0.0, [], []
        for k in range(6):
            cols = np.flatnonzero(ok[k])
            B = np.stack([np.stack([np.interp(tq, tf, raw[k, c, ip, jn], left=0, right=0)
                                    for ip in range(3) for jn in range(len(nlist))])
                          for c in cols])                 # (ncol, 3*21, nt)
            A = shape(B).reshape(len(cols), -1).T
            y = tgt_s[k].reshape(-1)
            w, r = nnls(A, y, maxiter=2000)
            full = np.zeros(len(deltas))
            full[cols] = w
            W.append(full)
            fits.append(np.tensordot(w, B, axes=1).reshape(3, len(nlist), nt))  # unshaped
            res += r ** 2
        return res, np.array(W), np.array(fits)

    lo, hi, st = (float(x) for x in a.shift.split(","))
    slo, shi, sst = (float(x) for x in a.stretch.split(","))
    best = None
    for s in np.arange(slo, shi + 1e-9, sst):
        for dt in np.arange(lo, hi + 1e-9, st):
            r, _, _ = fit_all(dt, s)
            if best is None or r < best[0]:
                best = (r, dt, s)
    _, dt, s = best
    _, W, fits = fit_all(dt, s)

    # residual fractions per bin and plane, as the data see them (shaped)
    fits_s = shape(fits)
    rel = np.zeros((6, 3))
    for k in range(6):
        for ip in range(3):
            rel[k, ip] = (((tgt_s[k, ip] - fits_s[k, ip]) ** 2).sum()
                          / max((tgt_s[k, ip] ** 2).sum(), 1e-30))

    # write projection in the target's layout
    for ip, (pid, _, pos, cur) in enumerate(planes):
        paths = doc["FieldResponse"]["planes"][ip]["PlaneResponse"]["paths"]
        for k in range(6):
            for jn, n in enumerate(nlist):
                j = target_index(ip, n, k)
                paths[j]["PathResponse"]["current"]["array"]["elements"] = fits[k, ip, jn].tolist()
    with bz2.open(a.output, "wt") as fp:
        json.dump(doc, fp)

    # figure
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(3, 6)
    for k in range(6):
        ax = fig.add_subplot(gs[0, k])
        ax.bar((xk[k] + deltas) / pitch, W[k], width=h / pitch * 0.9)
        ax.axvline(xk[k] / pitch, color="k", lw=0.5, ls=":")
        ax.set_title(f"bin {0.1 * k:.1f}p: sum w = {W[k].sum():.3f}", fontsize=9)
        ax.set_xlabel("start |x| / pitch", fontsize=8)
        for row, ip in ((1, 2), (2, 1)):
            ax = fig.add_subplot(gs[row, k])
            c = int(np.flatnonzero(nlist == 0)[0])
            ax.plot(t, tgt_s[k, ip, c], label="target")
            ax.plot(t, fits_s[k, ip, c], "--", label="composition")
            ax.set_xlim(55, 95)
            ax.set_title(f"{PLANES[ip]} central (shaped), resid {rel[k, ip]:.1e}", fontsize=8)
    fig.axes[1].legend(fontsize=7, frameon=False)
    fig.suptitle(f"composition fit: firep time = {s:.2f} (t - {dt:.1f} us)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"proj-{a.tag}.pdf")

    lines = [f"% generated by {Path(__file__).name}; do not edit",
             f"\\newcommand{{\\{a.tag}Shift}}{{{dt:.1f}}}",
             f"\\newcommand{{\\{a.tag}Stretch}}{{{s:.2f}}}"]
    for k in range(6):
        lines.append(f"\\newcommand{{\\{a.tag}Sum{LETTERS[k]}}}{{{W[k].sum():.3f}}}")
        for ip in range(3):
            lines.append(f"\\newcommand{{\\{a.tag}Res{PLANES[ip]}{LETTERS[k]}}}"
                         f"{{{rel[k, ip]:.1e}}}")
    (out / f"proj-{a.tag}.tex").write_text("\n".join(lines) + "\n")
    print(f"time map: firep = {s:.2f} (t - {dt:.2f} us)")
    for k in range(6):
        print(f"bin {0.1*k:.1f}p  sum w {W[k].sum():.3f}  resid U {rel[k,0]:.1e} V {rel[k,1]:.1e} W {rel[k,2]:.1e}")


if __name__ == "__main__":
    main()
