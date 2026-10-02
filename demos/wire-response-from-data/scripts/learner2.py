#!/usr/bin/env python3
"""Learner 2: composition weights fitted to ADC by non-negative least squares.

    learner2.py DATA.npz WINDOWS.npz OUTDIR --tag TAG --columns COLS.json
                [--shift 6.5,8.5,0.25] [--noise 3.5] [--anode 0] [--weights-out W.json]

COLS.json lists the basis: [{"bin": K, "offset": D, "path": cropped.npz}, ...],
each the cropped analog WCT simulation (scaled to ADC) of one component field
response (component_fr.py).  The model is sum_c w_c S_c with w >= 0.  The data
(baseline-subtracted ADC) are moved by -dt ticks (profiled over --shift) to
firep's time origin, cropped with the same windows, and the fit solved from
per-plane Gram matrices G = S^T S and correlations b = S^T y:

    chi^2(w) = y.y - 2 b.w + w.G.w      (over all planes, / noise^2)

via NNLS on the Cholesky factor of G.  Reports per-plane residual relative to
the signal (y.y), the weights (per bin, by starting position), their sums (the
charge each bin carries) and writes the weights as [[bin, offset, w], ...].
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import l2_crop

PLANES = "UVW"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data")
    ap.add_argument("windows")
    ap.add_argument("outdir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--columns", required=True)
    ap.add_argument("--shift", default="6.5,8.5,0.25", help="data shift lo,hi,step [ticks]")
    ap.add_argument("--noise", type=float, default=3.5)
    ap.add_argument("--anode", type=int, default=0)
    ap.add_argument("--weights-out", required=True)
    ap.add_argument("--balance", action="store_true",
                    help="divide each plane's chi^2 by its own signal energy, so the "
                         "small W signal counts as much as U and V")
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    cols = json.load(open(a.columns))
    win = np.load(a.windows)
    evs = sorted(int(k[1:].split("_")[0]) for k in win.files if k.endswith("_ch"))
    C = [np.load(c["path"]) for c in cols]
    nc = len(cols)

    # Gram matrices per plane (shift independent)
    G = np.zeros((3, nc, nc))
    for ev in evs:
        for ip in range(3):
            S = np.stack([c[f"e{ev}_p{ip}"].astype(float) for c in C])
            G[ip] += S @ S.T

    # data: shift, crop, correlate
    fd = np.load(a.data)
    lo, hi, st = (float(x) for x in a.shift.split(","))
    shifts = np.arange(lo, hi + 1e-9, st)
    B = np.zeros((len(shifts), 3, nc))
    YY = np.zeros((len(shifts), 3))
    for ev in evs:
        rel, x = l2_crop.load(fd, a.anode, ev)
        S = [np.stack([c[f"e{ev}_p{ip}"].astype(float) for c in C]) for ip in range(3)]
        for js, s in enumerate(shifts):
            y = l2_crop.shift_ticks(x, -s)
            parts = l2_crop.crop_event(rel, y, win[f"e{ev}_ch"], win[f"e{ev}_t0"],
                                       win[f"e{ev}_t1"])
            for ip in range(3):
                v = parts[ip].astype(float)
                B[js, ip] += S[ip] @ v
                YY[js, ip] += v @ v

    # plane weights: 1, or 1 / (signal energy) with --balance
    pw = (1.0 / YY.mean(0)) if a.balance else np.ones(3)
    Gt = (G * pw[:, None, None]).sum(0)
    ridge = 1e-10 * np.trace(Gt) / nc
    L = np.linalg.cholesky(Gt + ridge * np.eye(nc))
    best = None
    chis = []
    for js in range(len(shifts)):
        bt = (B[js] * pw[:, None]).sum(0)
        z = np.linalg.solve(L, bt)                  # min |L^T w - z|^2 == chi^2 + const
        w, _ = nnls(L.T, z, maxiter=5000)
        chi = (YY[js] * pw).sum() - 2 * bt @ w + w @ Gt @ w
        chis.append(chi)
        if best is None or chi < best[0]:
            best = (chi, js, w)
    chi, js, w = best
    per = np.array([YY[js, ip] - 2 * B[js, ip] @ w + w @ G[ip] @ w for ip in range(3)])
    rel = per / YY[js]
    noise2 = a.noise ** 2

    json.dump([[c["bin"], c["offset"], float(wi)] for c, wi in zip(cols, w)],
              open(a.weights_out, "w"), indent=1)

    bins = sorted({c["bin"] for c in cols})
    fig, ax = plt.subplots(1, len(bins) + 1, figsize=(3 * (len(bins) + 1), 3.2))
    for ib, k in enumerate(bins):
        sel = [i for i, c in enumerate(cols) if c["bin"] == k]
        x = [0.1 * k + cols[i]["offset"] for i in sel]
        ax[ib].bar(x, w[sel], width=0.015)
        ax[ib].axvline(0.1 * k, color="k", lw=0.5, ls=":")
        ax[ib].set_title(f"bin {0.1 * k:.1f}p: sum {w[sel].sum():.3f}", fontsize=9)
        ax[ib].set_xlabel("start (pitch)", fontsize=8)
    ax[-1].plot(shifts, (np.array(chis) - chi) / (1.0 if a.balance else noise2), "ko-")
    ax[-1].set(xlabel="data shift [ticks]", ylabel=r"$\Delta\chi^2$",
               title=f"U {100*rel[0]:.2g}% V {100*rel[1]:.2g}% W {100*rel[2]:.2g}%")
    fig.tight_layout()
    fig.savefig(out / f"l2-{a.tag}.pdf")

    def pct(v):
        return f"{v:.0f}" if v >= 10 else f"{v:.2g}"

    lines = [f"% generated by {Path(__file__).name}; do not edit",
             f"\\newcommand{{\\{a.tag}Nshots}}{{{len(evs)}}}",
             f"\\newcommand{{\\{a.tag}Ncols}}{{{nc}}}",
             f"\\newcommand{{\\{a.tag}Shift}}{{{shifts[js]:.2f}}}"]
    for ip in range(3):
        lines.append(f"\\newcommand{{\\{a.tag}Rel{PLANES[ip]}}}{{{pct(100 * rel[ip])}}}")
    for k in bins:
        sel = [i for i, c in enumerate(cols) if c["bin"] == k]
        lines.append(f"\\newcommand{{\\{a.tag}Sum{'ABCDEF'[k]}}}{{{w[sel].sum():.3f}}}")
    (out / f"l2-{a.tag}.tex").write_text("\n".join(lines) + "\n")
    print(f"{a.tag}: shift {shifts[js]:.2f} ticks; residual/signal U {100*rel[0]:.3g}% "
          f"V {100*rel[1]:.3g}% W {100*rel[2]:.3g}%; chi2 {chi/noise2:.4g}")
    for k in bins:
        sel = [i for i, c in enumerate(cols) if c["bin"] == k]
        print(f"  bin {0.1*k:.1f}p sum {w[sel].sum():.3f}: " +
              " ".join(f"{cols[i]['offset']:+.2f}:{w[i]:.3f}" for i in sel if w[i] > 1e-3))


if __name__ == "__main__":
    main()
