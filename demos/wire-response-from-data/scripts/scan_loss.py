#!/usr/bin/env python3
"""chi^2 between reference frames and model frames over a parameter scan.

    scan_loss.py REF.npz OUTDIR --tag TAG --anode A --param NAME
                 VALUE=MODEL.npz [VALUE=MODEL.npz ...]
                 [--noise 3.5] [--tmax 12] [--upsample 4] [--true VALUE]

For each model (one WCT frame file per parameter value, same events as REF),
chi^2 = sum over events, anode A's channels and ticks of (d - m)^2 / noise^2,
with d and m baseline-subtracted ADC.  A common time shift t0 of the model is
profiled: chi^2(s) = sum d^2 + sum m^2 - 2 sum d m(s) with the correlation
for all shifts from one FFT per channel, upsampled for sub-tick steps, within
+-tmax ticks.  Channels and ticks with no activity in either frame are skipped.

The noise is an assumed white rms per sample (the frames themselves are
noise-free), so chi^2 differences read as the statistical power of this many
shots with that noise.  The best value and its 1-sigma error come from a
parabola through the three lowest points (delta chi^2 = 1).

Writes OUTDIR/scan-TAG.pdf and OUTDIR/scan-TAG.tex (macros \\TAG...).
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLANES = "UVW"
SPLIT = (800, 1600)


def events(f, anode):
    return sorted(int(k.rsplit("_", 1)[1]) for k in f.files
                  if k.startswith(f"frame_orig{anode}_"))


def frame(f, anode, ev):
    x = f[f"frame_orig{anode}_{ev}"].astype(float)
    x -= np.median(x, axis=1, keepdims=True)
    ch = f[f"channels_orig{anode}_{ev}"]
    return ch - ch.min(), x


def plane_masks(rel):
    return [rel < SPLIT[0], (rel >= SPLIT[0]) & (rel < SPLIT[1]), rel >= SPLIT[1]]


def chi2_vs_shift(d, m, tmax, up, thr=0.5):
    """Per-sample-unnormalised chi^2 for shifts in ticks (m moved by +s)."""
    act = (np.abs(d).max(1) > thr) | (np.abs(m).max(1) > thr)
    if not act.any():
        return None
    d, m = d[act], m[act]
    cols = np.flatnonzero((np.abs(d).max(0) > thr) | (np.abs(m).max(0) > thr))
    lo = max(0, cols[0] - 2 * tmax - 20)
    hi = min(d.shape[1], cols[-1] + 2 * tmax + 20)
    d, m = d[:, lo:hi], m[:, lo:hi]
    n = d.shape[1]
    L = 1 << int(np.ceil(np.log2(2 * n)))                 # no wrap for |s| < n
    D = np.fft.rfft(d, L, axis=1)
    M = np.fft.rfft(m, L, axis=1)
    corr = np.fft.irfft(np.conj(D) * M, L * up, axis=1).sum(0) * up
    # corr[k] = sum_t d(t) m(t + k/up): moving m by +s means lag k = -s*up
    lags = np.arange(-tmax * up, tmax * up + 1)
    cross = corr[(-lags) % (L * up)]
    return (d ** 2).sum() + (m ** 2).sum() - 2.0 * cross, lags / up


def parabola(x, y):
    """Vertex and 1-sigma half width of a parabola through the three lowest
    points; a minimum at the scan's edge is returned as is, with nan error."""
    i = int(np.argmin(y))
    if len(x) < 3 or i in (0, len(x) - 1):
        return float(x[i]), float("nan"), float(y[i])
    i = min(max(i, 1), len(y) - 2)
    c = np.polyfit(x[i - 1:i + 2], y[i - 1:i + 2], 2)
    if c[0] <= 0:
        return float(x[np.argmin(y)]), float("nan"), float(np.min(y))
    xb = -c[1] / (2 * c[0])
    return float(xb), float(np.sqrt(1.0 / c[0])), float(np.polyval(c, xb))


def pct(x):
    return f"{x:.0f}" if x >= 10 else f"{x:.2g}"


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ref")
    ap.add_argument("outdir")
    ap.add_argument("models", nargs="+", help="VALUE=frames.npz")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--anode", type=int, default=0)
    ap.add_argument("--param", default="$V_W$ [V]")
    ap.add_argument("--noise", type=float, default=3.5, help="ADC rms per sample")
    ap.add_argument("--tmax", type=int, default=12, help="ticks")
    ap.add_argument("--upsample", type=int, default=4)
    ap.add_argument("--true", type=float, default=None)
    a = ap.parse_args()
    if not a.tag.isalpha():
        raise ValueError("tag must be letters only")
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    fr = np.load(a.ref)
    evs = events(fr, a.anode)
    models = sorted(((float(s.split("=", 1)[0]), s.split("=", 1)[1]) for s in a.models))
    values = np.array([v for v, _ in models])

    # chi2 of a null model (m = 0): the scale of the signal itself
    null = np.zeros(3)
    for ev in evs:
        rel, d = frame(fr, a.anode, ev)
        for ip, mask in enumerate(plane_masks(rel)):
            null[ip] += (d[mask] ** 2).sum()
    null /= a.noise ** 2

    # chi2[model, plane, shift]
    chi2 = None
    for im, (val, path) in enumerate(models):
        fm = np.load(path)
        if events(fm, a.anode) != evs:
            raise ValueError(f"{path}: events differ from {a.ref}")
        for ev in evs:
            rel, d = frame(fr, a.anode, ev)
            _, m = frame(fm, a.anode, ev)
            for ip, mask in enumerate(plane_masks(rel)):
                got = chi2_vs_shift(d[mask], m[mask], a.tmax, a.upsample)
                if got is None:
                    continue
                c, shifts = got
                if chi2 is None:
                    chi2 = np.zeros((len(models), 3, len(shifts)))
                chi2[im, ip] += c
    chi2 /= a.noise ** 2

    total = chi2.sum(1)                                   # (model, shift)
    best_shift = shifts[np.argmin(total, axis=1)]          # per model
    prof = total.min(1)
    per_plane = np.array([chi2[im, :, np.argmin(total[im])] for im in range(len(models))])
    xb, err, cmin = parabola(values, prof)
    ib = int(np.argmin(prof))

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].plot(values, prof - prof.min(), "ko-", label="all planes")
    for ip in range(3):
        ax[0].plot(values, per_plane[:, ip] - per_plane[:, ip].min(), ".--",
                   label=PLANES[ip])
    ax[0].set(xlabel=a.param, ylabel=r"$\chi^2 - \chi^2_{\min}$", yscale="symlog",
              title=f"{len(evs)} shots, noise {a.noise:g} ADC")
    if a.true is not None:
        ax[0].axvline(a.true, color="C3", lw=0.8, ls=":")
    ax[0].legend(frameon=False, fontsize=8)
    sel = np.abs(values - xb) <= max(3 * err if np.isfinite(err) else 0, 5)
    sel |= np.arange(len(values)) == ib
    ax[1].plot(values[sel], prof[sel] - cmin, "ko")
    xx = np.linspace(values[sel].min(), values[sel].max(), 200)
    if np.isfinite(err):
        ax[1].plot(xx, ((xx - xb) / err) ** 2, "C0-")
    ax[1].axhline(1.0, color="0.5", lw=0.5)
    ax[1].set(xlabel=a.param, ylabel=r"$\Delta\chi^2$",
              title=f"best {xb:.2f} $\\pm$ {err:.2f}")
    ax[2].plot(values, best_shift, "ko-")
    ax[2].set(xlabel=a.param, ylabel="best model shift t0 [ticks]",
              title="profiled time shift")
    fig.tight_layout()
    fig.savefig(out / f"scan-{a.tag}.pdf")

    lines = [f"% generated by {Path(__file__).name}; do not edit",
             macro(f"{a.tag}Nshots", len(evs)),
             macro(f"{a.tag}Noise", f"{a.noise:g}"),
             macro(f"{a.tag}Best", f"{xb:.1f}"),
             macro(f"{a.tag}Err", f"{err:.2f}" if np.isfinite(err) else "n/a"),
             macro(f"{a.tag}Edge", "yes" if ib in (0, len(values) - 1) else "no"),
             macro(f"{a.tag}Shift", f"{best_shift[ib]:.2f}"),
             macro(f"{a.tag}ChiMin", f"{prof[ib]:.3g}")]
    for ip in range(3):
        lines.append(macro(f"{a.tag}Chi{PLANES[ip]}", f"{per_plane[ib, ip]:.3g}"))
        lines.append(macro(f"{a.tag}Rel{PLANES[ip]}",
                           pct(100 * per_plane[ib, ip] / null[ip])))
    lines.append(macro(f"{a.tag}Rel", pct(100 * prof[ib] / null.sum())))
    (out / f"scan-{a.tag}.tex").write_text("\n".join(lines) + "\n")
    print(f"{a.tag}: best {xb:.2f} +- {err:.2f}, shift {best_shift[ib]:.2f} ticks, "
          f"chi2min {prof[ib]:.4g} (U {per_plane[ib,0]:.3g} V {per_plane[ib,1]:.3g} "
          f"W {per_plane[ib,2]:.3g}) over {len(evs)} shots")
    print("  null-model chi2 (signal scale): U %.3g V %.3g W %.3g" % tuple(null))
    print("  residual / signal at best: U %.2g%% V %.2g%% W %.2g%% all %.2g%%" % (
        *(100 * per_plane[ib] / null), 100 * prof[ib] / null.sum()))
    for v, p, s in zip(values, prof, best_shift):
        print(f"  {v:8.2f}  chi2 {p:12.5g}  shift {s:6.2f}")


if __name__ == "__main__":
    main()
