#!/usr/bin/env python3
"""Where electrons land in the bad-APA cell, as a function of W's potential.

    firep_landing.py CONFIG OUTDIR [--vw 0,-20,...] [--n 942] [--y-start 100]

For each W potential, solve the drift potential with firep's analytic basis
alone (exactly harmonic; seconds, no training), trace electrons launched at
``--y-start`` mm uniformly across one pitch, and record which conductor each
reaches.  Folded onto 0 <= |x| <= p/2 this is the impact-position map that the
6 Garfield paths sample at 0, 0.1, ..., 0.5 pitch (others/CM_FieldResponse.pdf
slide 8: V collects 0-0.3, W 0.4, the mesh 0.5 at V_W = 0).

The floating-equilibrium estimate is the least negative V_W at which W
collects nothing: a floating W charges until it stops collecting.

Writes OUTDIR/firep-landing.{pdf,tex,json}; checkpoints go to OUTDIR/pt/.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from firep import physics
from firep import train as train_mod

LABELS = ["v", "w", "ground", "u", "g", "cathode", "stalled"]
NAMES = {"ground": "mesh"}
GARFIELD_IMPACTS = np.arange(6) * 0.1       # in units of pitch


def w_index(cfg_path):
    import yaml
    cfg = yaml.safe_load(open(cfg_path))
    return [e["name"] for e in cfg["electrodes"]].index("w")


def solve(cfg_path, vw, out, sets=()):
    firep = Path(sys.executable).with_name("firep")
    extra = [x for kv in sets for x in ("--set", kv)]
    subprocess.run([str(firep), "train", "-c", str(cfg_path), "--steps", "0", "-q",
                    "--set", "model.final_init_scale=0",
                    "--set", f"electrodes.{w_index(cfg_path)}.bias={vw}", *extra,
                    "-o", str(out)], check=True, stderr=subprocess.DEVNULL)


def landing(pt, n, y_start):
    cfg, geom, model, _ = train_mod.load(str(pt), "cpu")
    landed, _, _ = physics.drift_paths(cfg, geom, model, n=n, y_start=y_start)
    x = geom.xlo + (np.arange(n) + 0.5) * geom.width / n
    pitch = geom.width
    return np.abs(x) / pitch, np.array(landed, dtype=str)


def band_edges(u, landed):
    """Sorted by |x|/p: list of (label, u_lo, u_hi) runs."""
    order = np.argsort(u, kind="stable")
    u, lab = u[order], landed[order]
    runs, start = [], 0
    for i in range(1, len(u) + 1):
        if i == len(u) or lab[i] != lab[start]:
            runs.append((lab[start], float(u[start]), float(u[i - 1])))
            start = i
    return runs


def label_at(u, landed, u0):
    return landed[np.argmin(np.abs(u - u0))]


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("outdir")
    ap.add_argument("--vw", default="0,-20,-40,-60,-80,-100,-120,-140,-160,-200")
    ap.add_argument("--n", type=int, default=942, help="electrons across one pitch")
    ap.add_argument("--y-start", type=float, default=100.0, help="[mm]")
    ap.add_argument("--bisect", default=None, metavar="LO,HI,TOL",
                    help="instead of a scan, bisect for the floating equilibrium: "
                         "W collects at LO, not at HI [V]; stop at TOL")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="passed to firep train, e.g. domain.bounds.y=[-8,204.71]")
    a = ap.parse_args()
    out = Path(a.outdir)
    (out / "pt").mkdir(parents=True, exist_ok=True)

    def w_fraction(vw):
        pt = out / "pt" / f"vw{vw:+.2f}.pt"
        if not pt.exists():
            solve(a.config, vw, pt, a.set)
        _, lab = landing(pt, a.n, a.y_start)
        return float(np.mean(lab == "w"))

    if a.bisect:
        lo, hi, tol = (float(x) for x in a.bisect.split(","))
        flo, fhi = w_fraction(lo), w_fraction(hi)
        if flo == 0.0 or fhi > 0.0:
            raise SystemExit(f"bisect needs W landing at {lo} ({flo}) and not at {hi} ({fhi})")
        while abs(hi - lo) > tol:
            mid = 0.5 * (lo + hi)
            if w_fraction(mid) > 0.0:
                lo = mid
            else:
                hi = mid
            print(f"  bracket [{lo:.2f}, {hi:.2f}] V")
        json.dump([{"vfloat_estimate": hi, "bracket": [lo, hi]}],
                  open(out / "firep-landing.json", "w"), indent=1)
        print(f"floating equilibrium: V_W = {hi:.2f} V (W collects at {lo:.2f})")
        return

    vws = [float(v) for v in a.vw.split(",")]

    rows = []
    for vw in vws:
        pt = out / "pt" / f"vw{vw:+.0f}.pt"
        if not pt.exists():
            solve(a.config, vw, pt, a.set)
        u, lab = landing(pt, a.n, a.y_start)
        frac = {k: float(np.mean(lab == k)) for k in LABELS if np.any(lab == k)}
        rows.append(dict(vw=vw, frac=frac, bands=band_edges(u, lab),
                         garfield=[str(label_at(u, lab, g)) for g in GARFIELD_IMPACTS],
                         u=u.tolist(), landed=lab.tolist()))
        print(f"V_W {vw:+7.1f} V  " + "  ".join(f"{NAMES.get(k, k)} {100*v:5.1f}%"
              for k, v in frac.items()) + "   6 paths: " + " ".join(rows[-1]["garfield"]))

    # floating equilibrium: first V_W (scanning downward) with no W landing
    zero_w = [r["vw"] for r in rows if r["frac"].get("w", 0.0) == 0.0]
    vfloat = max(zero_w) if zero_w else None

    json.dump([{k: v for k, v in r.items() if k not in ("u", "landed")} for r in rows]
              + [{"vfloat_estimate": vfloat}],
              open(out / "firep-landing.json", "w"), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"v": "C0", "w": "C3", "ground": "C2", "u": "C4", "g": "C5",
              "cathode": "C7", "stalled": "k"}
    for i, r in enumerate(rows):
        for lab, lo, hi in r["bands"]:
            ax[0].barh(i, max(hi - lo, 1e-3), left=lo, color=colors[lab], height=0.8)
    for g in GARFIELD_IMPACTS:
        ax[0].axvline(g, color="k", lw=0.5, ls=":")
    ax[0].set_yticks(range(len(rows)), [f"{r['vw']:+.0f}" for r in rows])
    ax[0].set(xlabel="impact |x| / pitch  (dotted: the 6 Garfield paths)",
              ylabel="$V_W$ [V]", xlim=(0, 0.5))
    for k in ["v", "w", "ground"]:
        ax[1].plot([r["vw"] for r in rows], [100 * r["frac"].get(k, 0) for r in rows],
                   "o-", color=colors[k], label=NAMES.get(k, k))
    ax[1].set(xlabel="$V_W$ [V]", ylabel="fraction of impacts [%]",
              title="collected fraction")
    ax[1].legend(frameon=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in ["v", "w", "ground"]]
    ax[0].legend(handles, ["lands on V", "on W", "on mesh"], frameon=False, ncol=3,
                 loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "firep-landing.pdf")

    r0 = next(r for r in rows if r["vw"] == 0.0) if 0.0 in vws else rows[0]
    lines = [f"% generated by {Path(__file__).name}; do not edit",
             macro("flYstart", f"{a.y_start:.0f}"),
             macro("flVzero", f"{100 * r0['frac'].get('v', 0):.0f}"),
             macro("flWzero", f"{100 * r0['frac'].get('w', 0):.0f}"),
             macro("flMzero", f"{100 * r0['frac'].get('ground', 0):.0f}"),
             macro("flVfloat", "n/a" if vfloat is None else f"{vfloat:.0f}")]
    (out / "firep-landing.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
