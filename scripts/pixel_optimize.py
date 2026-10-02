#!/usr/bin/env python
"""Optimise a focusing-grid design by differentiating through firep (Sec. 17).

Parameters theta = (a, r, h, dV): the rounded-square pad's half-width (corner
radius 0.4 a), the grid's hole radius, its height above the pads, and its bias
below the pads; the drift field above the grid is held at 500 V/cm.  Bounds
keep the pads and the grid bars at least 0.4 mm apart, and a penalty keeps the
grid-to-pad field below E_max.  Three objectives:

  width      the rms duration of the pad's current averaged over its pitch
             square and smeared by longitudinal diffusion;
  combined   width / width_nominal + X / X_nominal, with X the swing (peak to
             peak, within 1.5 us of the collection) of the charge the same
             electrons induce on an edge neighbour, averaged and smeared;
  neighbour  X alone.

The model (firep.pixel_opt) is differentiable end to end, so each iteration
costs one forward and one backward pass; bounded L-BFGS-B takes the steps.

    python scripts/pixel_optimize.py [runs/pixel/opt] [docs/figs] [width|combined|neighbour|report|all]
"""

import json
import os
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize

from firep import pixel_opt as O

DEV = os.environ.get("FIREP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
NOMINAL = np.array([1.75, 1.9, 3.2, 2000.0])
LO = np.array([1.5, 1.3, 1.0, 500.0])
HI = np.array([2.0, 2.0, 5.0, 3000.0])
NAMES = ("pad half-width a (mm)", "hole radius r (mm)", "grid height h (mm)", "grid bias dV (V)")
FD_STEP = np.array([0.05, 0.05, 0.05, 50.0])
OBJECTIVES = ("width", "combined", "neighbour")


def config(objective="width", width_ref=1.0, x_ref=1.0):
    return O.Design(drift_res=(661, 60), weight_res=(881, 88), nodes=1200, launch_step=0.2, t_max=12.0,
                    device=DEV, objective=objective, width_ref=width_ref, x_ref=x_ref)


def state_path(out, obj):
    return os.path.join(out, "state.json" if obj == "width" else f"state-{obj}.json")


def save_series(out, tag, r):
    np.savez(os.path.join(out, f"{tag}.npz"), t=r["t"], Is=r["Is"], Iavg=r["Iavg"], I=r["I"], Qe=r["Qe"],
             Qd=r["Qd"])


def summary(r):
    return dict(J=r["J"], width=r["width"], peak=r["peak"], cap_pf=r["cap_pf"], caught=r["grid_caught"],
                landed=r["landed"], fit_drift=r["fit_drift"], fit_weight=r["fit_weight"], x_edge=r["x_edge"],
                x_diag=r["x_diag"])


def optimise(prob, out, st, key):
    hist = []
    z_of = lambda th: (np.asarray(th) - LO) / (HI - LO)
    th_of = lambda z: LO + np.asarray(z) * (HI - LO)
    t_start = time.perf_counter()

    def fun(z):
        th = th_of(z)
        r = prob.evaluate(th, grad=True)
        g = np.asarray(r["grad"]) * (HI - LO)
        hist.append(dict(theta=list(map(float, th)), J=r["J"], width=r["width"], x_edge=r["x_edge"],
                         x_diag=r["x_diag"], grad=list(map(float, r["grad"])), seconds=r["timing"]["total"],
                         t=time.perf_counter() - t_start))
        print(f"{key} eval {len(hist):3d}: theta {np.round(th, 3)}  J {r['J']:.5f}  width {r['width']:.4f}  "
              f"X {r['x_edge']:.4f}  |g| {np.linalg.norm(g):.3e}  {r['timing']['total']:.0f} s", flush=True)
        return r["J"], g

    res = minimize(fun, z_of(NOMINAL), jac=True, method="L-BFGS-B", bounds=[(0, 1)] * 4,
                   options=dict(maxiter=25, maxfun=40, ftol=1e-4, gtol=1e-5))
    return dict(theta=list(map(float, th_of(res.x))), J=float(res.fun), nit=int(res.nit), nfev=int(res.nfev),
                message=str(res.message), seconds=time.perf_counter() - t_start, history=hist)


def run(out, obj):
    """One objective's optimisation, cached in its own state file."""
    path = state_path(out, obj)
    st = json.load(open(path)) if os.path.exists(path) else {}
    if obj == "width":
        prob = O.Problem(config("width"))
        if "nominal" not in st:
            t0 = time.perf_counter()
            nom = prob.evaluate(NOMINAL, grad=True, full=True)
            fd = []
            ta = time.perf_counter()
            for i in range(4):
                tp, tm = NOMINAL.copy(), NOMINAL.copy()
                tp[i] += FD_STEP[i]
                tm[i] -= FD_STEP[i]
                fd.append((prob.evaluate(tp, grad=False)["J"] - prob.evaluate(tm, grad=False)["J"]) / (2 * FD_STEP[i]))
            st["nominal"] = dict(summary(nom), grad=list(map(float, nom["grad"])), fd=fd,
                                 fd_seconds=time.perf_counter() - ta, timing=nom["timing"])
            save_series(out, "nominal", nom)
            json.dump(st, open(path, "w"), indent=1)
            print(f"nominal: {time.perf_counter() - t0:.0f} s with the gradient check", flush=True)
    else:
        # normalise the combined objective at the nominal design
        prob0 = O.Problem(config("width"))
        if "nominal" not in st:
            nom = prob0.evaluate(NOMINAL, grad=False, full=True)
            st["nominal"] = summary(nom)
            json.dump(st, open(path, "w"), indent=1)
        n = st["nominal"]
        prob = O.Problem(config(obj, n["width"], n["x_edge"]))
    if "optimum" not in st:
        st["optimum"] = optimise(prob, out, st, obj)
        json.dump(st, open(path, "w"), indent=1)
    if "opt_full" not in st or "x_edge" not in st["opt_full"]:
        r = prob.evaluate(np.array(st["optimum"]["theta"]), grad=False, full=True)
        st["opt_full"] = summary(r)
        save_series(out, f"optimum-{obj}", r)
        json.dump(st, open(path, "w"), indent=1)
    th = st["optimum"]["theta"]
    js = json.dumps(dict(a=th[0], r=th[1], h=th[2], dV=th[3]))
    open(os.path.join(out, f"optimum-{obj}.json"), "w").write(js)
    if obj == "width":
        open(os.path.join(out, "optimum.json"), "w").write(js)
    return st


def report(out, figdir):
    """Figures and numbers from all the objectives that have been run."""
    st = {o: json.load(open(state_path(out, o))) for o in OBJECTIVES if os.path.exists(state_path(out, o))}
    # the width run predates the neighbour metric: re-evaluate its nominal and optimum
    sw = st["width"]
    if "x_edge" not in sw["nominal"]:
        prob = O.Problem(config("width"))
        r = prob.evaluate(NOMINAL, grad=False, full=True)
        sw["nominal"].update({k: v for k, v in summary(r).items() if k in ("x_edge", "x_diag")})
        save_series(out, "nominal", r)
        json.dump(sw, open(state_path(out, "width"), "w"), indent=1)
    if "x_edge" not in sw.get("opt_full", {}):
        prob = O.Problem(config("width"))
        r = prob.evaluate(np.array(sw["optimum"]["theta"]), grad=False, full=True)
        sw["opt_full"] = summary(r)
        save_series(out, "optimum-width", r)
        json.dump(sw, open(state_path(out, "width"), "w"), indent=1)
    if not os.path.exists(os.path.join(out, "optimum-width.npz")) and os.path.exists(os.path.join(out, "optimum.npz")):
        prob = O.Problem(config("width"))
        save_series(out, "optimum-width", prob.evaluate(np.array(sw["optimum"]["theta"]), grad=False, full=True))
    V = {}

    def value(name, text):
        m = re.fullmatch(r"(-?[0-9.]+)e([+-]?)0*([0-9]+)", text)
        if m:
            mant, sign, ex = m.groups()
            text = f"${mant}\\times10^{{{'-' if sign == '-' else ''}{ex}}}$"
        V[name] = text

    cfg = config()
    nom, opt, hist = sw["nominal"], sw["optimum"], sw["optimum"]["history"]
    # ---- the width optimisation ------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0), constrained_layout=True)
    J = [h_["J"] for h_ in hist]
    ax[0].plot(np.arange(1, len(J) + 1), J, "o-", ms=3, label="each evaluation")
    ax[0].plot(np.arange(1, len(J) + 1), np.minimum.accumulate(J), "k-", lw=1.5, label="best so far")
    ax[0].set_xlabel("objective evaluation")
    ax[0].set_ylabel("rms duration of the smeared average (µs)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    TH = np.array([h_["theta"] for h_ in hist])
    for i in range(4):
        ax[1].plot(np.arange(1, len(J) + 1), (TH[:, i] - LO[i]) / (HI[i] - LO[i]), "o-", ms=3, label=NAMES[i])
    ax[1].set_xlabel("objective evaluation")
    ax[1].set_ylabel("parameter, scaled to its bounds")
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    wn, wo = np.load(os.path.join(out, "nominal.npz")), np.load(os.path.join(out, "optimum-width.npz"))
    for w_, lab, c in ((wn, "nominal", "C0"), (wo, "optimum", "C3")):
        tpk = w_["t"][np.argmax(w_["Is"])]
        ax[2].plot(w_["t"] - tpk, w_["Iavg"], color=c, lw=0.8, alpha=0.6, label=f"{lab}, before diffusion")
        ax[2].plot(w_["t"] - tpk, w_["Is"], color=c, lw=2, label=f"{lab}, with diffusion")
    ax[2].set_xlim(-2.5, 2.5)
    ax[2].set_xlabel("time from the peak (µs)")
    ax[2].set_ylabel("average current (e/µs)")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3)
    fig.savefig(os.path.join(figdir, "pixel-opt.pdf"))
    fig.savefig(os.path.join(out, "opt.png"), dpi=120)
    plt.close(fig)

    # ---- the neighbour signal: waveforms and the trade-off ------------------------
    have = [o for o in OBJECTIVES if o in st and os.path.exists(os.path.join(out, f"optimum-{o}.npz"))]
    cols = {"nominal": "k", "width": "C3", "combined": "C2", "neighbour": "C4"}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    series = [("nominal", wn)] + [(o, np.load(os.path.join(out, f"optimum-{o}.npz"))) for o in have]
    for lab, w_ in series:
        tpk = w_["t"][np.argmax(w_["Is"])]
        tq = (np.arange(len(w_["Qe"]))) * cfg.dt
        name = "nominal" if lab == "nominal" else f"optimum for {lab}"
        ax[0].plot(tq - tpk, w_["Qe"], color=cols[lab], lw=1.8, label=name)
        ax[0].plot(tq - tpk, w_["Qd"], color=cols[lab], lw=1.0, ls="--")
        ax[1].plot(w_["t"] - tpk, w_["Is"], color=cols[lab], lw=1.8, label=name)
    ax[0].set_xlim(-3, 1.5)
    ax[0].set_xlabel("time from the collecting pad's peak (µs)")
    ax[0].set_ylabel("charge on a neighbour per electron collected")
    ax[0].set_title("unwanted charge on the edge (solid) and diagonal (dashed)\nneighbour, averaged, with diffusion",
                    fontsize=9)
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)
    ax[1].set_xlim(-2.5, 2.5)
    ax[1].set_xlabel("time from the peak (µs)")
    ax[1].set_ylabel("current on the collecting pad (e/µs)")
    ax[1].set_title("the collecting pad's average response, with diffusion", fontsize=9)
    ax[1].grid(alpha=0.3)
    a = ax[2]
    for o in ("combined", "neighbour"):
        if o in st:
            H = st[o]["optimum"]["history"]
            a.plot([h_["width"] for h_ in H], [h_["x_edge"] for h_ in H], "o", ms=3, color=cols[o], alpha=0.4,
                   label=f"evaluations, {o} objective")
    a.plot(nom["width"], nom["x_edge"], "k*", ms=12, label="nominal")
    for o in have:
        f_ = st[o]["opt_full"]
        a.plot(f_["width"], f_["x_edge"], "s", ms=9, color=cols[o], label=f"optimum for {o}")
    a.set_xlabel("rms duration with diffusion (µs)")
    a.set_ylabel("peak unwanted charge on the edge neighbour")
    a.set_title("the trade-off", fontsize=9)
    a.legend(fontsize=7)
    a.grid(alpha=0.3)
    fig.savefig(os.path.join(figdir, "pixel-opt-neighbour.pdf"))
    fig.savefig(os.path.join(out, "opt-neighbour.png"), dpi=120)
    plt.close(fig)

    # ---- numbers ----------------------------------------------------------------
    tags = {"width": "Opt", "combined": "Comb", "neighbour": "Nbr"}
    for k, th in [("Nom", NOMINAL)] + [(tags[o], np.array(st[o]["optimum"]["theta"])) for o in have]:
        value(f"opt{k}A", f"{2 * th[0]:.2f}")
        value(f"opt{k}R", f"{th[1]:.2f}")
        value(f"opt{k}H", f"{th[2]:.2f}")
        value(f"opt{k}V", f"{th[3]:.0f}")
        value(f"opt{k}Field", f"{th[3] / th[2]:.0f}")
    for k, d in [("Nom", nom)] + [(tags[o], st[o]["opt_full"]) for o in have]:
        value(f"opt{k}Width", f"{d['width']:.3f}")
        value(f"opt{k}Peak", f"{d['peak']:.2f}")
        value(f"opt{k}Ratio", f"{d['peak'] / d['width']:.2f}")
        value(f"opt{k}Cap", f"{d['cap_pf']:.2f}")
        value(f"opt{k}FitDrift", f"{100 * d['fit_drift']:.2f}")
        value(f"opt{k}FitWeight", f"{d['fit_weight']:.1e}")
        value(f"opt{k}Xe", f"{100 * d['x_edge']:.2f}")
        value(f"opt{k}Xd", f"{100 * d['x_diag']:.2f}")
        value(f"opt{k}Caught", f"{100 * d['caught']:.1f}")
    for o in have:
        value(f"opt{tags[o]}Nfev", f"{st[o]['optimum']['nfev']}")
        value(f"opt{tags[o]}Nit", f"{st[o]['optimum']['nit']}")
        value(f"opt{tags[o]}Minutes", f"{st[o]['optimum']['seconds'] / 60:.0f}")
        value(f"opt{tags[o]}PerEval", f"{np.mean([h_['seconds'] for h_ in st[o]['optimum']['history']]):.0f}")
    value("optSigmaT", f"{cfg.sigma_t:.3f}")
    value("optDL", f"{cfg.D_L * 1e4:.0f}")
    value("optEmax", f"{cfg.E_max:.0f}")
    for i, key in enumerate(("A", "R", "H", "V")):
        value(f"optGrad{key}", f"{nom['grad'][i]:.3e}")
        value(f"optFd{key}", f"{nom['fd'][i]:.3e}")
    tm = nom["timing"]
    for k in ("build", "fit", "trace", "backward", "total"):
        value(f"optT{k.capitalize()}", f"{tm[k]:.1f}")
    value("optPeakGb", f"{tm.get('peak_gb', 0):.1f}")
    for k in ("cols_drift", "cols_weight", "rows_drift", "rows_weight"):
        value({"cols_drift": "optColsD", "cols_weight": "optColsW", "rows_drift": "optRowsD",
               "rows_weight": "optRowsW"}[k], f"{tm[k]}")
    value("optFdSeconds", f"{nom['fd_seconds']:.0f}")
    value("optForward", f"{tm['total'] - tm['backward']:.1f}")
    value("optNfev", f"{opt['nfev']}")
    value("optNit", f"{opt['nit']}")
    value("optMinutes", f"{opt['seconds'] / 60:.0f}")
    value("optPerEval", f"{np.mean([h_['seconds'] for h_ in hist]):.0f}")
    value("optCaught", f"{100 * sw['opt_full']['caught']:.1f}")
    value("optGain", f"{100 * (1 - sw['opt_full']['width'] / nom['width']):.0f}")
    dc = domain_check(out)
    dtag = {"nominal": "Nom", "width": "Opt", "combined": "Comb", "neighbour": "Nbr"}
    for key, v in dc.items():
        N, k = key.split("-")
        nn = {"5": "Five", "9": "Nine"}[N]
        value(f"optDom{nn}{dtag[k]}Xe", f"{100 * v[1]:.2f}")
        value(f"optDom{nn}{dtag[k]}Base", f"{100 * v[3]:.2f}")
    with open(os.path.join(figdir, "pixel-opt.tex"), "w") as f:
        f.write("% generated by scripts/pixel_optimize.py -- do not edit\n")
        for k, v in V.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    for k, v in V.items():
        print(f"{k:18s} {v}")


def domain_check(out):
    """The neighbour metric against the weighting domain's size (5 x 5 and
    9 x 9 pads) at the nominal design and each optimum: the neighbour's charge
    carries a small baseline from the finite domain's zero-slope walls, and the
    windowed swing should not."""
    path = os.path.join(out, "domain-check.json")
    if os.path.exists(path):
        return json.load(open(path))
    designs = {"nominal": list(NOMINAL)}
    for o in OBJECTIVES:
        p = os.path.join(out, f"optimum-{o}.json")
        if os.path.exists(p):
            designs[o] = list(json.load(open(p)).values())
    res = {}
    for N, wres in ((5, (881, 88)), (9, (1585, 158))):
        cfg = O.Design(drift_res=(661, 60), weight_res=wres, weight_cells=N, nodes=1200, launch_step=0.2,
                       t_max=12.0, device=DEV)
        prob = O.Problem(cfg)
        for k, th in designs.items():
            r = prob.evaluate(np.array(th), grad=False, full=True)
            res[f"{N}-{k}"] = [r["width"], r["x_edge"], r["x_diag"], float(r["Qe"][0])]
    json.dump(res, open(path, "w"), indent=1)
    return res


def main(out="runs/pixel/opt", figdir="docs/figs", mode="all"):
    os.makedirs(out, exist_ok=True)
    for o in (OBJECTIVES if mode == "all" else (mode,) if mode in OBJECTIVES else ()):
        run(out, o)
    if mode in ("all", "report"):
        report(out, figdir)


if __name__ == "__main__":
    main(*sys.argv[1:])
