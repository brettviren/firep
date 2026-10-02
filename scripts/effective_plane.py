#!/usr/bin/env python
"""Appendix B: a wire plane seen from afar.

Averaged across one period, Laplace's equation leaves only d^2<Phi>/dy^2 = 0,
so each wire row acts like a solid sheet: <Phi> is piecewise linear, its slope
jumps by Gauss's law at each row, and the row's wires sit gamma*Lambda above
the sheet's potential (Eq. selfpotential).  This script builds that
one-dimensional "effective-plane" model for the drift and for the three
weighting problems, checks it against the 2D solves, and asks what detector
observables it predicts and how they move with the as-built geometry.

The check uses the response table of the 2D model: summed over all 21 wires of
the weighting period and averaged over impact positions, it is the response to
a sheet of charge uniform across the wires (an isochronous track), which by the
weighting sum rule sees exactly the plane-averaged weighting potential.

    python scripts/effective_plane.py [runs] [docs/figs]
"""

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from firep import bias as bias_mod
from firep import config as config_mod
from firep import response as resp_mod

TWO_PI = 2.0 * math.pi
Y_LAUNCH = 99.1  # the response table's launch height
TEMPERATURE = 87.3


def speed(f):
    """Drift speed [mm/us] for a plane-averaged field of f V/mm."""
    return float(resp_mod.drift_speed(np.abs(f), "walkowiak", TEMPERATURE, 0.0))


class Planes:
    """The effective-plane model: faces at y_c > rows > y_g, one Lambda per row."""

    def __init__(self, y_c, rows, y_g, pitch, radius):
        self.y_c, self.rows, self.y_g = y_c, np.asarray(rows, float), y_g
        self.pitch, self.radius = pitch, radius
        self.nodes = np.concatenate(([y_c], self.rows, [y_g]))
        self.lengths = -np.diff(self.nodes)
        self.lam = math.log(pitch / (TWO_PI * radius))

    def solve(self, v_c, v_rows, v_g):
        """Gap fields F_j [V/mm] (marching down raises <V> by F_j d_j), node
        potentials of <V>, and the rows' line charges gamma_i [V]."""
        n = len(self.rows)
        a, b = np.zeros((n + 1, n + 1)), np.zeros(n + 1)
        w = self.lam * self.pitch / TWO_PI
        for i in range(n):
            a[i, : i + 1] = self.lengths[: i + 1]
            a[i, i] += w
            a[i, i + 1] -= w
            b[i] = v_rows[i] - v_c
        a[n, :] = self.lengths
        b[n] = v_g - v_c
        f = np.linalg.solve(a, b)
        avg = v_c + np.concatenate(([0.0], np.cumsum(f * self.lengths)))
        gamma = self.pitch * (f[:-1] - f[1:]) / TWO_PI
        return f, avg, gamma

    def average(self, avg, y):
        """<V>(y), linear between the nodes."""
        return np.interp(y, self.nodes[::-1], avg[::-1])


def sheet_charge(model, fields, weights, y0=Y_LAUNCH, dt=0.01):
    """Charge [e] a sheet uniform across the wires induces on one wire of each
    plane as it drifts from y0 to the collection row, and the times it crosses
    each row.  weights[p] are the node potentials of <Psi_p>."""
    ys, t, ts = [y0], 0.0, [0.0]
    y = y0
    crossings = {}
    for j in range(len(model.rows)):
        y_lo = model.rows[j]
        while y > y_lo:
            y -= speed(fields[j]) * dt
            t += dt
            ys.append(max(y, y_lo))
            ts.append(t)
        crossings[j] = t
    ys, ts = np.array(ys), np.array(ts)
    Q = np.array([model.average(w, ys) for w in weights])
    # landing on the collection row: its wires are at 1, the others' at 0
    final = np.zeros((len(weights), 1))
    final[-1] = 1.0
    return np.append(ts, t), ys, np.hstack((Q, final)), crossings


def main(rundir="runs", figdir="docs/figs"):
    os.makedirs(figdir, exist_ok=True)
    cfg = config_mod.load_with_overrides(None, ())
    sol = bias_mod.solve(cfg)
    planes = cfg.electrodes_by_y
    names = [e.name for e in planes]
    ylo, yhi = cfg.domain.bounds.y
    pitch, radius = planes[0].lattice.pitch, planes[0].shape.radius
    model = Planes(yhi, [e.plane for e in planes], ylo, pitch, radius)
    v_rows = np.array([sol.electrodes[n] for n in names])

    # ---- the drift problem: the effective-plane model is the bias solver's
    f, avg, gamma = model.solve(sol.cathode, v_rows, sol.ground)
    f_ref = np.array([g.field for g in sol.gaps]) * 0.1  # V/cm -> V/mm
    assert np.allclose(f, f_ref) and np.allclose(gamma, [sol.gamma[n] for n in names])

    V = {}

    def val(name, x, fmt):
        text = format(x, fmt)
        if "e" in fmt:  # 3e-18 -> $3\times10^{-18}$
            mant, ex = text.split("e")
            text = f"${mant}\\times10^{{{int(ex)}}}$"
        V[name] = text

    val("epLambda", model.lam, ".3f")
    val("epModeHalf", math.exp(-math.pi), ".3f")
    val("epModeOne", math.exp(-TWO_PI), ".4f")
    for n, a, g, vw in zip(names, avg[1:-1], gamma, v_rows):
        N = n.upper()
        val(f"epAvg{N}", a, ".1f")
        val(f"epGamma{N}", g, ".1f")
        val(f"epWire{N}", vw, ".1f")
        val(f"epOffset{N}", g * model.lam, ".1f")
        # the induced dipole: a <E_y> with <E_y> the mean of the two gap fields
    for j, gname in enumerate(["CU", "UV", "VW", "WG"]):
        val(f"epField{gname}", f[j] * 10.0, ".1f")  # V/cm
        val(f"epSpeed{gname}", speed(f[j]), ".3f")
    for i, n in enumerate(names):
        val(f"epDipole{n.upper()}", abs(radius * 0.5 * (f[i] + f[i + 1])) + 0.0, ".2f")

    # ---- the weighting problems: plane p at 1 V, everything else at 0
    W, G = [], []
    for p in range(len(names)):
        vp = np.zeros(len(names))
        vp[p] = 1.0
        _, a, g = model.solve(0.0, vp, 0.0)
        W.append(a)
        G.append(g)
    G = np.array(G)  # G[p, j]: line charge on row j with row p at 1 V
    val("epReciprocity", float(np.abs(G - G.T).max()), ".0e")
    for p, n in enumerate(names):
        for j, m in enumerate(names):
            val(f"epCap{n.upper()}{m.upper()}", G[p, j], ".4f")
        for i, m in enumerate(names):
            val(f"epPsi{n.upper()}At{m.upper()}", W[p][i + 1], ".4f")
        val(f"epPsi{n.upper()}Self", 1.0 - W[p][p + 1], ".4f")

    # ---- the 2D check: sum over wires, average over impacts
    ra, rm = resp_mod.load_npz(os.path.join(rundir, "response.npz"))
    t_c, cur = ra["time"], ra["current"]  # (plane, offset, impact, t)
    dt = float(t_c[1] - t_c[0])
    t_e = np.concatenate(([0.0], t_c + dt / 2))
    Qk = np.concatenate((np.zeros(cur.shape[:1] + cur.shape[2:3] + (1,)),
                         np.cumsum(cur.sum(axis=1) * dt, axis=-1)), axis=-1)  # (plane, impact, t_e)
    imp = ra["impact"]
    wk = np.ones(len(imp))
    wk[0] = wk[-1] = 0.5
    wk /= wk.sum()
    tt, ty, alive = ra["traj_t"], ra["traj_y"], ra["traj_alive"]
    # the table's charge is measured from the launch, where <Psi_p> is not quite
    # zero; far above every row the 2D potential is its plane average, so add it
    q_launch = np.array([model.average(W[p], Y_LAUNCH) for p in range(len(names))])
    Qk += q_launch[:, None, None]
    val("epLaunchPsiU", q_launch[0], ".4f")

    def crossing(k, y):
        yk = ty[:, k]
        ok = np.isfinite(yk)
        return float(np.interp(-y, -yk[ok], tt[ok]))

    probe = [50.0, 12.5, 7.5, 2.5]
    rows = []
    for y in probe:
        q2 = np.zeros(len(names))
        for k in range(len(imp)):
            tk = crossing(k, y)
            q2 += wk[k] * np.array([np.interp(tk, t_e, Qk[p, k]) for p in range(len(names))])
        q1 = np.array([model.average(W[p], y) for p in range(len(names))])
        rows.append((y, q1, q2))
        print(f"y = {y:5.1f}: effective plane {q1}, 2D {q2}")
    V["epProbeRows"] = "\n".join(
        f"{y:.1f} & " + " & ".join(f"{a:.4f} & {b:.4f}" for a, b in zip(q1, q2)) + r" \\"
        for y, q1, q2 in rows)
    err = max(float(np.abs(q1 - q2).max()) for _, q1, q2 in rows)
    val("epProbeMaxErr", err, ".0e")

    # the impact-averaged, wire-summed charge against the sheet model
    Qavg = np.einsum("k,pkt->pt", wk, Qk)
    ts, ys, Qs, cross = sheet_charge(model, f, W)
    # the times the rows are crossed, impact-averaged, against the sheet's
    t1 = [cross[0], cross[1], cross[2]]
    t2 = [sum(wk[k] * crossing(k, y) for k in range(len(imp))) for y in model.rows[:2]]
    t2.append(sum(wk[k] * float(tt[np.argmax(~alive[:, k])]) for k in range(len(imp))))
    for i, n in enumerate(names):
        val(f"epCross{n.upper()}Model", t1[i], ".2f")
        val(f"epCross{n.upper()}Solve", t2[i], ".2f")
    val("epTransitUVModel", t1[1] - t1[0], ".3f")
    val("epTransitUVSolve", t2[1] - t2[0], ".3f")
    # what a detector sees instead: the extrema of the induced charge
    tq = [float(t_e[np.argmax(Qavg[p])]) for p in range(2)]
    val("epExtremumU", tq[0], ".2f")
    val("epExtremumV", tq[1], ".2f")
    val("epExtremumShiftU", t2[0] - tq[0], ".2f")
    val("epExtremumShiftV", abs(t2[1] - tq[1]), ".2f")
    val("epLandLate", t2[2] - t1[2], ".2f")
    for i, n in enumerate(names[:2]):
        val(f"epPeak{n.upper()}Model", float(Qs[i].max()), ".4f")
        val(f"epPeak{n.upper()}Solve", float(Qavg[i].max()), ".4f")

    # ---- sensitivity: what the far-field observables say about the build
    def observables(m, v_c, vr):
        ff, _, gg = m.solve(v_c, vr, sol.ground)
        Ws = [m.solve(0.0, np.eye(len(names))[p], 0.0)[1] for p in range(len(names))]
        tu = sum(d / speed(x) for d, x in zip([Y_LAUNCH - m.rows[0]], ff[:1]))
        tuv = m.lengths[1] / speed(ff[1])
        tvw = m.lengths[2] / speed(ff[2])
        mid = 0.5 * (m.rows[:-1] + m.rows[1:])
        return np.array([ff[0] * 10.0, tuv, tvw,
                         m.average(Ws[0], mid[0]), m.average(Ws[1], mid[1]),
                         m.average(Ws[2], mid[1]), m.average(Ws[2], mid[0]), tu])

    obs_names = [r"$E_{\mathrm{drift}}$ [V/cm]", r"$T_{uv}$ [ns]", r"$T_{vw}$ [ns]",
                 r"$\langle\Psi_u\rangle(y_{uv})$", r"$\langle\Psi_v\rangle(y_{vw})$",
                 r"$\langle\Psi_w\rangle(y_{vw})$", r"$\langle\Psi_w\rangle(y_{uv})$"]
    obs_scale = np.array([1.0, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3])  # ns, and 10^-3
    base = observables(model, sol.cathode, v_rows)
    perturb = [
        (r"$y_u + 0.1$ mm", lambda: (Planes(yhi, model.rows + [0.1, 0, 0], ylo, pitch, radius), sol.cathode, v_rows)),
        (r"$y_v + 0.1$ mm", lambda: (Planes(yhi, model.rows + [0, 0.1, 0], ylo, pitch, radius), sol.cathode, v_rows)),
        (r"$r_w \times 1.1$", lambda: (Planes(yhi, model.rows, ylo, pitch, 1.1 * radius), sol.cathode, v_rows)),
        (r"$V_u + 5$ V", lambda: (model, sol.cathode, v_rows + [5, 0, 0])),
        (r"$V_v + 5$ V", lambda: (model, sol.cathode, v_rows + [0, 5, 0])),
        (r"$V_w + 5$ V", lambda: (model, sol.cathode, v_rows + [0, 0, 5])),
        (r"$V_c \times 1.01$", lambda: (model, 1.01 * sol.cathode, v_rows)),
    ]
    J = []
    lines = []
    for label, make in perturb:
        d = (observables(*make()) - base)[:7] * obs_scale
        J.append(d)
        lines.append(label + " & " + " & ".join(f"{x:+.2f}" if abs(x) >= 0.005 else "0" for x in d) + r" \\")
    V["epSensHead"] = " & ".join(obs_names)
    V["epSensRows"] = "\n".join(lines)
    V["epBaseRow"] = "nominal & " + " & ".join(
        f"{x:.1f}" if i == 0 else (f"{x:.0f}" if i < 3 else f"{x:.2f}")
        for i, x in enumerate(base[:7] * obs_scale)) + r" \\"
    # identifiability, with assumed measurement precisions per observable
    prec = np.array([1.0, 5.0, 5.0, 0.5, 0.5, 0.5, 0.5])  # V/cm, ns, ns, 10^-3
    s = np.linalg.svd(np.array(J) / prec[None, :], compute_uv=False)
    print("singular values (per unit perturbation, in units of the precision):", s)
    val("epSvdMax", s[0], ".0f")
    val("epSvdMin", s[-1], ".2f")
    val("epSvdCount", int((s > 1.0).sum()), "d")

    with open(os.path.join(figdir, "effplane.tex"), "w") as fh:
        for k, v in V.items():
            if "\n" in v or "&" in v:
                fh.write(f"\\newcommand{{\\{k}}}{{%\n{v}}}\n")
            else:
                fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    # ---- the figure
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    yy = np.linspace(ylo, yhi, 2001)
    cols = ["C0", "C1", "C2"]
    ax[0].plot(model.average(avg, yy), yy, "k-")
    for i, n in enumerate(names):
        ax[0].plot([avg[i + 1], v_rows[i]], [model.rows[i]] * 2, "-", color=cols[i], lw=2)
        ax[0].plot(v_rows[i], model.rows[i], "s", color=cols[i], label=f"{n} wires: {v_rows[i]:.0f} V")
        ax[0].plot(avg[i + 1], model.rows[i], "o", color=cols[i], mfc="none")
    ax[0].set_ylim(-10, 15)
    ax[0].set_xlim(-600, 1400)
    ax[0].set_xlabel(r"$\langle\Phi\rangle$ [V]")
    ax[0].set_ylabel("y [mm]")
    ax[0].set_title("drift: plane average (line), sheet (circle), wire (square)", fontsize=9)
    ax[0].legend(fontsize=7, loc="upper left")
    for p, n in enumerate(names):
        ax[1].plot(model.average(W[p], yy), yy, "-", color=cols[p], label=rf"$\langle\Psi_{n}\rangle$")
        ax[1].plot(1.0, model.rows[p], "s", color=cols[p])
        for y, q1, q2 in rows:
            ax[1].plot(q2[p], y, "x", color=cols[p], ms=7)
    ax[1].set_ylim(-10, 15)
    ax[1].set_xlabel("plane-averaged weighting potential")
    ax[1].set_title("weighting: sheet model (lines), 2D solve (crosses), wires (squares)", fontsize=9)
    ax[1].legend(fontsize=7)
    for p, n in enumerate(names):
        ax[2].plot(t_e, Qavg[p], "-", color=cols[p], label=f"{n}: 2D solve")
        ax[2].plot(ts, Qs[p], "--", color=cols[p], label=f"{n}: sheet model")
    ax[2].set_xlim(t1[0] - 8, t2[2] + 2.5)
    ax[2].set_ylim(-0.15, 1.05)
    ax[2].set_xlabel(r"time since launch [$\mu$s]")
    ax[2].set_ylabel("charge on one wire [e], per electron per pitch")
    ax[2].set_title("an isochronous sheet: wire-summed, impact-averaged", fontsize=9)
    ax[2].legend(fontsize=7, ncol=2)
    for a in ax:
        a.grid(alpha=0.3)
    fig.savefig(os.path.join(figdir, "effplane.pdf"))
    plt.close(fig)
    print(f"wrote {figdir}/effplane.pdf and effplane.tex")


if __name__ == "__main__":
    main(*sys.argv[1:])
