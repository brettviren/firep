"""Matplotlib rendering of solutions to PNG/PDF (or anything matplotlib knows)."""

from __future__ import annotations

import numpy as np

FIELDS = {
    "potential": ("potential", "V", "viridis"),
    "ex": ("transverse field $E_x$", "V/cm", "RdBu_r"),
    "ey": ("drift field $E_y$", "V/cm", "RdBu_r"),
    "emag": ("field magnitude $|E|$", "V/cm", "magma"),
    "residual": (r"Laplace residual $\nabla^2 V$", "V/mm$^2$", "RdBu_r"),
}


def _conductors(meta: dict) -> list[dict]:
    return meta.get("conductors", [])


def draw_conductors(ax, meta: dict, color="white", lw=0.8) -> None:
    from matplotlib.patches import Circle

    for c in _conductors(meta):
        ax.add_patch(
            Circle(
                (c["x"], c["y"]),
                c["radius"],
                facecolor="0.35",
                edgecolor=color,
                linewidth=lw,
                zorder=5,
            )
        )


def plot_solution(
    arrays: dict[str, np.ndarray],
    meta: dict,
    out: str,
    field: str = "potential",
    log: bool = False,
    symmetric: bool = False,
    contours: int = 0,
    streamlines: float = 0.0,
    drift: int = 0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    cmap: str | None = None,
    equal: bool = False,
    dpi: int = 150,
    figsize: tuple[float, float] = (6.0, 8.0),
    title: str | None = None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize, SymLogNorm

    if field not in arrays:
        raise KeyError(f"no field {field!r} in solution; have {sorted(arrays)}")

    x, y = arrays["x"], arrays["y"]
    z = np.array(arrays[field], dtype=float)
    label, unit, default_cmap = FIELDS.get(field, (field, "", "viridis"))
    if "inside" in arrays:
        z = np.where(arrays["inside"], np.nan, z)

    finite = z[np.isfinite(z)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if symmetric:
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m
    if log:
        if vmin > 0:
            norm = LogNorm(vmin=max(vmin, vmax * 1e-6), vmax=vmax)
        else:
            lin = max(abs(vmin), abs(vmax)) * 1e-4
            norm = SymLogNorm(linthresh=lin, vmin=vmin, vmax=vmax, base=10)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    mesh = ax.pcolormesh(x, y, z, norm=norm, cmap=cmap or default_cmap,
                         shading="auto", rasterized=True)
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label(f"{label} [{unit}]" if unit else label)

    if contours:
        levels = np.linspace(vmin, vmax, contours + 2)[1:-1]
        ax.contour(x, y, z, levels=levels, colors="k", linewidths=0.4, alpha=0.5)

    if (streamlines or drift) and {"ex", "ey"} <= set(arrays):
        # Electrons move along -E.
        ux, uy = -np.array(arrays["ex"]), -np.array(arrays["ey"])
        if "inside" in arrays:
            ux = np.where(arrays["inside"], np.nan, ux)
            uy = np.where(arrays["inside"], np.nan, uy)
        kw = dict(color="w", linewidth=0.7, arrowsize=0.7)
        if drift:
            top = y[-1] - 0.02 * (y[-1] - y[0])
            seeds = np.column_stack(
                (np.linspace(x[0], x[-1], drift + 2)[1:-1], np.full(drift, top))
            )
            ax.streamplot(x, y, ux, uy, start_points=seeds, integration_direction="forward",
                          broken_streamlines=False, **kw)
        else:
            ax.streamplot(x, y, ux, uy, density=streamlines, **kw)

    draw_conductors(ax, meta)
    ax.set_xlim(*(xlim if xlim else (x[0], x[-1])))
    ax.set_ylim(*(ylim if ylim else (y[0], y[-1])))
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]   (electrons drift downward)")
    if equal:
        ax.set_aspect("equal")
    ax.set_title(title if title is not None else _default_title(meta, field))
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def _default_title(meta: dict, field: str) -> str:
    name = meta.get("name", "")
    kind = meta.get("problem", "")
    bits = [b for b in (name, kind and f"{kind} solve", FIELDS.get(field, (field,))[0]) if b]
    return " - ".join(bits)


def plot_geometry(meta: dict, out: str, dpi: int = 150,
                  figsize: tuple[float, float] = (6.0, 8.0)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = meta.get("config", {})
    bounds = cfg.get("domain", {}).get("bounds", {})
    xlo, xhi = bounds.get("x", [-1, 1])
    ylo, yhi = bounds.get("y", [-1, 1])
    faces = meta.get("faces", {})

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.axhline(yhi, color="C3", lw=2)
    ax.axhline(ylo, color="C0", lw=2)
    ax.text(xhi, yhi, f"  cathode {faces.get('cathode', 0):.0f} V",
            va="bottom", ha="right", color="C3")
    ax.text(xhi, ylo, f"  ground {faces.get('ground', 0):.0f} V",
            va="top", ha="right", color="C0")
    for xb in (xlo, xhi):
        ax.axvline(xb, color="0.6", lw=1, ls="--")
    draw_conductors(ax, meta, color="k")
    seen = set()
    for c in _conductors(meta):
        if c["electrode"] in seen:
            continue
        seen.add(c["electrode"])
        ax.text(xlo, c["y"], f"{c['electrode']}  {c['potential']:.1f} V ",
                va="center", ha="right", fontsize=8)
    pad = 0.05 * (yhi - ylo)
    ax.set_xlim(xlo - 0.15 * (xhi - xlo), xhi + 0.05 * (xhi - xlo))
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"{meta.get('name', 'geometry')} (dashed = periodic boundary)")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_drift(meta: dict, tracks, landed, out: str, ylim=None, dpi: int = 150,
               figsize: tuple[float, float] = (6.0, 8.0)):
    """Draw traced electron paths, coloured by which electrode caught them."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = meta.get("config", {})
    bounds = cfg.get("domain", {}).get("bounds", {})
    xlo, xhi = bounds.get("x", [-1, 1])
    ylo, yhi = bounds.get("y", [-1, 1])

    ends = sorted(set(map(str, landed)))
    colors = {name: f"C{i}" for i, name in enumerate(ends)}
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    seen = set()
    for track, end in zip(tracks or [], landed):
        arr = np.array(track)
        # Break the line where a path wraps across the periodic edge.
        jump = np.flatnonzero(np.abs(np.diff(arr[:, 0])) > 0.5 * (xhi - xlo))
        for seg in np.split(arr, jump + 1):
            ax.plot(seg[:, 0], seg[:, 1], color=colors[str(end)], lw=0.7,
                    label=str(end) if str(end) not in seen else None)
            seen.add(str(end))
    draw_conductors(ax, meta, color="k")
    ax.axhline(yhi, color="0.5", lw=1.5)
    ax.axhline(ylo, color="0.5", lw=1.5)
    for xb in (xlo, xhi):
        ax.axvline(xb, color="0.7", lw=1, ls="--")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(*(ylim if ylim else (ylo, yhi)))
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.legend(title="caught by", fontsize=8, loc="upper right")
    ax.set_title(f"{meta.get('name', '')} - electron drift paths")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_response(arrays: dict[str, np.ndarray], meta: dict, out: str,
                  style: str = "curves", impact: int = 0, n_wires: int = 5,
                  tmax: float | None = None, tmin: float | None = None,
                  dpi: int = 150):
    """Draw a field response: induced current on each plane against time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = arrays["time"]
    cur = arrays["current"]  # (nplane, nwire, nimpact, ntime)
    offsets = arrays["offsets"]
    impacts = arrays["impact"]
    planes = meta.get("planes", [str(i) for i in range(cur.shape[0])])
    npl = len(planes)

    lo = 0 if tmin is None else int(np.searchsorted(t, tmin))
    hi = len(t) if tmax is None else int(np.searchsorted(t, tmax)) + 1
    keep = slice(lo, hi)
    t = t[keep]

    if impact >= cur.shape[2]:
        raise IndexError(
            f"impact index {impact} out of range (0..{cur.shape[2] - 1})")

    fig, axes = plt.subplots(npl, 1, figsize=(8, 2.6 * npl + 1), sharex=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ip, plane in enumerate(planes):
        ax = axes[ip]
        if style == "curves":
            half = n_wires // 2
            sel = [i for i, o in enumerate(offsets) if abs(o) <= half]
            for i in sel:
                ax.plot(t, cur[ip, i, impact, keep], lw=1.2,
                        label=f"wire {offsets[i]:+d}" if offsets[i] else "wire 0")
            ax.legend(fontsize=7, ncol=2)
            sub = f"impact {impacts[impact]:.3f} mm"
        elif style == "impacts":
            mid = int(np.argmin(np.abs(offsets)))
            for ii in range(cur.shape[2]):
                ax.plot(t, cur[ip, mid, ii, keep], lw=1.0,
                        color=plt.cm.viridis(ii / max(1, cur.shape[2] - 1)),
                        label=f"{impacts[ii]:.2f} mm" if ii in (0, cur.shape[2] - 1) else None)
            ax.legend(fontsize=7, title="impact")
            sub = "wire 0, all impacts"
        elif style == "heatmap":
            z = cur[ip, :, impact, keep]
            m = np.abs(z).max() or 1.0
            im = ax.pcolormesh(t, offsets, z, cmap="RdBu_r", vmin=-m, vmax=m,
                               shading="auto", rasterized=True)
            fig.colorbar(im, ax=ax, label="current [e/$\\mu$s]")
            ax.set_ylabel("wire offset")
            sub = f"impact {impacts[impact]:.3f} mm"
        else:
            raise KeyError(f"unknown response plot style {style!r}")

        if style != "heatmap":
            ax.axhline(0, color="0.7", lw=0.6)
            ax.set_ylabel("current [e/$\\mu$s]")
            ax.grid(alpha=0.25)
        ax.set_title(f"{plane} plane - {sub}", fontsize=10)

    axes[-1].set_xlabel("time [$\\mu$s]")
    vel = meta.get("velocity", "")
    extra = (f"{meta.get('temperature_K', '')} K" if vel == "walkowiak"
             else f"{meta.get('mobility_cm2_per_Vs', '')} cm$^2$/(V s)")
    fig.suptitle(f"field response (Ramo) - {vel} drift velocity, {extra}", fontsize=11)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_adc(arrays: dict[str, np.ndarray], meta: dict, out: str,
             style: str = "image", wire: int | None = None,
             tmin: float | None = None, tmax: float | None = None,
             cmap: str = "RdBu_r", dpi: int = 150, figsize=None):
    """ADC sample time against wire, one panel per plane.

    ``style='image'`` is the usual LArTPC event display: wire across, time down,
    colour the digitised sample.  ``style='waveforms'`` overlays the individual
    channel waveforms instead.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    adc = arrays["adc"]
    wires = arrays["wires"]
    t = arrays["adc_time"]
    planes = meta.get("planes", [str(i) for i in range(adc.shape[0])])
    npl = len(planes)

    lo = 0 if tmin is None else int(np.searchsorted(t, tmin))
    hi = len(t) if tmax is None else int(np.searchsorted(t, tmax)) + 1
    t = t[lo:hi]
    adc = adc[:, :, lo:hi]

    if style == "waveforms":
        fig, axes = plt.subplots(npl, 1, figsize=figsize or (9, 2.6 * npl + 1),
                                 sharex=True, constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ip, plane in enumerate(planes):
            ax = axes[ip]
            for iw, w in enumerate(wires):
                if np.abs(adc[ip, iw]).max() < 1:
                    continue
                ax.plot(t, adc[ip, iw], lw=0.9,
                        color=plt.cm.viridis(iw / max(1, len(wires) - 1)),
                        label=f"wire {w:+d}" if len(wires) <= 12 else None)
            ax.axhline(0, color="0.7", lw=0.6)
            ax.set_ylabel("ADC [count]")
            ax.set_title(f"{plane} plane", fontsize=10)
            ax.grid(alpha=0.25)
            if len(wires) <= 12:
                ax.legend(fontsize=7, ncol=3)
        axes[-1].set_xlabel("time [$\\mu$s]")
    else:
        fig, axes = plt.subplots(1, npl, figsize=figsize or (4.2 * npl, 6.5),
                                 sharey=True, constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ip, plane in enumerate(planes):
            ax = axes[ip]
            m = np.abs(adc[ip]).max() or 1
            im = ax.pcolormesh(wires, t, adc[ip].T, cmap=cmap, vmin=-m, vmax=m,
                               shading="nearest", rasterized=True)
            fig.colorbar(im, ax=ax, label="ADC [count]" if ip == npl - 1 else None,
                         pad=0.02)
            ax.set_xlabel("wire")
            ax.set_title(f"{plane} plane", fontsize=11)
            ax.invert_yaxis()  # time increases downward, as in an event display
        axes[0].set_ylabel("ADC sample time [$\\mu$s]")

    e = meta.get("electronics", {})
    fig.suptitle(
        f"{meta.get('electrons', 0):,.0f} e  -  {e.get('gain_mV_per_fC', '')} mV/fC, "
        f"{e.get('filter', '')} filter $\\tau$ = {e.get('tau_us', '')} $\\mu$s, "
        f"{e.get('adc_rate_MHz', '')} MHz {e.get('adc_bits', '')}-bit "
        f"$\\pm${e.get('adc_range_mV', 0) / 1000:g} V",
        fontsize=11)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_history(history: dict[str, np.ndarray], out: str, dpi: int = 150):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step = history["step"]
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True, constrained_layout=True)
    for key, lab in (("total", "total"), ("pde", "PDE"), ("face", "faces"),
                     ("surface", "conductors")):
        if key in history:
            axes[0].semilogy(step, np.maximum(history[key], 1e-300), label=lab)
    axes[0].set_ylabel("loss (dimensionless)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    for key, lab in (("rms_face_v", "face RMS"), ("rms_surface_v", "conductor RMS")):
        if key in history:
            axes[1].semilogy(step, np.maximum(history[key], 1e-300), label=lab)
    axes[1].set_ylabel("boundary error [V]")
    axes[1].set_xlabel("step")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# explanatory figures (documentation as well as diagnostics)
# --------------------------------------------------------------------------


def plot_electronics(elec, out: str, dpi: int = 150, figsize=(10.5, 3.2)):
    """Inputs and assumptions of the front end: shape, gain, quantisation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .electronics import E_IN_FC, Electronics, shaping_kernel

    dt = 0.005
    span = 6.0 * elec.tau
    t = np.arange(0.0, span, dt)

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    # (a) the shaping function, against the generic alternatives
    ax = axes[0]
    for kind, order, style, lab in (("uboone", 2, "-", "MicroBooNE cold"),
                                    ("rc2", 2, "--", "CR-RC$^2$"),
                                    ("rc2", 4, ":", "CR-RC$^4$")):
        k = shaping_kernel(kind, elec.tau, dt, order=order, n_tau=10.0,
                           normalise="peak")
        n = min(len(k), len(t))
        ax.plot(t[:n], k[:n], style, lw=1.7 if kind == "uboone" else 1.1, label=lab)
    ax.axvline(elec.tau, color="0.6", lw=0.8, ls="-.")
    ax.set_ylim(-0.06, 1.16)
    ax.text(elec.tau + 0.15, 1.08, f"$\\tau$ = {elec.tau:g} $\\mu$s",
            fontsize=8, color="0.3")
    ax.set_xlabel("time since impulse [$\\mu$s]")
    ax.set_ylabel("$s_\\tau(t)$  [unit peak]")
    ax.set_title("(a) shaping function", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # (b) response to impulses of charge: the gain calibration
    ax = axes[1]
    for q_fc in (0.5, 1.0, 2.0):
        cur = np.zeros_like(t)
        cur[2] = q_fc / E_IN_FC / dt
        v = elec.shape(cur, dt)
        ax.plot(t, v, lw=1.3, label=f"{q_fc:g} fC  ({q_fc / E_IN_FC:,.0f} e)")
    ax.axhline(elec.gain, color="0.6", lw=0.8, ls="--")
    ax.text(span * 0.98, elec.gain, f"{elec.gain} mV ", ha="right", va="bottom",
            fontsize=8, color="0.3")
    ax.set_xlabel("time since impulse [$\\mu$s]")
    ax.set_ylabel("output [mV]")
    ax.set_title(f"(b) gain {elec.gain} mV/fC (peak)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # (c) the ADC: quantisation and dynamic range
    ax = axes[2]
    lo, hi = elec.count_limits
    q = np.arange(0, 40) * elec.lsb
    ax.step(q / elec.lsb, np.rint(q / elec.lsb), where="mid", lw=1.2)
    ax.plot(q / elec.lsb, q / elec.lsb, "--", color="0.6", lw=0.8)
    ax.set_xlabel("input [LSB]")
    ax.set_ylabel("ADC count")
    ax.set_title(f"(c) {elec.adc_bits}-bit, $\\pm${elec.adc_range / 1000:g} V, "
                 f"{elec.adc_rate:g} MHz", fontsize=10)
    ax.grid(alpha=0.25)
    ax.text(0.04, 0.96,
            f"1 LSB = {elec.lsb * 1e3:.1f} $\\mu$V\n"
            f"        = {elec.lsb / elec.mv_per_electron:.1f} e\n"
            f"range = [{lo}, {hi}]\n"
            f"sample every {elec.sample_period:g} $\\mu$s",
            transform=ax.transAxes, va="top", fontsize=7.5,
            bbox=dict(fc="white", ec="0.8", alpha=0.9))

    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_drift_velocity(props, out: str, velocity: str = "walkowiak",
                        temperature: float = 87.3, mobility: float = 320.0,
                        marks=(), dpi: int = 150, figsize=(5.2, 3.4)):
    """Electron drift speed against field, with the operating points marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .response import WALKOWIAK_E_RANGE, drift_speed

    e_kv = np.logspace(np.log10(0.05), np.log10(30.0), 400)
    v = drift_speed(e_kv * 100.0, velocity, temperature, mobility)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.plot(e_kv, v, lw=1.6, label=f"{velocity}, {temperature} K")
    ax.axvspan(*WALKOWIAK_E_RANGE, color="C2", alpha=0.10,
               label="fitted range")
    for e_val, label in marks:
        vv = drift_speed(np.array([e_val * 100.0]), velocity, temperature,
                         mobility)[0]
        ax.plot([e_val], [vv], "o", ms=5, color="C3")
        ax.annotate(f"{label}\n{e_val:.2f} kV/cm, {vv:.2f} mm/$\\mu$s",
                    (e_val, vv), textcoords="offset points", xytext=(6, -14),
                    fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("electric field [kV/cm]")
    ax.set_ylabel("drift speed [mm/$\\mu$s]")
    ax.set_title("electron drift in liquid argon", fontsize=10)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7, loc="upper left")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_bias_profile(cfg, sol, out: str, dpi: int = 150, figsize=(10.0, 4.0)):
    """Plane-averaged potential and field, and the wire self-potential offset."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ylo, yhi = cfg.domain.bounds.y
    planes = cfg.electrodes_by_y
    # <V> is piecewise linear with a kink at each plane.
    ys = [yhi] + [e.plane for e in planes] + [ylo]
    vs = [sol.cathode] + [sol.plane_average[e.name] for e in planes] + [sol.ground]

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex="col",
                             constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.4, 1]})
    for col, (lo, hi, tag) in enumerate([(ylo, yhi, "full domain"),
                                         (ylo, 15.0, "wire region")]):
        axp, axe = axes[0][col], axes[1][col]
        axp.plot(ys, vs, "-o", ms=3, lw=1.4, color="C0",
                 label=r"$\langle \Phi \rangle (y)$")
        for e in planes:
            v_avg = sol.plane_average[e.name]
            v_wire = sol.electrodes[e.name]
            axp.plot([e.plane], [v_wire], "s", ms=5, color="C3")
            axp.annotate("", xy=(e.plane, v_wire), xytext=(e.plane, v_avg),
                         arrowprops=dict(arrowstyle="->", color="C3", lw=1.0))
            if col == 1:
                axp.annotate(
                    f"$\\gamma\\Lambda$ = {sol.offset(e.name):+.0f} V",
                    xy=(e.plane, 0.5 * (v_wire + v_avg)),
                    xytext=(10, 0), textcoords="offset points",
                    fontsize=7.5, color="C3", va="center")
        axp.plot([], [], "s", color="C3", label="wire potential $V_i$")
        if col == 1:
            near = [sol.plane_average[e.name] for e in planes]
            near += [sol.electrodes[e.name] for e in planes] + [sol.ground]
            pad = 0.35 * (max(near) - min(near))
            axp.set_ylim(min(near) - pad, max(near) + pad)
        axp.set_ylabel("potential [V]")
        axp.set_title(tag, fontsize=10)
        axp.grid(alpha=0.25)
        if col == 0:
            axp.legend(fontsize=7)

        for g in sol.gaps:
            axe.plot([g.y_lo, g.y_hi], [g.field, g.field], lw=1.6, color="C1")
        for e in planes:
            axe.axvline(e.plane, color="0.85", lw=0.8)
        axe.axhline(0, color="0.6", lw=0.8)
        axe.set_ylabel(r"$\langle E_y \rangle$ [V/cm]")
        axe.set_xlabel("y [mm]   (electrons drift toward $-y$)")
        axe.set_xlim(lo, hi)
        axe.grid(alpha=0.25)

    fig.suptitle(r"$V_i = \langle \Phi \rangle(y_i) + \gamma_i \Lambda_i$: "
                 "a wire is not its plane", fontsize=11)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_basis(model, geom, out: str, dpi: int = 150, figsize=(10.5, 6.6),
               index: int = 0):
    """The analytic basis columns, and what each actually contributes.

    The top row is the basis *functions* -- the columns of the design matrix of
    the warm start, each on its own colour scale, which is what shows their
    shape.  The bottom row is the same column multiplied by its fitted
    coefficient, all three on one shared scale in volts, which is what shows
    their size.  The distinction matters: the transverse dipole column has a
    large left-right structure and a coefficient of essentially zero, because
    the geometry is mirror-symmetric about every wire and there is no transverse
    external field for it to cancel.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from matplotlib.colors import SymLogNorm

    nc = len(geom.conductors)
    ic = index % nc
    name = geom.conductors[ic].electrode
    cx, cy = geom.cx[ic], geom.cy[ic]
    x = np.linspace(geom.xlo, geom.xhi, 301)
    y = np.linspace(cy - 2.5, cy + 2.5, 301)
    xx, yy = np.meshgrid(x, y)
    pts = torch.tensor(np.column_stack((xx.ravel(), yy.ravel())),
                       dtype=next(model.parameters()).dtype)
    with torch.no_grad():
        cols = model.baseline.basis(pts).numpy()
        coeff = model.baseline.coeff.detach().numpy()

    zs = [cols[:, ic + k * nc].reshape(len(y), len(x)) for k in range(3)]
    cs = [float(coeff[ic + k * nc]) for k in range(3)]
    sub = f"{{{name}}}"
    shapes = [rf"\tilde\psi_{sub}",
              rf"\partial\tilde\psi_{sub} / \partial x_{sub}",
              rf"\partial\tilde\psi_{sub} / \partial y_{sub}"]
    syms = [rf"c_{sub}", rf"d_{sub}", rf"e_{sub}"]

    def _fmt(v):
        if not v:
            return "0"
        if abs(v) >= 0.01:
            return f"{v:.3g}"
        e = int(np.floor(np.log10(abs(v))))
        return rf"{v / 10 ** e:.1f}\times 10^{{{e}}}"

    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    mshare = np.nanpercentile(np.abs(np.array([c * z for c, z in zip(cs, zs)])),
                              99.9)
    for k in range(3):
        for row in (0, 1):
            ax = axes[row, k]
            z = zs[k] if row == 0 else cs[k] * zs[k]
            m = np.nanpercentile(np.abs(z), 99.5) if row == 0 else mshare
            m = max(m, 1e-30)
            thr = (0.02 if row == 0 else 0.002) * m
            im = ax.pcolormesh(x, y, z, cmap="RdBu_r", shading="auto",
                               rasterized=True,
                               norm=SymLogNorm(linthresh=thr, vmin=-m,
                                               vmax=m, base=10))
            if row == 0 or k == 2:
                fig.colorbar(im, ax=ax, pad=0.02)
            ax.add_patch(plt.Circle((cx, cy), geom.cr[ic], fc="0.3", ec="k",
                                    lw=0.6))
            ax.set_aspect("equal")
            if row == 1:
                ax.set_xlabel("x [mm]")
                ax.set_title(f"${syms[k]}\\,{shapes[k]}$"
                             f"    (${syms[k]} = {_fmt(cs[k])}$ V)", fontsize=9)
            else:
                ax.set_title(f"${shapes[k]}$", fontsize=10)
    axes[0, 0].set_ylabel("y [mm]")
    axes[1, 0].set_ylabel("y [mm]")
    fig.suptitle(f"analytic basis for conductor {name}\n"
                 "top: the columns, each on its own scale.   "
                 "bottom: each times its fitted coefficient, shared scale [V]",
                 fontsize=10)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_response_spectrum(arrays: dict, meta: dict, out: str, elec=None,
                           sigma_t: float = 0.0, impact: int = 0,
                           dpi: int = 150, figsize=(10.5, 3.6)):
    """Magnitude spectra of the response waveforms, against the sampling rate.

    The question this answers is whether the response tick resolves the
    waveforms.  Panel (a) shows the raw spectra with the tick's Nyquist
    frequency marked; panel (b) shows what survives the low-pass filters that
    every physical signal passes through downstream -- diffusion in the drift
    and the front-end shaper -- which is what actually decides the answer.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cur = arrays["current"]
    t = arrays["time"]
    tick = float(t[1] - t[0])
    planes = meta.get("planes", [str(i) for i in range(cur.shape[0])])
    iw0 = int(np.argmin(np.abs(arrays["offsets"])))
    n = cur.shape[-1]
    freq = np.fft.rfftfreq(n, d=tick)  # MHz
    nyq = 0.5 / tick

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    ax = axes[0]
    for ip, p in enumerate(planes):
        mag = np.abs(np.fft.rfft(cur[ip, iw0, impact, :]))
        mag /= mag.max()
        ax.semilogy(freq, np.maximum(mag, 1e-6), lw=1.2, label=f"{p} plane")
    ax.axvline(nyq, color="C3", lw=1.0, ls="--")
    ax.text(0.97, 0.06, f"Nyquist = {nyq:g} MHz\n({tick * 1000:.0f} ns tick)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            color="C3", bbox=dict(fc="white", ec="0.85", alpha=0.9))
    ax.set_xlim(0, nyq)
    ax.set_ylim(1e-4, 2)
    ax.set_xlabel("frequency [MHz]")
    ax.set_ylabel("$|R(f)|$ / peak")
    ax.set_title("(a) raw field response", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    # The two physical low-pass stages.
    filt = np.ones_like(freq)
    if sigma_t > 0:
        diff = np.exp(-2.0 * np.pi**2 * freq**2 * sigma_t**2)
        ax.semilogy(freq, np.maximum(diff, 1e-12), "--", color="0.45", lw=1.1,
                    label=f"diffusion, $\\sigma_t$ = {sigma_t:.2f} $\\mu$s")
        filt = filt * diff
    if elec is not None:
        k = elec.kernel(tick)
        sh = np.abs(np.fft.rfft(k, n=n))
        sh /= sh[0]
        ax.semilogy(freq, np.maximum(sh, 1e-12), ":", color="0.2", lw=1.3,
                    label=f"{elec.filter} shaper, $\\tau$ = {elec.tau:g} $\\mu$s")
        filt = filt * sh
    for ip, p in enumerate(planes):
        mag = np.abs(np.fft.rfft(cur[ip, iw0, impact, :]))
        mag = mag / mag.max() * filt
        ax.semilogy(freq, np.maximum(mag, 1e-12), lw=1.2, label=f"{p} plane, filtered")
    ax.axvline(nyq, color="C3", lw=1.0, ls="--")
    ax.set_xlim(0, nyq)
    ax.set_ylim(1e-10, 2)
    ax.set_xlabel("frequency [MHz]")
    ax.set_ylabel("relative amplitude")
    ax.set_title("(b) after diffusion and shaping", fontsize=10)
    ax.legend(fontsize=6.5, loc="lower left")
    ax.grid(alpha=0.25, which="both")

    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_electrode_plan(groups, out: str | None, xlim, zlim, title="", n=420,
                        dpi: int = 150, figsize=(5.0, 5.0), legend=True, ax=None):
    """Looking along the drift: the footprint of each electrode plane.

    ``groups`` is a list of ``(label, colour, shapes, y)`` or
    ``(label, colour, shapes, y, style)``.  Conductors (``style='fill'``, the
    default) are rendered by evaluating their signed distance functions on a
    grid at that plane's own depth and outlining the zero contour, so the
    picture is drawn from the same geometry the solver uses.  Apertures
    (``style='aperture'``) are drawn as the openings they are: for a hole the
    conductor is the *sheet*, which would otherwise flood the figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    x = np.linspace(*xlim, n)
    z = np.linspace(*zlim, n)
    xx, zz = np.meshgrid(x, z)
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for group in groups:
        label, colour, shapes, y = group[:4]
        style = group[4] if len(group) > 4 else "fill"
        if style == "aperture":
            for sh in shapes:
                c = np.asarray(sh.centre, float)
                ax.add_patch(Circle((c[0], c[2]), sh.radius, facecolor="white",
                                    edgecolor="0.25", lw=0.8, zorder=6))
            ax.plot([], [], "o", mfc="white", mec="0.25", ms=7,
                    label=f"{label}  (y = {y:g} mm)")
            continue
        pts = np.column_stack((xx.ravel(), np.full(xx.size, y), zz.ravel()))
        d = np.full(xx.size, np.inf)
        for sh in shapes:
            d = np.minimum(d, sh.sdf(pts))
        d = d.reshape(xx.shape)
        ax.contourf(x, z, d, levels=[-1e9, 0.0], colors=[colour], alpha=0.40)
        ax.contour(x, z, d, levels=[0.0], colors=[colour], linewidths=0.8)
        ax.plot([], [], color=colour, lw=4, alpha=0.6,
                label=f"{label}  (y = {y:g} mm)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    if title:
        ax.set_title(title, fontsize=10)
    if legend:
        ax.legend(fontsize=6.8, loc="upper right", framealpha=0.92)
    if fig is not None:
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
    return out


def plot_response_table(table: dict, out: str, tmin=None, tmax=None,
                        dpi: int = 150, figsize=(9.2, 8.8),
                        linthresh_frac: float = 1e-3, suptitle=None):
    """The impact/time tabulation of :func:`firep.response.impact_table`.

    One panel per plane: current induced on a single wire as a function of the
    signed offset of the electron's launch point from that wire, and of time.
    Colour is a signed logarithm, which is the only scale on which a collection
    peak and a long-range induction tail are visible in the same picture.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    u = np.asarray(table["impact"], dtype=float)
    t = np.asarray(table["time"], dtype=float)
    tab = np.asarray(table["table"], dtype=float)
    planes = list(table["planes"])
    pitch = float(table.get("pitch", 5.0))

    keep = np.ones(len(t), dtype=bool)
    if tmin is not None:
        keep &= t >= tmin
    if tmax is not None:
        keep &= t <= tmax
    t, tab = t[keep], tab[:, :, keep]

    fig, axes = plt.subplots(len(planes), 1, figsize=figsize, sharex=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ip, (ax, name) in enumerate(zip(axes, planes)):
        z = tab[ip]
        m = float(np.abs(z).max()) or 1.0
        im = ax.pcolormesh(t, u, z, cmap="RdBu_r", shading="auto",
                           rasterized=True,
                           norm=SymLogNorm(linthresh=linthresh_frac * m,
                                           vmin=-m, vmax=m, base=10))
        cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.045)
        cb.ax.set_title(r"$e/\mu$s", fontsize=8, pad=4)
        cb.ax.tick_params(labelsize=7)
        for k in range(int(u.min() // pitch), int(u.max() // pitch) + 2):
            y = k * pitch
            if u.min() <= y <= u.max():
                ax.axhline(y, color="0.5", lw=0.25, alpha=0.5)
        ax.set_ylim(u.min(), u.max())
        ax.set_yticks(np.arange(-50, 51, 25))
        ax.set_ylabel("impact offset [mm]")
        ax.set_title(f"plane {name}   (peak $|i|$ = {m:.3g} "
                     r"$e/\mu$s)", fontsize=10, loc="left")
    axes[-1].set_xlabel(r"time [$\mu$s]")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
