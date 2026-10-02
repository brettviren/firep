"""Command line interface for ``firep``."""

from __future__ import annotations

import json
import os
import sys

import click
import numpy as np

from . import bias as bias_mod
from . import perf as perf_mod
from . import config as config_mod
from . import plot as plot_mod
from . import sample as sample_mod
from . import train as train_mod

CONFIG_HELP = "YAML configuration file; omit for the built-in 2D three-plane wire model."
SET_HELP = "Override a config value, e.g. --set train.steps=5000 --set electrodes.0.bias=-100"


def config_options(fn):
    fn = click.option("-c", "--config", "config_path", type=click.Path(exists=True,
                      dir_okay=False), default=None, help=CONFIG_HELP)(fn)
    fn = click.option("-s", "--set", "sets", multiple=True, metavar="KEY=VALUE",
                      help=SET_HELP)(fn)
    return fn


def _load(config_path, sets):
    try:
        return config_mod.load_with_overrides(config_path, sets)
    except config_mod.ConfigError as err:
        raise click.ClickException(str(err)) from err


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="firep")
def main():
    """SIREN solutions of the Laplace equation for LArTPC field response.

    A typical first session:

    \b
      firep config -o my.yaml          # write the default model out
      firep bias -c my.yaml            # check the transparency/bias arithmetic
      firep train -c my.yaml -o m.pt   # fit the SIREN
      firep check m.pt                 # how well are PDE and BCs satisfied?
      firep verify m.pt                # cross-check against untrained-on physics
      firep drift m.pt -n 200          # trace electrons; measure transparency
      firep sample m.pt -o sol.npz     # potential and field on a grid
      firep plot sol.npz -o sol.png    # look at it
    """


# --------------------------------------------------------------------------


@main.command("config")
@config_options
@click.option("-o", "--output", type=click.Path(dir_okay=False), default=None,
              help="Write to this file instead of stdout.")
def config_cmd(config_path, sets, output):
    """Print the fully expanded configuration."""
    cfg = _load(config_path, sets)
    text = config_mod.to_yaml(cfg)
    if output:
        with open(output, "w") as fp:
            fp.write(text)
        click.echo(f"wrote {output}")
    else:
        click.echo(text, nl=False)


@main.command("bias")
@config_options
def bias_cmd(config_path, sets):
    """Report the derived electrode potentials and transparency check."""
    cfg = _load(config_path, sets)
    try:
        sol = bias_mod.solve(cfg)
    except config_mod.ConfigError as err:
        raise click.ClickException(str(err)) from err
    click.echo(bias_mod.report(cfg, sol))


@main.command("geometry")
@config_options
@click.option("-o", "--output", type=click.Path(dir_okay=False), default=None,
              help="Also draw the layout to this PNG/PDF.")
@click.option("--dpi", default=150, show_default=True)
def geometry_cmd(config_path, sets, output, dpi):
    """Describe (and optionally draw) the conductor layout.

    This works for any configuration the schema accepts, including the 3D and
    strip/hole/pad geometries the Laplace solver cannot yet solve: the point of
    separating the two is to be able to write a geometry down and look at it
    before anything can solve it.
    """
    from . import shapes as shapes_mod
    from .geometry import Geometry

    cfg = _load(config_path, sets)
    try:
        cfg.require_solvable()
    except config_mod.ConfigError as err:
        click.echo(shapes_mod.describe(cfg))
        click.echo(f"\nnot solvable yet: {err}")
        if output:
            b = cfg.domain.bounds
            groups = [
                (f"{e.name} ({e.kind})", f"C{i}",
                 shapes_mod.lattice_shapes(e, cfg.domain.dimension), e.plane,
                 "aperture" if e.kind == "hole" else "fill")
                for i, e in enumerate(cfg.electrodes_by_y)
            ]
            plot_mod.plot_electrode_plan(
                groups, output, tuple(b.x), tuple(b.z or b.x), title=cfg.name,
                dpi=dpi)
            click.echo(f"wrote {output}")
        return

    sol = bias_mod.solve(cfg)
    geom = Geometry(cfg, sol)
    click.echo(geom.describe())
    if output:
        plot_mod.plot_geometry(sample_mod.metadata(cfg, geom), output, dpi=dpi)
        click.echo(f"wrote {output}")


# --------------------------------------------------------------------------


@main.command("train")
@config_options
@click.option("-o", "--output", type=click.Path(dir_okay=False), default="model.pt",
              show_default=True, help="Checkpoint to write.")
@click.option("--steps", type=int, default=None, help="Override train.steps.")
@click.option("--device", default=None, help="Override train.device (auto|cpu|cuda).")
@click.option("--precision", type=click.Choice(["float32", "float64"]), default=None)
@click.option("--baseline", type=click.Choice(["none", "linear", "wires"]), default=None,
              help="Override model.baseline.mode.")
@click.option("--seed", type=int, default=None, help="Override train.seed and sampling.seed.")
@click.option("-q", "--quiet", is_flag=True, help="Suppress the per-step log.")
def train_cmd(config_path, sets, output, steps, device, precision, baseline, seed, quiet):
    """Fit the SIREN to the Laplace problem and save a checkpoint."""
    extra = []
    if steps is not None:
        extra.append(f"train.steps={steps}")
    if device is not None:
        extra.append(f"train.device={device}")
    if precision is not None:
        extra.append(f"train.precision={precision}")
    if baseline is not None:
        extra.append(f"model.baseline.mode={baseline}")
    if seed is not None:
        extra += [f"train.seed={seed}", f"sampling.seed={seed}"]
    cfg = _load(config_path, tuple(sets) + tuple(extra))

    sol = bias_mod.solve(cfg)
    if not quiet:
        click.echo(bias_mod.report(cfg, sol))
        click.echo("")
        click.echo(f"device: {train_mod.resolve_device(cfg.train.device)}  "
                   f"precision: {cfg.train.precision}  baseline: {cfg.model.baseline.mode}")

    log = None if quiet else (lambda msg: click.echo(msg))
    label = cfg.name
    if cfg.problem.kind == "weighting":
        label += f" ({cfg.problem.weighting.electrode})"
    verb = "warm start" if cfg.train.steps == 0 else "train"
    with perf_mod.step(f"{verb} {label}", "train", steps=cfg.train.steps,
                       problem=cfg.problem.kind,
                       precision=cfg.train.precision,
                       points=cfg.sampling.n_bulk + cfg.sampling.n_near) as d:
        model, geom, history, sol = train_mod.train(cfg, sol, on_log=log)
        d["conductors"] = len(geom.conductors)
        d["params"] = sum(q.numel() for q in model.net.parameters())
        d["wall_harmonics"] = int(model.baseline.n_wall)
    train_mod.save(output, cfg, model, sol, history)
    click.echo(f"wrote {output}")


@main.command("check")
@click.argument("checkpoint", type=click.Path(exists=True, dir_okay=False))
@click.option("-n", "--points", default=200000, show_default=True,
              help="Collocation points used for the residual check.")
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON.")
@click.option("--device", default=None, help="Evaluate on this device.")
def check_cmd(checkpoint, points, as_json, device):
    """Measure how well a checkpoint satisfies the PDE and the boundary conditions."""
    cfg, geom, model, sol = train_mod.load(checkpoint, device)
    diag = sample_mod.diagnostics(cfg, geom, model, points)
    if as_json:
        click.echo(json.dumps(diag, indent=2))
        return
    click.echo(f"collocation points: {diag['n_collocation']}")
    click.echo(f"Laplace residual   rms {diag['residual_rms']:.4e} V/mm^2   "
               f"max {diag['residual_max']:.4e} V/mm^2")
    click.echo(f"  dimensionless (x pitch^2 / span): {diag['residual_rms_scaled']:.4e}")
    for key in ("ground", "cathode"):
        click.echo(f"face {key:<8} rms {diag[f'face_{key}_rms_V']:9.4f} V   "
                   f"max {diag[f'face_{key}_max_V']:9.4f} V")
    if "surface_rms_V" in diag:
        click.echo(f"conductors    rms {diag['surface_rms_V']:9.4f} V   "
                   f"max {diag['surface_max_V']:9.4f} V")
        for name, rec in diag["conductors"].items():
            click.echo(f"  {name:<8} rms {rec['rms_V']:9.4f} V   max {rec['max_V']:9.4f} V"
                       f"   mean {rec['mean_V']:+9.4f} V")


@main.command("verify")
@click.argument("checkpoint", type=click.Path(exists=True, dir_okay=False))
@click.option("--device", default=None)
def verify_cmd(checkpoint, device):
    """Cross-check a solution against physics it was never trained on.

    Compares the transverse-averaged drift field in each gap with the design
    value, and the line charge on each plane obtained two independent ways:
    from the jump in the averaged field (Gauss's law) and from the fitted
    analytic monopole coefficient.
    """
    from . import physics

    cfg, geom, model, sol = train_mod.load(checkpoint, device)

    if cfg.problem.kind == "drift":
        click.echo("transverse-averaged E_y at each gap midpoint [V/cm]")
        click.echo(f"  {'gap':<18}{'y [mm]':>9}{'design':>12}{'solved':>12}{'rel':>10}")
        for rec in physics.gap_field_check(cfg, geom, model):
            rel = (rec["measured"] - rec["expected"]) / rec["expected"]
            click.echo(f"  {rec['gap']:<18}{rec['y']:9.2f}{rec['expected']:12.2f}"
                       f"{rec['measured']:12.2f}{rel:10.2e}")
        click.echo("")

    click.echo("line charge per plane, lambda/(2 pi eps0) [V]")
    click.echo(f"  {'plane':<8}{'dy [mm]':>9}{'E above':>11}{'E below':>11}"
               f"{'Gauss':>12}{'model':>12}{'rel':>10}")
    for rec in physics.gauss_check(cfg, geom, model):
        cm = rec.get("coeff_model", float("nan"))
        rel = (cm - rec["coeff_gauss"]) / rec["coeff_gauss"] if rec["coeff_gauss"] else float("nan")
        click.echo(f"  {rec['electrode']:<8}{rec['offset_mm']:9.2f}{rec['e_above']:11.2f}"
                   f"{rec['e_below']:11.2f}{rec['coeff_gauss']:12.4f}{cm:12.4f}{rel:10.2e}")


@main.command("drift")
@click.argument("checkpoint", type=click.Path(exists=True, dir_okay=False))
@click.option("-n", "--count", default=64, show_default=True,
              help="Electrons launched, spread evenly across one period.")
@click.option("-o", "--output", type=click.Path(dir_okay=False), default=None,
              help="Draw the traced paths to this PNG/PDF.")
@click.option("--ylim", nargs=2, type=float, default=None)
@click.option("--device", default=None)
def drift_cmd(checkpoint, count, output, ylim, device):
    """Trace electron drift paths and measure the transparency of each plane."""
    from . import physics

    cfg, geom, model, sol = train_mod.load(checkpoint, device)
    landed, tracks, _ = physics.drift_paths(cfg, geom, model, n=count, store=bool(output))
    click.echo(physics.transparency_report(cfg, geom, landed))
    if output:
        plot_mod.plot_drift(sample_mod.metadata(cfg, geom), tracks, landed, output,
                            ylim=ylim or None)
        click.echo(f"wrote {output}")


@main.command("response")
@config_options
@click.option("-d", "--drift", "drift_path", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Checkpoint of the drift (real-potential) solution.")
@click.option("-w", "--weight", "weight_paths", multiple=True, required=True,
              metavar="[PLANE=]PATH",
              help="Weighting checkpoint; repeat once per sensing plane. "
                   "The plane is read from the checkpoint unless given as PLANE=PATH.")
@click.option("-o", "--output", type=click.Path(dir_okay=False), default="response.npz",
              show_default=True)
@click.option("--plot", "plot_path", type=click.Path(dir_okay=False), default=None,
              help="Also draw the response curves to this PNG/PDF.")
@click.option("--impacts", type=int, default=None, help="Override response.impacts.")
@click.option("--tick", type=float, default=None, help="Override response.tick [us].")
@click.option("--y-start", type=float, default=None, help="Override response.y_start [mm].")
@click.option("--velocity", type=click.Choice(["walkowiak", "constant"]), default=None)
@click.option("--temperature", type=float, default=None, help="Liquid argon temperature [K].")
@click.option("--mobility", type=float, default=None,
              help="Electron mobility [cm^2/(V s)] for --velocity constant.")
@click.option("--wires", type=int, default=None, help="Wire offsets to report (odd count).")
@click.option("--device", default=None)
def response_cmd(config_path, sets, drift_path, weight_paths, output, plot_path,
                 impacts, tick, y_start, velocity, temperature, mobility, wires, device):
    """Combine a drift solution and weighting solutions into a field response.

    Drifts electrons through the real field and applies the Shockley-Ramo
    theorem against each plane's weighting potential, giving the induced current
    on every wire as a function of time and of where the electron started.

    \b
      firep response -d runs/drift.pt \\
          -w runs/weight-u.pt -w runs/weight-v.pt -w runs/weight-w.pt \\
          -o runs/response.npz --plot runs/response.png
    """
    from . import response as resp_mod

    cfg = _load(config_path, sets)
    rc = cfg.response
    paths: dict[str, str] | list[str]
    if any("=" in w for w in weight_paths):
        if not all("=" in w for w in weight_paths):
            raise click.ClickException(
                "either give every --weight as PLANE=PATH or none of them")
        paths = dict(w.split("=", 1) for w in weight_paths)
    else:
        paths = list(weight_paths)

    try:
        cm = resp_mod.load_combined(
            drift_path, paths, device=device,
            velocity=velocity or rc.velocity,
            temperature=temperature if temperature is not None else rc.temperature,
            mobility=mobility if mobility is not None else rc.mobility,
        )
        with perf_mod.step("field response", "response") as d:
            result = resp_mod.response(
                cm,
                impacts=impacts if impacts is not None else rc.impacts,
                y_start=y_start if y_start is not None else rc.y_start,
                tick=tick if tick is not None else rc.tick,
                max_time=rc.max_time,
                n_wires=wires if wires is not None else rc.wires,
                half_pitch=rc.half_pitch,
                substeps=rc.substeps,
                impact_nudge=rc.impact_nudge,
            )
            d["impacts"] = len(result.impact)
            d["wires"] = len(result.offsets)
            d["ticks"] = len(result.time)
    except (ValueError, KeyError) as err:
        raise click.ClickException(str(err)) from err

    click.echo(resp_mod.report(cm, result))
    resp_mod.save_npz(output, result, cm)
    click.echo(f"\nwrote {output}")
    if plot_path:
        plot_mod.plot_response(*resp_mod.load_npz(output), plot_path)
        click.echo(f"wrote {plot_path}")


@main.command("learn-response")
@config_options
@click.option("-d", "--drift", "drift_path", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Checkpoint of the drift (real-potential) solution.")
@click.option("-w", "--weight", "weight_paths", multiple=True, required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Weighting checkpoint; repeat once per sensing plane.")
@click.option("-o", "--output", type=click.Path(dir_okay=False), default="direct.pt",
              show_default=True)
@click.option("--steps", type=int, default=None, help="Override direct.steps.")
@click.option("--device", default=None, help="Override direct.device.")
def learn_response_cmd(config_path, sets, drift_path, weight_paths, output, steps, device):
    """Learn the field response directly, from the transport equation.

    Trains a SIREN on the residual of the backward transport equation for the
    induced charge (technical note Sec. 15.4).  The drift and weighting
    solutions enter only as its coefficients; no response is a target, so the
    traced response of `firep response` remains an independent check.

    \b
      firep learn-response -c examples/drift-2d-pdsp.yaml -d runs/pdsp/drift.pt \\
          -w runs/pdsp/weight-u.pt -w runs/pdsp/weight-v.pt -w runs/pdsp/weight-w.pt \\
          -o runs/pdsp/direct.pt
    """
    from . import direct as direct_mod
    from . import response as resp_mod

    cfg = _load(config_path, sets)
    spec = cfg.direct
    if steps is not None:
        spec.steps = steps
    if device is not None:
        spec.device = device
    try:
        cm = resp_mod.load_combined(drift_path, list(weight_paths), device=spec.device,
                                    velocity=cfg.response.velocity,
                                    temperature=cfg.response.temperature,
                                    mobility=cfg.response.mobility)
        with perf_mod.step("learn response", "train", steps=spec.steps) as d:
            model, cell, pool, hist = direct_mod.fit(cm, spec, log=click.echo)
            d["outputs"] = model.nout
            d["pool"] = len(pool.xy)
    except (ValueError, KeyError) as err:
        raise click.ClickException(str(err)) from err
    direct_mod.save(output, model, spec, hist, drift_path, list(weight_paths))
    click.echo(f"wrote {output}")


@main.command("direct-response")
@config_options
@click.argument("checkpoint", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=click.Path(dir_okay=False),
              default="response-direct.npz", show_default=True)
@click.option("--impacts", type=int, default=None, help="Override response.impacts.")
@click.option("--tick", type=float, default=None, help="Override response.tick [us].")
@click.option("--y-start", type=float, default=None,
              help="Launch height [mm]; default just below the cathode, as `firep response`.")
@click.option("--device", default=None)
def direct_response_cmd(config_path, sets, checkpoint, output, impacts, tick, y_start, device):
    """Evaluate a learned response on the grid `firep response` uses.

    Writes the same `.npz` layout, so `plot-response` and `response-table`
    work on it unchanged.  `current` is the charge form (the change of the
    learned charge over each tick); `current_field` is the network's own dQ/dt.
    """
    import torch

    from . import direct as direct_mod
    from . import response as resp_mod

    cfg = _load(config_path, sets)
    rc = cfg.response
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    dev = device or blob["spec"]["device"]
    cm = resp_mod.load_combined(blob["drift"], blob["weights"], device=dev,
                                velocity=rc.velocity, temperature=rc.temperature,
                                mobility=rc.mobility)
    model, cell, spec, blob, tdev = direct_mod.load(checkpoint, cm, device=dev)
    n = impacts if impacts is not None else rc.impacts
    x0 = np.linspace(0.0, 0.5 * cell.pitch, n)
    if y_start is None:
        top = max(e.plane for e in cm.cfg.electrodes)
        y_start = cm.geom.yhi - 0.01 * (cm.geom.yhi - top)
    with perf_mod.step("direct response", "response") as d:
        res = direct_mod.response(model, cm, x0, float(y_start),
                                  tick if tick is not None else rc.tick, tdev)
        d["impacts"] = len(x0)
    resp_mod.save_npz(output, res, cm)
    iw = int(np.flatnonzero(res.offsets == 0)[0])
    click.echo("total induced charge on the central wire [e] per plane and impact")
    for ip, p in enumerate(res.planes):
        click.echo(f"  {p}  " + " ".join(f"{q:8.4f}" for q in res.integrated[ip, iw]))
    click.echo(f"wrote {output}")


@main.command("plot-response")
@click.argument("response_file", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=click.Path(dir_okay=False), default="response.png",
              show_default=True)
@click.option("--style", type=click.Choice(["curves", "heatmap", "impacts"]),
              default="curves", show_default=True,
              help="curves: current vs time per wire.  heatmap: wire x time.  "
                   "impacts: the central wire at each starting position.")
@click.option("--impact", type=int, default=0, show_default=True,
              help="Impact index to draw for --style curves/heatmap.")
@click.option("--wires", type=int, default=5, show_default=True,
              help="Number of wire offsets to draw for --style curves.")
@click.option("--tmax", type=float, default=None, help="Upper limit of the time axis [us].")
@click.option("--tmin", type=float, default=None, help="Lower limit of the time axis [us].")
@click.option("--dpi", default=150, show_default=True)
def plot_response_cmd(response_file, output, style, impact, wires, tmax, tmin, dpi):
    """Render a field response computed by ``firep response``."""
    from . import response as resp_mod

    arrays, meta = resp_mod.load_npz(response_file)
    try:
        plot_mod.plot_response(arrays, meta, output, style=style, impact=impact,
                               n_wires=wires, tmax=tmax, tmin=tmin, dpi=dpi)
    except (IndexError, KeyError) as err:
        raise click.ClickException(str(err)) from err
    click.echo(f"wrote {output}")


@main.command("response-table")
@click.argument("response_file", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=click.Path(dir_okay=False),
              default="response-table.npz", show_default=True,
              help="Where to write the tabulation.")
@click.option("--per-side", type=int, default=5, show_default=True,
              help="Impact positions kept on each side of every wire centre.")
@click.option("--plot", "plot_path", type=click.Path(dir_okay=False), default=None,
              help="Also draw the table here (PNG or PDF).")
@click.option("--tmin", type=float, default=None, help="Lower limit of the time axis [us].")
@click.option("--tmax", type=float, default=None, help="Upper limit of the time axis [us].")
@click.option("--dpi", default=150, show_default=True)
def response_table_cmd(response_file, output, per_side, plot_path, tmin, tmax, dpi):
    """Re-index a response as current vs (impact offset, time).

    The result is a single function per plane, covering every wire region the
    weighting solve reaches, on a grid of ``per_side`` launch positions either
    side of each wire centre.
    """
    from . import response as resp_mod

    arrays, meta = resp_mod.load_npz(response_file)
    try:
        table = resp_mod.impact_table(arrays, meta, per_side=per_side)
    except (ValueError, KeyError) as err:
        raise click.ClickException(str(err)) from err
    np.savez_compressed(
        output, impact=table["impact"], time=table["time"],
        table=table["table"], planes=np.array(table["planes"]),
        pitch=table["pitch"], per_side=table["per_side"])
    click.echo(f"wrote {output}  table{tuple(table['table'].shape)} "
               f"(plane, impact, tick)")
    if plot_path:
        plot_mod.plot_response_table(table, plot_path, tmin=tmin, tmax=tmax,
                                     dpi=dpi)
        click.echo(f"wrote {plot_path}")


@main.command("signal")
@config_options
@click.option("-r", "--response", "response_file", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Field response from `firep response`.")
@click.option("-o", "--output", type=click.Path(dir_okay=False), default="signal.npz",
              show_default=True)
@click.option("--plot", "plot_path", type=click.Path(dir_okay=False), default=None,
              help="Also draw the ADC image to this PNG/PDF.")
@click.option("--drift", type=float, default=1000.0, show_default=True,
              help="Drift distance of the track centre to the response plane [mm].")
@click.option("--length", type=float, default=60.0, show_default=True,
              help="Track length [mm].")
@click.option("--angle", type=float, default=45.0, show_default=True,
              help="Track angle from the transverse axis [degrees]; "
                   "0 = parallel to the wire plane, 90 = along the drift.")
@click.option("--x-centre", type=float, default=0.0, show_default=True)
@click.option("--per-mm", type=float, default=5000.0, show_default=True,
              help="Ionization electrons per mm of track.")
@click.option("--step", type=float, default=0.1, show_default=True,
              help="Track sampling step [mm].")
@click.option("--lifetime", type=float, default=None,
              help="Electron lifetime [us]; 0 or negative disables absorption.")
@click.option("--gain", type=float, default=None, help="Amplifier gain [mV/fC].")
@click.option("--tau", type=float, default=None, help="Filter time constant [us].")
@click.option("--filter", "filt", type=click.Choice(["none", "rc", "rc2", "gaussian", "uboone"]),
              default=None)
@click.option("--mode", type=click.Choice(["shaper", "integrator"]), default=None,
              help="Front-end model: shaping amplifier (default) or charge integrator.")
def signal_cmd(config_path, sets, response_file, output, plot_path, drift, length,
               angle, x_centre, per_mm, step, lifetime, gain, tau, filt, mode):
    """Simulate what the wires read out for a line ionization source.

    Drifts an ideal straight track to the response plane (diffusing and
    absorbing along the way), folds the resulting Gaussian electron groups
    through the field response, and pushes the induced current through the
    amplifier, anti-aliasing filter and ADC.

    \b
      firep signal -r runs/response.npz -o runs/signal.npz \\
          --drift 1000 --length 60 --angle 45 --plot runs/signal.png
    """
    from . import electronics as elec_mod
    from . import ionization as ion_mod
    from . import response as resp_mod
    from . import signal as sig_mod

    cfg = _load(config_path, sets)
    ic, ec = cfg.ionization, cfg.electronics

    speed = ic.drift_speed
    if speed == "auto":
        speed = float(resp_mod.drift_speed(
            np.array([ic.drift_field * 0.1]), cfg.response.velocity,
            cfg.response.temperature, cfg.response.mobility)[0])
    props = ion_mod.ArgonProperties(
        drift_speed=float(speed), d_long=ic.d_long, d_tran=ic.d_tran,
        lifetime=(lifetime if lifetime is not None else ic.lifetime),
    )
    electronics = elec_mod.Electronics(
        gain=gain if gain is not None else ec.gain,
        mode=mode or ec.mode,
        filter=filt or ec.filter,
        order=ec.order,
        tau=tau if tau is not None else ec.tau,
        decay=ec.decay, adc_rate=ec.adc_rate, adc_bits=ec.adc_bits,
        adc_range=ec.adc_range, baseline=ec.baseline,
    )

    groups = ion_mod.line_source(drift, props, length=length, angle=angle,
                                 x_centre=x_centre, per_mm=per_mm, step=step)
    click.echo(f"drift {drift} mm at {props.drift_speed:.4f} mm/us "
               f"({ic.drift_field} V/cm, {cfg.response.velocity})")
    click.echo(groups.describe())
    click.echo("")
    click.echo(electronics.describe())
    click.echo("")

    arrays, meta = resp_mod.load_npz(response_file)
    try:
        with perf_mod.step("signal simulation", "signal",
                           groups=len(groups.n)) as d:
            frame = sig_mod.simulate(arrays, meta, groups, electronics,
                                     n_sigma=ic.n_sigma)
            d["wires"] = int(frame.adc.shape[1])
            d["samples"] = int(frame.adc.shape[2])
    except ValueError as err:
        raise click.ClickException(str(err)) from err
    click.echo(frame.describe())
    sig_mod.save_npz(output, frame)
    click.echo(f"\nwrote {output}")
    if plot_path:
        plot_mod.plot_adc(*sig_mod.load_npz(output), plot_path)
        click.echo(f"wrote {plot_path}")


@main.command("plot-signal")
@click.argument("signal_file", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=click.Path(dir_okay=False), default="signal.png",
              show_default=True)
@click.option("--style", type=click.Choice(["image", "waveforms"]), default="image",
              show_default=True)
@click.option("--tmin", type=float, default=None)
@click.option("--tmax", type=float, default=None)
@click.option("--cmap", default="RdBu_r", show_default=True)
@click.option("--dpi", default=150, show_default=True)
def plot_signal_cmd(signal_file, output, style, tmin, tmax, cmap, dpi):
    """Render ADC sample time against wire for a simulated signal."""
    from . import signal as sig_mod

    arrays, meta = sig_mod.load_npz(signal_file)
    plot_mod.plot_adc(arrays, meta, output, style=style, tmin=tmin, tmax=tmax,
                      cmap=cmap, dpi=dpi)
    click.echo(f"wrote {output}")


@main.command("sample")
@click.argument("checkpoint", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=click.Path(dir_okay=False), default="solution.npz",
              show_default=True)
@click.option("--nx", type=int, default=None, help="Override grid.nx.")
@click.option("--ny", type=int, default=None, help="Override grid.ny.")
@click.option("--xlim", nargs=2, type=float, default=None, help="Sub-range in x [mm].")
@click.option("--ylim", nargs=2, type=float, default=None, help="Sub-range in y [mm].")
@click.option("--residual", is_flag=True, help="Also evaluate the Laplace residual (slower).")
@click.option("--format", "fmt", type=click.Choice(["npz", "npy"]), default="npz",
              show_default=True, help="npz bundle, or one .npy per array plus a .json.")
@click.option("--device", default=None)
def sample_cmd(checkpoint, output, nx, ny, xlim, ylim, residual, fmt, device):
    """Evaluate a checkpoint on a regular grid and save NumPy arrays."""
    cfg, geom, model, sol = train_mod.load(checkpoint, device)
    with perf_mod.step(f"sample {os.path.basename(checkpoint)}", "sample") as d:
        arrays = sample_mod.evaluate_grid(
            cfg, geom, model,
            nx=nx or cfg.grid.nx, ny=ny or cfg.grid.ny,
            xlim=xlim or None, ylim=ylim or None, residual=residual,
        )
        d["grid"] = "x".join(str(n) for n in arrays["potential"].shape[::-1])
    meta = sample_mod.metadata(cfg, geom)
    if fmt == "npz":
        sample_mod.save_npz(output, arrays, meta)
        click.echo(f"wrote {output}  ({arrays['potential'].shape[1]} x "
                   f"{arrays['potential'].shape[0]} grid, fields: {sorted(arrays)})")
    else:
        stem = output[:-4] if output.endswith(".npz") else output
        for path in sample_mod.save_npy_set(stem, arrays, meta):
            click.echo(f"wrote {path}")


@main.command("plot")
@click.argument("solution", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=click.Path(dir_okay=False), default="solution.png",
              show_default=True, help="Extension picks the format (.png, .pdf, .svg ...).")
@click.option("-f", "--field", default="potential", show_default=True,
              help=f"One of {', '.join(plot_mod.FIELDS)} (or any array in the file).")
@click.option("--log", is_flag=True, help="Logarithmic colour scale.")
@click.option("--symmetric", is_flag=True, help="Symmetric colour limits about zero.")
@click.option("--contours", type=int, default=0, help="Draw this many equipotential contours.")
@click.option("--streamlines", type=float, default=0.0,
              help="Draw electron drift streamlines at this density (e.g. 1.0).")
@click.option("--drift", type=int, default=0,
              help="Trace this many electron drift paths seeded across the top.")
@click.option("--xlim", nargs=2, type=float, default=None)
@click.option("--ylim", nargs=2, type=float, default=None)
@click.option("--cmap", default=None)
@click.option("--equal", is_flag=True, help="Equal aspect ratio (use with --ylim for zooms).")
@click.option("--dpi", default=150, show_default=True)
@click.option("--figsize", nargs=2, type=float, default=(6.0, 8.0), show_default=True)
@click.option("--title", default=None)
def plot_cmd(solution, output, field, log, symmetric, contours, streamlines, drift,
             xlim, ylim, cmap, equal, dpi, figsize, title):
    """Render a sampled solution to PNG/PDF."""
    arrays, meta = sample_mod.load_npz(solution)
    try:
        plot_mod.plot_solution(
            arrays, meta, output, field=field, log=log, symmetric=symmetric,
            contours=contours, streamlines=streamlines, drift=drift,
            xlim=xlim or None, ylim=ylim or None, cmap=cmap, equal=equal,
            dpi=dpi, figsize=tuple(figsize), title=title,
        )
    except KeyError as err:
        raise click.ClickException(str(err)) from err
    click.echo(f"wrote {output}")


@main.command("losses")
@click.argument("checkpoint", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=click.Path(dir_okay=False), default="losses.png",
              show_default=True)
@click.option("--dpi", default=150, show_default=True)
def losses_cmd(checkpoint, output, dpi):
    """Plot the training history stored in a checkpoint."""
    import torch

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hist = blob.get("history") or {}
    if "step" not in hist:
        raise click.ClickException(f"{checkpoint} has no training history")
    plot_mod.plot_history(hist, output, dpi=dpi)
    click.echo(f"wrote {output}")


@main.command("run")
@config_options
@click.option("--prefix", default="out", show_default=True,
              help="Write <prefix>.pt, <prefix>.npz, <prefix>.png, <prefix>.losses.png.")
@click.option("--steps", type=int, default=None)
@click.option("--device", default=None)
@click.option("-q", "--quiet", is_flag=True)
@click.pass_context
def run_cmd(ctx, config_path, sets, prefix, steps, device, quiet):
    """Convenience: train, sample and plot in one go."""
    extra = []
    if steps is not None:
        extra.append(f"train.steps={steps}")
    if device is not None:
        extra.append(f"train.device={device}")
    all_sets = tuple(sets) + tuple(extra)
    ctx.invoke(train_cmd, config_path=config_path, sets=all_sets,
               output=f"{prefix}.pt", steps=None, device=None, precision=None,
               baseline=None, seed=None, quiet=quiet)
    ctx.invoke(check_cmd, checkpoint=f"{prefix}.pt", points=100000,
               as_json=False, device=None)
    ctx.invoke(sample_cmd, checkpoint=f"{prefix}.pt", output=f"{prefix}.npz",
               nx=None, ny=None, xlim=None, ylim=None, residual=False,
               fmt="npz", device=None)
    ctx.invoke(plot_cmd, solution=f"{prefix}.npz", output=f"{prefix}.png",
               field="potential", log=False, symmetric=False, contours=24,
               streamlines=0.0, drift=9, xlim=None, ylim=None, cmap=None,
               equal=False, dpi=150, figsize=(6.0, 8.0), title=None)
    ctx.invoke(losses_cmd, checkpoint=f"{prefix}.pt",
               output=f"{prefix}.losses.png", dpi=150)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
