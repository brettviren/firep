"""Configuration schema for ``firep``.

The schema is deliberately a little more general than the 2D wire model that is
implemented today, so that it can grow toward 3D and toward electrode kinds
other than wires (strips, holes, pixel pads) without a breaking rewrite.

Conventions
-----------
* Lengths are in **millimetres**, potentials in **volts**, fields in **V/cm**
  (the unit LArTPC people quote drift fields in).  The ``units`` block records
  this; alternative units are not implemented but the block reserves the space.
* ``x`` is the transverse coordinate (across the wire pitch), ``y`` is the drift
  coordinate (electrons drift toward *decreasing* y), and ``z`` -- unused in 2D
  -- is along the wire axis.

Everything in a user config file is optional: values are deep-merged onto the
defaults returned by :func:`default_config_dict`.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Literal, get_type_hints

import yaml

# --------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------


def deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge ``over`` onto a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


class ConfigError(ValueError):
    """Raised for a malformed configuration."""


_HINTS: dict[type, dict[str, Any]] = {}


def _hints(cls) -> dict[str, Any]:
    """Resolved (not stringified) annotations for a dataclass."""
    if cls not in _HINTS:
        _HINTS[cls] = get_type_hints(cls)
    return _HINTS[cls]


def _build(cls, data: Any, path: str = ""):
    """Construct dataclass ``cls`` from ``data``, rejecting unknown keys.

    Values that are already dataclass instances (or non-dataclass field types
    such as ``list[Electrode]``) are passed through untouched.
    """
    if not is_dataclass(cls) or is_dataclass(data):
        return data
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path or 'config'}: expected a mapping, got {type(data).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"{path or 'config'}: unknown key(s) {sorted(unknown)}; "
            f"known keys are {sorted(known)}"
        )
    hints = _hints(cls)
    kwargs = {}
    for name in known:
        if name not in data:
            continue
        sub = f"{path}.{name}" if path else name
        kwargs[name] = _build(hints[name], data[name], sub)
    return cls(**kwargs)


# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------


@dataclass
class Units:
    length: str = "mm"
    potential: str = "V"
    field: str = "V/cm"

    def __post_init__(self):
        if (self.length, self.potential, self.field) != ("mm", "V", "V/cm"):
            raise ConfigError(
                "only units {length: mm, potential: V, field: V/cm} are implemented"
            )


@dataclass
class Face:
    """A Dirichlet condition on one face of the bounding box."""

    type: Literal["dirichlet"] = "dirichlet"
    role: str = "electrode"  # informational: "cathode", "ground", ...
    potential: Any = "auto"  # number, or "auto" to be filled in by bias solving

    def __post_init__(self):
        if self.type != "dirichlet":
            raise ConfigError(f"face type {self.type!r} not implemented (use 'dirichlet')")


@dataclass
class TransverseBoundary:
    """Condition on the transverse (x, and in 3D z) faces."""

    type: Literal["periodic"] = "periodic"

    def __post_init__(self):
        if self.type != "periodic":
            raise ConfigError(f"transverse boundary {self.type!r} not implemented")


@dataclass
class Bounds:
    x: list[float] = field(default_factory=lambda: [-2.5, 2.5])
    y: list[float] = field(default_factory=lambda: [-10.0, 100.0])
    z: list[float] | None = None

    def __post_init__(self):
        for name in ("x", "y", "z"):
            v = getattr(self, name)
            if v is None:
                continue
            if len(v) != 2 or not v[0] < v[1]:
                raise ConfigError(f"bounds.{name} must be [lo, hi] with lo < hi, got {v}")
            setattr(self, name, [float(v[0]), float(v[1])])


@dataclass
class Boundaries:
    x: TransverseBoundary = field(default_factory=TransverseBoundary)
    z: TransverseBoundary = field(default_factory=TransverseBoundary)
    ylo: Face = field(default_factory=lambda: Face(role="ground", potential=0.0))
    yhi: Face = field(default_factory=lambda: Face(role="cathode", potential="auto"))


@dataclass
class Domain:
    dimension: int = 2
    bounds: Bounds = field(default_factory=Bounds)
    boundaries: Boundaries = field(default_factory=Boundaries)

    def __post_init__(self):
        if self.dimension not in (2, 3):
            raise ConfigError("domain.dimension must be 2 or 3")
        if self.dimension == 2 and self.bounds.z is not None:
            raise ConfigError("domain.bounds.z given but domain.dimension is 2")
        if self.dimension == 3 and self.bounds.z is None:
            raise ConfigError("domain.dimension is 3 but domain.bounds.z is missing")

    @property
    def width(self) -> float:
        """Transverse period of the domain."""
        return self.bounds.x[1] - self.bounds.x[0]

    @property
    def height(self) -> float:
        return self.bounds.y[1] - self.bounds.y[0]


@dataclass
class Shape:
    """Kind-specific electrode cross-section parameters.

    Only the keys relevant to the electrode ``kind`` are consulted:

    ============ ==========================================================
    kind         keys
    ============ ==========================================================
    ``wire``     ``radius`` (circle in the x-y plane; cylinder in 3D)
    ``strip``    ``width``, ``thickness``  *(reserved, not implemented)*
    ``hole``     ``radius``, ``thickness`` *(reserved, not implemented)*
    ``pad``      ``width``, ``length``, ``thickness`` *(reserved)*
    ============ ==========================================================
    """

    radius: float | None = 0.075
    width: float | None = None
    length: float | None = None
    thickness: float | None = None


@dataclass
class Lattice:
    """Repetition of an electrode within its plane.

    In 2D only ``pitch``/``count``/``offset`` matter.  ``angle`` and ``pitch_z``
    are reserved for 3D wire/strip planes at a stereo angle.
    """

    pitch: float = 5.0
    count: int = 1
    offset: float = 0.0
    angle: float = 0.0  # degrees from +z, about the drift axis
    pitch_z: float | None = None  # second lattice direction (pads, hole arrays)
    count_z: int | None = None
    offset_z: float = 0.0
    stagger: bool = False  # offset alternate rows by pitch_z/2: a triangular
                           # (close-packed) lattice rather than a rectangular one

    def __post_init__(self):
        if self.count < 1:
            raise ConfigError("lattice.count must be >= 1")
        if self.count_z is not None and self.count_z < 1:
            raise ConfigError("lattice.count_z must be >= 1")


@dataclass
class Electrode:
    name: str = "electrode"
    kind: Literal["wire", "strip", "hole", "pad"] = "wire"
    role: str = "induction"  # "induction" | "collection" | "grid" | ...
    plane: float = 0.0  # y position of the electrode plane
    shape: Shape = field(default_factory=Shape)
    lattice: Lattice = field(default_factory=Lattice)
    bias: Any = "auto"  # volts, or "auto" to be filled in by bias solving

    def __post_init__(self):
        if self.kind not in ("wire", "strip", "hole", "pad"):
            raise ConfigError(f"electrode kind {self.kind!r} unknown")
        need = {"wire": ("radius",), "strip": ("width",), "hole": ("radius",),
                "pad": ("width", "length")}[self.kind]
        for attr in need:
            if not getattr(self.shape, attr):
                raise ConfigError(
                    f"electrode {self.name!r}: a {self.kind} needs shape.{attr}")


@dataclass
class Bias:
    """How to choose electrode potentials.

    ``mode: transparency`` derives the wire bias voltages from the
    Bunemann-Cranshaw-Harvey grid transparency criterion; see :mod:`firep.bias`.
    ``mode: explicit`` uses the ``bias:`` value given on each electrode and the
    ``potential:`` given on each face (no ``auto`` allowed).
    """

    mode: Literal["transparency", "explicit"] = "transparency"
    drift_field: float = 500.0  # V/cm, nominal
    anchor: Literal["cathode_uniform", "drift_region"] = "cathode_uniform"
    margin: float = 1.0
    collection_reverse_ratio: float = 1.0

    def __post_init__(self):
        if self.mode not in ("transparency", "explicit"):
            raise ConfigError(f"bias.mode {self.mode!r} unknown")
        if self.anchor not in ("cathode_uniform", "drift_region"):
            raise ConfigError(f"bias.anchor {self.anchor!r} unknown")
        if self.margin < 1.0:
            raise ConfigError("bias.margin < 1 would violate the transparency criterion")


@dataclass
class Weighting:
    """Which electrode is held at unit potential for a weighting-field solve."""

    electrode: str = "w"
    index: int | None = None  # index within the lattice; None -> central wire
    unit: float = 1.0


@dataclass
class Problem:
    kind: Literal["drift", "weighting"] = "drift"
    weighting: Weighting = field(default_factory=Weighting)

    def __post_init__(self):
        if self.kind not in ("drift", "weighting"):
            raise ConfigError(f"problem.kind {self.kind!r} unknown")


@dataclass
class Encoding:
    """Input feature map fed to the SIREN.

    The transverse coordinate enters only through ``sin``/``cos`` harmonics of
    the domain period, which makes the periodic boundary condition *exact* by
    construction rather than a soft penalty.
    """

    x_harmonics: int = 4
    y_scale: Any = "auto"  # mm; "auto" -> half the domain height
    y_center: Any = "auto"  # mm; "auto" -> centre of the domain


@dataclass
class Baseline:
    """Analytic harmonic ansatz added to the network output.

    ``none``
        ``V = out``
    ``linear``
        ``V = V_lin(y) + out`` with ``V_lin`` the linear ramp matching the two
        face potentials.
    ``wires``
        ``V = V_lin(y) + sum_i c_i psi_i(x, y) + out`` where ``psi_i`` is the
        exact potential of an infinite periodic row of line charges through
        electrode *i*, de-trended so that it vanishes on both faces, and ``c_i``
        is a learned coefficient (physically, the line-charge density).  Every
        term is harmonic, so the PDE residual involves only the network.
    """

    mode: Literal["none", "linear", "wires"] = "wires"
    multipole: int = 1  # 0: line charge per conductor; 1: also the two dipoles
    # Transverse harmonics kept in the correction that zeroes each column on the
    # two faces.  Only the q = 0 term (a linear ramp in y) is needed when
    # exp(-2 pi dy / P) << 1, which holds for a drift solve and not for a
    # weighting solve, where P is many pitches.  Terms whose coefficients fall
    # below 1e-10 are dropped, so this is a cap and not a cost.
    wall_harmonics: int = 64


@dataclass
class Model:
    hidden: int = 128
    layers: int = 4  # number of hidden layers
    omega0: float = 30.0  # first-layer frequency scale (Sitzmann et al. 2020)
    omega_hidden: float = 30.0
    encoding: Encoding = field(default_factory=Encoding)
    baseline: Baseline = field(default_factory=Baseline)
    output_scale: Any = "auto"  # volts; "auto" -> potential span of the problem
    final_init_scale: float = 0.01  # shrink last layer so training starts at the baseline


@dataclass
class Sampling:
    n_bulk: int = 4096  # uniform collocation points per step
    n_near: int = 4096  # collocation points in log-radial annuli around electrodes
    near_factor: float = 20.0  # annulus outer radius, in electrode radii
    n_face: int = 512  # boundary points per face (ylo, yhi)
    n_surface: int = 256  # boundary points per electrode
    resample_every: int = 1  # steps between redraws; 1 = fresh points every step
    seed: int = 20260918
    backend: Literal["torch", "numpy"] = "torch"  # torch draws on the training device

    def __post_init__(self):
        if self.backend not in ("torch", "numpy"):
            raise ConfigError(f"sampling.backend {self.backend!r} unknown (torch | numpy)")


@dataclass
class Train:
    steps: int = 20000
    lr: float = 1e-3
    lr_final: float = 1e-5  # exponential decay target at the last step
    # Adam's denominator floor.  Keep it at the torch default: when the
    # analytic baseline already solves the problem the remaining gradients are
    # ~1e-9, and a smaller eps turns Adam into sign-descent with a fixed,
    # lr-sized step, which random-walks high-frequency noise into the network
    # and drives the PDE residual back up.  Lower it only if training stalls
    # with a *large* boundary error.
    adam_eps: float = 1e-8
    w_pde: float = 1.0
    w_face: float = 10.0
    w_surface: float = 10.0
    lbfgs_steps: int = 0  # optional L-BFGS polish after Adam
    log_every: int = 200
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "cuda:0" ...
    precision: Literal["float32", "float64"] = "float32"
    # "explicit": the SIREN Laplacian by forward propagation (exact, faster);
    # "autograd": nested automatic differentiation, kept as the reference.
    laplacian: Literal["explicit", "autograd"] = "explicit"
    seed: int = 20260918


@dataclass
class ResponseCfg:
    """Field-response calculation (see :mod:`firep.response`).

    This block describes how to turn a drift solution plus one weighting
    solution per plane into induced current; it is not part of the Laplace
    problem itself.
    """

    velocity: Literal["walkowiak", "constant"] = "walkowiak"
    temperature: float = 87.3  # K, for the Walkowiak parameterisation
    mobility: float = 320.0  # cm^2/(V s), for velocity: constant
    y_start: Any = "auto"  # mm where electrons launch; auto -> just below the cathode
    impacts: int = 11  # starting transverse positions
    half_pitch: bool = True  # spread them over half a pitch (the symmetric half)
    tick: float = 0.1  # us, output sampling period
    max_time: float = 200.0  # us, safety cap on a trajectory
    wires: int | None = None  # wire offsets to report; None -> the weighting lattice
    substeps: int = 4  # integration steps per tick

    def __post_init__(self):
        if self.velocity not in ("walkowiak", "constant"):
            raise ConfigError(f"response.velocity {self.velocity!r} unknown")
        if self.impacts < 1:
            raise ConfigError("response.impacts must be >= 1")
        if self.tick <= 0:
            raise ConfigError("response.tick must be positive")


@dataclass
class IonizationCfg:
    """Transport of ionization electrons from a deposit to the response plane.

    ``drift_speed: auto`` evaluates the response's velocity model at
    ``drift_field``.  Note that ``drift_field`` is the field in the *bulk* drift
    volume, upstream of the wire region; it need not equal the realised
    cathode-gap field of the wire model.
    """

    drift_field: float = 500.0  # V/cm, in the bulk drift volume
    drift_speed: Any = "auto"  # mm/us
    d_long: float = 6.63  # cm^2/s, Li et al. PRD 94 (2016) 082004 at 500 V/cm
    d_tran: float = 13.0  # cm^2/s
    lifetime: float = 3000.0  # us, electron lifetime; null disables absorption
    n_sigma: float = 3.0  # Gaussian truncation


@dataclass
class ElectronicsCfg:
    """One readout channel's front end (see :mod:`firep.electronics`)."""

    gain: float = 14.0  # mV/fC (peak gain in shaper mode)
    mode: Literal["shaper", "integrator"] = "shaper"
    filter: Literal["none", "rc", "rc2", "gaussian", "uboone"] = "uboone"
    order: int = 2  # CR-RC^n order
    tau: float = 2.0  # us, shaping / filter characteristic time
    decay: float | None = None  # us, amplifier feedback decay; null = pure integrator
    adc_rate: float = 2.0  # MHz
    adc_bits: int = 14
    adc_range: float = 1000.0  # mV half-range, so +-1 V
    baseline: float = 0.0  # mV pedestal

    def __post_init__(self):
        if self.filter not in ("none", "rc", "rc2", "gaussian", "uboone"):
            raise ConfigError(f"electronics.filter {self.filter!r} unknown")


@dataclass
class DirectCfg:
    """Learning the field response directly (see :mod:`firep.direct`).

    The induced charge is learned as a field over launch point and time from the
    residual of the backward transport equation alone; no response is a target.
    """

    y_top: Any = "auto"  # y_R, mm; auto -> topmost plane + top_pitches pitches
    top_pitches: float = 1.5  # uniform translation above this (field uniform to ~1e-4)
    y_bottom: Any = "auto"  # mm; auto -> lowest plane - below
    below: float = 1.0  # mm
    duration: float = 20.0  # us learned below y_R
    tau: float = 0.05  # us, the initial-condition gate 1 - exp(-t/tau)
    ell: float = 0.3  # mm, conductor gate D = tanh(d / ell)
    sigma: float = 1.0  # e, network output scale
    hidden: int = 256
    layers: int = 4
    omega0: float = 200.0
    omega_hidden: float = 30.0
    final_init_scale: float = 0.1
    log_features: bool = True  # log-distance to each wire and log-time inputs
    t_log: float = 0.01  # us, scale of the log-time feature and sampling
    log_time_fraction: float = 0.25  # of each batch, times drawn log-uniform
    arrival: bool = True  # learn the arrival time first and give it as an input
    arrival_steps: int = 60000  # an accurate T matters once arrival_log is on
    arrival_hidden: int = 128
    arrival_layers: int = 3
    arrival_omega0: float = 30.0
    arrival_log: float = 0.03  # us; > 0 adds ln(1 + relu(T - t)/arrival_log) as an input
    arrival_time_fraction: float = 0.25  # of each batch, times drawn close to T(r)
    pool: int = 300000  # collocation points with precomputed coefficients
    near_fraction: float = 0.5  # of the pool, in log-radial half annuli
    near_factor: float = 40.0  # annulus outer radius, in wire radii
    batch: int = 16384
    steps: int = 60000
    lr: float = 5e-4
    lr_final: float = 1e-5
    causal_bins: int = 32
    causal_eps: float = 0.02  # 0 disables causal weighting
    normalise: str = "time"  # residual per unit time ("time") or path length ("path")
    v_floor: float = 0.25  # |v| floor for "path", in units of v0
    clip: float = 1.0  # gradient-norm clip; 0 disables
    log_every: int = 500
    device: str = "auto"
    seed: int = 20260924


@dataclass
class Grid:
    """Default evaluation grid for the ``sample`` sub-command."""

    nx: int = 201
    ny: int = 441


@dataclass
class Config:
    version: int = 1
    name: str = "lartpc-2d-wires"
    units: Units = field(default_factory=Units)
    domain: Domain = field(default_factory=Domain)
    electrodes: list[Electrode] = field(default_factory=list)
    bias: Bias = field(default_factory=Bias)
    problem: Problem = field(default_factory=Problem)
    model: Model = field(default_factory=Model)
    sampling: Sampling = field(default_factory=Sampling)
    train: Train = field(default_factory=Train)
    response: ResponseCfg = field(default_factory=ResponseCfg)
    ionization: IonizationCfg = field(default_factory=IonizationCfg)
    electronics: ElectronicsCfg = field(default_factory=ElectronicsCfg)
    direct: DirectCfg = field(default_factory=DirectCfg)
    grid: Grid = field(default_factory=Grid)

    def __post_init__(self):
        if self.version != 1:
            raise ConfigError(f"config version {self.version} not supported (expected 1)")
        if not self.electrodes:
            raise ConfigError("no electrodes configured")
        names = [e.name for e in self.electrodes]
        if len(set(names)) != len(names):
            raise ConfigError(f"duplicate electrode names in {names}")
        # Electrodes must sit inside the drift gap, and their lattice must tile
        # the transverse period -- otherwise the periodic boundary is a lie.
        ylo, yhi = self.domain.bounds.y
        for e in self.electrodes:
            if not ylo < e.plane < yhi:
                raise ConfigError(
                    f"electrode {e.name!r} plane y={e.plane} is outside "
                    f"the domain y range [{ylo}, {yhi}]"
                )
            if e.lattice.angle == 0.0:
                span = e.lattice.pitch * e.lattice.count
                if not math.isclose(span, self.domain.width, rel_tol=1e-9,
                                    abs_tol=1e-9):
                    raise ConfigError(
                        f"electrode {e.name!r}: lattice pitch*count = {span} does "
                        f"not tile the transverse period {self.domain.width}; a "
                        "periodic boundary requires them to match"
                    )
        if self.problem.kind == "weighting":
            if self.problem.weighting.electrode not in names:
                raise ConfigError(
                    f"problem.weighting.electrode {self.problem.weighting.electrode!r} "
                    f"is not one of {names}"
                )

    # -- what the solver can actually do ------------------------------------

    SOLVER_SUPPORTED = ("2D domains of round wires between parallel planes",)

    def require_solvable(self) -> None:
        """Raise unless the Laplace solver can handle this configuration.

        The schema deliberately describes more than the solver implements, so
        that geometries can be written down, validated and drawn before the
        solver grows to meet them.  This is the one place that boundary is
        enforced.
        """
        if self.domain.dimension != 2:
            raise ConfigError(
                "the Laplace solver is 2D; domain.dimension: 3 can be described "
                "and drawn (`firep geometry`) but not yet solved"
            )
        bad = sorted({e.kind for e in self.electrodes} - {"wire"})
        if bad:
            raise ConfigError(
                f"the Laplace solver implements round wires only; electrode "
                f"kind(s) {bad} can be described and drawn (`firep geometry`) "
                "but not yet solved"
            )
        stereo = sorted({e.name for e in self.electrodes if e.lattice.angle})
        if stereo:
            raise ConfigError(
                f"electrode(s) {stereo} have a stereo angle, which needs a 3D "
                "solve"
            )

    # -- ordering helpers ---------------------------------------------------

    @property
    def electrodes_by_y(self) -> list[Electrode]:
        """Electrodes ordered from the cathode side down toward the ground."""
        return sorted(self.electrodes, key=lambda e: -e.plane)

    def electrode(self, name: str) -> Electrode:
        for e in self.electrodes:
            if e.name == name:
                return e
        raise ConfigError(f"no electrode named {name!r}")


# --------------------------------------------------------------------------
# defaults and I/O
# --------------------------------------------------------------------------


def default_config_dict() -> dict:
    """The built-in 2D three-plane wire model described in the README."""
    return {
        "version": 1,
        "name": "lartpc-2d-wires",
        "units": {"length": "mm", "potential": "V", "field": "V/cm"},
        "domain": {
            "dimension": 2,
            "bounds": {"x": [-2.5, 2.5], "y": [-10.0, 100.0]},
            "boundaries": {
                "x": {"type": "periodic"},
                "ylo": {"type": "dirichlet", "role": "ground", "potential": 0.0},
                "yhi": {"type": "dirichlet", "role": "cathode", "potential": "auto"},
            },
        },
        "electrodes": [
            {
                "name": "u",
                "kind": "wire",
                "role": "induction",
                "plane": 10.0,
                "shape": {"radius": 0.075},
                "lattice": {"pitch": 5.0, "count": 1, "offset": 0.0},
                "bias": "auto",
            },
            {
                "name": "v",
                "kind": "wire",
                "role": "induction",
                "plane": 5.0,
                "shape": {"radius": 0.075},
                "lattice": {"pitch": 5.0, "count": 1, "offset": 0.0},
                "bias": "auto",
            },
            {
                "name": "w",
                "kind": "wire",
                "role": "collection",
                "plane": 0.0,
                "shape": {"radius": 0.075},
                "lattice": {"pitch": 5.0, "count": 1, "offset": 0.0},
                "bias": "auto",
            },
        ],
        "bias": {
            "mode": "transparency",
            "drift_field": 500.0,
            "anchor": "cathode_uniform",
            "margin": 1.0,
            "collection_reverse_ratio": 1.0,
        },
        "problem": {"kind": "drift", "weighting": {"electrode": "w", "index": None}},
        "model": {},
        "sampling": {},
        "train": {},
        "response": {},
        "ionization": {},
        "electronics": {},
        "direct": {},
        "grid": {},
    }


def from_dict(data: dict) -> Config:
    """Build a :class:`Config`, merging ``data`` onto the defaults."""
    if not isinstance(data, dict):
        raise ConfigError("configuration must be a mapping")
    merged = deep_merge(default_config_dict(), data)
    # A list-of-dataclass field needs building element by element.  Note that a
    # user-supplied "electrodes" list *replaces* the default list wholesale
    # (deep_merge does not merge lists), which is what one wants here.
    elec = merged.get("electrodes")
    if not isinstance(elec, list):
        raise ConfigError("electrodes: expected a list")
    merged["electrodes"] = [
        _build(Electrode, e, f"electrodes[{i}]") for i, e in enumerate(elec)
    ]
    return _build(Config, merged)


def load(path: str | None) -> Config:
    """Load a YAML config file, or the built-in defaults when ``path`` is None."""
    if path is None:
        return from_dict({})
    with open(path) as fp:
        data = yaml.safe_load(fp) or {}
    return from_dict(data)


def set_path(data: Any, dotted: str, value: Any) -> None:
    """Set ``data["a"]["b"][0]["c"] = value`` from the string ``"a.b.0.c"``."""
    parts = dotted.split(".")
    node = data
    for p in parts[:-1]:
        if isinstance(node, list):
            node = node[int(p)]
        else:
            if not isinstance(node.get(p), (dict, list)):
                node[p] = {}
            node = node[p]
    if isinstance(node, list):
        node[int(parts[-1])] = value
    else:
        node[parts[-1]] = value


def load_with_overrides(path: str | None, sets: tuple[str, ...] = ()) -> Config:
    """Load a config and apply ``key.path=yaml_value`` command-line overrides."""
    data = {}
    if path is not None:
        with open(path) as fp:
            data = yaml.safe_load(fp) or {}
    merged = deep_merge(default_config_dict(), data)
    for item in sets:
        key, sep, val = item.partition("=")
        if not sep:
            raise ConfigError(f"--set expects KEY=VALUE, got {item!r}")
        try:
            parsed = yaml.safe_load(val)
        except yaml.YAMLError as err:
            raise ConfigError(f"--set {item!r}: cannot parse value: {err}") from err
        try:
            set_path(merged, key.strip(), parsed)
        except (KeyError, IndexError, ValueError) as err:
            raise ConfigError(f"--set {item!r}: bad key path: {err}") from err
    return from_dict(merged)


def _plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def to_dict(cfg: Config) -> dict:
    """Fully expanded plain-data form of a config (round-trips through YAML)."""
    return _plain(cfg)


def to_yaml(cfg: Config) -> str:
    return yaml.safe_dump(to_dict(cfg), sort_keys=False, default_flow_style=False)
