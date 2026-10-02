# firep

**FI**eld **RE**sponse via **P**hysics-informed networks.

SIREN ([Sitzmann et al. 2020](https://arxiv.org/abs/2006.09661)) solutions of the
Laplace equation for LArTPC electrode geometries, built with PyTorch (so it runs
on a GPU when one is available) behind a Click CLI.

The first model is 2D: three rows of wires between a cathode and a ground plane,
periodic across one wire pitch. The configuration schema has deliberate room for
3D and for electrode kinds other than wires.

---

## Install

```bash
uv venv
uv pip install -e ".[test]"
```

### CPU or GPU

Everything runs on the CPU, and `device: auto` (the default) uses a GPU only when
PyTorch can actually drive one. Nothing needs a GPU except `learn-response`,
which is impractically slow without one. PyTorch wheels are built for a CUDA
version, and a wheel newer than the installed driver falls back to the CPU with a
warning. To use an older driver, install the matching wheel, for example for a
CUDA 12.5 driver:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch
```

Measured per training step (float32, default batch), before and after the two
changes below:

| model | CPU, 8 threads | GPU (RTX 4090) |
|---|---|---|
| 3-wire drift, old → new | 88 → 52 ms | 9.1 → 5.2 ms |
| 21-wire weighting, old → new | 192 → 117 ms | 33.4 → 6.9 ms |

The two changes, both on by default and both exact:

* `train.laplacian: explicit` computes the SIREN's Laplacian by propagating
  each unit's value, gradient and Laplacian forward through the sine layers
  (∇² sin z = cos z ∇²z − sin z |∇z|²). This is the same exact derivative
  autograd gives, agreeing to 10⁻¹⁵ in float64, without nested backward passes.
  `autograd` remains available as the reference.
* `sampling.backend: torch` draws collocation points on the training device.
  For the 21-wire weighting model the NumPy sampler and its host-to-device copy
  were half of every GPU step. The distributions are the same; the random
  streams differ, so `numpy` reproduces older runs exactly.

float64 does not converge faster. The loss follows float32 step for step, and
the conductor error (0.022 V) is set by the analytic warm start, not by
precision. It costs 1.6× per step on the CPU and 2.8× on the GPU.

## Quick start

```bash
firep bias                       # derived electrode potentials + transparency check
firep geometry -o geometry.png   # what the domain looks like
firep run --prefix out           # train, check, sample and plot in one go
```

or step by step:

```bash
firep config -o my.yaml          # write the fully expanded default config
firep train -c my.yaml -o m.pt   # fit the SIREN
firep check m.pt                 # PDE residual and boundary-condition errors
firep verify m.pt                # cross-checks against physics it never trained on
firep drift m.pt -n 200 -o d.png # trace electrons; measure plane transparency
firep sample m.pt -o sol.npz     # potential and field on a regular grid
firep plot sol.npz -o sol.png --contours 24 --drift 9
```

and, once you have a drift solve and a weighting solve per plane:

```bash
firep response -d drift.pt -w w-u.pt -w w-v.pt -w w-w.pt -o response.npz
firep plot-response response.npz -o response.png --wires 5
```

Every sub-command takes `--set KEY=VALUE` to override any config value without
editing a file, e.g. `--set train.steps=50000 --set model.hidden=256`.

---

## The model

### Geometry and conventions

`y` is the drift coordinate and increases toward the cathode, so **electrons
drift downward** and the potential *decreases* with `y`. `x` is transverse,
across the wire pitch. Lengths are mm, potentials V, fields V/cm.

| | |
|---|---|
| domain | `x ∈ [-2.5, 2.5] mm` (periodic), `y ∈ [-10, 100] mm` |
| wires | three rows at `y = 10, 5, 0 mm`, 0.15 mm diameter, 5 mm pitch |
| cathode | `y = 100 mm`, Dirichlet |
| ground | `y = -10 mm`, Dirichlet, 0 V |

One wire per row is modelled and the side boundaries at `x = ±2.5 mm` sit halfway
to the neighbours. The periodic condition there is **exact by construction** —
the network sees `x` only through `sin`/`cos` harmonics of the domain period, so
no penalty term is needed and none can be violated.

### Bias voltages

`firep bias` derives the potentials from the Bunemann–Cranshaw–Harvey grid
transparency criterion. For a plane of wires of radius `r` at pitch `p`, no field
line terminates on the plane when

$$\frac{E_\text{below}}{E_\text{above}} \ge \frac{1+\rho}{1-\rho},
\qquad \rho = \frac{2\pi r}{p}$$

which for `r = 0.075 mm`, `p = 5 mm` gives `ρ = 0.0942` and a required ratio of
**1.2081** at each induction plane. The collection plane must instead be
*opaque*: holding it above the ground-plane potential reverses the field in the
gap below it, so every arriving field line must end on a collection wire.

#### The wire is not its plane

The fields in that criterion are **transverse averages** `<E_y>`. Averaging
Laplace's equation over one period kills the `d²/dx²` term, so `<V>` is piecewise
linear in `y` and `<E_y>` is piecewise constant, with a kink at each wire plane.

A wire's own potential is *not* the plane-averaged potential at its height. A row
of line charges of strength `γ` (in volts, `λ/2πε₀`) contributes `γ·ln2/2` to the
plane average but `γ·(ln(p/2πr) + ln2/2)` at its own surface, so

$$V_\text{wire} = \langle V\rangle(y_i) + \gamma_i \Lambda_i,
\qquad \Lambda_i = \ln\frac{p_i}{2\pi r_i},
\qquad \gamma_i = \frac{p_i}{2\pi}\left(E_\text{above} - E_\text{below}\right)$$

the last from Gauss's law across the row. **This is not a refinement.** For the
default model the collection row carries `γ = 144 V` and `Λ = 2.36`, so its bias
sits **340 V** above its own plane-averaged potential. Setting the wire voltages
as if they were plane potentials gives realised gap fields of 621 → 671 → 515
V/cm — ratios of 1.08 and 0.77 against a requirement of 1.208, with the v–w gap
field *lower* than u–v. The planes would not have been transparent at all.

`firep verify` is what caught this: it compares the transverse-averaged `E_y` in
the solved field against the design value, and the two disagreed by 43 % in the
v–w gap. Both directions of the relation are implemented — `mode: transparency`
goes forward (pick the fields, get the voltages) and `mode: explicit` goes
backward (given the voltages, solve the linear system for the fields), and they
round-trip to round-off.

#### Anchoring

The cathode voltage, the drift field and the wire biases cannot all be chosen
independently, because the cascade multiplies the field by 1.2081 at each plane.
`bias.anchor` picks what is held fixed:

| `bias.anchor` | cathode | drift field in the cathode gap |
|---|---|---|
| `cathode_uniform` *(default)* | −5500 V | 619.8 V/cm |
| `drift_region` | −4437 V | 500.0 V/cm |

`cathode_uniform` is the literal reading of "the cathode potential should be such
that a nominal 500 V/cm field would exist if not for the wires": remove the
wires, and 500 V/cm spans the whole 110 mm cathode-to-ground gap, so
`V_cathode = −500 V/cm × 11 cm = −5500 V`. Because the wire region then needs
more volts per mm than its geometric share, the realised drift field comes out
24 % high. Use `drift_region` if you would rather pin the drift field itself and
let the cathode voltage follow. Either way `firep bias` reports both numbers.

The default solution:

```
conductor potentials [V]
  cathode    y =  100.000 mm      -5500.00
  u          y =   10.000 mm         53.67   (induction)
  v          y =    5.000 mm        423.00   (induction)
  w          y =    0.000 mm       1244.59   (collection)
  ground     y =  -10.000 mm          0.00

wire self-potential: V_wire = <V> + gamma * Lambda
  plane          <V> [V]   gamma [V]    Lambda  offset [V]
  u                77.91     -10.264    2.3618      -24.24
  v               452.28     -12.400    2.3618      -29.29
  w               904.57     143.967    2.3618      340.02

transverse-averaged gap fields [V/cm]
  cathode-u           90.000 mm       619.77
  u-v                  5.000 mm       748.75
  v-w                  5.000 mm       904.57
  w-ground            10.000 mm      -904.57
```

The reversed field below the collection plane is what guarantees collection. The
two induction rows carry only ~10 V of line charge each — a nearly transparent
wire holds almost no net charge — while the collection row carries 144 V, which
is the charge terminating every field line from the drift volume.

Set `bias.mode: explicit` and give each electrode a number to bypass all of this.

### How the Laplace equation is solved

The potential is written

```
V(x, y) = B(x, y) + s · N(x, y)
```

with `N` a SIREN and `B` an **analytic, exactly harmonic ansatz**:

```
B = V_linear(y) + Σ_i [ c_i ψ_i + d_i ∂ψ_i/∂x_i + e_i ∂ψ_i/∂y_i ]
```

where `ψ_i` is the exact potential of an infinite periodic row of line charges
through conductor *i*,

$$\psi_i(x,y) = -\tfrac12 \ln\!\left[\cosh\frac{2\pi(y-y_i)}{W}
   - \cos\frac{2\pi(x-x_i)}{W}\right]$$

evaluated in a numerically stable form, and each column is made to **vanish on
both faces** so that the basis leaves the cathode and ground conditions
untouched while staying harmonic.  That subtraction has a closed form: writing
`ψ_i` as a Fourier series in `x`, the `q = 0` term is a linear ramp in `y` and
the `q ≥ 1` terms are hyperbolic ramps times `cos q v`.  Only `q = 0` matters
when `exp(-2π Δy / P) ≪ 1`, which holds for a drift solve (`P` = one pitch,
factor `3e-6`) and emphatically not for a weighting solve (`P` = 21 pitches,
factor `0.55`) — dropping the rest there leaves `7e-2` on the ground plane and
poisons the far field of every weighting potential.  `model.baseline.wall_harmonics`
caps how many are kept (default 64; terms below `1e-10` are dropped, so a drift
solve pays for one). `ψ_i` carries the `ln r` behaviour at the wire surface; its two
derivatives with respect to the *source* position carry the `cos θ/ρ` and
`sin θ/ρ` dipole behaviour, and a derivative of a harmonic function with respect
to a source coordinate is still harmonic.

Because `B` is harmonic for any coefficients, `∇²V = s ∇²N`: the PDE residual is
evaluated on the network alone and never has to fight the baseline. Training
starts from a least-squares fit of the `c, d, e` coefficients to the conductor
surface potentials, and the network's output scale is then set to the size of
what that fit left over.

`model.baseline.mode` selects how much of this to use:

| mode | `V` is |
|---|---|
| `none` | `s·N` — the network does everything, including the 5.5 kV ramp |
| `linear` | `V_linear(y) + s·N` — the network does everything but the ramp |
| `wires` *(default)* | the full ansatz above |

and `model.baseline.multipole` is `0` (line charge only) or `1` (also the two
dipoles, the default).

### Reproducing everything

```bash
scripts/run-all.sh          # ~1 h: 4 solves concurrently, then response,
                            # signal, figures and the technical note PDF
STEPS_DRIFT=200 STEPS_WEIGHT=100 scripts/run-all.sh   # a 5-minute smoke test
```

The drift solve and the three weighting solves share nothing, so they run
concurrently with `THREADS` (default 8) each.  Every step appends a JSON record
— wall time, CPU time, peak RSS, thread count — to `runs/perf.jsonl` when
`FIREP_PERF` is set, which `run-all.sh` does; `scripts/perf_table.py` turns that
log into the table in the technical note.  Set `FIREP_PERF` yourself to time
individual commands.

### What actually converges — and what does not

This matters if you are here to evaluate SIREN, so it is worth stating plainly.
All numbers below are for the default 2D model, whose potential span is 6745 V.

The domain spans 110 mm while the wires are 0.075 mm in radius: a scale ratio of
1500. A SIREN sees `y` scaled to `[-1, 1]`, so near-wire structure lives at
`10⁻³` of its input range.

**Monopole-only basis: stalls at 2.6 V.** With `multipole: 0` the conductor
surface error sticks at 2.6 V and *nothing* moves it — `ω₀ ∈ {3, 10, 30}`, Adam
`eps ∈ {10⁻⁸, 10⁻¹²}`, output scale `∈ {30 V, 6745 V}`, `y`-scale
`∈ {5 mm, 55 mm}` and boundary weight up to 1000 all give 2.6–2.7 V. The residual
is the wires' **dipole moment**, which the network cannot resolve at that scale
ratio.

**Adding the dipole columns: 0.022 V.** A factor of 120, achieved in the
least-squares warm start before a single gradient step.

**Pure SIREN (`baseline: linear`): does not converge.** After 20 000 steps the
conductor error is 727 V and the face error 662 V — about 10 % of the span each.
It is not a tuning failure; it is a very robust local minimum:

| variant (2 000 steps) | conductor error |
|---|---|
| defaults | 727.49 V |
| boundary weights ×100 | 727.49 V |
| constant learning rate | 727.49 V |
| `hidden=256, layers=5` | 733.10 V *(PDE residual diverged)* |
| `ω₀ = 10` | 723.51 V |
| `ω₀ = 10` + weights ×100 + constant lr | 727.49 V |

The network drives the PDE residual to ~10⁻⁹ — it becomes an excellent *harmonic*
function — and then sits on the wrong one, splitting the difference between the
face and conductor conditions rather than localising a 1745 V spike onto a
0.075 mm wire.

**The default configuration, after 20 000 steps:**

```
Laplace residual   rms 3.2e-03 V/mm^2   (dimensionless 1.2e-05)
face ground        rms 0.0004 V         face cathode  rms 0.0000 V
conductors         rms 0.0220 V         max           0.0536 V
```

and the network's own contribution to `V` is **1.4·10⁻⁸ V**. The SIREN correctly
learned that it has nothing to add: the analytic basis already satisfies every
boundary condition to 3·10⁻⁶ of the span, which by the maximum principle bounds
the interior error at the same level.

Two of the fitted coefficients are worth reading as physics. The x-dipoles come
out *exactly* zero (the geometry is x-symmetric about every wire), and the
collection wire's y-dipole is 0.006 V ≈ 0 because `collection_reverse_ratio: 1`
makes the field magnitude equal above and below it, restoring up-down symmetry.
The monopoles are −10.3, −12.4 and +144.0 V: a nearly transparent wire carries
almost no net charge, while the collection wire carries the charge terminating
every field line from the drift volume.

So the honest summary for this 2D wire geometry is that the **analytic basis
solves the problem and the SIREN is not needed**. That is a result about the
geometry, not a defect in the method: the basis is exact precisely because
infinite periodic rows of round wires between parallel planes is the case where
the classical solution is known. `baseline: linear` and `baseline: none` are
there so the pure-SIREN question can be asked on its own, in one flag:

```bash
firep train --baseline linear -o linear.pt && firep check linear.pt
firep train --baseline wires  -o wires.pt  && firep check wires.pt
```

The near-wire sampling is what makes even that much work: half the collocation
points are drawn in log-radial annuli around the conductors (`sampling.n_near`,
`sampling.near_factor`), because the potential varies as `ln r` there and uniform
sampling starves exactly the region that matters.

#### Where the network does have work to do

The 21-wire weighting problem is the more interesting test. There the transverse
period is 105 mm rather than 5 mm, so the de-trending of the analytic basis —
which only removes the transverse *mean* — leaves an x-dependent residual on the
ground face of order `exp(-2π·10/105) = 0.55`. The basis fits the 63 conductors
to 3·10⁻⁵ V but leaves **0.022 V on the ground face, 2.2 % of the unit weighting
potential**, and that is the network's job.

It does not do it well. Over 1 500 steps the network takes the face error from
0.0216 V to 0.0176 V — an 18 % improvement — while pushing the conductor error
*up* from 3·10⁻⁵ V to 8·10⁻³ V. Raising `model.encoding.x_harmonics` from 4 to
16 or 42 changes nothing (0.01762, 0.01763, 0.01762 V), so the transverse
resolution of the encoding is not the limit.

The reason is the same multiscale obstruction wearing a different hat. The face
residual itself is smooth and long-wavelength, and a SIREN would represent it
happily — but any correction that fixes it must simultaneously *vanish on 63
conductor surfaces of radius 0.075 mm*. A smooth correction in a domain full of
tiny holes is not a smooth function. That, rather than anything about the
boundary data, is what a plain SIREN cannot represent here.

A trained 21-wire weighting solution therefore lands at roughly 2.5 % error on
the ground face and 0.8 % on the conductors, with a Laplace residual of
2·10⁻⁷ V/mm². Good enough to look at (and the weighting potential has exactly the
expected shape), not good enough to integrate for a field response.

**The clear next step**, and the reason this is worth writing down: the analytic
basis should carry *image rows* in the two faces. Reflecting each row in
`y = y_face` with alternating sign makes the basis satisfy the face conditions by
construction rather than by de-trending, and the series converges geometrically
— images at 220 mm contribute `exp(-2π·220/105) ≈ 2·10⁻⁶`, so two or three
orders suffice. That is the same move that the dipole columns made for the drift
problem, and it should take the weighting case to the same 10⁻⁵ level. It is not
implemented here because it is a design decision about how much of the solution
should be analytic, which is worth agreeing on first.

### Checking a solution

`firep check` reports the Laplace residual and the boundary errors — but those
are the training objective, so passing them is necessary, not sufficient.
`firep verify` cross-checks against relations the model never saw:

* the transverse-averaged `E_y` at each gap midpoint, against the design value;
* the line charge on each plane, obtained two independent ways — from the jump in
  the averaged field across the plane (Gauss's law) and from the fitted analytic
  monopole coefficient.

`firep drift` integrates electron trajectories through the solved field and
reports which electrode caught each one. That measures the transparency
condition instead of assuming it.

---

## Field response

`firep response` combines a drift solution with one weighting solution per plane
and drifts electrons through the real field, applying the Shockley-Ramo theorem
to get the induced current on every wire:

```bash
firep response -d runs/drift.pt \
    -w runs/weight-u-analytic.pt -w runs/weight-v-analytic.pt -w runs/weight-w-analytic.pt \
    -o runs/response.npz
firep plot-response runs/response.npz -o runs/response.png --wires 5 --tmin 48
firep plot-response runs/response.npz -o impacts.png --style impacts --tmin 48
firep plot-response runs/response.npz -o heat.png --style heatmap --impact 5
```

`firep.response.CombinedModel` is that combination as an object: the drift SIREN,
the weighting SIRENs, and the velocity model, exposing `drift_field`,
`weighting_potential`, `weighting_field` and `speed`. Note that this is a
*combination of trained models*, not a newly trained network — induced current is
a functional of a trajectory, not a field, so there is nothing for a SIREN to
represent.

### Shockley-Ramo

The charge induced on electrode *k* is `Q_k = -q φ_k(r)`, so

$$i_k(t) = \frac{dQ_k}{dt} = -q\,\frac{d\phi_k}{dt}
        = q\,\mathbf{v}\cdot\mathbf{E}_{w,k},\qquad
  \mathbf{E}_{w,k} = -\nabla\phi_k .$$

Both forms are computed. The **charge form** is what is reported: it is exactly
conservative and its time integral depends only on the endpoints, so an electron
that starts where `φ = 0` and lands on sensing wire *k* must induce exactly `+e`
there and exactly `0` everywhere else, whatever the path did in between. The
**field form** is evaluated independently (with the velocity taken from the field,
not from differencing the path) as a cross-check; the two agree to 1.4 % of the
peak current, the difference sitting in the last tick where the electron slams
into the wire and both `v` and `E_w` are singular.

Because the drift field is periodic over one pitch, one trajectory gives the
response of *every* wire — the wire `n` pitches away sees the same path read
against the weighting potential shifted by `n` pitches. That is what the 21-wire
weighting solve is for.

### Drift velocity

Electrons move along `-E` at a speed set by the field. The default is the
Walkowiak parameterisation (NIM A **449** (2000) 288), which gives **1.628 mm/µs
at 500 V/cm and 87.3 K**; `response.velocity: constant` uses `v = µE` instead.
Diffusion is deliberately absent — a field response is a single-electron,
drift-only quantity.

Fields at a wire surface reach ~19 kV/cm, beyond the 12.6 kV/cm Walkowiak fit
range, so `firep response` reports the largest field any trajectory saw.

### Launch height matters

`response.y_start: auto` launches just below the cathode. This is not
conservatism: the u-plane weighting potential reaches a long way up (φ = 0.056 at
y = 30 mm, a broad ~68 mm bump) because there is no grounded screen above it,
only 90 mm of drift. Launching lower silently truncates Ramo's integral by
exactly `φ(start)`, which the report prints for each plane. At `y = 14 mm` the
u-plane residual is −0.24 e; from the cathode it is −0.0006 e.

### Validation

The conservation checks are not part of the objective, so they measure something:

```
total induced charge [e] per plane and wire offset
  plane         -3       -2       -1        0        1        2        3
  u       -0.00056 -0.00061 -0.00062 -0.00062 -0.00063 -0.00061 -0.00056
  v       -0.00011 -0.00009  0.00006  0.00026  0.00002 -0.00010 -0.00011
  w       -0.00000  0.00006  0.00037  0.99740  0.00031  0.00005 -0.00001
```

Every impact from 0 to 2.25 mm collects on wire 0 with 0.997–0.998 e; the one at
exactly 2.50 mm sits on the saddle between two cells and goes to wire +1 with
0.998 e. The sum over all 21 wires is 0.998 for every impact.

**Training the weighting solves buys nothing — and used to cost a lot.** The
Ramo residual is a direct measure of weighting-solution quality at the
*conductors*, and once the basis clears both faces exactly (`wall_harmonics`,
above) it makes no difference whether the network is trained:

| weighting models | collection wire | worst induction residual | min φ |
|---|---|---|---|
| trained (6000 steps) | 0.997694 e | 3.81·10⁻⁴ e | −2.0·10⁻⁸ |
| analytic warm start only | 0.997673 e | 3.80·10⁻⁴ e | **−5.2·10⁻¹⁰** |

Before the `q ≥ 1` wall harmonics were carried, the same comparison read 0.9765 e
against 0.9974 e — training was spending the network's capacity smoothing away
the `7e-2` the basis was leaving on the ground plane, and the conductors paid
for it.  Fixing the basis removed the symptom along with the cause.  The
analytic checkpoints are still what the pipeline uses, because they are free:

```bash
firep train -c examples/weighting-2d-21wire.yaml --steps 0 \
    --set model.final_init_scale=0 --set problem.weighting.electrode=w \
    -o runs/weight-w-analytic.pt
```

(`final_init_scale=0` zeroes the network head, so the model is exactly the
analytic basis.)

---

## Example: ProtoDUNE-SP, four wire planes

`examples/drift-2d-pdsp.yaml` and `examples/weighting-2d-pdsp.yaml` describe
the ProtoDUNE single-phase anode, which is also the field response used for
ProtoDUNE-HD. A **grid plane** G above the three sensing planes partly shields
U from long-range induction, and a grounded **mesh** M closes the domain below W.
All potentials are explicit (`bias.mode: explicit`):

| plane | y [mm] | V [V] |
|---|---|---|
| C (cathode, face) | 204.71 | −10235.5 |
| G (grid) | 14.13 | −665 |
| U | 9.42 | −370 |
| V | 4.71 | 0 |
| W (collection) | 0.00 | +820 |
| M (mesh, face) | −4.79 | 0 |

The pitch is 4.71 mm on every plane and the wire radius is 0.076 mm (152 µm
CuBe), matching the Garfield model in Wire-Cell's
`dune-garfield-1d565.json.bz2`. The cathode is −10235.5 V = −500 V/cm × 20.471 cm
from the mesh: electrons drift toward −y, so the cathode is the most negative
electrode.

```bash
scripts/run-pdsp.sh                  # ~25 min: trained drift solve, everything else
STEPS_DRIFT=0 scripts/run-pdsp.sh    # ~2 min: analytic drift solve
```

It writes everything into `runs/pdsp/`: the drift maps (`drift-zoom.png` with
streamlines), the weighting maps (`weight-{u,v,w}{,-zoom}.png`), the response
waveforms (`response-{curves,impacts,heatmap}.png`), the wire-vs-time tables
with their zooms (`response-table{,-zoom,-zoom2}.png`) and the Garfield
comparison (`compare-waveforms{,-zoom}.png`, `garfield-table{,-zoom}.png`
beside `firep-y100-table{,-zoom}.png`, and `compare.txt`).
`scripts/pdsp_figs.py` looks for the Garfield file on `$WIRECELL_PATH`, or takes
the path as its second argument (`GARFIELD=` for the run script).

**Bias.** Gap fields are 503 → 635 → 791 → 962 V/cm, then −1007 V/cm below W.
The BCH ratios are 1.262 (G), 1.245 (U) and **1.216 (V), just under the
1.226 required**. BCH is a sufficient condition, not a necessary one: all 200
traced electrons still collect on W and none stop on G, U or V.

**Solve.** The analytic warm start alone is 0.023 V on the conductors
(span 11 kV) and exact on the faces. Twenty thousand training steps leave that
unchanged, and the gap fields and line charges agree to 1e-3 and 1e-7. The
weighting solves are analytic only (≤ 5e-4 V on any conductor).

**Response** (launched at the cathode): the collecting wire gets 0.998 e and no
induction wire gets more than 3e-4 e. G does what it is there for. U's
weighting potential at the cathode is 8e-5, compared with 6e-4 for the
unshielded 3-plane toy.

**Against Garfield.** Both are launched at y = 100 mm (Garfield's origin). Once
the arrival time is aligned (Garfield comes in 3.7 µs later, about 2 µs of it
from its 1.565 mm/µs drift speed against Walkowiak's 1.628), the waveforms agree
closely:

```
plane  u[mm]   peak+ firep  Garfield   peak- firep  Garfield     Q firep  Garfield   rms diff / peak
  u    0.00       0.3095    0.3045     -0.3022   -0.3033     -0.0045   -0.0021       0.078
  u    1.88       0.0938    0.0940     -0.1006   -0.1026     -0.0045   -0.0027       0.037
  v    0.00       0.2267    0.2296     -0.5568   -0.4694     -0.0007    0.0004       0.020
  v    4.71       0.0299    0.0305     -0.2112   -0.1878     -0.0009   -0.0009       0.017
  w    0.00       3.7828    3.3928     -0.0035   -0.0021      0.9977    0.9861       0.015
  w    1.88       3.3976    2.7657      0.0000    0.0000      0.9968    1.0059       0.045
```

Currents are in e/µs and charges in e. Wire-Cell stores e/ns with the sign of
the electron charge, so the conversion factor is −1000. The last-tick spikes,
where the electron hits a wire, are what differ most. Their height depends on
where the singular final approach falls within a 0.1 µs tick, in both codes;
Garfield's own collection charge scatters by ±2% between impacts. At the
cell edge (u = 2.355 mm) the two codes send the electron to different
wires. That line is a saddle of the drift field, so either answer is correct.
Launching at 100 mm truncates U's Ramo integral by φ(100 mm) = 4.5e-3 in
both codes. That is why U's Q is −0.0045 rather than 0, and Garfield's is
similar.

### Learning the same response directly

`firep learn-response` learns the PDSP field response without tracing a single
electron (technical note Sec. 15.4). The induced charge is treated as a field
of launch point and elapsed time, 𝒬(r, t) = −q φ(X_t(r)). It obeys the backward
transport equation ∂_t𝒬 = v·∇𝒬 with 𝒬(r, 0) = −q φ(r), and a SIREN is trained on
that residual alone. No response is ever a target: the drift and weighting
solutions enter only as the coefficient v and the initial data. So the traced
response, and Ramo's sum rule (the long-time limit of 𝒬, which is not in the
loss), are independent checks.

```bash
firep learn-response -c examples/drift-2d-pdsp.yaml -d runs/pdsp/drift.pt \
    -w runs/pdsp/weight-u.pt -w runs/pdsp/weight-v.pt -w runs/pdsp/weight-w.pt \
    -o runs/pdsp/direct.pt                          # ~20 min on one RTX 4090
firep direct-response -c examples/drift-2d-pdsp.yaml runs/pdsp/direct.pt \
    --impacts 11 -o runs/pdsp/response-direct.npz  # same .npz layout as `firep response`
scripts/pdsp_direct_figs.py runs/pdsp              # plots and the comparison
```

`scripts/run-pdsp.sh` runs all of this as its last stage (`DIRECT=0` skips it).
The same plots are made for the learned response
(`direct-response-{curves,impacts,heatmap}.png`,
`direct-response-table{,-zoom,-zoom2}.png`). The comparison with the traced
response is in `compare-direct-{waveforms,waveforms-zoom,table,charge}.png` and
`compare-direct.txt`. `direct-field.png` shows the learned field 𝒬_w0(r, t)
itself, `direct-arrival.png` the arrival-time check, and `direct-losses.png` the
training history.

**What it took.** Five things beyond the bare equation, each physics rather than
tuning (`firep/direct.py` has the details, technical note Secs. 15.4 and 16 the
reasons):

* **The invariant half cell.** v_x = 0 on the line through a wire and midway
  between two, so no trajectory crosses either. The half cell between them is
  solved alone, with no side conditions. The separatrix is then a side of the
  domain rather than a jump inside it. The 21-pitch field of one wire unfolds
  onto the half cell as 63 outputs (3 planes × 21 wire offsets).
* **The weighting of the residual.** The error in 𝒬 grows as ∫R dt = ∫R/|v| ds
  along a path, which argues for weighting per unit path. That was needed while
  the next item was wrong, because per unit time the false boundary layers
  dominated and training blew up. Once they were gone, per unit time
  (`normalise: time`, the default) is better. It stops the fast final approach,
  which makes the collection spike, from being down-weighted.
* **The conductor condition only where electrons arrive.** This is measured
  from the solved field as the sign of v·n on each wire. G and U repel all round
  and W absorbs all round. V absorbs on 9% of its surface, which is its 0.8%
  shortfall on the BCH ratio seen from the other side; no drift-volume electron
  reaches that arc. Imposing the condition on the repelling wires, as the first
  version did, was the largest single error: a false boundary layer every
  passing path had to cross.
* **The arrival time as an input.** The collection kink sits on t = T(r), a
  surface through the whole domain that a smooth network rounds. T is learned
  first from its own stationary equation, v·∇T = −1 with T = 0 on W, again from
  the residual alone. The time-to-go T(r) − t is then a network input, which
  aligns the kink with one coordinate. Checked against traced paths afterwards,
  the learned T is within 0.022 µs of the traced time-to-go everywhere below y_R.
* **The wire's logarithm, handed over.** Near W, φ ~ ln ρ, and the electron closes
  the last stretch at speed v. So 𝒬 ~ ln(1 + v s/a) in the time-to-go s: the
  collection spike is the wire's logarithm carried into time, tens of ns wide.
  Neither a higher ω₀ nor more points resolves it. `arrival_log: 0.03` supplies
  ln(1 + relu(T − t)/τ_a) as an input, the response's counterpart of the ψ
  columns in B_c. That input needs matched sampling, or the network rings between
  collocation points (aliasing): `arrival_time_fraction: 0.25` draws a quarter of
  each batch at times close to each point's own arrival.

Above y_R = G + 1.5 pitches the drift is a uniform translation, so there 𝒬 is
the weighting potential shifted in closed form. The network works only in the
22 mm × 2.355 mm × 20 µs region below it.

**Against the traced response**, on the centred impacts. The two stagnation
lines x = 0 and x = p/2, where an exact electron never arrives, are excluded:

```
Ramo sum rule (not in the loss): W collects 1.001-1.003 e (traced 0.997-0.998);
                                 U and V net charge |Q| < 0.004 e
waveforms, 0.1 us ticks:  rms difference 0.42 % (u), 0.34 % (v), 0.23 % (w) of the peak
waveforms, 0.5 us ADC:    rms difference 0.31 % (u), 0.48 % (v), 0.21 % (w) of the peak
collection peak, 0.1 us:  3.38 e/us learned, 4.04 traced
spike on a 0.02 us tick:  4.4 e/us, 84 ns FWHM learned; 7.4 e/us, 47 ns traced
V dip on a 0.02 us tick:  -0.65 e/us learned, -0.96 traced
timing:                   no offset (cross-correlation)
```

The spike is now resolved to within a factor of two in width. The previous
default (kept as `runs/pdsp/direct-v1.pt`, with its outputs as `*-v1.*`) reached
1.9 e/µs and 264 ns on the same fine tick. The price is on U. There the residual
ringing leaves an error of 0.42% of peak against 0.30% before, and 0.28% for
per-time weighting without the logarithmic input. The default favours the
collection signal, which carries the charge. Set `arrival_log: 0` and
`arrival_time_fraction: 0` for the smoothest induction waveforms. The traced
peak is itself unreliable at 0.1 µs binning: from one launch height to another
it moves between 2.3 and 4.1 e/µs, depending only on where the spike falls
within a tick. The learned table also carries speckle at the 10⁻³-of-peak level
where the translation hands over to the network, and a small tail after arrival
where the traced current is exactly zero.

`runs/pdsp/exp/` holds the experiments that led here, with `summary.txt` and
`spike-experiments.png`. Section 16 of the technical note sets them out:
ω₀ = 400, the same with 4× the samples, per-time weighting, the logarithmic
input, matched sampling, and their combinations.

Choosing between trained models by the sum rule is legitimate, because it uses
no traced response. By that measure, longer training is not automatically
better. A 150k-step run with twice the batch (kept as `runs/pdsp/direct-150k.pt`)
collected only 0.981-0.986 e, against 0.993-0.999 for 60k steps at the same
settings.


---

## Pixel anodes in 3D

`firep.pixel` solves a plane of pixel pads in three dimensions, and
`scripts/pixel_pochoir.py` reproduces a colleague's pochoir finite-difference
field response with it: 4.4 mm pitch, 3.8 mm pads with about 0.4 mm rounded
corners, and pochoir's boundary conditions as they stand.

```bash
scripts/pixel_pochoir.py runs/pixel docs/figs   # ~15 min on a GPU the first time; cached after
```

The analytic part is a **spectral single layer**. Each pad carries surface
densities with the `1/sqrt(d)` edge singularity, in pad-wide and edge-band
forms. They are symmetrised over the eight symmetries of the square and summed
as a double cosine series in x and z. Each transverse mode gets the slab Green's
function of the face types (drift: zero slope below and the fixed cathode above;
weighting: grounded below and zero slope above). The cosine series makes the
sides exact: periodic for the drift cell, pochoir's zero-slope walls for an
N x N weighting domain. The coefficients are fitted by least squares to the pad
surfaces, which are two sheets, the pad's top and bottom faces.

Against pochoir (`runs/pixel/*.png`, technical note Sec. 17):

* **Drift.** The uniform field is 50.059 against 50.057 V/mm. The potentials
  agree to 0.3-0.5 V rms of 7000 V. The 100 paths reach 20 mm above the pads
  within 2 ns of pochoir's.
* **Weighting.** The near field agrees (0.9579 against 0.9580 at 0.1 mm above
  the sensing pad). **pochoir's far field is not converged.** Its zero-slope top
  and sides require the plane-averaged weighting potential to be constant above
  the pads (0.0123 for 9 x 9), but its average falls to 2e-3 at 20 mm and 8e-7
  at 90 mm: 5e6 Jacobi iterations started from zero never filled it in.
* **Currents.** Over the interior of the sensing pad they agree to 0.12% of
  the peak. At the gap crossing pochoir's collection pulse is smeared (1.1
  against 4.8 e/us). Across the pad, 99% of the charge is collected within a
  0.40 us window here and a 0.95 us window in pochoir. The analytic solution
  is converged to a few ns under a fourfold change of resolution, so the
  smearing is the finite-difference discretisation: six voxels across the
  0.6 mm gap, staircase corners, and velocity zeroed on conductor voxels.
* **The SIREN correction** (`pixel.train_correction`, on a D4-invariant cosine
  encoding) leaves the pad-surface residual unchanged, 0.0143 V rms. What the
  single layer leaves is at the pad edges. For flat pads the analytic part is
  the solution; the network is there for pads that are not flat.

Problems met on the way, each found by a check:
* the pad needs two sheets, because its thickness moves the effective plane;
* an odd term placed on every pad of a class cancels, so it must go on one
  representative;
* the charge piles up within a gap width of facing edges, so the basis needs
  edge bands;
* the single layer converges only by cancellation on its own sheet, so the fit
  uses full-grid FFT values there;
* round-off columns left over from symmetry-removed terms had to be dropped.

### What lies under the pads

`scripts/pixel_gap.py` traces drift lines through one inter-pad gap on the exact
field, for three treatments of the region under the pads. `PadArray(pcb=r)` puts
a dielectric of relative permittivity `r` below the pads; `pcb=0` is a charged-up
insulator, whose surface is zero-slope between pads.

* In pochoir's free space, 7.3% of the drift flux passes below the pad plane and
  lands on the pads' undersides; lines near the gap midline dive to a stagnation
  point 2 mm down and take up to ~175 us. That is the model, not the stepper:
  halving the step moves landings by 0.1 um, while sampling the same field at
  0.1 mm voxels moves them by ~90 um.
* A fresh FR4 board catches 11.6% of the electrons; once charged, none, and the
  drift field no longer depends on the board's permittivity. The weighting field
  does, and needs the true dielectric.

### Against a newer pochoir solve with a PCB

`scripts/pixel_rado.py` reproduces R. Fan's pochoir store
(`RADO_STORE`, LArPix-v2a: 3.5 mm pads, 0.7 mm corners, the PCB as a no-flux
surface in both solves). It agrees on the uniform field (49.934 against 49.936
V/mm), the near weighting field (1e-3), interior currents (0.03% of the peak) and
the collection-time dispersion (1.20 against 1.10 us). The difference is the far
weighting field: pochoir's plane average is not linear in height as Gauss's law
requires, because the solve's stopping rule (change per sweep < 2e-8) permits
errors ~0.1 in the slowest mode.

### Pad shape and size

`scripts/pixel_shapes.py` compares the 3.8 mm rounded square with flat disks and
hemispheres of 3.8, 2.5 and 1.0 mm diameter, over a charged PCB with a 9 x 9 FR4
weighting domain (about 2 hours on a GPU the first time). Hemispheres use
`PadShape("hemisphere", R)`: smooth charge sheets inside each dome
(`pixel.dome_sheets`), fitted on the dome surface (`pixel.dome_nodes`,
`pixel.fit_design`). The rounded square has the smallest spread of collection
times (0.08 us rms over the pitch square) and the tallest average pulse. Smaller
disks sharpen single-launch pulses (up to 9.5 e/us for 1 mm) but spread the
arrivals (0.61 us rms), because electrons that miss the pad travel over bare
board. Hemispheres add a spread from their height, ~0.67 us rms for all three.

A pixel of four ganged 1.1 mm dots (`PadShape("quad", 0.55, 1.1)`, every dot on
one 2.2 mm lattice, gaps equal to the diameter) gives the tallest and narrowest
average pulse of all (1.47 e/us, 0.25 us FWHM, against 1.29 e/us and 0.35 us for
the square), at 0.23 us rms spread: the pixel centre, between its own dots, is a
second stagnation point on the charged board.

A fifth dot at the pixel centre (`PadShape("quint", 0.55, 1.1)`) removes that
stagnation point but not the spread (0.22 us rms) and lowers the average pulse
(1.29 e/us); a flat 3.3/1.1 mm ring (`PadShape("annulus", 1.65, 0.55)`, half
conductor along the axes, gaps equal to the hole) behaves like a pad with a hole
(0.28 us rms, 1.16 e/us).

### A focusing grid

A grid is one more sheet: `Sheet(y, PadShape("hole", pitch/2, r), "edge",
tile=True)`, a tile per pad cell outside a round hole, fitted with its own nodes
and voltage (`pixel.grid_nodes`, per-plane `nodes={plane: (i, j)}`,
`pixel.fit_rows`); `trace(..., grid=(y, r))` stops paths that land on it. Over
the 3.5 mm standard pad, a grid 3.2 mm up, 2000 V below the pads, with 1.9 mm
holes and the drift field held at 500 V/cm above it, is transparent (no launch
lands on it), makes every launch's pulse alike (1.8-2.0 e/us), screens the
far-field plateau (0.999 e collected), and lifts the average pulse a little
(1.27 against 1.17 e/us), but doubles the collection-time spread (0.29 against
0.14 us rms): electrons over the bars travel sideways above the grid to reach a
hole. Larger holes help most (0.26 us rms at r = 2.1 mm).

### Optimising the grid design by differentiation

`firep.pixel_opt` rebuilds the pad-plus-grid model so that everything depends
differentiably on theta = (pad half-width, hole radius, grid height, grid bias).
Densities are exact scalings about their centres, so their spectra's
derivatives follow from a scaling identity; the three geometric derivatives of
the fit are propagated forward and the fit linearised; the drift, induction,
diffusion smearing and objective are reverse-mode differentiated with
checkpointing. `scripts/pixel_optimize.py` (needs `pip install -e .[opt]` for
scipy) minimises the rms duration of the diffusion-smeared average response with
bounded L-BFGS-B, subject to a grid-to-pad field limit (1000 V/mm assumed):

```bash
scripts/pixel_optimize.py runs/pixel/opt docs/figs   # ~30 min on one GPU
```

From the nominal grid (3.5 mm pad, r 1.9 mm, 3.2 mm up, 2000 V) it converges in
24 evaluations (22 min, 55 s each with the gradient) to a 3.37 mm pad, 1.66 mm
holes, 2.73 mm up and 2739 V, at the field limit. The gradient agrees with
finite differences to 2-5%. In the full model (`gridE50opt` in
`scripts/pixel_shapes.py`) the objective falls 3% (0.502 to 0.485 us), and both
grid designs beat bare pads (0.77 us) once diffusion is counted, though the
spread of 99% collection times ranks them the other way.

### The signal induced on neighbours

An electron collected on one pad induces a transient charge on its neighbours
that returns to zero at collection, and can shift a neighbour's threshold
crossing or make it trigger. `pixel_opt` measures it as X: the charge on the
edge (and diagonal) neighbour, averaged over the pitch square and smeared by
diffusion, peak to peak within 1.5 us of the collection, per electron collected
(the centre pad's weighting potential along the paths shifted by a pitch, so no
second trace). `scripts/pixel_optimize.py ... combined` minimises J/J_nom + X/X_nom
and `... neighbour` X alone. In the full model: bare pads give X = 3.6% (edge) and
1.8% (diagonal); the nominal grid 1.6%; the combined optimum (3.53 mm pad, 1.44 mm
holes, grid 1.54 mm up at 1552 V) 0.27% and 0.12%, for J 0.530 against 0.502 us.

### The standard design against the latest pochoir solves

`scripts/pixel_hybrid.py` compares R. Fan's two-grid pochoir solves of the
standard pad (`RADO_RESULTS`, 5 x 5, 9 x 9 and 17 x 17 weighting domains). The
currents agree to 0.03% of the peak over the pad and on distant pads, and both
find 9 x 9 needed and 17 x 17 incremental. The far weighting field is still short
of Gauss's law, more so for larger domains, and the stitched drift field carries
an offset of up to 13 V from the coarse grid's pad plane.

## Ionization, electronics and readout

`firep signal` takes a field response, drifts an ideal ionization track to the
response plane, folds it through the response, and pushes the result through the
front end:

```bash
firep signal -r runs/response.npz -o runs/signal.npz \
    --drift 1000 --length 60 --angle 45 --per-mm 5000 --step 0.1 \
    --plot runs/signal.png
firep plot-signal runs/signal.npz -o waveforms.png --style waveforms --tmin 25 --tmax 78
```

### Post-drift ionization

A deposit that has drifted a distance `L` arrives as a 3D Gaussian group, thinned
by attachment:

$$\sigma_L = \sqrt{2 D_L t_d},\quad \sigma_T = \sqrt{2 D_T t_d},\quad
N = N_0 e^{-t_d/\tau_e},\quad t_d = L/v_d$$

with `D_L = 6.63 cm²/s` (Li et al. 2016 at 500 V/cm), `D_T = 13.0 cm²/s` and
`τ_e = 3 ms` by default. Over 1 m that is **σ_T = 1.26 mm, σ_t = 0.55 µs and
81.5 % survival**. Gaussians are truncated at `n_sigma` (3) and renormalised, so
truncation reshapes the tails without losing charge.

`σ_z` is carried but integrates out: the 2D wires are infinite along `z`.

### Composition

$$I_{p,w}(t) = \int\! dx_0 \int\! dt'\; n(x_0,t')\, r_p(x_0, x_w; t-t')$$

Every group is first deposited onto one common `(x, t)` grid, so the cost does
not grow with the number of groups — a 600-sample track costs the same as a
60 000-sample one. The `x` integral collapses into shifted FFT convolutions using
the pitch-periodicity and mirror symmetry of the response. Charge landing exactly
on a cell boundary is split evenly between the two neighbouring cells; without
that the half-pitch cells are lopsided about their own wire and the composition
loses its mirror symmetry.

### Front end

`electronics.mode` picks how the channel responds to an impulse of charge:

| mode | response to impulse `Q` | collection | induction |
|---|---|---|---|
| `shaper` *(default)* | pulse of peak `G·Q` at `t = τ` | unipolar | bipolar |
| `integrator` | step of height `G·Q` | step, never returns | bipolar |

`electronics.filter` picks the shape. The default `uboone` is the MicroBooNE /
BNL cold-electronics response — one real pole and two complex-conjugate pairs,

$$f(x) = 4.31054\,e^{-2.94809x} - 5.2404\,e^{-2.82833x}\cos(1.19361x)
+ 1.524912\,e^{-2.82833x}\sin(1.19361x)$$
$$\qquad + 0.929848\,e^{-2.40318x}\cos(2.5928x)
- 0.655368\,e^{-2.40318x}\sin(2.5928x)$$

with `x = t/T_p`. It starts at zero, is unipolar, and has died away by
`x = 10`. Its maximum is at `x = 1.1735`, so `T_p` is *not* the peaking time;
`tau` is quoted as the peaking time (the convention behind "2 µs shaping") and
the kernel is rescaled accordingly. `rc`, `rc2` (CR-RC^n) and `gaussian` are
also available.

$$V(t) = G\!\int\! I(t')\, s_\tau(t-t')\,dt',\qquad
s_\tau(t) = (t/\tau)^n e^{n(1-t/\tau)}$$

`shaper` normalises `s` to unit *peak*, so `G` is a peak gain in mV/fC — verified
exactly: a 1 fC impulse gives 14.000 mV at t = 2.00 µs. `G = 14 mV/fC` with
`τ = 2 µs` is the nominal MicroBooNE cold-electronics setting. `integrator` is
the literal "charge amplifier then anti-aliasing filter" reading; it leaves the
collection signal permanently raised, which an event display renders as a filled
wedge behind the track rather than a track.

Then the ADC samples (it does not average — which is what makes the shaping
necessary) and quantises:

$$a_k = \mathrm{round}\!\left(V(kT)/\Delta\right),\quad T = 1/f_s,\quad
\Delta = 2V_\mathrm{range}/2^b$$

At 2 MHz, 14 bit, ±1 V: one count is 122 µV, or **54.4 electrons**.

### The worked line source

5000 e/mm sampled every 0.1 mm along a 60 mm track at 45°, centre drifting 1 m:

```
groups: 600   electrons: 244,458        (5000 x 60 x 0.815 after attachment)
  u   peak |V|  18.4 mV   peak |ADC| 151   net charge over all wires   -1813 e
  v   peak |V|  12.4 mV   peak |ADC| 101   net charge over all wires    -235 e
  w   peak |V|  43.0 mV   peak |ADC| 350   net charge over all wires  243,988 e
```

The collection plane recovers 99.8 % of the arriving charge, matching the Ramo
efficiency of the response. The induction planes retain ~0.7 % and ~0.1 % of the
total rather than zero; that residual is the accumulated launch-point truncation
(`φ(start) ≈ 7·10⁻⁴` summed over 21 wires), not a bug in the composition.

`runs/signal.png` is the ADC-sample-time against wire image, one panel per plane;
`runs/signal-waveforms.png` overlays the channel waveforms;
`runs/signal-integrator.png` is the same track through the integrator front end,
for comparison.

---

## Configuration schema

`firep config` prints the full expanded form. A user file need only contain what
it changes; everything else is deep-merged onto the defaults. Unknown keys are an
error rather than a silent typo.

```yaml
version: 1
name: lartpc-2d-wires
units: {length: mm, potential: V, field: V/cm}

domain:
  dimension: 2                    # 3 is reserved in the schema, not implemented
  bounds:
    x: [-2.5, 2.5]
    y: [-10.0, 100.0]
    z: null                       # for dimension: 3
  boundaries:
    x: {type: periodic}           # transverse faces
    z: {type: periodic}
    ylo: {type: dirichlet, role: ground,  potential: 0.0}
    yhi: {type: dirichlet, role: cathode, potential: auto}

electrodes:                       # a list; a user list replaces the default one
  - name: u
    kind: wire                    # wire | strip | hole | pad  (only wire implemented)
    role: induction               # induction | collection | grid | ...
    plane: 10.0                   # y position of the plane
    shape:  {radius: 0.075}       # kind-specific: width/length/thickness reserved
    lattice: {pitch: 5.0, count: 1, offset: 0.0, angle: 0.0, pitch_z: null}
    bias: auto                    # a number, or auto to be derived
  # ... v at y=5, w at y=0 with role: collection

bias:
  mode: transparency              # transparency | explicit
  drift_field: 500.0              # V/cm, nominal
  anchor: cathode_uniform         # cathode_uniform | drift_region
  margin: 1.0                     # safety factor on the transparency ratio
  collection_reverse_ratio: 1.0   # reversed field below the collection plane

problem:
  kind: drift                     # drift | weighting
  weighting: {electrode: w, index: null, unit: 1.0}

model:
  hidden: 128
  layers: 4
  omega0: 30.0                    # SIREN first-layer frequency scale
  omega_hidden: 30.0
  encoding: {x_harmonics: 4, y_scale: auto, y_center: auto}
  baseline: {mode: wires, multipole: 1}
  output_scale: auto              # volts; auto adapts to the warm-start residual
  final_init_scale: 0.01

sampling:
  n_bulk: 4096                    # uniform collocation points per step
  n_near: 4096                    # log-radial annuli around the conductors
  near_factor: 20.0               # annulus outer radius, in wire radii
  n_face: 512                     # boundary points per face
  n_surface: 256                  # boundary points per conductor
  resample_every: 1
  seed: 20260918
  backend: torch                  # torch draws on the training device | numpy

train:
  steps: 20000
  lr: 1.0e-3
  lr_final: 1.0e-5
  adam_eps: 1.0e-12
  w_pde: 1.0
  w_face: 10.0
  w_surface: 10.0
  lbfgs_steps: 0                  # optional L-BFGS polish after Adam
  log_every: 200
  device: auto                    # auto | cpu | cuda | cuda:1
  precision: float32              # float64 is worth it for residual studies
  laplacian: explicit             # explicit (forward-propagated) | autograd
  seed: 20260918

grid: {nx: 201, ny: 441}          # default evaluation grid for `firep sample`
```

### Other geometries

`firep.shapes` implements electrode forms as signed distance functions, in 2D
and 3D. Everything downstream — near-surface sampling, the Dirichlet boundary
term, the adaptive drift step, the collision test — asks only `sdf(points)`,
`sample_surface(n)` and `bounds()`, so adding a kind is local to that module.

| kind | parameters | unbounded along | detector |
|---|---|---|---|
| `wire` | radius, angle | its own axis | MicroBooNE, single-phase |
| `strip` | width, thickness, angle | its own axis | DUNE vertical drift |
| `hole` | radius, thickness | — | DUNE vertical drift (the conductor is the **sheet**) |
| `pad` | width, length, thickness | — | pixel readout |

The schema deliberately describes more than the solver implements, so geometries
can be written down, validated and drawn before the solver grows to meet them.
`Config.require_solvable()` is the single place that boundary is enforced, and
it says what is missing:

```bash
firep geometry -c examples/geom-dune-vd.yaml -o vd.pdf   # works
firep bias     -c examples/geom-dune-vd.yaml             # "the Laplace solver is 2D..."
```

`examples/geom-microboone-3d.yaml`, `geom-dune-vd.yaml` and `geom-pixels.yaml`
describe the three anode styles.

### Learning the response directly

`firep.surrogate` fits a SIREN to the response itself,
`R(x0, µ) -> r[wire, time]`, optionally conditioned on `µ`. Emitting the whole
waveform per wire from one trunk keeps the wires consistent and replaces one
forward pass per time sample with one per impact. On the 2D model it reproduces
the tabulated response to 2.4×10⁻⁵ relative rms, and conditioned on a diffusion
width it reproduces a width **held out of training** to 3.1×10⁻⁴ — provided the
training grid in σ is refined to ~0.05 µs (at 0.2 µs spacing the held-out error
is 2.1×10⁻²).

### Room that was left on purpose

* **3D.** `domain.dimension`, `domain.bounds.z`, `boundaries.z`,
  `lattice.angle` (stereo wire planes) and `lattice.pitch_z` are in the schema
  and validated; `dimension: 3` currently raises "reserved but not implemented".
* **Other electrode kinds.** `kind: strip | hole | pad` are accepted by the
  schema and rejected by the builder. Adding one means implementing a signed
  distance and a surface sampler in `firep/geometry.py`; nothing else in the
  package knows that a conductor is a circle. The analytic baseline in
  `firep/siren.py` is wire-specific and would fall back to `mode: linear`.
* **Weighting fields.** `problem.kind: weighting` already works: it grounds every
  conductor and both faces and raises one chosen wire to 1 V. The 21-wires-per-
  plane configuration is a config change, not a code change:

  ```bash
  firep train --set problem.kind=weighting \
      --set domain.bounds.x='[-52.5, 52.5]' \
      --set electrodes.0.lattice.count=21 \
      --set electrodes.1.lattice.count=21 \
      --set electrodes.2.lattice.count=21 \
      -o weight.pt
  ```

  Note that the periodic boundary is then a period of 21 pitches, and that
  `problem.weighting.index` selects which wire of the row is the sensing one
  (the central one by default).

---

## Output files

`firep sample` writes an `.npz` holding

| array | shape | units |
|---|---|---|
| `x`, `y` | `(nx,)`, `(ny,)` | mm |
| `potential` | `(ny, nx)` | V |
| `ex`, `ey`, `emag` | `(ny, nx)` | V/cm |
| `inside` | `(ny, nx)` | bool, true inside a conductor |
| `residual` | `(ny, nx)` | V/mm², with `--residual` |
| `metadata` | JSON string | full config, conductor list, units |

`--format npy` writes each array as a separate `.npy` with a `.json` sidecar
instead. `firep plot` renders any of them; the output format follows the file
extension (`.png`, `.pdf`, `.svg`).

The `runs/` directory in this checkout holds worked examples: a trained drift
model, and the weighting potential of the central wire of each of the three
planes, with their sampled grids, training logs and plots.

`scripts/weighting-all.sh [STEPS] [OUTDIR] [PLANE ...]` regenerates the
weighting set — train, sample and plot for each plane, on a common zoom window
so the three maps are directly comparable:

```bash
scripts/weighting-all.sh              # all three planes, 6000 steps each
scripts/weighting-all.sh 20000 runs u # just the u plane, longer
```

The accuracy of a weighting solve depends on how far the sensing plane sits from
the ground plane, because that sets how much of the analytic basis's
un-de-trended transverse structure survives (`exp(-2π Δy / W)`):

| sensing plane | distance to ground | ground-face error | conductor error |
|---|---|---|---|
| `u` (y = 10 mm) | 20 mm | < 0.0001 V | < 0.0001 V |
| `v` (y = 5 mm) | 15 mm | 0.0054 V | 0.0016 V |
| `w` (y = 0 mm) | 10 mm | 0.0255 V | 0.0076 V |

against a unit weighting potential of 1 V. The `u` solve is essentially exact;
`w` is the one that would benefit from image rows in the basis.

## Status

Implemented and validated: the 2D wire model, the bias/transparency solver in
both directions, the SIREN + analytic-basis solver, sampling, plotting, the
physics cross-checks, the weighting-potential mode, the Shockley-Ramo field
response with a liquid-argon drift velocity, and the ionization + electronics
chain down to ADC counts.

## The technical note

`docs/` holds a technical note giving the whole method in one consistent
notation, with a worked end-to-end section on the 2D toy model and sections on
the caveats and how to improve them.

```bash
cd docs && make        # regenerate figures and numbers, then build the PDF
make figs              # figures and values only
make pdf               # PDF only
```

It is built to stay true as the models improve. `scripts/make_figs.py` reads the
solved models in `runs/` and writes both `docs/figs/*.pdf` and
`docs/figs/values.tex`, a set of LaTeX macros for **every number the note
quotes**. Nothing in the prose is transcribed by hand, so re-running the
pipeline updates the text as well as the plots. The source is split into
`docs/sections/*.tex`, one file per section.

Appendix B (`scripts/effective_plane.py`, CPU, seconds) reduces the 2D model to
its transverse average: each wire row becomes a charged sheet at the potential
`V - c Lambda`, for the drift and the weighting problems. It checks that
picture against the response table, summed over wires and averaged over
impacts, and tabulates how drift times and plateau charges from tracks parallel
to the planes respond to changes in the as-built geometry and bias.

Describable, validated and drawable but not yet solvable: 3D domains and the
`strip`/`hole`/`pad` electrode kinds. Identified but not implemented: image rows
in the analytic basis (see above), and a 3D solve.

## Testing

```bash
uv run pytest
```

## References

* V. Sitzmann et al., *Implicit Neural Representations with Periodic Activation
  Functions*, [arXiv:2006.09661](https://arxiv.org/abs/2006.09661) (2020).
* O. Bunemann, T. E. Cranshaw, J. A. Harvey, *Design of grid ionization
  chambers*, Can. J. Res. A **27** (1949) 191.
