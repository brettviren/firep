# Learning a wire field response from (simulated) detector data — design

This demo tests the laser leg of App. A of `docs/siren-for-lartpc.tex`
("Learning the response from detector data") on **ProtoDUNE-HD** (PDHD). An
earlier draft wrongly targeted ProtoDUNE-VD. The PDVD-specific notes about the
response files `protodunevd_FR_*` no longer apply and have been removed.

## The problem

PDHD has four APAs. The first (hardware/online "APA1", WCT **anode 0**; here
just "the bad APA") has its W plane disconnected and electrically floating. W
charges up, acts as an induction plane, and V collects. Online "APA2" (WCT
anode 2) is coplanar with it, side by side in the same drift volume. The
other two APAs face them across the common cathode.

| model | file (`reference/wire-cell-data/`) | role here |
|---|---|---|
| Garfield, nominal APA | `dune-garfield-1d565.json.bz2` | anodes 1–3 |
| Garfield V_W=0, **tuned to data** (MCMC) | `np04hd-garfield-6paths-mcmc-bestfit.json.bz2` | **truth** of the sim-to-sim test |
| Garfield V_W=0, untuned | not yet in hand | the natural prior (stage B2) |
| wires | `protodunehd-wires-larsoft-v1.json.bz2` | geometry |

**Numbering: we always use WCT anode numbers** unless a name is written
"online APAn". From DUNE's channel-map code (`duneprototypes
Protodune/hd/Tool/PdhdChannelRanges.h`): WCT anode 0 = online APA1, 1 = APA3,
2 = APA2, 3 = APA4; anode n owns channels 2560n to 2560n+2559. Anodes 0 and 2
share the x<0 drift volume (0 at low z); 1 and 3 face them across the cathode.
Figure: `note/figs/apa-numbering.pdf` (`apa_numbering` rule).

What we ultimately want is nature's bad-APA response, learned from real data as
a **correction to the Garfield bad-APA model**. The Garfield models are used
here only to evaluate the method in simulation.

### How the truth was made (X. Ning; `others/CM_FieldResponse.pdf`)

- Garfield 2D cell (slide 7): G, U, V, W ("X" on the slide), then M (mesh).
  Pitch and plane gap 4.71 mm, wire diameter 150 µm. Potentials G −665, U −370,
  V 0, W 0, M 0 V. Electrons start 10 cm from W, at **6 impact positions**
  (0 to ½ pitch in steps of 0.1 pitch). The mesh position is not stated; we
  assume y = −4.71 mm with W at 0 (`config.yaml:bad_apa.mesh_y_mm`).
- With V_W = 0, the W signal combines three kinds of path (slide 8): charge
  collected on V (bipolar on W), on W (unipolar), and on the mesh (bipolar).
  The 0.4-pitch path is W-collected and the 0.5-pitch path mesh-collected.
- Changing V_W alone (0 to −100 V) moves the data comparison in the right
  direction but "won't work well" (slides 10–11, 23).
- So the tuned model renormalises the 0.4-pitch path (f₅) and 0.5-pitch path
  (f₆), adds a start point and time-stretch factor per path (slide 14), fits
  all of it to 8 data tracks at different angles by MCMC, and normalises to
  "1 electron collected on V". The best-fit label on slide 15,
  `compare_check_0.01_2.10_1.00_1.90_550`, suggests f₅ ≈ 0.01 (the W-collected
  path almost removed) and f₆ ≈ 2.1 (the mesh path doubled); to be confirmed.

**Consequence: the truth is Garfield plus an empirical, data-driven
correction.** Reweighting two of six paths and stretching them in time is not
the response of any electrostatic configuration. That is exactly the kind of
correction we want the method to learn, and it shapes the tests below.

## Layout

| path | what |
|---|---|
| `design.md` | this file |
| `config.yaml` | every parameter that changes a result |
| `Snakefile` | the recipes; `snakemake -c4 note` rebuilds the note |
| `env.sh` | run-time environment (xerosere WCT install, `WIRECELL_PATH`, python) |
| `scripts/` | `laser.py` (CSV reader), `wires.py` (geometry), `laser_depos.py`, `laser_coverage.py`, `fr_compare.py`, `check-env.sh` |
| `firep/` | firep configs: `bad-apa-drift.yaml` (the bad-APA cell) |
| `cfg/` | WCT Jsonnet for this demo (first on `WIRECELL_PATH`) |
| `inputs/` | files provided by hand (laser CSVs and their source URL) |
| `note/` | the tech note: `wrfd.tex`, `sections/`, generated `figs/` |
| `runs/` | run products, not version controlled |

Every number quoted in the note is a LaTeX macro written by a recipe into
`note/figs/<step>.tex`. The note `\IfFileExists`-includes each step's macros
and section, so it builds at every stage.

**What runs today** (all through `snakemake`):

- S1 `fr_good_bad`, S2 `laser_coverage`, `laser_depos` (497 shots, 2.07 M depos).
- S3 `data` and `sim_prior`: WCT simulation of anodes 0 and 2 via
  `cfg/sim-laser.jsonnet`, about 1.3 s per shot. `frames_data` plots one shot.
- S4 firep: `firep_landing`, `firep_vfloat`, `firep_weight`, `firep_drift`,
  `firep_response`, and `prior`, which writes WCT response files
  (`scripts/firep_to_wct.py`). `fr_bone` compares the B1 pair.
- `snakemake -c4 note` builds the note with all of the above.

S5 (differentiable forward model) and later are skeletons.

## FR names (note Sec. 2.3) and where they live

The note names every field response by provenance letter; use these names in
prose and plots. Recipe/file names map as follows.

| note name | recipe / file |
|---|---|
| G-nom | `good_fr` = `dune-garfield-1d565` |
| G-bad | `bad_untuned_fr` = `np04hd-garfield-6paths` → `runs/prior/garfield-untuned` |
| G-bad-T | `bad_fr` = `np04hd-garfield-6paths-mcmc-bestfit` (hidden truth) |
| P(V_W) | `runs/prior/vw<V>.json.bz2` (e.g. `vw0`, `vw-100`) |
| P(float) | `runs/prior/vwfloat.json.bz2` |
| P_b(V_W; m) | `runs/prior/vw<V>[m<m>]b.json.bz2`, e.g. `vwfloatm4.71b` = P_b(float) |
| X[G-bad-T] | `runs/truth/proj-vwfloat.json.bz2` (hidden; rule `proj_truth`) |
| X[G-bad] | `runs/firep/proj-untuned.json.bz2` (rule `proj_untuned`) |
| L2[T] | `runs/l2/learned-<tag>.json.bz2`: `lbone`→T=P(float), `lbproj`→X[G-bad-T], `lbtwo`→G-bad-T; `…bal` = L2bal |

Reserved letters: D (SIREN direct response), L3 (response residual), LS
(SIREN through a differentiable forward model), P-nom (physics solve of the
nominal APA); real data as target: `[data]`.

## Hidden-truth discipline

`config.yaml:bad_fr` is the truth. Only `data` (which makes the pseudo-data)
and the scoring rules (`fr_good_bad`, `score`) read it. A fit's inputs are the
prior, the depos and `runs/frames/data.npz`. Check with `snakemake --dag fit`.

Knowing the untuned Garfield model and the bad-APA voltages is legitimate: in
the real problem they are the prior. What must stay hidden is the tuned
response and its tuning parameters (f₅, f₆, stretches).

## Evaluating the sim-to-sim tests

From `fr_compare.py` (`note/figs/fr-frgb.tex`). Both files have 126 paths
(21 wires × 6 impacts) at 100 ns, pitch 4.71 mm, response plane 100 mm,
speed 1.565 mm/µs. Median net charge on central-wire paths and peak current:

| | U | V | W |
|---|---|---|---|
| net charge, good | 0.003 | 0.003 | −0.994 |
| net charge, bad (tuned) | 0.003 | −0.999 | 0.002 |
| peak ×10⁻³, good | 0.305 | 0.579 | 4.36 |
| peak ×10⁻³, bad | 0.522 | 3.49 | 0.475 |

### A. Learn the bad APA starting from the good one — ruled out

- **It is not a correction.** V and W swap roles: V's net charge goes 0 → −1 and
  its peak grows sixfold, W goes the other way, U's peak grows 70%. The
  difference is as large as the response. A residual on the nominal response,
  or a SIREN surrogate trained around it, starts no closer than zero, and the
  switch between unipolar and bipolar is out of distribution.
- **It tests the wrong regime.** The real task is a small correction to the
  bad-APA model. The slides now show what that correction looks like, and
  good→bad is nothing like it.
- In a boundary-condition parametrisation it collapses to one voltage
  crossing a transparency threshold. That could be an optional stress test
  (S8), not the demo.

### B. Start from an untuned bad-APA model, recover the tuned one — adopted

The prior has the bad APA's physics (V collects, W floats); the truth adds
the data-driven correction. Three stages:

- **B0 — solver closure.** firep's V_W=0 cell (`firep/bad-apa-drift.yaml`) vs
  the *untuned* Garfield V_W=0 response. This measures the solver floor. firep vs
  Garfield on the nominal APA differs by 1.4–9.6% of peak rms per path at
  0.1 µs (`runs/pdsp/compare.txt`), including a 3.7 µs time-origin shift from
  different drift speeds, so this must be measured, not assumed.
- **B1 — method, exactly representable.** Truth and prior are both firep
  solves, e.g. truth = firep(V_W at its floating equilibrium, below) and prior =
  firep(V_W = 0). This tests identifiability and the fit machinery with no model
  mismatch.
- **B2 — the real question in simulation.** Prior = untuned Garfield V_W=0 (or
  firep V_W=0 if that file can't be had); truth = the tuned bestfit. The
  prior→truth difference is *exactly the correction real data demanded*. The
  learner has to find it from laser ADC waveforms alone.

What the learner is (S6), in order of preference:

1. **Physical parameters with continuous impact position.** θ = (V_W, mesh
   position, maybe wire radius and the U/G potentials), response from firep.
   Hypothesis worth testing: much of the f₅/f₆ reweighting emulates physics
   that 6 impact samples can't resolve. The 0.4–0.5-pitch region is where the
   separatrices between V-, W- and mesh-collection lie, and their positions move
   with V_W. firep's response is continuous in impact, so it can put them where
   the data want. Supporting evidence (`firep bias` scan of V_W): making W more
   negative raises the average field below W (36 → 165 V/cm from 0 to −100 V),
   pushing more electrons to the mesh, and lowers it above W, reversing near
   −136 V. That is the direction of the fit: less W collection (f₅ ≈ 0.01), more
   mesh collection (f₆ ≈ 2).
2. **Physics prior: the floating equilibrium.** A floating W charges until it
   stops collecting electrons. So "V_W = the potential at which W's collected
   fraction → 0" is a self-consistent boundary condition, found by a root-find
   on firep drift traces (`transparency_report` counts where electrons land). It
   replaces a free V_W with a physical constraint, and its value is a
   prediction that can be checked against the fit.
3. **A response-level residual** on top of 1 for what physics can't express
   (the time stretch may be one such term). Keep it small and regularised, and
   report it as the "unexplained" part.

Ramo sum rules are left free and checked afterwards (App. A). Note that the
tuned truth is normalised to one electron on V by construction, and path
reweighting need not respect the zero-sum rule on U and W.

## Findings so far (S3, S4)

**The firep bad-APA cell** (`firep/bad-apa-drift.yaml`, analytic basis only:
exactly harmonic, conductors within 17 mV; seconds per solve).

- At V_W = 0, electrons launched 10 cm from W land on V for 71.5% of impacts,
  on W for 20.4% and on the mesh for 7.6%. The six Garfield paths fall exactly as
  in Ning's slide 8: 0–0.3 pitch on V, 0.4 on W, 0.5 on the mesh.
- Making W more negative closes the W band and widens the mesh band. W first
  collects nothing at **V_W ≈ −46 V** (the floating-equilibrium estimate;
  `firep_vfloat`), where the mesh band peaks at 18.7% (2.5× its V_W = 0
  width). At −100 V the mesh band is back to 7.6%, which may be why Ning found
  −100 V better than 0 V but not good enough.
- The mesh position (−4.71 vs −4.79 mm) changes these fractions by ≤ 0.2%.
- Stagnation line: an impact at exactly x = 0 lies on the line above the
  aligned G/U/V/W wires and ends on G. firep now has `response.impact_nudge`
  (mm, default 0) to launch that path slightly off the line; we use 0.025 mm,
  far below the ~1 mm of transverse diffusion over 10 cm. With it, V_W = 0 gives
  V×4, W×1, mesh×1 on the six paths; the floating equilibrium gives V×5, mesh×1.
- B1 pair (`note/figs/fr-frbone`): the impact-averaged W response goes from net
  −0.14 e (the W-collected band) to pure induction, and W's peak current drops
  from 2.63e-3 to 0.50e-3.
- *Scoring-side observation, not used to choose anything:* the floating-
  equilibrium peaks (U 0.513, V 3.23, W 0.498 ×10⁻³) are close to those of the
  tuned truth (0.522, 3.49, 0.475), and far from V_W = 0 on W (2.63). This
  supports the hypothesis that the MCMC tuning emulates a floating W at its
  equilibrium. The fit must establish this; the peaks alone don't.

**WCT simulation.**

- On anode 0 with the bad-APA response, V is unipolar (~280 ADC on a crossed
  channel) and W bipolar and small (−6 to +15 ADC), as in Ning's data; with the
  nominal response the roles reverse. Anode 2 is normal. Gain 14 mV/fC, no
  noise, no charge fluctuation (a deterministic closed loop).
- Gotcha: `DepoFileSource` aborts on any array in the depo file besides
  `depo_data_N`/`depo_info_N`, so the track table is a separate file.
- Time origin: firep's Walkowiak speed at 500 V/cm is 1.628 mm/µs vs Garfield's
  1.565. So firep responses arrive ~3.7 µs earlier from the 10 cm response
  plane. This is harmless within B1 (firep vs firep); for B2 it is a t0 the fit
  must absorb or that we align explicitly. Seen directly in `frames-data`: both
  firep priors lead the pseudo-data by ~7 ticks. On W, the V_W = 0 prior shows a
  75 ADC collection bump the data lack; the floating prior has the data's
  bipolar shape.

## Findings so far (S6: learner 1 with WCT in the loop)

The simulated ADC is linear in the field response, and learner 1 has only a
few physical parameters, so WCT itself is the forward model. Each trial is a
firep solve, written as a WCT response file, then a WCT simulation of the fit
sample: every 20th shot, 25 shots, anode 0 only, about 20 s. There is no
surrogate, so no closure problem. `scripts/scan_loss.py` computes χ² over
anode 0 with an assumed 3.5 ADC white noise; a common time shift is profiled
via FFT cross-correlation at quarter-tick steps.

- **B1** (pseudo-data from firep at V_W = −46 V): the V_W scan has its minimum
  at −46 V with zero time shift and χ² ≈ 0. It is very sharp (±2 V costs
  Δχ² ≈ 2–4×10⁴), so V_W is identified to ~0.01 V in this noise-free idealisation.
  Model mismatch, not statistics, will set the real limit.
- **B2** (tuned-Garfield pseudo-data), point-sampled firep tables: the fitted time
  shift is 7.25–7.5 ticks, exactly the known drift-speed offset. But the χ²
  landscape is shallow and bumpy, and its residual is dominated by W. U and V
  match well at any V_W. On W the data show a small positive bump and a weak
  slow tail, whereas firep at −46 or −100 V has a negative lobe the data lack.
  The "best" −30 V model is an unphysical compromise: its W-collection bump
  partly fills that lobe.
- The tuned truth's 0.5-pitch (mesh-collected) path has a long, slow positive
  current (slide 28). firep's mesh electrons arrive ~10 µs after V collection;
  Garfield's take ~12–15 µs. A saddle-point explanation is ruled out: at 51
  impacts firep's mesh-band transit is uniform (68.1–68.9 µs over 0.42–0.50
  pitch). Remaining suspects: the region below W (mesh distance, unconfirmed)
  and the low-field drift speed.
- Point sampling misrepresents band edges: at the floating equilibrium the
  0.4-pitch point lands on V, but its 0.35–0.45 bin is ~60% V, 10% W, 30% mesh.
  `firep_to_wct.py --bin-average` builds the six-path table from 51 impacts,
  each path the average over its bin. Prior names now encode variants:
  `vw<V>[m<mesh>][b]`, e.g. `vwfloatm8b`.
- The floating equilibrium, now found by bisection (`--bisect`, to 0.5 V), is
  −45.4 V with the mesh at 4.71 mm, and moves with mesh depth: −57.2 (6 mm),
  −62.4 (8), −64.6 (10), −67.6 (12), −76.3 V (15 mm).
- **B1 again, truth off-grid** (−45.43 V): recovered −45.41 ± 0.01 V.
- **B2 with learner 1 fails on W.** Residual χ² as a fraction of each plane's
  signal χ² (a null model): U 0.6%, V 0.3–0.7%, but W 240–720%. On W every
  physical model does worse than predicting zero. Bin-averaging halves the
  total residual; a deeper mesh (W floating for each depth) makes W worse; the
  best mesh is the shallowest scanned.
- **Why (scoring side, `paths_truth`):** the truth's 0.4- and 0.5-pitch paths,
  the ones the MCMC renormalised and stretched, break the Ramo sum rules. W at
  0.5 pitch carries −0.108 e net, with a slow positive tail still running at the
  window's end; at 0.4 pitch the electron is collected by neither V (+0.002 e) nor
  W (+0.041 e). No electrostatic model reaches that. This is App. A's sum-rule
  diagnostic doing its job: a data-driven correction that violates the sum
  rules is compensating for something that is not the field.
- **Hypothesis for real data (user to judge):** a long W tail with non-zero net
  charge is what a *different AC coupling of the floating W channels* would
  look like (e.g. a changed RC high-pass with no bias resistor). The tuning may
  have absorbed an electronics effect into the field response. If so, learner 1
  should gain a per-plane (W) RC time constant alongside V_W. WCT's PIR already
  takes `long_responses` (RC) per plane trio, so this is a config-level change.

**Correction (after comparing with the untuned file):** the RC hypothesis
above does not survive. `np04hd-garfield-6paths.json.bz2` is the untuned model
(believed V_W = 0; to be confirmed). Fitting tuned(t) ≈ f · untuned(t_s + (t − t_s)/s)
path by path:

- Almost every path: f ≈ 1.02 and s ≈ 0.97–0.98, a global renormalisation fitted
  to ≤ 0.1%. 0.3 pitch: ×1.36, s = 0.80.
- W at 0.5 pitch: ×1.7 with stretch 1.87 about ~62 µs (the "1.90" of the slide-15
  label). A stretched copy of the untuned path, including its abrupt,
  never-closing end.
- W at 0.4 pitch: −0.049 × the untuned W-collection path, a *negative* weight.
- The untuned file's own 0.5-pitch W path already fails to close (net −0.073 e):
  its Garfield drift line stops short of ground. In firep that path closes, and
  the exact symmetry line x = p/2 is not trapped at V_W = 0 or at the floating
  equilibrium. So a missing nudge onto a saddle is not the cause in firep's field;
  it may be a Garfield termination or integration detail (ask X. Ning).

An RC high-pass would pull the net charge toward zero with an exponential tail.
The tuned tail is a plateau that stops exactly where the stretched, truncated
Garfield path ends, which is an artifact signature.

**Is the tuning a composition?** (`project_truth.py`, `proj_truth`,
`proj_untuned`.) Fit each impact bin as a non-negative distribution of electron
starting positions within ±0.1 pitch, with weights shared by every plane and
wire (the same electrons induce all signals), on a 51-impact firep basis, one
global time map, both sides shaped by the cold electronics (raw 0.1 µs
comparisons are dominated by spikes and jitter).

- Tuned truth: bins 0–0.3 pitch are compositions (Σw ≈ 1.00, shaped residuals
  ≲ 1–2%). Bins 0.4 and 0.5 are not: W residual 36–40× and 61–67% of the signal,
  Σw 0.78 and 0.47. There the W paths were rescaled and stretched while V and U
  at the same impacts were left alone, which no set of electrons can produce.
  (Slide 12 says the same factors were applied to V and W; the file doesn't show
  that. Ask.)
- Untuned file on firep at V_W = 0: bins 0–0.4 are compositions (0.4: Σw 1.11,
  W residual 3%); only the non-closing 0.5 path fails (Σw 0.43, 30%).

**B0** (WCT frames from the untuned file vs firep V_W scans): with bin-averaged
tables the minimum is bracketed at **V_W = +2.7 V** (scan to +20 V), consistent
with the untuned file being V_W = 0. Residuals are U 0.6%, V 0.13%, W 28%, the
last largely the 0.5-pitch artifact. Point-sampled tables give a bumpy landscape.

**B2′** (truth = the tuned file projected onto physical compositions, hidden
side, `data_proj`): learner 1 with bin-averaged tables reaches W residual 16%
(vs 244% against the raw tuned file), best V_W ≈ −30 V. The projection is itself
a compromise at 0.4/0.5 pitch, so this is not yet a clean target.

**Proposed next: learner 2, composition weights fitted to ADC.** The simulated
ADC is linear in the composition weights w_k(d), so fitting them to the data is a
non-negative least-squares problem whose columns are WCT simulations of single
basis components. For bins 0.4 and 0.5 with 21 starting positions each, that's
42 WCT runs of ~20 s. This is "learning the response from data" with the
physical constraints built in (sum rules per component, cross-plane
consistency). Against B2′ it is a closure test (the truth is exactly
representable); against the raw tuned truth it finds the physical part of the
tuning, and the leftover is the non-physical remainder.

## Findings so far (S6: learner 2)

Learner 2 fits composition weights w_k(d) ≥ 0 (per impact bin k, over starting
offsets d, shared by all planes and wires) to the ADC. WCT is linear in the
field response, so the model is Σ w_c S_c, with S_c the analog (undigitized)
WCT simulation of one *component* response: all paths zero except bin k,
filled with firep's path for offset d (`component_fr.py`). There are 50
components: d every 0.05 pitch in bins 0–0.3, every 0.01 in bins 0.4 and 0.5,
where the W-landing sliver is ~0.01 pitch wide. Frames are cropped to windows
around the tracks (`l2_crop.py`), and NNLS runs on the Gram matrix with the
data's time shift profiled (`learner2.py`). Analog → ADC is exactly
2¹⁴ / 1.4 V (checked: residual 0.32 ADC rms, i.e. quantisation). Residuals
below are fractions of each plane's signal energy.

| fit | U | V | W |
|---|---|---|---|
| B1 closure (firep point-sampled floating W) | 0.003% | 0.003% | 0.10% |
| **B2′** (physical projection of the tuned truth) | **0.009%** | **0.006%** | **0.21%** |
| B2 (tuned truth) | 0.18% | 0.25% | 76% |
| B2, planes balanced (each χ² / own signal) | 0.63% | 1.2% | 24% |
| held-out, B2′ learned → B2′ | 0.009% | 0.004% | 0.24% |
| held-out, prior → B2′ | 0.98% | 0.83% | 14% |
| held-out, B2 learned → B2 | 0.18% | 0.26% | 78% |
| held-out, prior → B2 | 0.68% | 1.2% | 260% |

- **The method works on a physical target.** B2′ is fitted and predicted on
  held-out shots at the digitisation floor, from 14% (prior) to 0.24% on W.
- **The tuned truth is not the response of any set of electrons.** Even with W
  weighted like U and V, the best composition misses W by 24%, degrades U/V
  ~5×, and the per-bin weights stop summing to one electron.
- **Identifiability:** a track crosses each wire at every impact position, so
  the ADC pins down the response as tracks sample it (close to the
  impact-averaged one), not each impact bin. Ad hoc check: the B2′ learned table
  differs from the truth by 20–100% (shaped rms) bin by bin but 2.5–17% in the
  impact average, while predicting the ADC to 0.01–0.2%. Scoring must be by
  prediction (App. A's third criterion), not table vs table. Separating bins
  needs tracks nearly parallel to a plane's wires, or point sources (³⁹Ar).
- Pitfalls fixed along the way: profile the time shift per reference (firep- vs
  Garfield-timed data); keep the projection's time map shift-only (learner 2
  has no stretch); name components by offset, not list index; symmetrise the
  nudged x = 0 path over mirror wires (`firep_to_wct.py`).

## The laser

The CSVs (`inputs/`, source in the `.url` file) are `;`-separated, one row per
shot. What the data show (`scripts/laser.py` documents the assumptions):

- Timestamps step by 6.25e6 counts: 10 Hz if that is the 62.5 MHz DAQ clock.
  The large run is a 494 s scan of 4946 shots, the small one 20 shots.
- Laser position is given in two frames that differ by a constant translation
  (36.005, 327.837, 234.804) cm. We use the "LArSoft" columns, the frame of the
  WCT wires file. `LaserDir` always has norm 0.1; we normalise it and
  **assume** it is in the same axes.
- The large run scans one rotary stage (`ANGLE RNN600` 93.5°→129.1°) with the
  vertical direction component fixed; the small run has 5 settings × 4 shots.
- The periscope is at LArSoft (−247.7, 583.8, 351.7) cm: over anode 2 (online
  APA2), 105 cm from the anode plane. **You confirmed it can fire into the bad
  APA.** These two runs happen to fire toward +z, away from it: 0% reach anode 0.

**Sim-to-sim treatment:** keep the real port and mirror the beams in z
(`laser.flip: [1, 1, -1]`). Every shot then crosses anode 2 and then anode 0,
with a median 2.7 m of track in the bad APA's volume. Anode 2 gives the null
test for free. Mirroring swaps the U and V angle coverage (angle from drift in
the (drift, pitch) plane: U 31–84°, V 54–87°, W 51–86°). The earlier "shift by
one APA" option remains (`laser.shift_mm`).

Full understanding of the CSVs isn't needed for sim-to-sim. For sim-to-real
we'll want:

1. runs that actually fire into anode 0 (from the `.url` folder);
2. the frame and sign convention of `LaserDir`, and whether the position is the
   beam's entry point into the argon.

## Evaluation of `~/dune/xerosere`

**Sufficient for simulation; no C++ needed so far.** The sandbox mounts it
writable.

- Branch `wctfrdata` (from `master` 79c4043d) is checked out at
  `worktrees/wctfrdata/wire-cell-toolkit`, and variant `wctfrdata` is
  registered in `.xerosere/config.toml` but not built. Until C++ changes are
  needed, `env.sh` runs the `gcc15-wctmaster` install (the same commit) with
  `WIRECELL_PATH` taking `cfg/` from the `wctfrdata` worktree.

- `installs/envs/gcc15/bin/wire-cell` runs with `env.sh`'s library paths.
  Variants: `gcc15` (WCT `devel/` at `084d087e`, edep/spng feature branch),
  `gcc15-wctmaster`, `gcc15-wctspng`.
- WCT's `cfg/pgrapher/experiment/pdhd/` already assigns one response per anode
  (`params.files.fields` = [bestfit, 1d565, 1d565, 1d565]). Its `sim.jsonnet`
  uses `tools.pirs[n]` per anode, so `cfg/sim-laser.jsonnet` only overrides
  `fields`.
- `DepoFileSource` reads our depo `.npz` ((N,7) float, (N,4) int, checked
  against its source). `FrameFileSink` writes frames.

**Gaps.**

1. The `wirecell-*` commands on `PATH` come from `~/newsp/python`, not
   `~/dune/xerosere/python/wire-cell-python`, and neither is importable from
   firep's `.venv`. Plan: install xerosere's copy editable into firep's `.venv`.
2. Done: the `wctfrdata` branch, and `scripts/firep_to_wct.py` (firep → WCT
   response file, 21 wires × 6 impacts).
3. GPU: use the cu124 side environment for firep training (S4, S6).

## Things to provide

- [ ] **From X. Ning**: the untuned Garfield V_W=0 response in WCT format (and
      V_W=−100 if handy); the tuned parameters (f₅, f₆, start points, stretch
      factors, with MCMC widths), including what the label
      `0.01_2.10_1.00_1.90_550` means; the mesh position in her Garfield cell; and
      how the per-path stretch is defined (slide 14). With these we can also
      rebuild the truth from the untuned file ourselves, which checks our
      understanding.
- [ ] The 8 data tracks' directions (from imaging) and her data waveforms, if
      shareable: a ready-made sim-to-real benchmark for later.
- [ ] Laser runs that fire into anode 0 (not blocking).
- [ ] Choices: which wire-cell-python checkout; electronics settings
      (defaults: 14 mV/fC, 2.2 µs, PDHD params).

## Plan

**S0 — environment** (`check_env`). wire-cell-python importable from the demo
python; `wctfrdata` variant if wanted.

**S1 — the target** (`fr_good_bad` → note Sec. 3). Done. Next: once the untuned
file arrives, `fr_compare` untuned vs tuned (the correction to learn), on the
scoring side.

**S2 — laser** (`laser_coverage` → Sec. 4; `anode_box`, `laser_depos`). Done
for sim-to-sim. Next: coverage of impact position within the pitch per plane.

**S3 — WCT simulation** (`data`, `sim_prior`, `frames_data`). Done for the
noise-free, fluctuation-free case. Next: a pre-digitizer (float) frame output
for the S5 closure, and a per-shot "how different is data from the prior"
metric.

**S4 — firep bad-APA models** (`firep_*`, `prior`). Done: the landing map, the
floating equilibrium, and priors at V_W = 0, −100 V and floating. Waiting on
Ning's untuned file for B0. Next: finer impact sampling (11 or 21 impacts) to
see how much the 6-path tabulation misrepresents the 0.4–0.5-pitch region.

**S5 — differentiable forward model** (`forward_closure` → Sec. 5). A torch copy
of the WCT chain: diffusion onto WCT's impact grid, convolution with field and
electronics response, resampling, gain, soft digitisation. Validate against WCT
until the residual is far below the prior–truth difference.

**S6 — fit** (`fit`). B1 done by WCT-in-the-loop scans. B2 with learner 1
fails on W for a reason we understand (see Findings; decision needed). Learner
1 → 2 → 3 above. Electronics, t0,
drift speed, diffusion and dQ/dx fixed at true values. Output a WCT response
file.

**S7 — score** (`score` → Sec. 6). Learned vs truth per plane and impact.
Fraction of the prior–truth difference recovered. Sum rules. WCT re-simulation
vs data. Null test on anode 2. For learner 1: fitted V_W and mesh position vs
the floating-equilibrium prediction.

**S8 — robustness.** Noise and laser repeats. Free t0, gain and diffusion.
Pointing error. Fewer shots or fewer angles (Ning used 8 tracks: how few
suffice?). Optionally the good→bad stress test.
