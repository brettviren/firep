"""Front-end electronics: shaping amplifier, anti-aliasing filter, ADC.

Two chains are available, because "a 14 mV/fC amplifier into a filter with a
2 us characteristic time" can mean either.

``mode = "shaper"`` (default)
    The front end responds to an impulse of charge ``Q`` with a *pulse* of peak
    height ``G Q`` and width set by ``tau``:

    .. math::

        V(t) = G \int I(t')\, s_\tau(t - t')\, dt',
        \qquad s_\tau(t) = \left(\tfrac{t}{\tau}\right)^{\!n}
                             e^{\,n(1 - t/\tau)} ,

    a CR-RC^n shaper normalised to unit *peak* at ``t = tau``.  This is what a
    gain quoted in mV/fC alongside a shaping time means in practice, and it is
    the standard LArTPC front end: 14 mV/fC with 2 us shaping is the nominal
    MicroBooNE cold-electronics setting.  A collection signal comes out
    unipolar, an induction signal bipolar, and both return to baseline.

``mode = "integrator"``
    The literal reading: a charge amplifier that integrates, followed by a
    unit-area low-pass of time constant ``tau``:

    .. math::

        V(t) = G \left(h_\tau * \!\int^t\! I\, dt'\right)(t) .

    Its response to an impulse of charge is a *step*, so a collection signal
    rises and stays there for ever and an event display fills in behind the
    track.  Physically it is what an ideal charge amplifier with no feedback
    resistor does; ``decay`` adds that resistor and restores a falling edge.

Both end at the same ADC:

.. math::

    a_k = \mathrm{round}\!\left(V(k\,T)/\Delta\right),
    \qquad \Delta = \frac{2 V_\mathrm{range}}{2^{b}} ,

sampling (not averaging) at ``adc_rate`` -- which is what makes the
anti-aliasing shaping upstream necessary rather than decorative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

E_IN_FC = 1.602176634e-4  # one electron in femtocoulomb

# MicroBooNE / BNL cold-electronics shaper.  The transfer function has one real
# pole and two complex-conjugate pairs; the impulse response is therefore a sum
# of a damped exponential and two damped sinusoids, in units of t/Tp:
#
#   f(x) = 4.31054 e^{-2.94809x}
#        - 5.2404  e^{-2.82833x} cos(1.19361x) + 1.524912 e^{-2.82833x} sin(1.19361x)
#        + 0.929848 e^{-2.40318x} cos(2.5928x) - 0.655368 e^{-2.40318x} sin(2.5928x)
#
# It starts at zero, is unipolar, and has died away by x = 10.  Its maximum sits
# at x = 1.1735, so the shaping-time parameter Tp is *not* the peaking time; we
# quote `tau` as the peaking time (the convention behind "2 us shaping") and
# rescale accordingly.  The overall amplitude of the published coefficients is a
# convention; normalising to unit peak makes the quoted mV/fC a peak gain.
UBOONE_PEAK_X = 1.173540  # position of the maximum, in units of Tp


def uboone_shape(x: np.ndarray) -> np.ndarray:
    """Cold-electronics impulse response against ``x = t / Tp`` (unnormalised)."""
    x = np.asarray(x, dtype=float)
    e1 = np.exp(-2.94809 * x)
    e2 = np.exp(-2.82833 * x)
    e3 = np.exp(-2.40318 * x)
    return (4.31054 * e1
            - 5.2404 * e2 * np.cos(1.19361 * x)
            + 1.524912 * e2 * np.sin(1.19361 * x)
            + 0.929848 * e3 * np.cos(2.5928 * x)
            - 0.655368 * e3 * np.sin(2.5928 * x))


def shaping_kernel(kind: str, tau: float, dt: float, order: int = 2,
                   n_tau: float = 10.0, normalise: str = "peak") -> np.ndarray:
    """Impulse response of the front end.

    ``normalise='peak'`` scales the kernel to unit maximum, so that a gain in
    mV/fC is a peak gain (the shaper convention).  ``normalise='area'`` scales
    it to unit integral, so that it smooths without rescaling (the
    anti-aliasing convention, used after an integrator).
    """
    if tau <= 0:
        raise ValueError("filter time constant must be positive")
    if kind == "none":
        return np.array([1.0 if normalise == "peak" else 1.0 / dt])
    span = max(int(round(n_tau * tau / dt)), 2)
    t = np.arange(span) * dt
    if kind == "rc":  # single pole
        h = np.exp(-t / tau)
    elif kind in ("rc2", "crrc"):  # CR-RC^n, peak at t = tau
        n = max(int(order), 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            h = (t / tau) ** n * np.exp(n * (1.0 - t / tau))
        h[0] = 0.0
    elif kind == "gaussian":
        t0 = 0.5 * n_tau * tau
        h = np.exp(-0.5 * ((t - t0) / tau) ** 2)
    elif kind == "uboone":
        # tau is the peaking time, so Tp = tau / UBOONE_PEAK_X.
        h = uboone_shape(t / (tau / UBOONE_PEAK_X))
        h[t > 10.0 * tau / UBOONE_PEAK_X] = 0.0
    else:
        raise ValueError(f"unknown filter kind {kind!r}")
    if normalise == "peak":
        scale = h.max()
    else:
        scale = h.sum() * dt
    if scale <= 0:
        raise ValueError(f"filter {kind!r} has non-positive normalisation")
    return h / scale


@dataclass
class Electronics:
    """One readout channel's front end."""

    gain: float = 14.0  # mV/fC (peak gain in shaper mode)
    mode: str = "shaper"  # shaper | integrator
    filter: str = "uboone"  # none | rc | rc2 | gaussian | uboone
    order: int = 2  # CR-RC^n order, for filter = rc2
    tau: float = 2.0  # us, shaping / filter characteristic time
    decay: float | None = None  # us, amplifier feedback decay (integrator mode)
    adc_rate: float = 2.0  # MHz
    adc_bits: int = 14
    adc_range: float = 1000.0  # mV, half-range (so +-1 V)
    baseline: float = 0.0  # mV, pedestal added before digitising

    def __post_init__(self):
        if self.adc_bits < 2:
            raise ValueError("adc_bits must be at least 2")
        if self.adc_rate <= 0 or self.adc_range <= 0:
            raise ValueError("adc_rate and adc_range must be positive")
        if self.mode not in ("shaper", "integrator"):
            raise ValueError(f"unknown electronics mode {self.mode!r}")

    # -- derived ------------------------------------------------------------

    @property
    def sample_period(self) -> float:
        """ADC sampling period in us."""
        return 1.0 / self.adc_rate

    @property
    def lsb(self) -> float:
        """Volts per count, in mV."""
        return 2.0 * self.adc_range / (2 ** self.adc_bits)

    @property
    def count_limits(self) -> tuple[int, int]:
        half = 2 ** (self.adc_bits - 1)
        return -half, half - 1

    @property
    def mv_per_electron(self) -> float:
        return self.gain * E_IN_FC

    def describe(self) -> str:
        chain = (f"shaper {self.filter}" + (f"^{self.order}" if self.filter == "rc2" else "")
                 + f", peak at tau = {self.tau} us"
                 if self.mode == "shaper" else
                 f"integrator + {self.filter} filter, tau = {self.tau} us"
                 + (f", amplifier decay {self.decay} us" if self.decay
                    else ", pure integrator"))
        return "\n".join([
            f"gain {self.gain} mV/fC  ({self.mv_per_electron * 1e3:.4f} uV per electron)",
            chain,
            f"ADC {self.adc_rate} MHz ({self.sample_period} us), {self.adc_bits} bit, "
            f"+-{self.adc_range} mV  ->  {self.lsb * 1e3:.3f} uV/count, "
            f"{self.lsb / self.mv_per_electron:.1f} electrons/count",
        ])

    # -- the chain ----------------------------------------------------------

    def kernel(self, dt: float) -> np.ndarray:
        """Impulse response on a grid of step ``dt``."""
        return shaping_kernel(
            self.filter, self.tau, dt, order=self.order,
            normalise="peak" if self.mode == "shaper" else "area",
        )

    def shape(self, current: np.ndarray, dt: float) -> np.ndarray:
        """Current in e/us -> front-end output voltage in mV.

        The last axis is time.
        """
        if self.mode == "shaper":
            # V = G * (I * s), with s at unit peak: an impulse of charge Q
            # gives a pulse of height G*Q.
            charge_rate = current * E_IN_FC  # fC/us
            return self.gain * _convolve_last(charge_rate, self.kernel(dt) * dt)

        # integrator: accumulate charge, then smooth with a unit-area low-pass.
        if self.decay:
            span = max(int(round(8.0 * self.decay / dt)), 2)
            k = np.exp(-np.arange(span) * dt / self.decay) * dt
            q = _convolve_last(current, k)
        else:
            q = np.cumsum(current, axis=-1) * dt  # electrons
        volts = q * self.mv_per_electron
        if self.filter == "none":
            return volts
        return _convolve_last(volts, self.kernel(dt) * dt)

    def digitise(self, volts: np.ndarray, t: np.ndarray):
        """Sample at the ADC rate and quantise.  Returns ``(counts, sample_times)``."""
        t0, t1 = float(t[0]), float(t[-1])
        n = int(np.floor((t1 - t0) / self.sample_period)) + 1
        ts = t0 + np.arange(n) * self.sample_period
        # An ADC samples; it does not average.  The anti-aliasing filter above
        # is what makes that safe.
        idx = np.clip(np.searchsorted(t, ts), 0, len(t) - 1)
        sampled = np.take(volts, idx, axis=-1) + self.baseline
        lo, hi = self.count_limits
        counts = np.clip(np.rint(sampled / self.lsb), lo, hi).astype(np.int16)
        return counts, ts

    def readout(self, current: np.ndarray, t: np.ndarray):
        """Full chain: current in e/us -> ``(counts, sample_times, volts)``."""
        dt = float(t[1] - t[0])
        volts = self.shape(current, dt)
        counts, ts = self.digitise(volts, t)
        return counts, ts, volts


def _convolve_last(a: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Causal convolution along the last axis, truncated to the input length."""
    n = a.shape[-1]
    m = len(kernel)
    nf = 1 << int(math.ceil(math.log2(n + m - 1)))
    fa = np.fft.rfft(a, n=nf, axis=-1)
    fk = np.fft.rfft(kernel, n=nf)
    out = np.fft.irfft(fa * fk, n=nf, axis=-1)
    return out[..., :n]
