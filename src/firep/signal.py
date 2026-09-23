"""Compose ionization groups with the field response and the electronics.

Given the arrival density of electrons at the response plane, ``n(x, t)``, and
the single-electron field response ``r_p(x_0, x_w; t)`` -- the current on the
wire at ``x_w`` of plane ``p`` due to an electron launched at ``x_0`` -- the
current on that wire is a convolution in time and a superposition over the
transverse coordinate,

.. math::

    I_{p,w}(t) = \\int\\! dx_0 \\int\\! dt' \\;
        n(x_0, t')\\; r_p(x_0, x_w;\\, t - t') .

Two symmetries make this cheap.  The drift field is periodic over one pitch, so
``r`` depends on the launch position only through its offset within a cell and
on the wire only through how many pitches away it is; and the geometry is
mirror-symmetric about each wire, so only half a pitch of launch positions need
be tabulated.  Both are exactly what ``firep response`` tabulates, so the
superposition reduces to a set of shifted convolutions, done here by FFT.

Crucially the cost does not grow with the number of groups: every group is first
deposited onto one common ``(x, t)`` grid, and it is that grid which is
convolved.  A track of 600 samples costs the same as a track of 60 000.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from .electronics import Electronics
from .ionization import Groups, arrival_grid


@dataclass
class Frame:
    """Digitised waveforms for every plane and wire."""

    planes: list[str]
    wires: np.ndarray  # (nw,) wire index, 0 = the wire at x = 0
    time: np.ndarray  # (nt,) us, response-tick grid
    current: np.ndarray  # (np, nw, nt) e/us
    volts: np.ndarray  # (np, nw, nt) mV, after amplifier + filter
    adc: np.ndarray  # (np, nw, ns) counts
    adc_time: np.ndarray  # (ns,) us
    pitch: float
    t_offset: float = 0.0  # us, drift time removed from the group times
    meta: dict = field(default_factory=dict)

    @property
    def wire_x(self) -> np.ndarray:
        return self.wires * self.pitch

    def describe(self) -> str:
        lo, hi = self.adc.min(), self.adc.max()
        lines = [
            f"planes {self.planes}   wires {self.wires[0]}..{self.wires[-1]}"
            f"   ADC samples {self.adc.shape[-1]} at "
            f"{self.adc_time[1] - self.adc_time[0]:.3f} us",
            f"time window [{self.time[0]:.2f}, {self.time[-1]:.2f}] us "
            f"(group times had {self.t_offset:.2f} us removed)",
            f"ADC counts in [{lo}, {hi}]",
        ]
        for ip, p in enumerate(self.planes):
            peak = np.abs(self.volts[ip]).max()
            tot = self.current[ip].sum() * (self.time[1] - self.time[0])
            lines.append(
                f"  {p:<3} peak |V| {peak:9.3f} mV   peak |ADC| "
                f"{np.abs(self.adc[ip]).max():6d}   net charge over all wires "
                f"{tot:12.1f} e"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------


def trim_response(current: np.ndarray, tick: float, threshold: float = 1e-5,
                  pad: int = 4):
    """Cut the response kernels down to where they are actually non-zero.

    A response computed from the cathode carries tens of microseconds of silence
    while the electron crosses the drift volume.  That is a pure delay: keeping
    it in the convolution kernel costs time and buys nothing.  Returns the
    trimmed kernel and the delay that was removed.
    """
    mag = np.abs(current).max(axis=(0, 1, 2))
    peak = mag.max()
    if peak <= 0:
        return current, 0.0
    live = np.flatnonzero(mag > threshold * peak)
    lo = max(int(live[0]) - pad, 0)
    hi = min(int(live[-1]) + pad + 1, current.shape[-1])
    return current[..., lo:hi], lo * tick


def _impact_grid(impact: np.ndarray, pitch: float):
    """Validate the impact tabulation and return its sub-pitch step."""
    ni = len(impact)
    if ni < 2:
        raise ValueError("the response needs at least two impact positions")
    if not np.isclose(impact[0], 0.0) or not np.isclose(impact[-1], 0.5 * pitch):
        raise ValueError(
            "signal composition needs a half-pitch impact tabulation "
            f"(0 .. {0.5 * pitch} mm); got {impact[0]} .. {impact[-1]}. "
            "Re-run `firep response` with response.half_pitch: true"
        )
    step = np.diff(impact)
    if not np.allclose(step, step[0]):
        raise ValueError("impact positions must be uniformly spaced")
    return float(step[0])


def simulate(resp_arrays: dict, resp_meta: dict, groups: Groups,
             electronics: Electronics, n_sigma: float = 3.0,
             margin_wires: int = 2, trim: float = 1e-5) -> Frame:
    """Fold ``groups`` through the field response and the electronics."""
    time = resp_arrays["time"]
    tick = float(time[1] - time[0])
    pitch = float(resp_meta["pitch_mm"])
    planes = list(resp_meta["planes"])
    offsets = np.asarray(resp_arrays["offsets"], int)
    impact = np.asarray(resp_arrays["impact"], float)
    dx = _impact_grid(impact, pitch)
    ni = len(impact)

    kernels, delay = trim_response(np.asarray(resp_arrays["current"]), tick, trim)
    npl, noff, _, nk = kernels.shape

    # ---- time grid on the response ticks, covering all the arrivals.
    t_lo = float(groups.t.min() - n_sigma * groups.sigma_t.max())
    t_hi = float(groups.t.max() + n_sigma * groups.sigma_t.max())
    nt_a = max(int(math.ceil((t_hi - t_lo) / tick)) + 1, 2)
    t_edges = t_lo + (np.arange(nt_a + 1) - 0.5) * tick

    # ---- transverse grid, aligned so every node maps onto a tabulated impact.
    # Nodes sit at multiples of dx.  Node k belongs to the cell of the nearest
    # wire, at sub-pitch index s in [-(ni-1), ni-1]; the two extreme values are
    # the same physical point on the cell boundary, so a node landing there has
    # its charge split evenly between the two neighbouring cells.  Without that
    # split each cell would be lopsided about its own wire and the response
    # would lose its mirror symmetry.
    per_pitch = 2 * (ni - 1)  # fine steps per pitch
    reach = n_sigma * float(groups.sigma_x.max())
    k_lo = int(math.floor((groups.x.min() - reach) / dx)) - per_pitch
    k_hi = int(math.ceil((groups.x.max() + reach) / dx)) + per_pitch
    kk = np.arange(k_lo, k_hi + 1)
    x_edges = (kk[0] - 0.5) * dx + np.arange(len(kk) + 1) * dx

    dens1d = arrival_grid(groups, x_edges, t_edges, n_sigma)  # (nk, nt)

    m_of = np.floor_divide(kk + per_pitch // 2, per_pitch)
    s_of = kk - m_of * per_pitch  # in [-per_pitch/2, per_pitch/2 - 1]
    m_base = int(m_of.min()) - 1  # room for the boundary split into m-1
    nm = int(m_of.max()) - m_base + 1
    ns = per_pitch + 1  # s from -(ni-1) to +(ni-1)
    s = np.arange(-(ni - 1), ni)

    dens = np.zeros((nm, ns, nt_a))
    edge = -(per_pitch // 2)
    for k in range(len(kk)):
        row = dens1d[k]
        if not row.any():
            continue
        mi = int(m_of[k]) - m_base
        si = int(s_of[k])
        if si == edge:
            dens[mi, si + (ni - 1)] += 0.5 * row
            dens[mi - 1, -si + (ni - 1)] += 0.5 * row
        else:
            dens[mi, si + (ni - 1)] += row
    m_lo = m_base

    # ---- shifted convolutions.
    nmax = int(np.abs(offsets).max())
    nw = nm + 2 * nmax
    nt_out = nt_a + nk - 1
    nf = 1 << int(math.ceil(math.log2(max(nt_out, 2))))
    dens_hat = np.fft.rfft(dens, n=nf, axis=-1)  # (nm, ns, nf)
    kern_hat = np.fft.rfft(kernels, n=nf, axis=-1)  # (npl, noff, ni, nf)
    out_hat = np.zeros((npl, nw, nf // 2 + 1), dtype=complex)

    off_index = {int(o): i for i, o in enumerate(offsets)}
    for si, sv in enumerate(s):
        j = abs(int(sv))  # impact index
        sgn = 1 if sv >= 0 else -1  # mirror symmetry about the wire
        for o in offsets:
            io = off_index[int(o)]
            # Charge in cell m contributes to the wire m + sgn*o.
            start = nmax + sgn * int(o)
            out_hat[:, start:start + nm, :] += (
                dens_hat[None, :, si, :] * kern_hat[:, io, j, None, :]
            )
    current = np.fft.irfft(out_hat, n=nf, axis=-1)[..., :nt_out]

    wires = np.arange(m_lo - nmax, m_lo + nm + nmax)
    if len(wires) != nw:  # pragma: no cover - guarded by construction
        raise AssertionError(f"wire bookkeeping: {len(wires)} != {nw}")
    keep = np.ones(nw, bool)
    if margin_wires >= 0:
        busy = np.abs(current).max(axis=(0, 2))
        live = np.flatnonzero(busy > 1e-9 * max(busy.max(), 1e-30))
        if len(live):
            lo = max(int(live[0]) - margin_wires, 0)
            hi = min(int(live[-1]) + margin_wires + 1, nw)
            keep = np.zeros(nw, bool)
            keep[lo:hi] = True
    current = current[:, keep, :]
    wires = wires[keep]

    t_out = t_lo + delay + np.arange(nt_out) * tick
    volts = electronics.shape(current, tick)
    adc, adc_time = electronics.digitise(volts, t_out)

    meta = {
        "planes": planes,
        "pitch_mm": pitch,
        "response_delay_us": delay,
        "electrons": groups.total,
        "n_groups": len(groups),
        "n_sigma": n_sigma,
        "electronics": {
            "gain_mV_per_fC": electronics.gain,
            "mode": electronics.mode,
            "filter": electronics.filter,
            "tau_us": electronics.tau,
            "decay_us": electronics.decay,
            "adc_rate_MHz": electronics.adc_rate,
            "adc_bits": electronics.adc_bits,
            "adc_range_mV": electronics.adc_range,
            "lsb_mV": electronics.lsb,
            "baseline_mV": electronics.baseline,
        },
        "units": {"time": "us", "current": "e/us", "volts": "mV", "adc": "count"},
    }
    return Frame(planes=planes, wires=wires, time=t_out, current=current,
                 volts=volts, adc=adc, adc_time=adc_time, pitch=pitch,
                 t_offset=groups.t_offset, meta=meta)


# --------------------------------------------------------------------------


def save_npz(path: str, frame: Frame) -> None:
    np.savez_compressed(
        path,
        metadata=np.array(json.dumps(frame.meta, indent=1)),
        wires=frame.wires, time=frame.time, adc_time=frame.adc_time,
        current=frame.current, volts=frame.volts, adc=frame.adc,
    )


def load_npz(path: str):
    blob = np.load(path, allow_pickle=False)
    meta = json.loads(str(blob["metadata"]))
    arrays = {k: blob[k] for k in blob.files if k != "metadata"}
    return arrays, meta
