"""Read the PDHD laser "angles_pos_cib" CSV files.

What is known about these files (design.md, "Laser files"), and what the
readers here assume:

- ';'-separated, one header row, one row per laser shot at 10 Hz (TIMESTAMP
  steps by 6.25e6 counts of the 62.5 MHz DAQ clock = 0.1 s).
- Laser position is given twice, in cm: a detector-centred frame and a
  "LArSoft" frame; the two differ by a constant translation
  (36.005, 327.837, 234.804) cm.  We use the LArSoft columns because the WCT
  wires file is in LArSoft coordinates.
- LaserDir.{X,Y,Z} has norm 0.1 in every row seen; we treat it as a direction
  only and normalise it.  ASSUMED: it is expressed in the same axes as the
  LArSoft position (the two position frames differ only by a translation).
- RNN800 / RNN600 / LSTAGE are raw stage encoders with ANGLE columns derived
  from two of them; they are kept but not interpreted.

Everything returned is in WCT system units (mm, ns).
"""
import numpy as np

CM = 10.0   # mm

COLUMNS = {
    "timestamp": "TIMESTAMP",
    "rnn800": "RNN800",
    "rnn600": "RNN600",
    "angle_rnn600": "ANGLE RNN600",
    "lstage": "LSTAGE",
    "angle_lstage": "ANGLE LSTAGE",
    "x": "LaserPos.X LArSoft (cm)",
    "y": "LaserPos.Y LArSoft (cm)",
    "z": "LaserPos.Z LArSoft (cm)",
    "dx": "LaserDir.X()",
    "dy": "LaserDir.Y()",
    "dz": "LaserDir.Z()",
}


def read_csv(path):
    """Return dict of column arrays: ``origin`` (N,3) mm, ``direction`` (N,3)
    unit, ``timestamp`` (int64 DAQ counts) and the raw stage columns."""
    with open(path) as fp:
        header = [h.strip() for h in fp.readline().split(";")]
    index = {name: header.index(col) for name, col in COLUMNS.items()}
    ts = np.loadtxt(path, delimiter=";", skiprows=1, usecols=[index["timestamp"]],
                    dtype=np.int64, ndmin=1)
    a = np.loadtxt(path, delimiter=";", skiprows=1, ndmin=2)
    col = {name: a[:, i] for name, i in index.items()}
    d = np.column_stack([col["dx"], col["dy"], col["dz"]])
    out = dict(timestamp=ts,
               origin=np.column_stack([col["x"], col["y"], col["z"]]) * CM,
               direction=d / np.linalg.norm(d, axis=1)[:, None])
    for k in ("rnn800", "rnn600", "angle_rnn600", "lstage", "angle_lstage"):
        out[k] = col[k]
    return out


def transform(origin, direction, shift=(0.0, 0.0, 0.0), flip=(1.0, 1.0, 1.0)):
    """Move recorded shots onto the anode under study (sim-to-sim only).

    ``shift`` (mm) translates the port; ``flip`` multiplies direction
    components, e.g. (1, 1, -1) mirrors the beam in z about the port.
    """
    return (np.asarray(origin, float) + np.asarray(shift, float),
            np.asarray(direction, float) * np.asarray(flip, float))


def shots(paths, shift=(0.0, 0.0, 0.0), flip=(1.0, 1.0, 1.0), stride=1, per_event=1):
    """Shots from one or more CSV files as dicts for laser_depos.make_depos.

    ``shift`` and ``flip`` are applied by ``transform``.  ``stride`` keeps
    every n-th shot; ``per_event`` shots share one depo set, each at t = 0
    (the laser is the trigger).
    """
    if isinstance(paths, str):
        paths = [paths]
    out = []
    for path in paths:
        c = read_csv(path)
        o, d = transform(c["origin"], c["direction"], shift, flip)
        for i in range(0, len(c["timestamp"]), stride):
            out.append(dict(origin=o[i], direction=d[i], time=0.0,
                            event=len(out) // per_event))
    return out
