#!/usr/bin/env python3
"""Anode geometry from a WCT wires file.

    wires.py WIRES.json.bz2 ANODE [--cathode-x X_MM]   # prints the active box

The active box of an anode is the yz extent of its wire planes on the face
looking at the cathode, and in x the span from that face's outermost wire
plane to the cathode.  Printed as ``x0,y0,z0,x1,y1,z1`` in mm, the form
``laser_depos.py --box`` takes.
"""
import argparse
import bz2
import json

import numpy as np


def load(path):
    fp = bz2.open(path) if str(path).endswith(".bz2") else open(path)
    return json.load(fp)["Store"]


def anode_faces(store, ident):
    """For the anode with WCT ident, a list per face of planes, each a dict of
    ``ident``, ``x`` (plane position), ``dir`` (unit wire direction, the mean
    over wires), ``yz`` (min, max) extent of the wire endpoints."""
    pts = np.array([[p["Point"][k] for k in "xyz"] for p in store["points"]])
    anode = next(a["Anode"] for a in store["anodes"] if a["Anode"]["ident"] == ident)
    faces = []
    for fi in anode["faces"]:
        planes = []
        for pi in store["faces"][fi]["Face"]["planes"]:
            pl = store["planes"][pi]["Plane"]
            wires = [store["wires"][wi]["Wire"] for wi in pl["wires"]]
            tails = pts[[w["tail"] for w in wires]]
            heads = pts[[w["head"] for w in wires]]
            d = heads - tails
            d /= np.linalg.norm(d, axis=1)[:, None]
            d = np.where(d[:, 1:2] < 0, -d, d).mean(0)   # orient +y, average
            ends = np.vstack([tails, heads])
            planes.append(dict(ident=pl["ident"], x=float(ends[:, 0].mean()),
                               dir=d / np.linalg.norm(d),
                               yz=(ends[:, 1:].min(0), ends[:, 1:].max(0))))
        faces.append(planes)
    return faces


def active_face(store, ident, cathode_x=0.0):
    """The face of the anode whose planes are nearest the cathode."""
    faces = anode_faces(store, ident)
    return min(faces, key=lambda planes: min(abs(p["x"] - cathode_x) for p in planes))


def active_box(store, ident, cathode_x=0.0):
    planes = active_face(store, ident, cathode_x)
    xs = [p["x"] for p in planes]
    xnear = min(xs, key=lambda x: abs(x - cathode_x))
    lo = np.min([p["yz"][0] for p in planes], axis=0)
    hi = np.max([p["yz"][1] for p in planes], axis=0)
    x0, x1 = sorted([xnear, cathode_x])
    return np.array([x0, lo[0], lo[1]]), np.array([x1, hi[0], hi[1]])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("wires")
    ap.add_argument("anode", type=int)
    ap.add_argument("--cathode-x", type=float, default=0.0, help="[mm]")
    a = ap.parse_args()
    lo, hi = active_box(load(a.wires), a.anode, a.cathode_x)
    print(",".join(f"{v:.2f}" for v in (*lo, *hi)))


if __name__ == "__main__":
    main()
