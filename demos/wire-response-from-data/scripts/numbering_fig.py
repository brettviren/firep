#!/usr/bin/env python3
"""PDHD anode numbering: WCT (offline) anode vs online APA, top view.

    numbering_fig.py WIRES OUTPUT.pdf [--laser X,Z]  (laser port, LArSoft cm)

Geometry (x and z extents of each anode's wire planes) comes from the WCT
wires file.  The online names follow DUNE's channel-map code
(duneprototypes Protodune/hd/Tool/PdhdChannelRanges.h and
ChannelMap/mapmakers/MakePD2HDChannelMap_WIBEth_v1_visiblewires.C), which
viewed from above gives

    --->   TPS1  TPS3        APA3  APA4
    beam   TPS0  TPS2        APA1  APA2

with TPC set (TPS) n equal to WCT anode n, first offline channel 2560 n.
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wires

ONLINE = {0: "APA1", 1: "APA3", 2: "APA2", 3: "APA4"}
NOTE = {0: "the bad APA: W floating"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("wires")
    ap.add_argument("output")
    ap.add_argument("--laser", default="-247.7,351.7", help="port x,z [cm]")
    a = ap.parse_args()
    store = wires.load(a.wires)
    lx, lz = (float(v) for v in a.laser.split(","))

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.axhline(0.0, color="0.3", lw=2)
    ax.text(470, 0, " cathode", va="center", fontsize=9, color="0.3")
    for ident in range(4):
        faces = wires.anode_faces(store, ident)
        xs = [p["x"] for f in faces for p in f]
        z0 = min(p["yz"][0][1] for f in faces for p in f) / 10
        z1 = max(p["yz"][1][1] for f in faces for p in f) / 10
        xc = np.mean(xs) / 10
        bad = ident == 0
        ax.add_patch(plt.Rectangle((z0, xc - 6), z1 - z0, 12,
                                   color="C3" if bad else "C0", alpha=0.9))
        # its drift volume, toward the cathode
        ax.add_patch(plt.Rectangle((z0, min(0, xc)), z1 - z0, abs(xc),
                                   color="C3" if bad else "C0", alpha=0.07))
        ymid = xc / 2
        zc = (z0 + z1) / 2
        ax.text(zc, ymid + 40, f"WCT anode {ident}", ha="center", fontsize=13,
                weight="bold")
        ax.text(zc, ymid + 5, f"online {ONLINE[ident]}", ha="center", fontsize=11)
        ax.text(zc, ymid - 28, f"channels {2560 * ident}-{2560 * ident + 2559}",
                ha="center", fontsize=8, color="0.3")
        if ident in NOTE:
            ax.text(zc, ymid - 60, NOTE[ident], ha="center", fontsize=9,
                    style="italic", color="C3" if bad else "0.2")
    ax.plot([lz], [lx], "k*", ms=12)
    ax.text(lz + 8, lx - 5, "laser port", fontsize=8, va="top")
    ax.annotate("", xy=(40, -420), xytext=(-60, -420),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(-60, -440, "beam", fontsize=9, va="top")
    ax.set(xlim=(-80, 520), ylim=(-470, 420), xlabel="z [cm]  (LArSoft)",
           ylabel="x [cm]  (drift)", title="ProtoDUNE-HD from above")
    fig.tight_layout()
    fig.savefig(a.output)


if __name__ == "__main__":
    main()
