#!/usr/bin/env python

# Some efficiency plots...
# TODO proper docstring
# stdlib imports
import argparse
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

# Non DESI imports
from astropy.table import Table, vstack
import fitsio
import matplotlib.pyplot as plt
import numpy as np

# TODO remove this at some point to point to a generic simassign import.
import sys
sys.path.append("/pscratch/sd/d/dylang/repos/simassign/src/")
from simassign.mtl import deduplicate_mtl
from simassign.io import *
from simassign.util import get_nobs_arr

parser = argparse.ArgumentParser()
parser.add_argument("--mtls", required=True, nargs='+', default=[], help="list of mtls to load and compare.")
parser.add_argument("--goals", required=True, nargs='+', default=[], help="target number of exposures for each associated fiberassign run/mtl.")
parser.add_argument("--labels", required=True, nargs='+', default=[], help="labels to use for each mtl in the output plot.")
parser.add_argument("--verbose", required=False, action="store_true", help="be verbose when doing everything or not.")
parser.add_argument("--pertile", required=False, action="store_true", help="plot results per num tile/exposure.")
parser.add_argument("--perpass", required=False, action="store_true", help="plot results per pass.")
parser.add_argument("--hist", required=False, action="store_true", help="hist of number of exposures at goal.")
parser.add_argument("--nproc", required=False, type=int, default=1, help="number of multiprocessing processes to use for loading tables.")
parser.add_argument("--density_assign", required=False, action="store_true", help="use density of assigned targets as y axis instead of fraction.")
parser.add_argument("--targs", required=False, action="store_true", help="plots of targets colored by number observations.")
parser.add_argument("-o", "--outdir", required=True, type=str, help="where to save generated plots.")
args = parser.parse_args()

out_dir = Path(args.outdir)
colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"] # Okabe and Ito colorbline friendly.

# TODO these functions should be moved to the main package.
def get_all_mtl_locs(top_dir):
    hp_base = top_dir / "hp" / "main" / "dark"
    return list(hp_base.glob("*.fits"))

def load_mtl(mtl_loc):
    temp_tbl = Table.read(mtl_loc)
    return temp_tbl

def get_num_tiles(top_dir, n_passes):
    n_tiles = [0]
    for fname in top_dir.glob("tiles-*.fits"):
    # for i in range(1, n_passes + 1):
    #     tile_file = top_dir / f"tiles-pass-{i}.fits"
        with fitsio.FITS(fname) as h:
            n_tiles.append(len(h[1][:]))
    return n_tiles


# all_mtls = [] # TODO might need this for other summary plots might not...
nobs_arrs = []
at_least_arrs = []
fraction_arrs = []
ntiles_arrs = []
mtls = []
for mtl_loc in args.mtls:
    print(f"Loading {mtl_loc}...")
    # mtl = load_mtl_all(Path(mtl_loc), args.verbose)
    mtl_locs = get_all_mtl_locs(Path(mtl_loc))
    # print(mtl_locs)
    with Pool(args.nproc) as p:
         mtl_tbls = p.map(load_mtl, mtl_locs)

    mtl = vstack(mtl_tbls)
    mtls.append(mtl)

    # Each row is a pass, each column is number of objects with that many exposures
    nobs, at_least = get_nobs_arr(mtl)
    nobs_arrs.append(nobs)
    at_least_arrs.append(at_least)

    # [0,0] is the "at least zero exposures with zero iterations" which should include
    # every single object at this point.
    fraction_arrs.append(at_least / at_least[0, 0])

    if args.pertile:
        ntiles_arrs.append(np.cumsum(get_num_tiles(Path(mtl_loc), 50)))

if args.perpass:
    # Making the comparison plot between the different runs
    fig, ax = plt.subplots(figsize=(12, 5))

    for i in range(len(fraction_arrs)):
        goal = int(args.goals[i])
        name = Path(args.mtls[i]).name
        plt.plot(fraction_arrs[i][:, goal], "-o", label=f"{name} (Goal {goal})", c=colors[i])

    plt.legend()
    ax.grid(alpha=0.5)
    ax.set(xlim=(0, 30), ylim=(0, 1), xlabel="Num Passes", ylabel="Fraction", title="Fraction of targets with at least $n$ obs")
    plt.axhline(y=0.9, c="k")
    plt.axhline(y=0.95, c="r")
    plt.savefig(out_dir / "efficiency_per_pass.jpg", dpi=256, bbox_inches="tight")

if args.pertile:
    # Comparison plot by number of tiles in each exposure.
    fig, ax = plt.subplots(figsize=(12, 5))

    tile_maxes = []
    ylbl = "Fraction"
    densities = []
    for i in range(len(fraction_arrs)):
        goal = int(args.goals[i])
        name = Path(args.mtls[i]).name

        # Use this to determine how far to plot in the xlimits
        efficiency = 0.95
        hit_goal = fraction_arrs[i][:, goal] >= efficiency
        idx_goal = np.argmax(hit_goal)
        tile_maxes.append(ntiles_arrs[i][idx_goal] * (5 - goal))

        density = 1

        if args.density_assign:
            density = int(name.split("-")[3])
            ylbl = "Assigned Targets Per Sq. Deg."

        densities.append(density)

        # Scale the 1 goal to be x4 and the 4 goal to be x1 so they're on the same scale.
        if i < 2:
            lw = 2
            plt.plot(ntiles_arrs[i] * (5 - goal), fraction_arrs[i][:, goal] * density, "-o", lw=lw, label=args.labels[i], c=colors[i])
        else:
            lw = 2
            plt.plot(ntiles_arrs[i] * (5 - goal), fraction_arrs[i][:, goal] * density, "-o", lw=lw, label=args.labels[i], c=colors[i], markerfacecolor='white', mew=2)
    # np.max(tile_maxes) * 1.1
    plt.legend()
    ax.grid(alpha=0.5)
    ax.set(xlim=(0, 1200), ylim=(0, np.max(densities)), xlabel="Num. Exposures", ylabel=ylbl)#, title="Fraction of targets with at least $n$ obs")

    if not args.density_assign:
        plt.axhline(y=0.9 * np.max(densities), c="k")
        plt.axhline(y=0.95 * np.max(densities), c="r")

    z_ax = ax.twiny()
    z_ax.set(xlim=ax.get_xlim())
    scale = 14.62439
    xticks = ax.get_xticks()
    zticks = np.arange(0, np.max(xticks) * scale, 2000, dtype=int)
    z_ax.set(xticks=zticks, xticklabels=zticks, xlabel="Exposures Scaled to 5000 sq. deg.")


    plt.axvline(12600, ls="dashed", c="grey")
    plt.savefig(out_dir / "efficiency_per_tile.jpg", dpi=256, bbox_inches="tight")

if args.hist:
    fig, ax = plt.subplots(figsize=(12, 5))

    hatchings = ["/", "\\", "+"]
    for i in range(len(nobs_arrs)):
        goal = int(args.goals[i])
        efficiency = 0.9

        hit_goal = fraction_arrs[i][:, goal] >= efficiency
        idx_goal = np.argmax(hit_goal)

        name = Path(args.mtls[i]).name

        x = np.arange(nobs_arrs[1].shape[-1])
        # plt.plot(x, nobs_arrs[i][idx_goal, :], "-o", label=f"{name} (Goal {goal})", c=colors[i],)

        plt.bar(x, nobs_arrs[i][idx_goal, :], width=1, label=f"{name} (Goal {goal})", ec=colors[i], fill=False, hatch=hatchings[i])

    plt.legend()
    ax.grid(alpha=0.5)
    ax.set(xticks=np.arange(0, 40, 4), xlim=(0, 20), xlabel="Num Exposures", ylabel="Num Targets", title="Number of Targets with $n$ Exposures")
    plt.savefig(out_dir / "nobs_at_goal_hist.jpg", dpi=256, bbox_inches="tight")

if args.targs:
    for j, mtl in enumerate(mtls):

        timestamps = np.array(mtl["TIMESTAMP"])
        ts = np.array([datetime.fromisoformat(x.decode()) for x in timestamps])
        unique_timestamps = np.sort(np.unique(ts))

        name = Path(args.mtls[j]).name

        for i in [4, 7, 16]:
            this_ts = unique_timestamps[i]
            keep_rows = ts <= this_ts

            trunc_mtl = deduplicate_mtl(mtl[keep_rows])
            color = np.where(trunc_mtl["NUMOBS"] > 4, 4, trunc_mtl["NUMOBS"])

            fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")

            ra_min = 205
            ra_max = 210
            dec_min = 25
            dec_max = 30

            in_zone_ra = (trunc_mtl["RA"] >= ra_min) & (trunc_mtl["RA"] <= ra_max)
            in_zone_dec = (trunc_mtl["DEC"] >= dec_min) & (trunc_mtl["DEC"] <= dec_max)
            in_zone = in_zone_ra & in_zone_dec

            plt.scatter(trunc_mtl["RA"][in_zone], trunc_mtl["DEC"][in_zone], c=color[in_zone], cmap="bwr", alpha=0.25, s=2)
            ax.axis("off")
            plt.savefig(out_dir / f"{name}_targs_{i}.jpg", dpi=256, bbox_inches="tight")


# python summary_plots.py  --mtls /pscratch/sd/d/dylang/fiberassign/mtl-4exp-lae-1000-big-nodither-nproc-32/ /pscratch/sd/d/dylang/fiberassign/mtl-4exp-lae-1000-big-nproc-32/ /pscratch/sd/d/dylang/fiberassign/mtl-4exp-lae-1200-big-nodither-nproc-32/ /pscratch/sd/d/dylang/fiberassign/mtl-4exp-lae-1200-big-nproc-32/ --goals 4 4 4 4 -o /pscratch/sd/d/dylang/fiberassign/plots/ --pertile --nproc 32 --density_assign --labels "Movable Collimater (1000 sq. deg.)" "Offset Tiles (1000 sq. deg.)" "Movable Collimater (1200 sq. deg.)" "Offset Tiles (1200 sq. deg.)"

# python summary_plots.py  --mtls /pscratch/sd/d/dylang/fiberassign/mtl-4exp-lae-1000-big-nodither-nproc-32/ /pscratch/sd/d/dylang/fiberassign/mtl-4exp-lae-1000-big-nproc-32/ --goals 4 4 -o /pscratch/sd/d/dylang/fiberassign/plots/ --pertile --nproc 32 --labels "Movable Collimater" "Offset Tiles"


# python summary_plots.py  --mtls /pscratch/sd/d/dylang/fiberassign/mtl-4exp-lae-1200-big-nproc32-inputtiles/ /pscratch/sd/d/dylang/fiberassign/mtl-4exp-lae-1200-big-nproc-32/ --goals 4 4 -o /pscratch/sd/d/dylang/fiberassign/plots/ --pertile --nproc 32 --labels "Per Night" "Original"