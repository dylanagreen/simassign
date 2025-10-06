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

parser = argparse.ArgumentParser()
parser.add_argument("--mtls", required=True, nargs='+', default=[], help="list of mtls to load and compare.")
parser.add_argument("--goals", required=True, nargs='+', default=[], help="target number of exposures for each associated fiberassign run/mtl.")
parser.add_argument("--verbose", required=False, action="store_true", help="be verbose when doing everything or not.")
parser.add_argument("--pertile", required=False, action="store_true", help="plot results per num tile/exposure.")
parser.add_argument("--perpass", required=False, action="store_true", help="plot results per pass.")
parser.add_argument("--hist", required=False, action="store_true", help="hist of number of exposures at goal.")
parser.add_argument("--nproc", required=False, type=int, default=1, help="number of multiprocessing processes to use for loading tables.")
parser.add_argument("-o", "--outdir", required=True, type=str, help="where to save generated plots.")
args = parser.parse_args()

out_dir = Path(args.outdir)
colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"] # Okabe and Ito colorbline friendly.

# TODO these functions should be moved to the main package.
def load_mtl_all(top_dir, verbose=False):
    hp_base = top_dir / "hp" / "main" / "dark"
    mtl_all = Table()
    tbls = []
    for mtl_loc in hp_base.glob("*.fits"):
        if verbose: print(f"Loading {mtl_loc.name}")
        temp_tbl = Table.read(mtl_loc)
        tbls.append(temp_tbl)
    mtl_all = vstack(tbls)
    return mtl_all

def get_all_mtl_locs(top_dir):
    hp_base = top_dir / "hp" / "main" / "dark"
    return list(hp_base.glob("*.fits"))

def load_mtl(mtl_loc):
    temp_tbl = Table.read(mtl_loc)
    return temp_tbl

def get_nobs_arr(mtl):
    timestamps = np.array(mtl["TIMESTAMP"])
    ts = np.array([datetime.fromisoformat(x.decode()) for x in timestamps])
    unique_timestamps = np.unique(ts)

    # Timestamps correspond with when the MTL was created/updated
    # So we can loop over the timestamps to get information from each
    # fiberassign run.
    bins = np.arange(-0.5, len(unique_timestamps) - 0.4, 1) # "binning" for counting numbers of observations of targets.
    nobs = []
    for time in unique_timestamps:
        keep_rows = ts <= time
        # print(sum(keep_rows))

        trunc_mtl = deduplicate_mtl(mtl[keep_rows])
        h, _ = np.histogram(trunc_mtl["NUMOBS"], bins=bins)
        nobs.append(h)

    obs_arr = np.asarray(nobs)
    # Reverse to go max down to zero, then sum to get how many have at least that number exposures
    # i.e. at least 3 exposures should be the sum of n_3 and n_4. Since it's reversed this is true
    # since 4 will be the first element (not summed), the second is the sum of the first two (3 and 4)
    at_least_n = np.cumsum(obs_arr[:, ::-1], axis=1)[:, ::-1]

    return obs_arr, at_least_n

def get_num_tiles(top_dir, n_passes):
    n_tiles = [0]
    for i in range(1, n_passes + 1):
        tile_file = top_dir / f"tiles-pass-{i}.fits"
        with fitsio.FITS(tile_file) as h:
            n_tiles.append(len(h[1][:]))
    return n_tiles


# all_mtls = [] # TODO might need this for other summary plots might not...
nobs_arrs = []
at_least_arrs = []
fraction_arrs = []
ntiles_arrs = []
for mtl_loc in args.mtls:
    print(f"Loading {mtl_loc}...")
    # mtl = load_mtl_all(Path(mtl_loc), args.verbose)
    mtl_locs = get_all_mtl_locs(Path(mtl_loc))
    with Pool(args.nproc) as p:
         mtl_tbls = p.map(load_mtl, mtl_locs)

    mtl = vstack(mtl_tbls)

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
    for i in range(len(fraction_arrs)):
        goal = int(args.goals[i])
        name = Path(args.mtls[i]).name

        # Use this to determine how far to plot in the xlimits
        efficiency = 0.95
        hit_goal = fraction_arrs[i][:, goal] >= efficiency
        idx_goal = np.argmax(hit_goal)
        tile_maxes.append(ntiles_arrs[i][idx_goal] * (5 - goal))

        # Scale the 1 goal to be x4 and the 4 goal to be x1 so they're on the same scale.
        plt.plot(ntiles_arrs[i] * (5 - goal), fraction_arrs[i][:, goal], "-o", label=f"{name} (Goal {goal})", c=colors[i])

    plt.legend()
    ax.grid(alpha=0.5)
    ax.set(xlim=(0, np.max(tile_maxes) * 1.1), ylim=(0, 1), xlabel="Num Tile/Exposures", ylabel="Fraction", title="Fraction of targets with at least $n$ obs")
    plt.axhline(y=0.9, c="k")
    plt.axhline(y=0.95, c="r")
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

