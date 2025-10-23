#!/usr/bin/env python

# Some efficiency plots...
# TODO proper docstring
# stdlib imports
import argparse
from pathlib import Path
import time

# Non DESI imports
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--indir", required=True, type=str, help="parent directory where all processed data is saved.")
parser.add_argument("--suffixes", required=True, nargs='+', default=[], help="list of suffixes corresponding to processed mtls to load and compare.")
parser.add_argument("--goals", required=True, nargs='+', default=[], help="target number of exposures for each associated fiberassign run.")
parser.add_argument("--labels", required=True, nargs='+', default=[], help="labels to use for each mtl in the output plot.")
parser.add_argument("--verbose", required=False, action="store_true", help="be verbose when doing everything or not.")
parser.add_argument("--pertile", required=False, action="store_true", help="plot results per num tile/exposure.")
parser.add_argument("--density_assign", required=False, action="store_true", help="use density of assigned targets as y axis instead of fraction.")
parser.add_argument("--use_marker", required=False, action="store_true", help="use the marker when making the plots.")
parser.add_argument("-o", "--outdir", required=True, type=str, help="where to save generated plots.")
parser.add_argument("-n", "--name", required=False, type=str, help="name suffix to attach to saved plots.")
args = parser.parse_args()

out_dir = Path(args.outdir)
colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"] # Okabe and Ito colorbline friendly.

top_axis = False

parent_dir = Path(args.indir)

t_start = time.time()
print(f"Loading everything...")
nobs_arrs = [np.load(parent_dir / f"nobs_{suffix}.npy") for suffix in args.suffixes]
at_least_arrs = [np.load(parent_dir / f"at_least_{suffix}.npy") for suffix in args.suffixes]
fraction_arrs = [np.load(parent_dir / f"fraction_{suffix}.npy") for suffix in args.suffixes]
ntiles_arrs = [np.load(parent_dir / f"ntiles_{suffix}.npy") for suffix in args.suffixes]
ntiles_cum_arrs = [np.load(parent_dir / f"ntiles_cum_{suffix}.npy") for suffix in args.suffixes]
t_end = time.time()
print(f"Loading took {t_end - t_start} seconds...")


if args.pertile:
    # Comparison plot by number of tiles in each exposure.
    # Plots the fraction of assigned targets or assigned density per night/MTL update
    fig, ax = plt.subplots(figsize=(12, 5))

    ylbl = "Fraction"
    densities = [] # For scaling the y axis correctly
    for i in range(len(fraction_arrs)):
        goal = int(args.goals[i])
        name = args.suffixes[i]

        # If we're making the plot in assigned density rather than assigned
        # fraction space we need to keep track of each curve's unassigned (true)
        # density, but we can set that density to 1 for the case that we don't do that.
        density = 1
        if args.density_assign:
            density = int(name.split("-")[1])
            ylbl = "Assigned Targets Per Sq. Deg."

        densities.append(density)
        # Index in the arr should correspond with the goal, given the order
        # we load thigns in...
        # NOTE: If we were really worried we could do it as a dict and not a list.
        y = fraction_arrs[i][:, goal] * density
        x = ntiles_cum_arrs[i][:fraction_arrs[i].shape[0]] * (5 - goal)
        if args.use_marker:
            marker = "-o"
            plt.plot(x, y, marker, lw=1, label=args.labels[i], c=colors[i])
        else:
            plt.plot(x, y, lw=1, label=args.labels[i], c=colors[i])

    plt.legend()
    ax.grid(alpha=0.5)
    tile_max = np.max([arr[-1] for arr in ntiles_cum_arrs])
    ax.set(xlim=(0, tile_max), ylim=(0, np.max(densities)), xlabel="Num. Exposures", ylabel=ylbl)

    if not args.density_assign: # Visual guidelines for the eye.
        plt.axhline(y=0.9 * np.max(densities), c="k")
        plt.axhline(y=0.95 * np.max(densities), c="r")

    save_name = f"efficiency_per_tile_{args.name}.jpg" if args.name is not None else "efficiency_per_tile.jpg"
    plt.savefig(out_dir / save_name, dpi=256, bbox_inches="tight")


# python summary_plots.py --suffixes lae-1000-big-inputtiles-withstds lae-1000-big-inputtiles-withstds-test --goals 4 4 -o . --pertile --labels "Base" "Test" --name "lae_1000_300sqdeg" -i /pscratch/sd/d/dylang/fiberassign/processed/
