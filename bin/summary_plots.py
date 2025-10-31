#!/usr/bin/env python

# Some efficiency plots...
# TODO proper docstring
# stdlib imports
import argparse
from datetime import datetime
import math
from pathlib import Path
import time

# Non DESI imports
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--indir", required=True, type=str, help="parent directory where all processed data is saved.")
parser.add_argument("--suffixes", required=True, nargs='+', default=[], help="list of suffixes corresponding to processed mtls to load and compare.")
parser.add_argument("--goals", required=False, nargs='+', default=[], help="target number of exposures for each associated fiberassign run.")
parser.add_argument("--labels", required=True, nargs='+', default=[], help="labels to use for each mtl in the output plot.")
parser.add_argument("--verbose", required=False, action="store_true", help="be verbose when doing everything or not.")
parser.add_argument("--pertile", required=False, action="store_true", help="plot results per num tile/exposure.")
parser.add_argument("--density_assign", required=False, action="store_true", help="use density of assigned targets as y axis instead of fraction.")
parser.add_argument("--use_marker", required=False, action="store_true", help="use the marker when making the plots.")
parser.add_argument("-o", "--outdir", required=True, type=str, help="where to save generated plots.")
parser.add_argument("-n", "--name", required=False, type=str, help="name suffix to attach to saved plots.")
parser.add_argument("--tiles", required=False, nargs='+', default=[], help="file of tiles and observation dates to add date timestamping to the plots.")
parser.add_argument("--bydate", required=False, action="store_true", help="plot points by date and not cumulative number of exposures.")
parser.add_argument("--nobs", required=False, action="store_true", help="use the full nobs array for everything. If not will look for done arrays instead.")
args = parser.parse_args()

if args.nobs:
    assert args.goals, "Must provide goals if using the full nobs array."

out_dir = Path(args.outdir)
colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"] # Okabe and Ito colorbline friendly.

top_axis = False

parent_dir = Path(args.indir)

t_start = time.time()
print(f"Loading Tiles...")

if len(args.tiles) > 0:
    ntiles_cum_arrs = []
    time_arrs = []
    assert len(args.tiles) == len(args.suffixes), "Must provide a tiles file for each of the datasets"

    max_years = 1
    all_months = []
    for tilefile in args.tiles:
        # Use tiles file to generate the ntiles per night, if its passed.
        tbl = Table.read(tilefile)
        tbl = tbl[tbl["TILEDONE"]]

        unique_nights = np.unique(tbl["TIMESTAMP_YMD"])
        tiles_per_night = [0] + [np.sum(tbl["TIMESTAMP_YMD"] == n) for n in unique_nights]

        tbl["MONTH"] = [t[:6] for t in tbl["TIMESTAMP_YMD"]]
        unique_months = np.unique(tbl["MONTH"])
        all_months.append(unique_months)

        ntiles_cum_arrs.append(np.cumsum(tiles_per_night))
        max_years = max([max_years, ((datetime.strptime(unique_nights[-1], "%Y%m%d") - datetime.strptime(unique_nights[0], "%Y%m%d")).days) / 365])

        if args.bydate:
            night_0 = datetime.strptime(unique_nights[0][:4], "%Y") # Base everything around start of the year.
            nights = [datetime.strptime(n, "%Y%m%d") for n in unique_nights]
            delta_night = [0] + [(n - night_0).days for n in nights]

            time_arrs.append(delta_night)

    max_years = math.ceil(max_years)
else:
    ntiles_arrs = [np.load(parent_dir / f"ntiles_{suffix}.npy") for suffix in args.suffixes]
    ntiles_cum_arrs = [np.load(parent_dir / f"ntiles_cum_{suffix}.npy") for suffix in args.suffixes]
t_end = time.time()
print(f"Loading took {t_end - t_start} seconds...")

def generate_months(night_0, nyears=5):
    month_0 = night_0.strftime("%Y%m")
    cur_year = int(month_0[:4])
    cur_month = int(month_0[4:])

    months = []
    for i in range(nyears):
        while cur_month <= 12:
            months.append(str(cur_year) + str(cur_month).zfill(2))

            cur_month += 1
        cur_year += 1
        cur_month = 1

    # Bonus month for getting everything to align correctly (i.e. we need time
    # all the way through the final month, so need the last month as a grid line.)
    months.append(str(cur_year) + str(cur_month).zfill(2))

    return months

t_start = time.time()

print(f"Loading other files...")
if args.nobs:
    nobs_arrs = [np.load(parent_dir / f"nobs_{suffix}.npy") for suffix in args.suffixes]
    at_least_arrs = [np.load(parent_dir / f"at_least_{suffix}.npy") for suffix in args.suffixes]

else:
    done_arrs = [np.load(parent_dir / f"done_{suffix}.npy") for suffix in args.suffixes]

# Fraction arrs exist in both. For the nobs case though they're hughely multi dimensional.
fraction_arrs = [np.load(parent_dir / f"fraction_{suffix}.npy") for suffix in args.suffixes]

t_end = time.time()
print(f"Loading took {t_end - t_start} seconds...")

if args.pertile:
    # Comparison plot by number of tiles in each exposure.
    # Plots the fraction of assigned targets or assigned density per night/MTL update
    fig, ax = plt.subplots(figsize=(12, 5))

    ylbl = "Fraction"
    densities = [] # For scaling the y axis correctly
    min_x = 100
    max_x = 0
    for i in range(len(fraction_arrs)):
        name = args.suffixes[i]

        # If we're making the plot in assigned density rather than assigned
        # fraction space we need to keep track of each curve's unassigned (true)
        # density, but we can set that density to 1 for the case that we don't do that.
        density = 1
        if args.density_assign:
            if args.nobs:
                density = int(name.split("-")[-1])
            else:
                density = [1300, 1100]
            ylbl = "Assigned Targets Per Sq. Deg."

        densities.append(density)
        # Index in the arr should correspond with the goal, given the order
        # we load thigns in...
        # NOTE: If we were really worried we could do it as a dict and not a list.
        if args.nobs:
            goal = int(args.goals[i])
            y = fraction_arrs[i][:, goal] * density
        else:
            y = fraction_arrs[i] * np.atleast_1d(density)[:, None]

        print(y.shape, ntiles_cum_arrs[i].shape)
        if not args.bydate:
            if args.nobs:
                x = ntiles_cum_arrs[i][:fraction_arrs[i].shape[0]]
            else:
                x = ntiles_cum_arrs[i][:fraction_arrs[i].shape[1]]
        else:
            x = time_arrs[i]

        marker = "-"
        if args.use_marker:
            marker = "-o"

        if args.nobs:
            plt.plot(x, y, marker, lw=1, label=args.labels[i], c=colors[i])
        else:
            labels = ["LBG", "LAE"]
            linestyles = ["-", "-."]
            for j, l in enumerate(labels):
                plt.plot(x, y[j], linestyles[j], label=f"{args.labels[i]} {l}", c=colors[i])

        min_x = np.min([min_x, np.min(x)])
        max_x = np.max([max_x, np.max(x)])

    plt.legend()
    ax.grid(alpha=0.5)
    ax.set(xlabel="Num. Exposures", ylabel=ylbl, xlim=(min_x, max_x))

    if len(args.tiles) > 0:
        unique_months = generate_months(night_0, max_years)

        print("Unique months", len(unique_months))

        tiles_per_month = [np.sum(tbl["MONTH"] == m) for m in unique_months]
        cum_tiles_per_month = np.cumsum(tiles_per_month)
        # print(cum_tiles_per_month)
        if not args.bydate:
            month_ticks = cum_tiles_per_month
        else:
            month_ticks = [(datetime.strptime(n, "%Y%m") - night_0).days for n in unique_months]
            print(month_ticks)


        ax.set(xticks=month_ticks, xticklabels="")
        for i in range(1, len(month_ticks), 12):
            ax.axvline(month_ticks[i - 1], c="r")

        ax.set(xlabel="Month", xlim=(0, month_ticks[-1]))


    if not args.density_assign: # Visual guidelines for the eye.
        plt.axhline(y=0.9 * np.max(densities), c="k")
        plt.axhline(y=0.95 * np.max(densities), c="r")

    print("Saving...")
    save_name = f"efficiency_per_tile_{args.name}.jpg" if args.name is not None else "efficiency_per_tile.jpg"
    plt.savefig(out_dir / save_name, dpi=256, bbox_inches="tight")


# python summary_plots.py --suffixes lae-1000-big-inputtiles-withstds lae-1000-big-inputtiles-withstds-test --goals 4 4 -o . --pertile --labels "Base" "Test" --name "lae_1000_300sqdeg" -i /pscratch/sd/d/dylang/fiberassign/processed/

# python summary_plots.py --suffixes 5years-offsettiles-4exp-lae-1000 5years-offsettiles-4exp-lae-1200 --goals 4 4 --pertile --label "5 Years 1000 sq deg^-2" "5 Years 1200 sq deg^-2" -i /pscratch/sd/d/dylang/fiberassign/processed/ -o . --name "offsettiles_laeonly_5yrs_fraction"
# python summary_plots.py --suffixes 5years_corrected_reproc-offsettiles-4exp-lae-1975 5years_corrected_reproc-offsettiles-4exp-lae-1000 --goals 4 4 --pertile --label "5 Years 1975 sq deg^-2" "5 Years 1000 sq deg^-2" -i /pscratch/sd/d/dylang/fiberassign/processed/ -o . --name "offsettiles_laeonly_5yrs" --density_assign --tiles /global/cfs/cdirs/desi/users/dylang/fiberassign_desi2/exposures_processedv2_offset-tiles-5000deg-30pass.fits


# python summary_plots.py --suffixes 6years_offset-tiles-4exp-lae-1975 5years_corrected_reproc-offsettiles-4exp-lae-1975 --goals 4 4 --pertile --label "6 Years 1975 deg^-2" "5 Years 1975 deg^-2" -i /pscratch/sd/d/dylang/fiberassign/processed/ -o . --name "offsettiles_laeonly_by_yearc" --density_assign --tiles /global/cfs/cdirs/desi/users/dylang/fiberassign_desi2/exposures_processedv2_offset-tiles-5000deg-30pass.fits
