#!/usr/bin/env python

# python generate_tiling_file.py --npass 30 -o /pscratch/sd/d/dylang/fiberassign/ --fourex --collapse

import argparse
from datetime import datetime, timedelta

# Non-DESI Imports
import numpy as np
from astropy.table import Table, vstack, unique

# DESI imports
from desimodel.io import load_tiles
from desimodel.focalplane import get_tile_radius_deg

# stdlib imports
from pathlib import Path

# simassign imports
from simassign.util import *

parser = argparse.ArgumentParser()
parser.add_argument("--ramax", required=False, type=float, help="maximum RA angle to assign over.")
parser.add_argument("--ramin", required=False, type=float, help="minimum RA angle to assign over.")
parser.add_argument("--decmax", required=False, type=float, help="maximum DEC angle to assign over.")
parser.add_argument("--decmin", required=False, type=float, help="minimum DEC angle to assign over.")
parser.add_argument("-o", "--out", required=True, type=str, help="where to save generated tile file.")
parser.add_argument("--collapse", required=False, action="store_true", help="collapse to unique tileids. Useful if running fourex, but don't need to 4x duplicate every tile.")
parser.add_argument("--trim", required=False, action="store_true", help="trim tiling to survey area (that is, set IN_DESI=True only within survey area).")
parser.add_argument("--starttime", required=False, type=str, default="2025-09-16T00:00:00+00:00", help="starting timestamp for the first tile")
parser.add_argument("--survey", required=False, type=str, default=None, help="use the survey defined by the boundaries in this file rather than the nominal DESI 2 survey.")
parser.add_argument("--add_tiledone", required=False, action="store_true", help="add TILEDONE column (for running without a simulated survey).")
parser.add_argument("--stripes", required=False, action="store_true", help="use stripe tiling instead of DESI-I-like symmetrical tiling.")

group_trim = parser.add_mutually_exclusive_group(required=False)
group_trim.add_argument("--trim_rad", type=float, help="when trimming, keep only tiles if their center is at least trim_rad/2 away from the survey edge. This convention matches that of the matplotlib path")
group_trim.add_argument("--trim_scale", type=float, help="trim by this multiplier of the radius.")

group = parser.add_mutually_exclusive_group(required=False)
group.add_argument("--fourex", action="store_true", help="take four exposures of a single tiling rather than four unique tilings.")
group.add_argument("--twoex", action="store_true", help="take two exposures of a single tiling rather than two unique tilings.")

group_pass = parser.add_mutually_exclusive_group(required=True)
group_pass.add_argument("--npass", type=int, help="number of assignment passes to do.")
group_pass.add_argument("--ntiles", type=int, help="target number of tiles to achieve, using as many passes as possible to get there.")
args = parser.parse_args()


def add_tile_cols(tiles):
    tile_tbl = Table(tiles)
    tile_tbl["PROGRAM"] = "DARK"
    tile_tbl["OBSCONDITIONS"] = 1

    del tile_tbl["ROW"]
    del tile_tbl["COL"]
    return tile_tbl

trim_rad = 0
if args.trim_rad:
    trim_rad = args.trim_rad
elif args.trim_scale:
    trim_rad = args.trim_scale * get_tile_radius_deg()

survey = None
if args.survey is not None:
    survey = np.load(args.survey)

if args.stripes:
    base_tiles = generate_stripe_tiles()
else:
    # Load the geometry superset to get the tiling of the entire sky.
    tiles = load_tiles(onlydesi=False, tilesfile="tiles-geometry-superset.ecsv")
    tiles = Table(tiles)

    # This ensures we only get one tiling of the sky to use as a base.
    zero_pass = tiles["PASS"] == 0
    dark_tile = (tiles["PROGRAM"] == "DARK") | (tiles["PROGRAM"] == "DARK1B")
    base_tiles = tiles[zero_pass & dark_tile]

in_survey = check_in_survey_area(base_tiles, survey, trim_rad=trim_rad)
n_base = np.sum(in_survey)
# Use this to get all tiles that touch the given zone, not just ones that only
# have a center that falls inside the zone.
tile_rad =  get_tile_radius_deg()
margin = tile_rad - 0.4

pass_tilings = []

if args.npass:
    if args.twoex: max_pass = (args.npass + 1) // 2
    else: max_pass = args.npass + 1
else:
    print("n_base", n_base)
    max_pass = args.ntiles // n_base + 1 # Add one because we iterate from 1 upwards.
    print("max_pass", max_pass)
    print(args.ntiles)
    print("pred tiles", max_pass * n_base)
    if not args.stripes: max_pass += 1


if args.stripes:
    all_stripes = shift_stripes(max_pass, base_tiles)

cur_tiles = 0
for i in range(1, max_pass):
    print(f"Generating tiling for pass {i}...")
    if args.stripes:
        tiles = add_tile_cols(all_stripes[i - 1])
        tiles["PASS"] = i

        tileids = np.arange(len(tiles)) + i * 10000
        tiles["TILEID"] = tileids

    else:
        if args.fourex: # Repeat each tiling four times before moving to the next one
            passnum = (i + 3) // 4
        elif args.twoex:
            passnum = (i + 1) // 2
        else:
            passnum = i
        print("PASSNUM", passnum)
        tiles = rotate_tiling(base_tiles, passnum)

    # IF we're not collapsing but we are doing 4x, give each "pass" a unique
    # tileid, so that we keep all four passes on joins.
    if (args.fourex or args.twoex) and not args.collapse:
        tileids = np.arange(len(tiles)) + i * 10000
        tiles["TILEID"] = tileids

    # Booleans for determining which tiles to keep.
    # Margin makes sure we don't end up with tiles that are "in bounds"
    # but because of the circular shape are off the corner of the
    # region and don't actually cover any of the targets (which crashes fiberassign)
    if (args.ramin is not None) and (args.ramax is not None) and (args.decmin is not None) and (args.decmax is not None):
        # Only run this if the box is actually passed in.
        tiles_in_ra = (tiles["RA"] >= (args.ramin - margin)) & (tiles["RA"] <= (args.ramax + margin))
        tiles_in_dec = (tiles["DEC"] >= (args.decmin - margin)) & (tiles["DEC"] <= (args.decmax + margin))
        not_in_zone = ~(tiles_in_ra & tiles_in_dec)
        tiles["IN_DESI"][not_in_zone] = False

    if args.trim:
        in_survey = check_in_survey_area(tiles, survey, trim_rad=trim_rad)
        tiles["IN_DESI"] = False
        tiles["IN_DESI"][in_survey] = True

    cur_tiles += np.sum(tiles["IN_DESI"])

    # IF we're still below our requested target number of tiles, add this pass.
    # We do this by pass because we don't want to do any incomplete passes.
    if args.ntiles and (cur_tiles > args.ntiles):
        break
    else:
        pass_tilings.append(tiles)

tiles = vstack(pass_tilings)
if args.collapse:
    tiles = unique(tiles, "TILEID")
    tiles.sort("TILEID")

n_tiles = np.sum(tiles["IN_DESI"])
print(f"{n_tiles} tiles IN_DESI")
print(f"{len(tiles)} total tiles")

start_time = datetime.fromisoformat(args.starttime)
print(str(start_time))

# datetime.timedelta(days=10)
tiles["TIMESTAMP"] = start_time.isoformat()

# Adding some addigional necessary columns
tiles["PRIORITY"] = 1000 # Some good number... for this we want them to be all the same priority
tiles["STATUS"] = "unobs"
tiles["EBV_MED"] = 0.01 # TODO figure this out.
tiles["DESIGNHA"] = 0 # TODO figure this out
tiles["DONEFRAC"] = 0.0
tiles["AVAILABLE"] = True
tiles["PRIORITY_BOOSTFAC"] = 1.0

# Need these for survey sim
# tiles_bright = Table(tiles[~tiles["IN_DESI"]][1])
# tiles_bright["PROGRAM"] = tiles_bright["PROGRAM"].astype("<U15")
# tiles_bright["TILEID"] = np.max(tiles["TILEID"]) + 1
# tiles_bright["PROGRAM"] = "BRIGHT"
# tiles_bright["IN_DESI"] = True

# tiles_backup = Table(tiles[tiles["IN_DESI"]][-1])
# tiles_backup["PROGRAM"] = tiles_backup["PROGRAM"].astype("<U15")
# tiles_backup["TILEID"] = np.max(tiles["TILEID"]) + 2
# tiles_backup["PROGRAM"] = "BACKUP"
# tiles_backup["IN_DESI"] = True

# tiles = vstack([tiles, tiles_backup, tiles_bright])

tiles["IN_DESI"][-2:] = True
tiles["PROGRAM"] = tiles["PROGRAM"].astype("<U15")
tiles["PROGRAM"][-2] = "BRIGHT"
tiles["PROGRAM"][-1] = "BACKUP"

# Update timestamps last so that we only update those that are IN_DESI
timestamps = [str(start_time)] * np.sum(tiles["IN_DESI"])
cur_time = start_time + timedelta(seconds=1000)
n_days = 0

n_exps = 0
for i in range(len(timestamps)):

    timestamps[i] = cur_time.isoformat()
    cur_time = cur_time + timedelta(seconds=1000)
    n_exps += 1

    #  I dunno 15 exposures a night is a reasonable number for now.
    # Since there's no ordering to the tiles this is kind of irrelevant anyway.
    # After this 15 exposures increment by one day, but based on the
    # original start time of the night.
    if n_exps == 15:
        n_days += 1
        cur_time = start_time + timedelta(days= 1 * n_days)
        n_exps = 0

tiles["TIMESTAMP"][tiles["IN_DESI"]] = timestamps

# Will use this column for breaking down observation dates without time
ts = [datetime.fromisoformat(x).strftime("%Y%m%d") for x in tiles["TIMESTAMP"]]
tiles["TIMESTAMP_YMD"] = ts

if args.add_tiledone:
    tiles["TILEDONE"] = tiles["IN_DESI"]

print(tiles)
print(len(tiles["TILEID"]), len(np.unique(tiles["TILEID"])))
print(np.sum(tiles["IN_DESI"]), "in DESI")


print(f"Saving to... {args.out}")

tiles.write(args.out, format="ascii.ecsv", overwrite=True)

# %ECSV 1.0
# ---
# datatype:
# - {name: TILEID, datatype: int64}
# - {name: PASS, datatype: float64}
# - {name: RA, datatype: float64}
# - {name: DEC, datatype: float64}
# - {name: PROGRAM, datatype: string}
# - {name: IN_DESI, datatype: bool}

# - {name: PRIORITY, datatype: float64, format: '%10.3e', description: Tile observation priority}
# - {name: STATUS, datatype: string, description: 'unobs, obsstart, obsend, done'}
# - {name: EBV_MED, datatype: float64, format: '%6.3f', description: median E(B-V) on tile}
# - {name: DESIGNHA, datatype: float64, format: '%7.2f', description: Design hour angles}
# - {name: DONEFRAC, datatype: float64, format: '%7.4f', description: Tile completeness fraction}
# - {name: AVAILABLE, datatype: bool, description: Fiberassign file is available}
# - {name: PRIORITY_BOOSTFAC, datatype: float64, format: '%7.3f', description: Manual boost factor applied on top of computed priorities.}
