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

# TODO remove this at some point to point to a generic simassign import.
import sys
sys.path.append("/pscratch/sd/d/dylang/repos/simassign/src/")
from simassign.util import rotate_tiling

parser = argparse.ArgumentParser()
parser.add_argument("--ramax", required=False, type=float, help="maximum RA angle to assign over.")
parser.add_argument("--ramin", required=False, type=float, help="minimum RA angle to assign over.")
parser.add_argument("--decmax", required=False, type=float, help="maximum DEC angle to assign over.")
parser.add_argument("--decmin", required=False, type=float, help="minimum DEC angle to assign over.")
parser.add_argument("--npass", required=False, type=int, default=1, help="number of assignment passes to do.")
parser.add_argument("-o", "--outdir", required=True, type=str, help="where to save generated tile file.")
parser.add_argument("--fourex", required=False, action="store_true", help="take four exposures of a single tiling rather than four unique tilings.")
parser.add_argument("--collapse", required=False, action="store_true", help="collapse to unique tileids. Useful if running fourex, but don't need to 4x duplicate every tile.")
parser.add_argument("--starttime", required=False, type=str, default="2025-09-16T00:00:00+00:00", help="starting timestamp for the first tile")
args = parser.parse_args()

# Load the geometry superset to get the tiling of the entire sky.
tiles = load_tiles(onlydesi=False, tilesfile="tiles-geometry-superset.ecsv")
tiles = Table(tiles)

# This ensures we only get one tiling of the sky to use as a base.
zero_pass = tiles["PASS"] == 0
dark_tile = (tiles["PROGRAM"] == "DARK") | (tiles["PROGRAM"] == "DARK1B")
base_tiles = tiles[zero_pass & dark_tile]

# Use this to get all tiles that touch the given zone, not just ones that only
# have a center that falls inside the zone.
tile_rad =  get_tile_radius_deg()
margin = tile_rad - 0.2

pass_tilings = []
for i in range(1, args.npass + 1):
    print(f"Generating tiling for pass {i}...")
    if args.fourex: # Repeat each tiling four times before moving to the next one
         passnum = (i + 3) // 4
    else:
         passnum = i
    tiles = rotate_tiling(base_tiles, passnum)

    # IF we're not collapsing but we are doing 4x, give each "pass" a unique
    # tileid, so that we keep all four passes on joins.
    if args.fourex and not args.collapse:
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

# Adding some addigional necessary columns
tiles["PRIORITY"] = 1000 # Some good number... for this we want them to be all the same priority
tiles["STATUS"] = "unobs"
tiles["EBV_MED"] = 0.01 # TODO figure this out.
tiles["DESIGNHA"] = 0 # TODO figure this out
tiles["DONEFRAC"] = 0.0
tiles["AVAILABLE"] = True
tiles["PRIORITY_BOOSTFAC"] = 1.0

print(tiles[tiles["IN_DESI"]])
print(tiles)


# print(np.unique(ts))


base_dir = Path(args.outdir)
fname = f"tiles-{args.npass}pass-superset.ecsv"
if args.fourex:
    if args.collapse:
        fname = f"tiles-{args.npass // 4}pass-movable_collimator-superset.ecsv"
    else:
        fname = f"tiles-{args.npass // 4}pass-movable_collimator-fastmtl-superset.ecsv"

print(len(tiles["TILEID"]), len(np.unique(tiles["TILEID"])))

tiles.write(base_dir / fname, format="ascii.ecsv", overwrite=True)

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
