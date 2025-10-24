#!/usr/bin/env python

# Run simulated fiberassign over either a simulated or given catalog.
# python run_survey_mp.py --ramin 200 --ramax 210 --decmin 20 --decmax 30 -o /pscratch/sd/d/dylang/fiberassign/mtl-4exp-lae-1000-withstandards/ --npass 50 --catalog /pscratch/sd/d/dylang/fiberassign/lya-colore-lae-1000.fits --nproc 32  --fourex

# python run_survey_mp.py --ramin 190 --ramax 210 --decmin 15 --decmax 30 -o /pscratch/sd/d/dylang/fiberassign/mtl-4exp-lae-1200-big-nproc-32/ --npass 50 --catalog /pscratch/sd/d/dylang/fiberassign/lya-colore-lae-1200.fits --nproc 32  --fourex

# python run_survey_mp.py --ramin 190 --ramax 210 --decmin 15 --decmax 30 -o /pscratch/sd/d/dylang/fiberassign/mtl-4exp-lae-1000-big-nproc-32-inputtiles-withstds-test/ --catalog /pscratch/sd/d/dylang/fiberassign/lya-colore-lae-1000.fits --nproc 32  --tiles /pscratch/sd/d/dylang/fiberassign/tiles-30pass-superset.ecsv --stds /pscratch/sd/d/dylang/fiberassign/dark_stds_catalog.fits

# python run_survey_mp.py --ramin 190 --ramax 210 --decmin 15 --decmax 30 -o /pscratch/sd/d/dylang/fiberassign/mtl-4exp-lae-1000-big-nproc-32-inputtiles-withstds-test/ --catalog /pscratch/sd/d/dylang/fiberassign/lya-colore-lae-1000.fits --nproc 32  --tiles /pscratch/sd/d/dylang/fiberassign/tiles-2pass-superset.ecsv --stds /pscratch/sd/d/dylang/fiberassign/dark_stds_catalog.fits

# TODO proper docstring
import argparse
from datetime import datetime, timedelta
from multiprocessing import Pool
import time

# Non-DESI Imports
import numpy as np
from astropy.table import Table, vstack, unique
import healpy as hp
import fitsio

# DESI imports
from desimodel.focalplane import get_tile_radius_deg
from desimodel.footprint import tiles2pix
from fiberassign.scripts.assign import parse_assign, run_assign_full

# stdlib imports
from pathlib import Path

# TODO remove this at some point to point to a generic simassign import.
import sys
sys.path.append("/pscratch/sd/d/dylang/repos/simassign/src/")
from simassign.mtl import *
from simassign.util import *
from simassign.io import load_catalog


import logging
LEVEL = 15 # More thand bug less than info
logging.addLevelName(LEVEL, "DETAILS")

def details(self, message, *args, **kws):
    if self.isEnabledFor(LEVEL):
        self._log(LEVEL, message, args, **kws)
logging.Logger.details = details
log = logging.getLogger(__name__)
log.setLevel(LEVEL-1)

# I need to log things instead of print because of the way multiprocessing
# works all the text printed will be hijacked until the end of the script
# but it's actually of debug benefit to have it in the right place.
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(levelname)s: %(asctime)s: %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

parser = argparse.ArgumentParser()
parser.add_argument("--ramax", required=False, type=float, help="maximum RA angle to assign over.")
parser.add_argument("--ramin", required=False, type=float, help="minimum RA angle to assign over.")
parser.add_argument("--decmax", required=False, type=float, help="maximum DEC angle to assign over.")
parser.add_argument("--decmin", required=False, type=float, help="minimum DEC angle to assign over.")
parser.add_argument("-o", "--outdir", required=True, type=str, help="where to save the mtl* and fba* output files.")
parser.add_argument("-t", "--tiles", required=True, type=str, help="tiling to use for observations.")
parser.add_argument("--stds", required=False, type=str, help="base location of standards catalog.")
parser.add_argument("--nproc", required=False, type=int, default=1, help="number of multiprocessing processes to use.")
parser.add_argument("--fourex", required=False, action="store_true", help="take four exposures of a single tiling rather than four unique tilings.")

parser.add_argument("--danger", required=False, action="store_true", help="you want this to run as fast as possible, so do everything dangerously.")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--catalog", type=str, help="A catalog of objects to use for fiber assignment.")
group.add_argument("--density", type=int, help="number density per square degree of randomly generated targets.")

args = parser.parse_args()

t_start = time.time()
targetmask = load_target_yaml("targetmask.yaml")
log.details(f"Using {targetmask}")
log.details(f"Running with...")
log.details(args)

if args.danger:
    log.details("=" * 9)
    log.details("Running in danger mode. This means:")
    log.details("1. Will not save MTLs every night, only every year of the survey which has implications for checkpointing.")
    log.details("=" * 9)


# Generate the random targets
rng = np.random.default_rng(91701)
if args.density:
    ra, dec = generate_random_objects(args.ramin, args.ramax, args.decmin, args.decmax, rng, args.density)
elif (args.ramin is not None) and (args.ramax is not None) and (args.decmin is not None) and (args.decmax is not None):
    ra, dec = load_catalog(args.catalog, [args.ramin, args.ramax, args.decmin, args.decmax])
else:
    ra, dec = load_catalog(args.catalog)

log.details(f"Generated {len(ra)} targets...")

tbl = Table()
tbl["RA"] = ra
tbl["DEC"] = dec

nside = 64
theta, phi = np.radians(90 - dec), np.radians(ra)
pixlist = np.unique(hp.ang2pix(nside, theta, phi, nest=True))

log.details(f"{len(pixlist)} HEALpix covered by catalog.")

if args.stds is not None:
    stds_catalog = Table.read(args.stds)
    mtl_all = initialize_mtl(tbl, args.outdir, stds_catalog, as_dict=True)
else:
    mtl_all = initialize_mtl(tbl, args.outdir, as_dict=True)

# Directories for later
base_dir = Path(args.outdir)
hp_base = base_dir / "hp" / "main" / "dark"

tile_loc = Path(args.tiles)
tiles = Table.read(tile_loc)

# Use this to get all tiles that touch the given zone, not just ones that only
# have a center that falls inside the zone.
tile_rad =  get_tile_radius_deg()
margin = tile_rad - 0.2
fba_loc = str(base_dir / "fba")

def fiberassign_tile(targ_loc, tile_loc, runtime, tileid):
    params = ["--rundate",
              runtime,
              "--overwrite",
              "--write_all_targets",
              "--footprint", # Actually means "footprint" of tile centers...
              tile_loc,
              "--dir",
              fba_loc,
            #   "--sky_per_petal",
            #   40, # Use the default for this
            #   "--standards_per_petal",
            #   10,
              "--overwrite",
              "--targets",
              targ_loc,
            #   "--fba_use_fabs",
            #   "1",
    ]

    fba_args = parse_assign(params)
    run_assign_full(fba_args)

    # TODO find a way to do this without having to read (when we run fiberassign without io)
    # After assigning, load that fiber assignment and return the tids.
    fba_file = base_dir / "fba" / f"fba-{str(tileid).zfill(6)}.fits"
    with fitsio.FITS(fba_file) as h:
            tids = h["FASSIGN"]["TARGETID"][:] # Actually assigned TARGETIDS
            device = h["FASSIGN"]["DEVICE_TYPE"][:]
            # Cutting on "not ETC" probably not necessary but just to be safe.
            tids = tids[(tids > 0) & (device != "ETC")]
            log.details(f"Loaded {len(tids)} from {fba_file}")
    return tids

def save_mtl(mtl_to_save, hpx):
    log.details(f"Saving healpix {hpx}")
    mtl_to_save.write(hp_base / f"mtl-dark-hp-{hpx}.ecsv", overwrite=True)

n_nights = len(np.unique(tiles["TIMESTAMP_YMD"]))
times = {"gen_curr_mtl": [], "assign": [],  "get_last_time": [], "update_mtl": [], "save_mtl": [],}  # For profiling.
cur_year = tiles["TIMESTAMP_YMD"][0][:4]
log.details(f"Starting year: {cur_year}")
t2 = time.time()
with Pool(args.nproc) as p:

    for i, timestamp in enumerate(np.unique(tiles["TIMESTAMP_YMD"])):
        log.details(f"Beginning night {i} {timestamp} by loading tiling...")
        night_year = timestamp[:4]

        # Step 1: generate the subset of tiles that are run on this night
        # And the associated file of targes observable by that tile.
        this_date = tiles["TIMESTAMP_YMD"] == timestamp
        tiles_subset = unique(tiles[this_date & tiles["IN_DESI"]], "TILEID") # Unique to avoid 2 processes assigning the same tile.

        hpx_night = tiles2pix(nside, tiles_subset["TILEID", "RA", "DEC"]) # Already unique from the return of tiles2pix
        hpx_night = hpx_night[np.isin(hpx_night, pixlist)] # The "fuzzy" nature of tiles 2 pix might return healpix we don't have targets in

        log.details(f"Night {i} {timestamp}: {len(tiles_subset)} tiles ({len(hpx_night)} HPX) to run")

        # Deduplicate the MTL to get only the most recent information for each target.
        # TODO run fiberassign in a way that we can skip saving target files.
        t_start_curr = time.time()
        curr_mtl = deduplicate_mtl(vstack([mtl_all[hpx] for hpx in hpx_night]))
        t_end_curr = time.time()
        times["gen_curr_mtl"].append(t_end_curr - t_start_curr)
        log.details(f"Gen curr mtl took {t_end_curr - t_start_curr} seconds...")
        targ_files, tile_files = generate_target_files(curr_mtl, tiles_subset, base_dir, i)

        # Worthwhile to keep this for summary plot purposes
        tile_loc = base_dir / f"tiles-{timestamp}.fits"
        tiles_subset.write(tile_loc, overwrite=True)

        # Step 2: actually run the fiber assignment, and get back the assigned targetids
        t_start_assign = time.time()

        fiberassign_params = zip(targ_files, tile_files, tiles_subset["TIMESTAMP"], tiles_subset["TILEID"])
        assigned_tids = p.starmap(fiberassign_tile, fiberassign_params)
        assigned_tids = np.concatenate(assigned_tids)

        t_end_assign = time.time()
        times["assign"].append(t_end_assign - t_start_assign)
        log.details(f"Assignment took {t_end_assign - t_start_assign} seconds...")

        unique_tids, counts = np.unique(assigned_tids, return_counts=True)
        log.details(f"Sanity check on tid updates: {len(assigned_tids)}, {len(unique_tids)}, {np.unique(counts)}")

        # Step 3 update the MTL
        # Determining the timestamp to imprint on the MTL update
        t3 = time.time()
        ts = [datetime.fromisoformat(t) for t in tiles_subset["TIMESTAMP"]]
        last_time = max(ts)
        last_time += timedelta(hours=1)
        last_time = last_time.isoformat()

        t_mid = time.time()
        times["get_last_time"].append(t_mid - t3)
        # TODO parallelize
        update_params = [(mtl_all[hpx], assigned_tids, last_time, False) for hpx in hpx_night]
        updated_tbls = p.starmap(update_mtl, update_params) # Should return in same order as hpx_night
        for i, hpx in enumerate(hpx_night):
            mtl_all[hpx] = updated_tbls[i]
        t4 = time.time()
        times["update_mtl"].append(t4 - t3)
        log.details(f"MTL update took {t4 - t3} seconds...")

        # Step 4 save the updated MTLs
        # Write updated MTLs by healpix.
        # TODO If we keep per loop saved MTLS, use them to add checkpointing to the script.
        if not args.danger:
            save_params = [(mtl_all[hpx], hpx) for hpx in hpx_night]
            p.starmap(save_mtl, save_params)
        # In danger mode only save if the year crosses over or it's the last night.
        elif (args.danger and (night_year > cur_year) or (i == (n_nights - 1))):
            log.details(f"Saving on night {i} {timestamp}")
            save_params = [(mtl_all[hpx], hpx) for hpx in pixlist]
            p.starmap(save_mtl, save_params)

        t5 = time.time()
        times["save_mtl"].append(t5 - t4)
        log.details(f"Saving MTL took {t5 - t4} seconds...")

        cur_year = night_year

log.details("Done!")
t_end = time.time()
log.details(f"Init: \t\t\t{t2 - t_start} \t {(t2 - t_start) / 60}")
log.details(f"Full: \t\t\t{t_end - t_start} \t {(t_end - t_start) / 60}")
log.details(f"Average per night: \t{(t_end - t2) / n_nights}\t {(t_end - t2) / (n_nights * 60)}")

for k in times.keys():
    log.details(f"Average {k}: {np.mean(times[k])}")