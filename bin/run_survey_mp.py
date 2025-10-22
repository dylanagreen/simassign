#!/usr/bin/env python

# Run simulated fiberassign over either a simulated or given catalog.
# python run_survey_mp.py --ramin 200 --ramax 210 --decmin 20 --decmax 30 -o /pscratch/sd/d/dylang/fiberassign/mtl-4exp-lae-1000-withstandards/ --npass 50 --catalog /pscratch/sd/d/dylang/fiberassign/lya-colore-lae-1000.fits --nproc 32  --fourex

# python run_survey_mp.py --ramin 190 --ramax 210 --decmin 15 --decmax 30 -o /pscratch/sd/d/dylang/fiberassign/mtl-4exp-lae-1200-big-nproc-32/ --npass 50 --catalog /pscratch/sd/d/dylang/fiberassign/lya-colore-lae-1200.fits --nproc 32  --fourex

# python run_survey_mp.py --ramin 190 --ramax 210 --decmin 15 --decmax 30 -o /pscratch/sd/d/dylang/fiberassign/mtl-4exp-lae-1000-big-nproc-32-inputtiles-withstds-test/ --catalog /pscratch/sd/d/dylang/fiberassign/lya-colore-lae-1000.fits --nproc 64  --tiles /pscratch/sd/d/dylang/fiberassign/tiles-30pass-superset.ecsv --stds /pscratch/sd/d/dylang/fiberassign/dark_stds_catalog.fits

# TODO proper docstring
import argparse
from datetime import datetime, timedelta
from multiprocessing import Pool, Manager
import time

# Non-DESI Imports
import numpy as np
from astropy.table import Table, vstack, unique
import healpy as hp
import fitsio

# DESI imports
from desimodel.io import load_tiles
from desimodel.focalplane import get_tile_radius_deg
from desimodel.footprint import tiles2pix
from desitarget.mtl import make_mtl, make_ledger_in_hp
from desitarget.targetmask import desi_mask, obsconditions
from fiberassign.scripts.assign import parse_assign, run_assign_full, run_assign_bytile

# stdlib imports
from pathlib import Path

# TODO remove this at some point to point to a generic simassign import.
import sys
sys.path.append("/pscratch/sd/d/dylang/repos/simassign/src/")
from simassign.mtl import *
from simassign.util import *
from simassign.io import load_catalog


parser = argparse.ArgumentParser()
parser.add_argument("--ramax", required=False, type=float, help="maximum RA angle to assign over.")
parser.add_argument("--ramin", required=False, type=float, help="minimum RA angle to assign over.")
parser.add_argument("--decmax", required=False, type=float, help="maximum DEC angle to assign over.")
parser.add_argument("--decmin", required=False, type=float, help="minimum DEC angle to assign over.")
parser.add_argument("-o", "--outdir", required=True, type=str, help="where to save the mtl* and fba* output files.")
parser.add_argument("-t", "--tiles", required=True, type=str, help="tiling to use for observations.")
parser.add_argument("--stds", required=False, type=str, help="base location of standards catalog.")
# parser.add_argument("--npass", required=False, type=int, default=100,
#                     help="number of assignment passes to do. Script will run as many passes as possible up to npass or max(tiles[\"PASS\"]), whichever is lower.")
parser.add_argument("--nproc", required=False, type=int, default=1, help="number of multiprocessing processes to use.")
parser.add_argument("--fourex", required=False, action="store_true", help="take four exposures of a single tiling rather than four unique tilings.")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--catalog", type=str, help="A catalog of objects to use for fiber assignment.")
group.add_argument("--density", type=int, help="number density per square degree of randomly generated targets.")

args = parser.parse_args()

t_start = time.time()
targetmask = load_target_yaml("targetmask.yaml")
print(f"Using {targetmask}")

# Generate the random targets
rng = np.random.default_rng(91701)
if args.density:
    ra, dec = generate_random_objects(args.ramin, args.ramax, args.decmin, args.decmax, rng, args.density)
elif (args.ramin is not None) and (args.ramax is not None) and (args.decmin is not None) and (args.decmax is not None):
    ra, dec = load_catalog(args.catalog, [args.ramin, args.ramax, args.decmin, args.decmax])
else:
    ra, dec = load_catalog(args.catalog)

print(f"Generated {len(ra)} targets...")

tbl = Table()
tbl["RA"] = ra
tbl["DEC"] = dec

nside = 64
theta, phi = np.radians(90 - dec), np.radians(ra)
pixlist = np.unique(hp.ang2pix(nside, theta, phi, nest=True))

print(f"{len(pixlist)} HEALpix covered by catalog.")

if args.stds is not None:
    stds_catalog = Table.read(args.stds)
    mtl_all = initialize_mtl(tbl, args.outdir, stds_catalog, as_dict=True)
else:
    mtl_all = initialize_mtl(tbl, args.outdir, as_dict=True)

t2 = time.time()

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
            print(f"Loaded {len(tids)} from {fba_file}")

    return tids

def load_tids_from_fba(fba_loc):
    with fitsio.FITS(fba_loc) as h:

        tids = h["FASSIGN"]["TARGETID"][:] # Actually assigned TARGETIDS
        device = h["FASSIGN"]["DEVICE_TYPE"][:]
        # Cutting on "not ETC" probably not necessary but just to be safe.
        tids = tids[(tids > 0) & (device != "ETC")]
    return tids

def save_mtl(mtl_to_save, hpx):
    print(f"Saving healpix {hpx}")
    mtl_to_save.write(hp_base / f"mtl-dark-hp-{hpx}.ecsv", overwrite=True)


print(f"cols: {mtl_all[list(mtl_all.keys())[0]].colnames}")

# for i in range(1, args.npass + 1):
n_nights = len(np.unique(tiles["TIMESTAMP_YMD"]))
times = {"gen_curr_mtl": [], "assign": [],  "update_mtl": [], "save_mtl": [],}  # For profiling.
for i, timestamp in enumerate(np.unique(tiles["TIMESTAMP_YMD"])):
    print(f"Beginning night {i} {timestamp} by loading tiling...")

    this_date = tiles["TIMESTAMP_YMD"] == timestamp
    tiles_subset = unique(tiles[this_date & tiles["IN_DESI"]], "TILEID") # Unique to avoid 2 processes assigning the same tile.

    hpx_night = tiles2pix(nside, tiles_subset["TILEID", "RA", "DEC"]) # Already unique from the return of tiles2pix
    hpx_night = hpx_night[np.isin(hpx_night, pixlist)] # The "fuzzy" nature of tiles 2 pix might return healpix we don't have targets in

    print(f"Night {i} {timestamp}: {len(tiles_subset)} tiles ({len(hpx_night)} HPX) to run")

    # TODO run fiberassign in a way that we can skip saving target files.
    t_start_curr = time.time()
    curr_mtl = deduplicate_mtl(vstack([mtl_all[hpx] for hpx in hpx_night]))
    t_end_curr = time.time()
    times["gen_curr_mtl"].append(t_end_curr - t_start_curr)
    targ_files, tile_files = generate_target_files(curr_mtl, tiles_subset, base_dir, i)

    # Worthwhile to keep this for summary plot purposes
    tile_loc = base_dir / f"tiles-{timestamp}.fits"
    tiles_subset.write(tile_loc, overwrite=True)

    t_start_assign = time.time()
    fiberassign_params = zip(targ_files, tile_files, tiles_subset["TIMESTAMP"], tiles_subset["TILEID"])
    with Pool(args.nproc) as p:
         assigned_tids = p.starmap(fiberassign_tile, fiberassign_params)

    # assigned_tids = []

    # TODO we can parallelize this because the order of assigned tids is irrelevant

    # for tileid in tiles_subset["TILEID"]:
    #     tileid = str(tileid)
        # fba_file = base_dir / "fba" / f"fba-{tileid.zfill(6)}.fits"
        # print(f"Loading tids from {fba_file.name}")
        # with fitsio.FITS(fba_file) as h:
        #         tids = h["FASSIGN"]["TARGETID"][:] # Actually assigned TARGETIDS
        #         device = h["FASSIGN"]["DEVICE_TYPE"][:]
        #         # Cutting on "not ETC" probably not necessary but just to be safe.
        #         tids = tids[(tids > 0) & (device != "ETC")]
        #         assigned_tids.append(tids)

        #         print(f"Loaded {len(tids)} from {fba_file}")

    assigned_tids = np.concatenate(assigned_tids)
    t_end_assign = time.time()
    times["assign"].append(t_end_assign - t_start_assign)

    unique_tids, counts = np.unique(assigned_tids, return_counts=True)
    print(f"Sanity check on tid updates: {len(assigned_tids)}, {len(unique_tids)}, {np.unique(counts)}")

    ts = [datetime.fromisoformat(t) for t in tiles_subset["TIMESTAMP"]]
    last_time = max(ts)
    last_time += timedelta(hours=1)

    t3 = time.time()
    # TODO parallelize
    for hpx in hpx_night:
        mtl_all[hpx] = update_mtl(mtl_all[hpx], assigned_tids, timestamp=last_time.isoformat(), use_desitarget=False)
    t4 = time.time()
    times["update_mtl"].append(t4 - t3)
    print(f"MTL update took {t4 - t3} seconds...")

    # Write updated MTLs by healpix.
    # TODO If we keep per loop saved MTLS, use them to add checkpointing to the script.
    mtls_to_save = [mtl_all[hpx] for hpx in hpx_night]
    save_params = zip(mtls_to_save, hpx_night)
    with Pool(args.nproc) as p:
         p.starmap(save_mtl, save_params)
    t5 = time.time()
    times["save_mtl"].append(t5 - t4)
    print(f"Saving MTL took {t5 - t4} seconds...")

    # curr_mtl = deduplicate_mtl(mtl_all)
    # curr_mtl.write(base_dir / "targets.fits.gz", overwrite=True)

print("Done!")
t_end = time.time()
print(f"Init: \t\t\t{t2 - t_start} \t {(t2 - t_start) / 60}")
print(f"Full: \t\t\t{t_end - t_start} \t {(t_end - t_start) / 60}")
print(f"Average per night: \t{(t_end - t2) / n_nights}\t {(t_end - t2) / (n_nights * 60)}")

for k in times.keys():
    print(f"Average {k}: {np.mean(times[k])}")