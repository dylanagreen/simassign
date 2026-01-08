#!/usr/bin/env python

# Run simulated fiberassign over either a simulated or given catalog.
# TODO proper docstring

import argparse
from datetime import datetime, timedelta
from multiprocessing import Pool
import sys
import time

# Non-DESI Imports
import numpy as np
from astropy.table import Table, vstack, unique
import healpy as hp
import fitsio
import yaml

# DESI imports
from desimodel.focalplane import get_tile_radius_deg
from desimodel.footprint import tiles2pix
from fiberassign.scripts.assign import parse_assign, run_assign_full

# stdlib imports
from pathlib import Path

from simassign.mtl import *
from simassign.util import *
from simassign.io import load_mtl_all
from simassign.logging import get_log

parser = argparse.ArgumentParser()
parser.add_argument("--ramax", required=False, type=float, help="maximum RA angle to assign over.")
parser.add_argument("--ramin", required=False, type=float, help="minimum RA angle to assign over.")
parser.add_argument("--decmax", required=False, type=float, help="maximum DEC angle to assign over.")
parser.add_argument("--decmin", required=False, type=float, help="minimum DEC angle to assign over.")
parser.add_argument("-o", "--outdir", required=True, type=str, help="where to save the mtl* and fba* output files.")
parser.add_argument("-t", "--tiles", required=True, type=str, help="tiling to use for observations.")
parser.add_argument("--stds", required=False, type=str, help="base location of standards catalog.")
parser.add_argument("--nproc", required=False, type=int, default=1, help="number of multiprocessing processes to use.")
parser.add_argument("--config", required=False, type=str, help="configuration yaml file with target parameters. At minimum this should contain everything in targetmask.yaml, but in the future could contain additional run parameters.")
parser.add_argument("--danger", required=False, action="store_true", help="you want this to run as fast as possible, so do everything dangerously.")
parser.add_argument("--catalog_b", type=str, help="A catalog of objects to use for fiber assignment, that will be added later in the survey.")
parser.add_argument("--b_start_date", type=str, help="the date on which targets in catalog b get added to the survey. Should be of form YYYYMMDD")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--catalog", type=str, help="A catalog of objects to use for fiber assignment.")
group.add_argument("--density", type=int, help="number density per square degree of randomly generated targets.")

args = parser.parse_args()

if args.catalog_b or args.b_start_date:
    assert args.catalog_b and args.b_start_date, "If providing --catalog_b or a --b_start_date, you must provide both!"

t_start = time.time()

if args.config is not None:
    with open(args.config) as f:
        targetmask = yaml.safe_load(f)
else:
    targetmask = load_target_yaml("targetmask.yaml")

sciencemask = target_mask_to_int(targetmask)

log = get_log()
log.details(f"Using {targetmask}")
log.details(f"sciencemask: {sciencemask}")
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

    tbl = Table()
    tbl["RA"] = ra
    tbl["DEC"] = dec
else:
    tbl = Table.read(args.catalog)

# TARGETID will be reset in initi_mtl, Z_COSMO doesn't exist in the standars
# table, so it breaks the stacking of the two.
if "TARGETID" in tbl.colnames:
    del tbl["TARGETID"]

if "Z_COSMO" in tbl.colnames:
    del tbl["Z_COSMO"]

ra = tbl["RA"]
dec = tbl["DEC"]

log.details(f"Using {len(tbl)} targets...")

nside = 64
theta, phi = np.radians(90 - dec), np.radians(ra)
pixlist = np.unique(hp.ang2pix(nside, theta, phi, nest=True))

log.details(f"{len(pixlist)} HEALpix covered by catalog.")

# Directories for later
base_dir = Path(args.outdir)
hp_base = base_dir / "hp" / "main" / "dark"
fba_base = base_dir / "fba"

tile_loc = Path(args.tiles)
tiles = Table.read(tile_loc)

loaded_from_checkpoint = False
# Check for healpixels AND fiber assignments, if there's only the former the
# script may have interrupted when the catalog was still being generated, and
# we may attempt an incomplete checkpoint load.
if hp_base.is_dir() and fba_base.is_dir():
    # Attempt to checkpoint
    mtl_all = load_mtl_all(hp_base, as_dict=True, nproc=args.nproc)
    last_timestamp = np.sort([np.sort(tbl["TIMESTAMP"])[-1] for tbl in mtl_all.values()])[-1]
    last_timestamp = last_timestamp[:10] # Only need the date, not the time
    last_timestamp = last_timestamp.replace("-", "")

    loaded_from_checkpoint = True
    log.details(f"Loaded Checkpointed MTLs with last timestamp: {last_timestamp}")
else:
    if args.catalog_b:
        # Do not load standards for catalog b. Since it gets added later to the mtl_all, the
        # stadards would be duplicated if we did. We create it first mostly just because,
        # it makes more sense to me to make it in memory at the start of everything.
        # But we could make the second MTL at the time its supposed to be added if we wanted.
        tbl_b = Table.read(args.catalog_b)
        # We do not need to save this, it just needs to exist in memory for appending later.
        mtl_all_b = initialize_mtl(tbl_b, save_dir=None, as_dict=True, targetmask=targetmask, nproc=args.nproc,
                                   start_id=len(tbl), timestamp=args.b_start_date)

    if args.stds is not None:
        stds_catalog = Table.read(args.stds)
        mtl_all = initialize_mtl(tbl, args.outdir, stds_catalog, as_dict=True, targetmask=targetmask, nproc=args.nproc)
    else:
        mtl_all = initialize_mtl(tbl, args.outdir, as_dict=True, targetmask=targetmask, nproc=args.nproc)

    if args.catalog_b:
        # Generate empty tables for healpixels that are in one catalog but not
        # the other.
        hp_a = list(mtl_all.keys())
        hp_b = list(mtl_all_b.keys())

        log.details("Generating dummy tables...")
        for hp in (hp_a + hp_b):
            if hp not in hp_b:
                log.details(f"Added dummy table to mtl_all_b for {hp}")
                mtl_all_b[hp] = Table(names=mtl_all[hp].colnames, dtype=mtl_all[hp].dtype)
            elif hp not in hp_a:
                log.details(f"Added dummy table to mtl_all for {hp}")
                mtl_all[hp] = Table(names=mtl_all_b[hp].colnames, dtype=mtl_all_b[hp].dtype)


# Use this to get all tiles that touch the given zone, not just ones that only
# have a center that falls inside the zone.
tile_rad =  get_tile_radius_deg()
margin = tile_rad - 0.2
fba_loc = str(fba_base)

def fiberassign_tile(targ_loc, tile_loc, runtime, tileid, tile_done=True):
    params = ["--rundate",
              runtime,
              "--obsdate",
              runtime[:10], # Only need the date, not the time
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
              "--fba_use_fabs",
              "1",
              "--sciencemask",
              str(sciencemask)
    ]

    fba_file = base_dir / "fba" / f"fba-{str(tileid).zfill(6)}.fits"
    # Only refiberassign if the file doesn't exist
    if not fba_file.is_file():
        fba_args = parse_assign(params)
        run_assign_full(fba_args)

    # Only update the MTL once this tile is done.
    if tile_done:
        # TODO find a way to do this without having to read (when we run fiberassign without io)
        # After assigning, load that fiber assignment and return the tids.
        with fitsio.FITS(fba_file) as h:
                tids = h["FASSIGN"]["TARGETID"][:] # Actually assigned TARGETIDS
                device = h["FASSIGN"]["DEVICE_TYPE"][:]
                # Cutting on "not ETC" probably not necessary but just to be safe.
                tids = tids[(tids > 0) & (device != "ETC")]
                log.details(f"Loaded {len(tids)} from {fba_file}")
        return tids
    return np.asarray([], dtype=int) # Force dtype = int to ensure stacking remains ints.

def save_mtl(mtl_to_save, hpx):
    log.details(f"Saving healpix {hpx}")
    mtl_to_save.write(hp_base / f"mtl-dark-hp-{hpx}.ecsv", overwrite=True)

n_nights = len(np.unique(tiles["TIMESTAMP_YMD"]))
times = {"gen_curr_mtl": [], "assign": [],  "get_last_time": [], "update_mtl": [], "save_mtl": [],}  # For profiling.
cur_year = tiles["TIMESTAMP_YMD"][0][:4]

# So we save at the correct points again later.
if loaded_from_checkpoint:
    cur_year = last_timestamp[:4]

not_added = True
log.details(f"Starting year: {cur_year}")
t2 = time.time()
with Pool(args.nproc) as p:
    for i, timestamp in enumerate(np.unique(tiles["TIMESTAMP_YMD"])):
        if loaded_from_checkpoint and timestamp <= last_timestamp:
            log.details(f"Skipped timestamp {timestamp} <= {last_timestamp} (checkpoint)")

            # If we are in this block, and enter this if condition, then we haven't yet reached
            # the checkpointed timestamp but we have passed the time at which the catalog
            # would be added, meaning the catalog was added in the checkpointed MTLs
            if timestamp > args.b_start_date: not_added = False
            continue
        if args.b_start_date:
            if not_added and (timestamp >= args.b_start_date):
                log.details(f"Adding catalog_b on {timestamp}")
                hpx_join = mtl_all.keys()
                concat_params = [(mtl_all[hp], mtl_all_b[hp]) for hp in hpx_join]
                res = p.starmap(concatenate_mtls, concat_params)

                for j, hp in enumerate(hpx_join):
                    mtl_all[hp] = res[j]

                log.details(f"Prev pixlist len: {len(pixlist)}")
                # The previous pixlist was generated from the healpixels of
                # only targets in the primary catalog. Dummy tables were added
                # to MTL all to account for healpixels that are in catalog
                # b but not a, but we need to update the pixlist now
                # that we've added catalog b to include those targets.
                pixlist = np.asarray(list(mtl_all.keys()))
                log.details(f"Updated pixlist len: {len(pixlist)}")
                del mtl_all_b # Free up some memory, now that those targets are in the main mtl.
                not_added = False

        log.details(f"Beginning night {i} {timestamp} by loading tiling...")
        night_year = timestamp[:4]

        # Step 1: generate the subset of tiles that are run on this night
        # And the associated file of targes observable by that tile.
        this_date = tiles["TIMESTAMP_YMD"] == timestamp

        # Unique to avoid 2 processes assigning the same tile.
        # Shouldn't be necessary with updated processing but that's fine.
        tiles_subset = unique(tiles[this_date & tiles["IN_DESI"]], "TILEID")

        hpx_night = tiles2pix(nside, tiles_subset["TILEID", "RA", "DEC"]) # Already unique from the return of tiles2pix
        hpx_night = hpx_night[np.isin(hpx_night, pixlist)] # The "fuzzy" nature of tiles 2 pix might return healpix we don't have targets in

        log.details(f"Night {i} {timestamp}: {len(tiles_subset)} tiles ({len(hpx_night)} HPX) to run")
        # if len(hpx_night) == 0: continue
        # Deduplicate the MTL to get only the most recent information for each target.
        # TODO run fiberassign in a way that we can skip saving target files.
        t_start_curr = time.time()
        curr_mtl = deduplicate_mtl(vstack([mtl_all[hpx] for hpx in hpx_night]))
        t_end_curr = time.time()
        times["gen_curr_mtl"].append(t_end_curr - t_start_curr)
        log.details(f"Gen curr mtl took {t_end_curr - t_start_curr} seconds...")
        # TODO send night as TIMESTAMP_YMD instead of i to save by night date instead of an arbitrary int.
        targ_files, tile_files, ntargs_on_tile = generate_target_files(curr_mtl, tiles_subset, base_dir, i)

        ntargs_on_tile = np.asarray(ntargs_on_tile)
        targ_files, tile_files = np.asarray(targ_files), np.asarray(tile_files)
        good_tile = np.where(ntargs_on_tile > 0)

        log.details(f"Good tile: {good_tile}, {ntargs_on_tile}")

        # Worthwhile to keep this for summary plot purposes
        tile_loc = base_dir / f"tiles-{timestamp}.fits"
        tiles_subset.write(tile_loc, overwrite=True)

        # Step 2: actually run the fiber assignment, and get back the assigned targetids
        t_start_assign = time.time()

        fiberassign_params = zip(targ_files[good_tile], tile_files[good_tile], tiles_subset["TIMESTAMP"][good_tile], tiles_subset["TILEID"][good_tile], tiles_subset["TILEDONE"][good_tile])
        assigned_tids = p.starmap(fiberassign_tile, fiberassign_params)
        assigned_tids = np.concatenate(assigned_tids)

        t_end_assign = time.time()
        times["assign"].append(t_end_assign - t_start_assign)
        log.details(f"Assignment took {t_end_assign - t_start_assign} seconds...")

        unique_tids, counts = np.unique(assigned_tids, return_counts=True)
        log.details(f"Sanity check on tid updates: {len(assigned_tids)}, {len(unique_tids)}, {np.unique(counts)}, {assigned_tids.dtype}")

        # Step 3 update the MTL
        # Determining the timestamp to imprint on the MTL update
        t3 = time.time()
        ts = [datetime.fromisoformat(t) for t in tiles_subset["TIMESTAMP"]]
        last_time = max(ts)
        last_time += timedelta(hours=1)
        last_time = last_time.isoformat()

        t_mid = time.time()
        times["get_last_time"].append(t_mid - t3)
        update_params = [(mtl_all[hpx], assigned_tids, targetmask, last_time, False) for hpx in hpx_night]
        updated_tbls = p.starmap(update_mtl, update_params) # Should return in same order as hpx_night
        for i, hpx in enumerate(hpx_night):
            mtl_all[hpx] = updated_tbls[i]
        t4 = time.time()
        times["update_mtl"].append(t4 - t3)
        log.details(f"MTL update took {t4 - t3} seconds...")

        # Step 4 save the updated MTLs
        # Write updated MTLs by healpix.
        if not args.danger:
            save_params = [(mtl_all[hpx], hpx) for hpx in hpx_night]
            p.starmap(save_mtl, save_params)
        # In danger mode only save if the year crosses over or it's the last night.
        elif (args.danger and (night_year > cur_year)):
            log.details(f"Saving on night {i} {timestamp}")
            save_params = [(mtl_all[hpx], hpx) for hpx in pixlist]
            p.starmap(save_mtl, save_params)

        t5 = time.time()
        times["save_mtl"].append(t5 - t4)
        log.details(f"Saving MTL took {t5 - t4} seconds...")

        cur_year = night_year

    t4 = time.time()
    log.details(f"Saving at conclusion...")
    save_params = [(mtl_all[hpx], hpx) for hpx in pixlist]
    p.starmap(save_mtl, save_params)
    t5 = time.time()
    times["save_mtl"].append(t5 - t4)
    log.details(f"Saving MTL took {t5 - t4} seconds...")

log.details("Done!")
t_end = time.time()
log.details(f"Init: \t\t\t{t2 - t_start} \t {(t2 - t_start) / 60}")
log.details(f"Full: \t\t\t{t_end - t_start} \t {(t_end - t_start) / 60}")
log.details(f"Average per night: \t{(t_end - t2) / n_nights}\t {(t_end - t2) / (n_nights * 60)}")

for k in times.keys():
    log.details(f"Average {k}: {np.mean(times[k])}")