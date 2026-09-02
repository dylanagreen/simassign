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
parser.add_argument("-o", "--outdir", required=True, type=str, help="where to save the mtl* and fba* output files.")
parser.add_argument("-t", "--tiles", required=True, type=str, help="tiling to use for observations.")
parser.add_argument("--stds", required=False, type=str, help="base location of standards catalog.")
parser.add_argument("--skies", required=False, type=str, help="base location of sky catalog.")
parser.add_argument("--nproc", required=False, type=int, default=1, help="number of multiprocessing processes to use.")
parser.add_argument("--config", required=False, type=str, help="configuration yaml file with target parameters. At minimum this should contain everything in targetmask.yaml, but in the future could contain additional run parameters.")
parser.add_argument("--danger", required=False, action="store_true", help="you want this to run as fast as possible, so do everything dangerously.")
parser.add_argument("--resetmtl", required=False, action="store_true", help="reset the mtl every other night for reassignment tests.")
parser.add_argument("--seed", required=False, type=int, default=100721, help="seed to use for randomness")
parser.add_argument("--catalog", type=str, nargs="*", required=True, help="Catalog(s) of objects to use for fiber assignment. Expect that the PROGRAM value is written to the fits headers.")
parser.add_argument("--catalog_later", type=str, nargs="*", help="catalog(s) of objects to use for fiber assignment, that will be added later in the survey.")
parser.add_argument("--later_starts", type=str, nargs="*", help="the date on which targets in catalog b get added to the survey. Should be of form YYYYMMDD")
# TODO rename catalog b to something more useful.
args = parser.parse_args()

if args.catalog_later or args.later_starts:
    assert args.catalog_later and args.later_starts, "If providing --catalog_later or --later_starts, you must provide both!"
if args.catalog_later and args.later_starts:
    assert len(args.catalog_later) == len(args.later_starts), "Must provide one start date for each catalog in catalog_later"

    # Sorting the dates to add means that at the comparison step in the loop
    # we only ever need to check against the first vslue of the list. We will
    # pop if off when we add it.
    dates_to_add = np.asarray(args.later_starts)
    catalogs_to_add = np.asarray(args.catalog_later)
    sorter = np.argsort(dates_to_add)
    dates_to_add = dates_to_add[sorter]
    catalogs_to_add = catalogs_to_add[sorter]
else:
    dates_to_add = np.array([])
    catalogs_to_add = np.array([])

t_start = time.time()

if args.config is not None:
    with open(args.config) as f:
        targetmask = yaml.safe_load(f)
else:
    targetmask = load_target_yaml("targetmask.yaml")

def load_calibration(cal_loc, cal_type, pixlist, start_id):
    tbl = Table.read(cal_loc)
    mtl = initialize_mtl(tbl, args.outdir, as_dict=True, cal_type=cal_type,
                                targetmask=targetmask, nproc=args.nproc,
                                rng=rng, start_id=start_id, healpixels_to_load=pixlist)
    # Some targs are cut by pixlist but this is fine, they're still unique
    return mtl, len(tbl)


sciencemask = target_mask_to_int(targetmask)
skymask = target_mask_to_int(targetmask, "SKY")
stdmask = target_mask_to_int(targetmask, "STD")

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

rng = np.random.default_rng(args.seed)

# Directories for later
base_dir = Path(args.outdir)
hp_base = base_dir / "hp" / "main" #/ "dark"
fba_base = base_dir / "fba"

tile_loc = Path(args.tiles)
tiles = Table.read(tile_loc)

nside = 64
def get_pixlist(ra, dec):
    theta, phi = np.radians(90 - dec), np.radians(ra)
    return np.unique(hp.ang2pix(nside, theta, phi, nest=True))

loaded_from_checkpoint = False
# Check for healpixels AND fiber assignments, if there's only the former the
# script may have interrupted when the catalog was still being generated, and
# we may attempt an incomplete checkpoint load.
mtl_all = {}
mtl_calib = {}
calib_progs = ["STD", "SKY"]
pixlist = {}
curr_tid = 0
if hp_base.is_dir(): #and fba_base.is_dir():
    # Attempt to checkpoint
    timestamps = []
    for prog_dir in hp_base.glob("*"):
        prog = prog_dir.name.upper()
        if prog in calib_progs:
            mtl_calib[prog] = load_mtl_all(prog_dir, as_dict=True, nproc=args.nproc)
        else:
            mtl_all[prog] = load_mtl_all(prog_dir, as_dict=True, nproc=args.nproc)
            timestamps.append([np.sort(tbl["TIMESTAMP"])[-1] for tbl in mtl_all[prog].values()])

    last_timestamp = np.sort(np.concatenate(timestamps))[-1]
    last_timestamp = last_timestamp[:10] # Only need the date, not the time
    last_timestamp = last_timestamp.replace("-", "")

    loaded_from_checkpoint = True
    log.details(f"Loaded Checkpointed MTLs with last timestamp: {last_timestamp}")
    pixlist = {k: list(v.keys()) for k, v in mtl_all.items()}

    # Anything that was added before the latest timestamp
    # will already be reflected in the loaded mtls. Remove them.
    if len(dates_to_add) > 0:
        keep = dates_to_add > last_timestamp
        dates_to_add = dates_to_add[keep]
        catalogs_to_add = catalogs_to_add[keep]

else:
    for catalog in args.catalog:
        tbl = Table.read(catalog)

        # TARGETID will be reset in initi_mtl, Z_COSMO doesn't exist in the standars
        # table, so it breaks the stacking of the two.
        if "TARGETID" in tbl.colnames:
            del tbl["TARGETID"]

        if "Z_COSMO" in tbl.colnames:
            del tbl["Z_COSMO"]

        ra = tbl["RA"]
        dec = tbl["DEC"]

        prog = tbl.meta["PROGRAM"]
        pixlist[prog] = get_pixlist(ra, dec)

        log.details(f"Using {len(tbl)} {prog=} targets...")
        log.details(f"{len(pixlist[prog])} HEALpix covered by catalog.")

        # if args.stds is not None:
        #     mtl_all[prog] = initialize_mtl(tbl, args.outdir, stds_catalog,
        #                                    as_dict=True, targetmask=targetmask,
        #                                    nproc=args.nproc, rng=rng, program=prog,
        #                                    start_id=curr_tid)
        # else:
        mtl_all[prog] = initialize_mtl(tbl, args.outdir, as_dict=True,
                                       targetmask=targetmask, nproc=args.nproc,
                                       rng=rng, program=prog, start_id=curr_tid)
        curr_tid += len(tbl)

    full_pixlist = np.unique(np.concatenate(list(pixlist.values())))
    if args.stds is not None:
        mtl_calib["STD"], ntarg = load_calibration(args.stds, "STD", full_pixlist, curr_tid)
        curr_tid += ntarg

    if args.skies is not None:
        mtl_calib["SKY"], ntarg = load_calibration(args.skies, "SKY", full_pixlist, curr_tid)
        curr_tid += ntarg

# Use this to get all tiles that touch the given zone, not just ones that only
# have a center that falls inside the zone.
tile_rad =  get_tile_radius_deg()
margin = tile_rad - 0.2
fba_loc = str(fba_base)

def fiberassign_tile(targ_loc, tile_loc, runtime, tileid, tile_done=True, design_ha=0):
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
              str(sciencemask),
              "--skymask",
              str(skymask),
              "--stdmask",
              str(stdmask),
              "--ha",
              str(design_ha) # Default is zero so this shouldn't change behaviour unless explicitly passed
    ]

    fba_file = base_dir / "fba" / f"fba-{str(tileid).zfill(6)}.fits"
    # Only refiberassign if the file doesn't exist
    if not fba_file.is_file():
        fba_args = parse_assign(params)
        run_assign_full(fba_args)

    # Only update the MTL once this tile is done.
    if tile_done:
        # TODO find a way to do this without having to read (when we run fiberassign without io)
        # Counterpoint: this makes checkpointing easy since it reloads assigned tiles
        # whose results didn't get saved to the MTL at the last checkpoint
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
    # TODO put the healpix in the metadata and we don't need to pass it in.
    prog = mtl_to_save.meta["PROGRAM"].lower()
    log.details(f"Saving healpix {hpx}, {prog=}")
    mtl_to_save.write(hp_base / prog / f"mtl-{prog}-hp-{hpx}.ecsv", overwrite=True)

def add_dummies(mtl_a, mtl_b):
    # TODO docstring, this mutates inputs
    # Generate empty tables for healpixels that are in one catalog but not
    # the other.
    hp_a = list(mtl_a.keys())
    hp_b = list(mtl_b.keys())

    log.details("Generating dummy tables...")
    for hp in (hp_a + hp_b):
        if hp not in hp_b:
            log.details(f"Added dummy table to mtl_b for {hp}")
            mtl_b[hp] = Table(names=mtl_a[hp].colnames, dtype=mtl_a[hp].dtype)
        elif hp not in hp_a:
            log.details(f"Added dummy table to mtl_a for {hp}")
            mtl_a[hp] = Table(names=mtl_b[hp].colnames, dtype=mtl_b[hp].dtype)

n_nights = len(np.unique(tiles["TIMESTAMP_YMD"]))
times = {"gen_curr_mtl": [], "assign": [],  "get_last_time": [], "update_mtl": [], "save_mtl": [],}  # For profiling.
cur_year = tiles["TIMESTAMP_YMD"][0][:4]

# So we save at the correct points again later.
if loaded_from_checkpoint:
    cur_year = last_timestamp[:4]

# Cut tiles for programs which we have no targets
tiles = tiles[np.isin(tiles["PROGRAM"], list(mtl_all.keys()))]

not_added = True
log.details(f"Starting year: {cur_year}")
# This is a helper variable for tracking what new healpixel coverage is added
# later for adding in calibration targets.
full_pixlist = np.unique(np.concatenate(list(pixlist.values())))
t2 = time.time()
with Pool(args.nproc) as p:
    for i, timestamp in enumerate(np.unique(tiles["TIMESTAMP_YMD"])):
        if loaded_from_checkpoint and timestamp <= last_timestamp:
            log.details(f"Skipped timestamp {timestamp} <= {last_timestamp} (checkpoint)")

            continue
        if (len(dates_to_add) > 0) and (timestamp >= dates_to_add[0]):
            log.details(f"Adding {catalogs_to_add[0]} on {timestamp}")

            # Do not load standards for catalog b. Since it gets added to mtl_all, the
            # stadards would be duplicated if we did.
            tbl_add = Table.read(catalogs_to_add[0])
            prog_add = tbl_add.meta["PROGRAM"]

            # Need to make the timestamp for this catalog. "timetsamp" is
            # > date to add because date to add is at 000 UTC, which is what
            # we will put into the catalog for the first row of these targets
            # (so they are "on" for that "night" of observing)
            timestamp_add = dates_to_add[0][:4] + "-" + dates_to_add[0][4:6] + "-" + dates_to_add[0][6:]
            timestamp_add += "T00:00:01+00:00"

            # If we are adding to a program that already exists, add
            # it to that mtl
            if prog_add in mtl_all.keys():
                mtl_add = initialize_mtl(tbl_add, save_dir=None, as_dict=True,
                                         targetmask=targetmask, nproc=args.nproc,
                                         start_id=curr_tid, timestamp=timestamp_add,
                                         rng=rng)

                # Adding dummies if they don't cover exactly the same healpixels.
                add_dummies(mtl_all[prog_add], mtl_add)

                hpx_join = mtl_add.keys()
                concat_params = [(mtl_all[prog_add][hp], mtl_add[hp]) for hp in hpx_join]
                res = p.starmap(concatenate_mtls, concat_params)

                for j, hp in enumerate(hpx_join):
                    mtl_all[prog_add][hp] = res[j]

                log.details(f"Prev pixlist len: {len(pixlist[prog_add])}")
                # The previous pixlist was generated from the healpixels of
                # only targets in the primary catalog. New pixlist may not be
                # the same due to the catalogs not covering entirely the same area
                pixlist[prog_add] = np.asarray(list(hpx_join))
                log.details(f"Updated pixlist len: {len(pixlist[prog_add])}")
            else:
                # We create with saving to make sure that the directory exists
                # for later when we save at ehe end of the run.
                # Also ensures it is checkpointed correctly.
                mtl_add = initialize_mtl(tbl_add, save_dir=args.outdir, as_dict=True,
                                         targetmask=targetmask, nproc=args.nproc,
                                         start_id=curr_tid, timestamp=timestamp_add,
                                         rng=rng)

                mtl_all[prog_add] = mtl_add
                pixlist[prog_add] = list(mtl_add.keys())

            curr_tid += len(tbl_add)

            # Remove this catalog from the "to add"
            dates_to_add = dates_to_add[1:]
            catalogs_to_add = catalogs_to_add[1:]

            # Add some calibration targets if necessary
            new_pixlist = np.unique(np.concatenate(list(pixlist.values())))
            added = ~np.isin(new_pixlist, full_pixlist)
            if np.sum(added) > 0:
                diff_pixlist = new_pixlist[added]
                if args.stds is not None:
                    mtl_add, ntarg = load_calibration(args.stds, "STD", diff_pixlist, curr_tid)
                    curr_tid += ntarg
                    # Since the healpixels are unique (we kept only new healpixels)
                    # this should concatenate without replacement
                    log.details(f"Adding STD, increasing from {len(mtl_calib["STD"].keys())} healpixels...")
                    mtl_calib["STD"] = mtl_calib["STD"] | mtl_add
                    log.details(f"..to {len(mtl_calib["STD"].keys())} healpixels")

                if args.skies is not None:
                    mtl_add, ntarg = load_calibration(args.skies, "SKY", diff_pixlist, curr_tid)
                    curr_tid += ntarg
                    # Since the healpixels are unique (we kept only new healpixels)
                    # this should concatenate without replacement
                    log.details(f"Adding SKY, increasing from {len(mtl_calib["SKY"].keys())} healpixels...")
                    mtl_calib["SKY"] = mtl_calib["SKY"] | mtl_add
                    log.details(f"..to {len(mtl_calib["SKY"].keys())} healpixels")

            del mtl_add # Free up some memory.

        log.details(f"Beginning night {i} {timestamp} by loading tiling...")
        night_year = timestamp[:4]

        # Step 1: generate the subset of tiles that are run on this night
        # And the associated file of targes observable by that tile.
        this_date = tiles["TIMESTAMP_YMD"] == timestamp

        # Unique to avoid 2 processes assigning the same tile.
        # Shouldn't be necessary with updated processing but that's fine.
        tiles_subset = unique(tiles[this_date & tiles["IN_DESI"]], "TILEID")

        # Already unique from the return of tiles2pix
        # The "if prog in tiles_subset" helps catch if a specific prog is not
        # observed on a given night.
        # NOTE: hpx_nights.keys() is guaranteed to always be a subset of mtl_all.keys() because of this.
        night_prog = tiles_subset["PROGRAM"]
        hpx_night = {prog: tiles2pix(nside, tiles_subset["TILEID", "RA", "DEC"][tiles_subset["PROGRAM"] == prog]) for prog in mtl_all.keys() if prog in night_prog}
        hpx_night = {k: v[np.isin(v, pixlist[k])] for k, v in hpx_night.items()} # The "fuzzy" nature of tiles 2 pix might return healpix we don't have targets in
        all_hpx_night = np.concatenate(list(hpx_night.values()))
        log.details(f"Night {i} {timestamp}: {len(tiles_subset)} tiles to run")
        # if len(hpx_night) == 0: continue
        # Deduplicate the MTL to get only the most recent information for each target.
        # TODO run fiberassign in a way that we can skip saving target files.
        t_start_curr = time.time()
        curr_mtl = {prog: deduplicate_mtl(vstack([mtl_all[prog][hpx] for hpx in hpx_night[prog]])) for prog in hpx_night.keys()}
        calib_mtl = {prog: vstack([v for k, v in mtl_calib[prog].items() if k in all_hpx_night]) for prog in mtl_calib.keys()}
        t_end_curr = time.time()
        times["gen_curr_mtl"].append(t_end_curr - t_start_curr)
        log.details(f"Gen curr mtl took {t_end_curr - t_start_curr} seconds...")
        # TODO send night as TIMESTAMP_YMD instead of i to save by night date instead of an arbitrary int.
        targ_files, tile_files, ntargs_on_tile = generate_target_files(curr_mtl, calib_mtl, tiles_subset, base_dir, i)

        ntargs_on_tile = np.asarray(ntargs_on_tile)
        targ_files, tile_files = np.asarray(targ_files), np.asarray(tile_files)
        good_tile = np.where(ntargs_on_tile > 0)

        # Some debugging lines I leave for posterity.
        # log.details(f"Good tile: {good_tile}, {ntargs_on_tile}")
        # log.details(np.array(tiles_subset["TILEID"][good_tile]))
        # log.details(tiles_subset[ntargs_on_tile == 0])

        # Worthwhile to keep this for summary plot purposes
        tile_loc = base_dir / f"tiles-{timestamp}.fits"
        tiles_subset.write(tile_loc, overwrite=True)

        # Step 2: actually run the fiber assignment, and get back the assigned targetids
        t_start_assign = time.time()

        fiberassign_params = zip(targ_files[good_tile], tile_files[good_tile], tiles_subset["TIMESTAMP"][good_tile],
                                 tiles_subset["TILEID"][good_tile], tiles_subset["TILEDONE"][good_tile],
                                 tiles_subset["DESIGNHA"][good_tile])
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
        update_params = [(mtl_all[prog][hpx], assigned_tids, targetmask, last_time, False) for prog in hpx_night.keys() for hpx in hpx_night[prog]]
        updated_tbls = p.starmap(update_mtl, update_params) # Should return in same order as prog then hpx_night
        j = 0
        for prog in hpx_night.keys():
            for hpx in hpx_night[prog]:
                mtl_all[prog][hpx] = updated_tbls[j]
                j += 1
        t4 = time.time()
        times["update_mtl"].append(t4 - t3)
        log.details(f"MTL update took {t4 - t3} seconds...")

        # Step 4 save the updated MTLs
        # Write updated MTLs by healpix.
        if not args.danger:
            save_params = [(mtl_all[prog][hpx], hpx) for prog in hpx_night.keys() for hpx in hpx_night[prog]]
            p.starmap(save_mtl, save_params)
        # In danger mode only save if the year crosses over or it's the last night.
        elif (args.danger and (night_year > cur_year)):
            log.details(f"Saving on night {i} {timestamp}")
            save_params = [(mtl_all[prog][hpx], hpx) for prog in mtl_all.keys() for hpx in pixlist[prog]]
            p.starmap(save_mtl, save_params)

        t5 = time.time()
        times["save_mtl"].append(t5 - t4)
        log.details(f"Saving MTL took {t5 - t4} seconds...")

        cur_year = night_year

        # TODO remove reset mtl.
        if args.resetmtl and (i % 2) == 1:
            log.details(f"Resetting MTL after Night {i}")
            mtl_all = initialize_mtl(tbl, None, stds_catalog, as_dict=True, targetmask=targetmask, nproc=args.nproc)

    t4 = time.time()
    log.details(f"Saving at conclusion...")
    save_params = [(mtl_all[prog][hpx], hpx) for prog in mtl_all.keys() for hpx in pixlist[prog]]
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