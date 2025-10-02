#!/usr/bin/env python

# Run simulated fiberassign over either a simulated or given catalog.
# python run_survey_mp.py --ramin 200 --ramax 210 --decmin 20 --decmax 30 -o /pscratch/sd/d/dylang/fiberassign/mtl-4exp-lae-1000-rerun/ --npass 50 --catalog /pscratch/sd/d/dylang/fiberassign/lya-colore-lae-1000.fits --nproc 32  --fourex


# TODO proper docstring
import argparse
from multiprocessing import Pool
import time

# Non-DESI Imports
import numpy as np
from astropy.table import Table, vstack, join
import healpy as hp
import fitsio

# DESI imports
from desimodel.io import load_tiles
from desimodel.focalplane import get_tile_radius_deg
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
parser.add_argument("--ramax", required=True, type=float, help="maximum RA angle to assign over.")
parser.add_argument("--ramin", required=True, type=float, help="minimum RA angle to assign over.")
parser.add_argument("--decmax", required=True, type=float, help="maximum DEC angle to assign over.")
parser.add_argument("--decmin", required=True, type=float, help="minimum DEC angle to assign over.")
parser.add_argument("-o", "--outdir", required=True, type=str, help="where to save the mtl* and fba* output files.")
parser.add_argument("--npass", required=False, type=int, default=1, help="number of assignment passes to do.")
parser.add_argument("--nproc", required=False, type=int, default=1, help="number of multiprocessing processes to use.")
parser.add_argument("--fourex", required=False, action="store_true", help="take four exposures of a single tiling rather than four unique tilings.")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--catalog", type=str, help="A catalog of objects to use for fiber assignemnt.")
group.add_argument("--density", type=int, help="number density per square degree of randomly generated targets.")

args = parser.parse_args()

t_start = time.time()
targetmask = load_target_yaml("targetmask.yaml")
print(f"Using {targetmask}")

# Generate the random targets
rng = np.random.default_rng(91701)
if args.density:
    ra, dec = generate_random_objects(args.ramin, args.ramax, args.decmin, args.decmax, rng, args.density)
else:
    ra, dec = load_catalog(args.catalog, [args.ramin, args.ramax, args.decmin, args.decmax])

print(f"Generated {len(ra)} randoms...")

tbl = Table()
tbl["RA"] = ra
tbl["DEC"] = dec

nside = 64
theta, phi = np.radians(90 - dec), np.radians(ra)
pixlist = np.unique(hp.ang2pix(nside, theta, phi, nest=True))

print(f"{len(pixlist)} HEALpix to write.")

mtl_all = initialize_mtl(tbl, args.outdir)
t2 = time.time()

# Directories for later
base_dir = Path(args.outdir)
hp_base = base_dir / "hp" / "main" / "dark"

# Generate the tiling for this patch of sky.
# Load from the FITs file to match fiber assign expectations.
# tiles = Table(load_tiles(surveyops=False)) # NOTE: By default loads only tiles in the DESI footprint.
# tiles = load_tiles()

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
fba_loc = str(base_dir / "fba")

def fiberassign_tile(targ_loc, tile_loc):
    params = ["--rundate",
            "2025-09-16T00:00:00+00:00",
            "--overwrite",
            "--write_all_targets",
            "--footprint", # Actually means "footprint" of tile centers...
            tile_loc,
            "--dir",
            fba_loc,
            # "--sky_per_petal",
            # 0, # Use the default for this
            "--standards_per_petal",
            0,
            "--overwrite",
            "--targets",
            targ_loc,
    ]

    fba_args = parse_assign(params)
    run_assign_full(fba_args)
    # Don't need to return anything because fiberassign writes to files
    # which we'll load later.

def load_tids_from_fba(fba_loc):
    with fitsio.FITS(fba_loc) as h:

        tids = h["FASSIGN"]["TARGETID"][:] # Actually assigned TARGETIDS
        device = h["FASSIGN"]["DEVICE_TYPE"][:]
        # Cutting on "not ETC" probably not necessary but just to be safe.
        tids = tids[(tids > 0) & (device != "ETC")]
    return tids

curr_mtl = mtl_all # It starts with unique rows per target id so this is fine.
for i in range(1, args.npass + 1):
    print(f"Beginning iteration {i} by generating tiling...")
    if args.fourex: # Repeat each tiling four times before moving to the next one
         passnum = (i + 3) // 4
    else:
         passnum = i
    tiles = rotate_tiling(base_tiles, passnum)

    # Booleans for determining which tiles to keep.
    # Margin makes sure we don't end up with tiles that are "in bounds"
    # but because of the circular shape are off the corner of the
    # region and don't actually cover any of the targets (which crashes fiberassign)
    tiles_in_ra = (tiles["RA"] >= (args.ramin - margin)) & (tiles["RA"] <= (args.ramax + margin))
    tiles_in_dec = (tiles["DEC"] >= (args.decmin - margin)) & (tiles["DEC"] <= (args.decmax + margin))
    tiles_subset = tiles[tiles_in_ra & tiles_in_dec]

    print(f"Iteration {i}: {len(tiles_subset)} tiles to run")
    targ_files, tile_files = generate_target_files(curr_mtl, tiles_subset, base_dir, i)

    # Worthwhile to keep this for summary plot purposes
    tile_loc = base_dir / f"tiles-pass-{i}.fits"
    tiles_subset.write(tile_loc, overwrite=True)

    fiberassign_locs = zip(targ_files, tile_files)
    with Pool(args.nproc) as p:
         p.starmap(fiberassign_tile, fiberassign_locs)

    assigned_tids = np.array([])
    # TODO we can parallelize this because the order of assigned tids is irrelevant
    for tileid in tiles_subset["TILEID"]:
        tileid = str(tileid)
        fba_file = base_dir / "fba" / f"fba-{tileid.zfill(6)}.fits"
        print(f"Loading tids from {fba_file.name}")
        with fitsio.FITS(fba_file) as h:

                tids = h["FASSIGN"]["TARGETID"][:] # Actually assigned TARGETIDS
                device = h["FASSIGN"]["DEVICE_TYPE"][:]
                # Cutting on "not ETC" probably not necessary but just to be safe.
                tids = tids[(tids > 0) & (device != "ETC")]

                assigned_tids = np.concatenate([assigned_tids, tids])

    mtl_all = update_mtl(mtl_all, assigned_tids, use_desitarget=False)

    # Write updated MTLs by healpix.
    # TODO Seems to me like the way to {do update the global MTL is iterate over it
    # by healpix, and save the updated healpix if there are updated observations
    # for that healpix.
    # TODO Use these per loop saved MTLS to add checkpointing to the script.
    for hpx in np.array(np.unique(mtl_all["HEALPIX"])):
        print(f"Saving healpix {hpx}")
        this_hpx = mtl_all["HEALPIX"] == hpx
        mtl_all[this_hpx].write(hp_base / f"mtl-dark-hp-{hpx}.fits", overwrite=True)

    curr_mtl = deduplicate_mtl(mtl_all)
    curr_mtl.write(base_dir / "targets.fits", overwrite=True)

print("Done!")
t_end = time.time()
print(f"Init: \t\t\t{t2 - t_start} \t {(t2 - t_start) / 60}")
print(f"Full: \t\t\t{t_end - t_start} \t {(t_end - t_start) / 60}")
print(f"Average per pass: \t{(t_end - t2) / args.npass}\t {(t_end - t2) / (args.npass * 60)}")