#!/usr/bin/env python

# Generate some randoms
# python run_randoms.py --ramin 200 --ramax 210 --decmin 20 --decmax 30
# TODO proper docstring
import argparse

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
from simassign.mtl import update_mtl, deduplicate_mtl
from simassign.util import generate_random_objects


parser = argparse.ArgumentParser()
parser.add_argument("--ramax", required=True, type=float, help="maximum RA angle to assign over.")
parser.add_argument("--ramin", required=True, type=float, help="minimum RA angle to assign over.")
parser.add_argument("--decmax", required=True, type=float, help="maximum DEC angle to assign over.")
parser.add_argument("--decmin", required=True, type=float, help="minimum DEC angle to assign over.")
parser.add_argument("-o", "--outdir", required=True, type=str, help="where the save the mtl* and fba* output files.")
parser.add_argument("--npass", required=False, type=int, default=1, help="number of assignment passes to do.")

args = parser.parse_args()


# Generate the random targets
rng = np.random.default_rng(91701)
ra, dec = generate_random_objects(args.ramin, args.ramax, args.decmin, args.decmaxrng, 1000)
print(f"Generated {len(ra)} randoms...")

tbl = Table()
tbl["RA"] = ra
tbl["DEC"] = dec

# Minimum set of columns necessary for make_mtl:
# TARGETID`, `DESI_TARGET`, `BGS_TARGET`, `MWS_TARGET`, `NUMOBS_INIT`, `PRIORITY_INIT`, `PRIORITY` (because we won't pass a zcat)
# For targetids we will just use the indices, they just need to be unique and non negative
tbl["TARGETID"] = np.arange(len(ra))

# TODO I'm going to invent my own target bit for this, 2**22 (unused by desitarget). Call it LAE/LBG if you like.
# In order to piggyback off make_mtl we need to use a DESI target type, e.g. QSOs (bit 2, 2**2)
tbl["DESI_TARGET"] = 2**2

# These two are unused but necessary to exist for mtl
tbl["BGS_TARGET"] = 0
tbl["MWS_TARGET"] = 0

tbl["NUMOBS_INIT"] = 4
tbl["PRIORITY_INIT"] = 3400
tbl["PRIORITY"] = 3400

# QSO: {UNOBS: 3400, MORE_ZGOOD: 3350, MORE_ZWARN: 3300, MORE_MIDZQSO: 100, DONE: 2, OBS: 1, DONOTOBSERVE: 0}
tbl["SUBPRIORITY"] = rng.uniform(size=len(ra))

nside = 64
theta, phi = np.radians(90 - dec), np.radians(ra)
pixlist = np.unique(hp.ang2pix(nside, theta, phi, nest=True))

print(f"{len(pixlist)} HEALpix to write.")

# TODO command line arg.
base_dir = Path(args.outdir)

# Run mtl to split them into healpix ledgers.
make_ledger_in_hp(tbl, str(base_dir / "hp"), nside=nside, pixlist=pixlist, obscon="DARK", verbose=True)

hp_base = base_dir / "hp" / "main" / "dark"
mtl_all = Table()
for mtl_loc in hp_base.glob("*.ecsv"):
    print(f"Moving {mtl_loc.name} to fits")
    temp_tbl = Table.read(mtl_loc)
    # Helpixels are not z filled to the same digit length otherwies I'd use a regex to pull this out.
    hpx = mtl_loc.name.split("-")[-1].split(".")[0]
    temp_tbl["HEALPIX"] = int(hpx)

    # Update to custom target type.
    # temp_tbl["DESI_TARGET"] = 2**22
    # temp_tbl["TARGET_STATE"] = "LAE|UNOBS"

    temp_tbl.write(hp_base / mtl_loc.name.replace(".ecsv", ".fits"), overwrite=True)
    mtl_loc.unlink()

    mtl_all = vstack([mtl_all, temp_tbl])

mtl_all.write(base_dir / "targets.fits", overwrite=True)
# Generate the tiling for this patch of sky.
# Load frmo the FITs file to match fiber assign expectations.
# tiles = Table(load_tiles(surveyops=False)) # NOTE: By default loads only tiles in the DESI footprint.
tiles = load_tiles()
# Use this to get all tiles that touch the given zone, not just ones that only
# have a center that falls inside the zone.
tile_rad =  get_tile_radius_deg()
margin = tile_rad - 0.2

for i in range(args.npass):
    print(f"Beginning iteration {i}")
    # Booleans for determining which tiles to keep. We're just assuming dark time
    # since we want four passes, but the 7 pass program is only designed for
    # dark tiles anyway.
    tiles_in_ra = (tiles["RA"] >= (args.ramin - margin)) & (tiles["RA"] <= (args.ramax + margin))
    tiles_in_dec = (tiles["DEC"] >= (args.decmin - margin)) & (tiles["DEC"] <= (args.decmax + margin))
    this_pass = tiles["PASS"] == i # NOTE change this after testing.
    dark_tile = (tiles["PROGRAM"] == "DARK")
    # test_id = tiles["TILEID"] == 9682
    tiles_subset = tiles[tiles_in_ra & tiles_in_dec & this_pass & dark_tile]

    tiles_subset["OBSCONDITIONS"] = obsconditions.mask("DARK") * np.ones(len(tiles_subset), dtype=int)
    print(f"{len(tiles_subset)} tiles to run")
    tiles_subset.write(base_dir / "tiles.fits", overwrite=True)

    params = ["--rundate",
            "2025-09-16T00:00:00+00:00",
            "--overwrite",
            "--write_all_targets",
            "--footprint", # Actually means "footprint" of tile centers...
            str(base_dir / "tiles.fits"),
            "--dir",
            str(base_dir / "fba"),
            # "--sky_per_petal",
            # 0, # Use the default for this
            "--standards_per_petal",
            0,
            "--overwrite",
            "--targets",
            str(base_dir / "targets.fits"),
        ]

    fba_args = parse_assign(params)
    run_assign_full(fba_args)

    # Seems to me like the way to do update the global MTL is iterate over it
    # by healpix, and save the updated healpix if there are updated observations
    # for that healpix.
    assigned_tids = np.array([])
    # for fba_file in (base_dir / "fba").glob("*.fits"):
    for tileid in tiles_subset["TILEID"]:
        tileid = str(tileid)
        fba_file = base_dir / "fba" / f"fba-{tileid.zfill(6)}.fits"
        print(f"Loading tids from {fba_file.name}|")
        with fitsio.FITS(fba_file) as h:

                tids = h["FASSIGN"]["TARGETID"][:] # Actually assigned TARGETIDS
                device = h["FASSIGN"]["DEVICE_TYPE"][:]
                # Cutting on "not ETC" probably not necessary but just to be safe.
                tids = tids[(tids > 0) & (device != "ETC")]

                assigned_tids = np.concatenate([assigned_tids, tids])
    print(len(assigned_tids), len(np.unique(assigned_tids)))
    # mtl_all = update_mtl(mtl_all, assigned_tids, use_desitarget=True)

    mtl_all = update_mtl(mtl_all, assigned_tids, use_desitarget=True)

    # Write updated MTLs by healpix.
    # TODO only save these after the entire assigning loop?
    for hpx in np.array(np.unique(mtl_all["HEALPIX"])):
        print(f"Saving healpix {hpx}")
        this_hpx = mtl_all["HEALPIX"] == hpx
        mtl_all[this_hpx].write(hp_base / f"mtl-dark-hp-{hpx}.fits", overwrite=True)

    curr_mtl = deduplicate_mtl(mtl_all)
    curr_mtl.write(base_dir / "targets.fits", overwrite=True)

print("Done!")