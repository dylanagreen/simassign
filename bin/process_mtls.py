#!/usr/bin/env python

# Process the MTLS into necessary arrays for summary plots, to avoid
# having to reprocess when making plots and iterating.

# python process_mtls.py  --mtls  /pscratch/sd/d/dylang/fiberassign/mtl-4exp-lae-1000-big-nproc-32-inputtiles-withstds/ -o /pscratch/sd/d/dylang/fiberassign/processed/  --nproc 32 --suffix "lae-1000-big-inputtiles-withstds"

# TODO proper docstring
# stdlib imports
import argparse
from multiprocessing import Pool
from pathlib import Path
import time

# Non DESI imports
import fitsio
import numpy as np

from simassign.io import *
from simassign.util import get_nobs_arr, get_targ_done_arr

parser = argparse.ArgumentParser()
parser.add_argument("--mtls", required=True, type=str, help="base directory of mtls to process")
parser.add_argument("--nproc", required=False, type=int, default=1, help="number of multiprocessing processes to use for loading tables.")
parser.add_argument("-o", "--outdir", required=True, type=str, help="where to save generated files.")
parser.add_argument("-s", "--suffix", required=True, type=str, help="suffix to attach to file names.")
parser.add_argument("--nobs", required=False, action="store_true", help="get and save the nobs array if set. Otherwise only get the array of targets that met their own goal.")
parser.add_argument("--split_subtype", required=False, action="store_true", help="split subtypes of targets into their own arrays. e.g. if there are multiple types of LBGs, do not aggregate their results as one LBG class.")
args = parser.parse_args()

out_dir = Path(args.outdir)
out_dir.mkdir(exist_ok=True)

def get_num_tiles(top_dir):
    n_tiles = [0]
    for fname in sorted(top_dir.glob("tiles-*.fits")):
        with fitsio.FITS(fname) as h:
            n_tiles.append(len(h[1][:]))
    return n_tiles

def get_tile_files(mtl_dir):
    return sorted(mtl_dir.glob("tiles-*.fits"))

def tiles_from_file(fname):
     with fitsio.FITS(fname) as h:
            return len(h[1][:])

top_dir = Path(args.mtls)
mtl_loc =  top_dir / "hp" / "main" / "dark"

# Load all the mtls for each of the input mtl bases
t_start = time.time()
print(f"Loading {mtl_loc}...")
mtl_all = load_mtl_all(mtl_loc, as_dict=True, nproc=args.nproc)

# mtl = vstack(mtl_tbls)
t_end = time.time()
print(f"Loading took {t_end - t_start} seconds...")

timestamps = [np.array(mtl["TIMESTAMP"], dtype=str) for mtl in mtl_all.values()]
timestamps = np.concatenate(timestamps)
timestamps = np.unique(timestamps)#[:15]
nobs = len(timestamps)

targs = [np.unique(mtl["DESI_TARGET"]) for mtl in mtl_all.values()]
targs = np.concatenate(targs)
targs = np.unique(targs)
targs = targs[targs < 2 ** 10] # Not gonna use anything more than bit 10 for this.

if args.split_subtype:
    all_targs = []
    targs_complete = []
    for mtl in mtl_all.values():
        for t in targs:
            # Already got all sub targs.
            if t in targs_complete:
                continue
            this_targ = mtl["DESI_TARGET"] == t
            # Name should be the same for all target states so just take the first and split that
            name = mtl['TARGET_STATE'][this_targ][0].split("|")[0]

            if np.sum(this_targ) == 0:
                continue

            if f"{name}_TARGET" in mtl.colnames:
                sub_bits = np.unique(mtl[f"{name}_TARGET"][this_targ])
                for s in sub_bits:
                    all_targs.append(f"{t}|{s}")
            else:
                all_targs.append(str(t))
            targs_complete.append(t) # We actually got something...

        if len(targs_complete) == len(targs): break

    targs = all_targs


print(f"Found targs {targs}")

print(f"Calculating values...")
# Each row is a pass, each column is number of objects with that many exposures
print("Generating num obs per update arr....")
t_start = time.time()

print("Num mtls:", len(list(mtl_all.values())))
print("Num obs:", nobs)

def get_nobs_mp(mtl):
     return get_nobs_arr(mtl, timestamps)

def get_done_mp(mtl):
     return get_targ_done_arr(mtl, args.split_subtype, targs, timestamps)

if args.nobs:
    with Pool(args.nproc) as p:
        res = p.map(get_nobs_mp, list(mtl_all.values()))

    # res = []
    # for k, v in mtl_all.items():
    #      res.append(get_nobs_mp(v))

    nobs = np.zeros(res[0][0].shape)
    at_least = np.zeros(res[0][1].shape)

    # Res is an array of tuples (tuples the return of get_nobs) so this just
    # unpacks all the tuples.
    for i, r in enumerate(res):
        nobs += r[0]
        at_least += r[1]


    # [0,0] is the "at least zero exposures at zero iterations" which should include
    # every single object at this point.
    fraction = at_least / at_least[0, 0]

    print("Writing nobs arrs...")
    np.save(out_dir / f"nobs_{args.suffix}.npy", nobs)
    np.save(out_dir / f"at_least_{args.suffix}.npy", at_least)
    np.save(out_dir / f"fraction_full_{args.suffix}.npy", fraction)
else:
    with Pool(args.nproc) as p:
        res = p.map(get_done_mp, list(mtl_all.values()))

    done = np.zeros(res[0][0].shape)
    n_tot = np.zeros(res[1][1].shape)

    for i, r in enumerate(res):
        done += r[0]
        n_tot += r[1]

    fraction = done / n_tot[:, None] # Should hopefully broadcast correctly.

    print(f"Done shape: {done.shape}")
    print(done)

    print(f"Max achieved: {targs} : {fraction[:, -1]}")

    print("Writing done arrs...")
    np.save(out_dir / f"done_{args.suffix}.npy", done)
    np.save(out_dir / f"n_tot_{args.suffix}.npy", n_tot)
    np.save(out_dir / f"fraction_{args.suffix}.npy", fraction)

t_end = time.time()
print(f"Nobs took {t_end - t_start} seconds...")


t_start = time.time()
del res

print("Getting n_tiles...")
with Pool(args.nproc) as p:
    ntiles = p.map(tiles_from_file, get_tile_files(top_dir))
ntiles.insert(0, 0)
ntiles_cum = np.cumsum(ntiles)

t_end = time.time()
print(f"Get Tiles took {t_end - t_start} seconds...")

print("Writing n_tiles...")

np.save(out_dir / f"ntiles_{args.suffix}.npy", ntiles)
np.save(out_dir / f"ntiles_cum_{args.suffix}.npy", ntiles_cum)