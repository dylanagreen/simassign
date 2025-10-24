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
from astropy.table import Table, vstack
import fitsio
import numpy as np

# TODO remove this at some point to point to a generic simassign import.
import sys
sys.path.append("/pscratch/sd/d/dylang/repos/simassign/src/")
from simassign.io import *
from simassign.util import get_nobs_arr

parser = argparse.ArgumentParser()
parser.add_argument("--mtls", required=True, type=str, help="base directory of mtls to process")
parser.add_argument("--nproc", required=False, type=int, default=1, help="number of multiprocessing processes to use for loading tables.")
parser.add_argument("-o", "--outdir", required=True, type=str, help="where to save generated plots.")
parser.add_argument("-s", "--suffix", required=True, type=str, help="suffix to attach to file names.")
args = parser.parse_args()

out_dir = Path(args.outdir)

# TODO these functions should be moved to the main package.
# def get_all_mtl_locs(top_dir):
#     hp_base = top_dir / "hp" / "main" / "dark"
#     fnames = list(hp_base.glob("*.fits"))
#     if len(fnames) == 0:
#         fnames = list(hp_base.glob("*.ecsv"))
#     return fnames

# def load_mtl(mtl_loc):
#     temp_tbl = Table.read(mtl_loc)
#     return temp_tbl

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
timestamps = np.unique(timestamps)
nobs = len(timestamps)

print(f"Calculating values...")
# Each row is a pass, each column is number of objects with that many exposures
print("Generating num obs per update arr....")
t_start = time.time()

print(len(list(mtl_all.values())))

def get_nobs_mp(mtl):
     return get_nobs_arr(mtl, timestamps)

# with Pool(args.nproc) as p:
#      res = p.map(get_nobs_mp, list(mtl_all.values()))

res = []
for k, v in mtl_all.items():
     res.append(get_nobs_mp(v))

nobs = np.zeros(res[0][0].shape)
at_least = np.zeros(res[0][1].shape)

for i, r in enumerate(res):
     nobs += res[0]
     at_least += res[1]

t_end = time.time()
print(f"Nobs took {t_end - t_start} seconds...")

# [0,0] is the "at least zero exposures with zero iterations" which should include
# every single object at this point.
fraction = at_least / at_least[0, 0]

t_start = time.time()
# del res

print("Getting n_tiles...")
with Pool(args.nproc) as p:
    ntiles = p.map(tiles_from_file, get_tile_files(top_dir))
ntiles.insert(0, 0)
ntiles_cum = np.cumsum(ntiles)

print(f"Get Tiles took {t_end - t_start} seconds...")

print("Writing all files...")
np.save(out_dir / f"nobs_{args.suffix}.npy", nobs)
np.save(out_dir / f"at_least_{args.suffix}.npy", at_least)
np.save(out_dir / f"fraction_{args.suffix}.npy", fraction)
np.save(out_dir / f"ntiles_{args.suffix}.npy", ntiles)
np.save(out_dir / f"ntiles_cum_{args.suffix}.npy", ntiles_cum)