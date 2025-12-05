# python trim_catalog_to_survey.py /pscratch/sd/d/dylang/fiberassign/tiles-50pass-superset.ecsv -o /global/cfs/cdirs/desi/users/dylang/fiberassign_desi2/tiles-50pass-5000deg-superset.ecsv --tiles

# python trim_catalog_to_survey.py /pscratch/sd/d/dylang/fiberassign/lya-colore-lae.fits -o /global/cfs/cdirs/desi/users/dylang/fiberassign_desi2/lya-colore-lae-density1975-area5000.fits --matchdensity 1975
import argparse
from pathlib import Path

from astropy.table import Table
import numpy as np

# TODO take in a survey area instead of harcoded.
# simassign imports
from simassign.util import check_in_survey_area, check_in_tile_area

parser = argparse.ArgumentParser()
parser.add_argument("catalog", type=str, help="A catalog of objects to use to trim.")
parser.add_argument("-o", "--out", required=True, type=str, help="where to save the resulting file.")
parser.add_argument("--seed", required=False, type=int, default=91701, help="seed for the random subsampling.")
parser.add_argument("--qsos", required=False, type=int, default=None, help="These are qsos and not lbgs, so select objects that would be excluded by the lbgs given by the matchdensity, and use this parameter for qso density.")
parser.add_argument("--desitarget", required=False, type=int, default=1, help="desitarget bit value encode into the catalo, default: 1.")
parser.add_argument("--matchdensity", required=False, type=int, default=1000, help="output density to match in n_targ per sq deg.")

group = parser.add_mutually_exclusive_group(required=False)
group.add_argument("--survey", type=str, default=None, help="use the survey defined by the boundaries in this file rather than the nominal DESI 2 survey.")
group.add_argument("--tiles", type=str, default=None, help="use the tiles in this file to trim the targets.")
args = parser.parse_args()

catalog_loc = Path(args.catalog)

data_tbl = Table.read(catalog_loc)

# Downsample to density first
sky_area = 360**2 / np.pi
density = len(data_tbl) / sky_area

idcs = np.arange(len(data_tbl))
rng = np.random.default_rng(args.seed)
rng.shuffle(idcs)

# Can't overample above the density we have.
target_density = np.min([density, args.matchdensity])
print(f"Sampling to density {target_density}...")

n_tot = int(args.matchdensity * sky_area)

if args.qsos is not None:
    qso_density = np.min([density, args.qsos])
    n_qso = int(qso_density * sky_area)
    keep_idcs = idcs[-n_qso:] # Take from the end so that we can later change our mind on how many lbgs we want to use.
else:
    keep_idcs = idcs[:n_tot]
data_tbl = data_tbl[keep_idcs]

print(f"Achieved density {len(data_tbl) / sky_area}.")

if args.tiles is not None:
    tiles = Table.read(args.tiles)
    in_survey = check_in_tile_area(data_tbl, tiles)
else:
    survey = None
    if args.survey is not None:
        survey = np.load(args.survey)
    in_survey = check_in_survey_area(data_tbl, survey)

data_tbl = data_tbl[in_survey]
data_tbl["DESI_TARGET"] = 2 ** args.desitarget

print(f"{len(data_tbl)} final rows.")
print("Saving...")
data_tbl.write(args.out, overwrite=True)
