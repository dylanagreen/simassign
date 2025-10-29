# python trim_catalog_to_survey.py /pscratch/sd/d/dylang/fiberassign/tiles-50pass-superset.ecsv -o /global/cfs/cdirs/desi/users/dylang/fiberassign_desi2/tiles-50pass-5000deg-superset.ecsv --tiles

# python trim_catalog_to_survey.py /pscratch/sd/d/dylang/fiberassign/lya-colore-lae.fits -o /global/cfs/cdirs/desi/users/dylang/fiberassign_desi2/lya-colore-lae-density1975-area5000.fits --matchdensity 1975
import argparse
from pathlib import Path

from astropy.table import Table
import numpy as np
from matplotlib.patches import Path as mpPath

# TODO take in a survey area instead of harcoded.
# simassign imports
from simassign.util import check_in_survey_area

parser = argparse.ArgumentParser()
parser.add_argument("catalog", type=str, help="A catalog of objects to use to trim.")
parser.add_argument("-o", "--out", required=True, type=str, help="where to save the resulting file.")
parser.add_argument("--seed", required=False, type=int, default=91701, help="seed for the random subsampling.")


group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--tiles", required=False, action="store_true", help="whether this catalog is tiles or not. If tiles, will set IN_DESI = False instead of truncating.")
group.add_argument("--matchdensity", required=False, type=int, default=1000, help="output density to match in n_targ per sq deg.")
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
keep_idcs = idcs[:n_tot]
data_tbl = data_tbl[keep_idcs]

print(f"Achieved density {len(data_tbl) / sky_area}.")

in_ngc, in_sgc = check_in_survey_area(data_tbl)

print("Saving...")
if args.tiles:
    data_tbl["IN_DESI"] = False
    data_tbl["IN_DESI"][in_ngc | in_sgc] = True
    print(np.sum(data_tbl["IN_DESI"]))
    data_tbl.sort("TILEID")
else:
    data_tbl = data_tbl[in_ngc | in_sgc]

print(f"{len(data_tbl)} final rows.")

data_tbl.write(args.out, overwrite=True)
