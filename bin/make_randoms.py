import argparse

from astropy.table import Table
import numpy as np

# simassign imports
from simassign.util import check_in_survey_area

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out", required=True, type=str, help="where to save the resulting file.")
parser.add_argument("--seed", required=False, type=int, default=91701, help="seed for the random subsampling.")
parser.add_argument("--density", required=False, type=int, default=1200, help="output density in n_targ per sq deg.")
parser.add_argument("--survey", type=str, default=None, help="use the survey defined by the boundaries in this file rather than the full sky.")
parser.add_argument("--desitarget", required=False, type=int, default=1, help="desitarget bit value encode into the catalo, default: 1.")
parser.add_argument("--program", required=False, type=str, default="DARK", help="program to encode in table metadata.")
args = parser.parse_args()

sky_area = 360**2 / np.pi
n_targs = int(args.density * sky_area)

rng = np.random.default_rng(args.seed)
ra = rng.uniform(0, 2 * np.pi, n_targs)
ra = np.rad2deg(ra)

dec = np.arccos(rng.uniform(-1, 1, n_targs)) - np.pi/2
dec = np.rad2deg(dec)

data_tbl = Table({"RA": ra, "DEC": dec,})
data_tbl["DESI_TARGET"] = 2 ** args.desitarget

data_tbl.meta["PROGRAM"] = args.program

if args.survey is not None:
    try:
        survey = np.load(args.survey)
    except ValueError: # Survey is multiple polygons and was saved as an object array.
        survey = np.load(args.survey, allow_pickle=True)
        survey = [s for s in survey] # Converts the ragged numpy array to list of numpy arrays.
    in_survey = check_in_survey_area(data_tbl, survey)

    data_tbl[in_survey].write(args.out, overwrite=True)
else:
    data_tbl.write(args.out, overwrite=True)