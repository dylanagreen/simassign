# python trim_catalog_to_survey.py /pscratch/sd/d/dylang/fiberassign/tiles-50pass-superset.ecsv -o /global/cfs/cdirs/desi/users/dylang/fiberassign_desi2/tiles-50pass-5000deg-superset.ecsv --tiles

# python trim_catalog_to_survey.py /pscratch/sd/d/dylang/fiberassign/lya-colore-lae.fits -o /global/cfs/cdirs/desi/users/dylang/fiberassign_desi2/lya-colore-lae-density1200-area5000.fits --matchdensity 1200
import argparse
from pathlib import Path

from astropy.table import Table
import numpy as np
from matplotlib.patches import Path as mpPath

# Points defining the hulls surrounding the survey area.
ngc_points = np.array([[ 1.49244e+02, -6.97500e+00],
                        [ 2.21244e+02, -6.97500e+00],
                        [ 2.30000e+02,  7.50000e-01],
                        [ 2.49268e+02,  1.18900e+00],
                        [ 2.50217e+02,  2.59200e+00],
                        [ 2.52665e+02,  7.16400e+00],
                        [ 2.55516e+02,  1.32070e+01],
                        [ 2.55862e+02,  1.48110e+01],
                        [ 2.34212e+02,  1.49840e+01],
                        [ 2.26749e+02,  1.49850e+01],
                        [ 1.54749e+02,  1.49850e+01],
                        [ 1.26212e+02,  1.49840e+01],
                        [ 1.21674e+02,  1.48000e+01],
                        [ 1.21241e+02,  1.43790e+01],
                        [ 1.20999e+02,  1.39530e+01],
                        [ 1.20934e+02,  1.34820e+01],
                        [ 1.24870e+02,  5.38000e+00],
                        [ 1.26725e+02,  2.03800e+00],
                        [ 1.27964e+02, -1.81000e-01],
                        [ 1.30860e+02, -5.09100e+00],
                        [ 1.32131e+02, -6.41800e+00],
                        [ 1.33345e+02, -6.75600e+00],
                        [ 1.40129e+02, -6.88500e+00]])

sgc_points = np.array([[304.844, -17.566],
                       [305.806, -17.985],
                       [413.806, -17.985],
                       [416.767, -17.934],
                       [418.013, -17.824],
                       [417.877, -16.953],
                       [416.752, -14.198],
                       [415.878, -12.355],
                       [414.643,  -9.925],
                       [413.231,  -7.502],
                       [411.103,  -4.543],
                       [409.966,  -2.981],
                       [408.302,  -0.898],
                       [406.496,   1.098],
                       [405.905,   1.743],
                       [403.939,   3.585],
                       [403.231,   4.162],
                       [401.65 ,   5.394],
                       [400.577,   5.691],
                       [399.417,   5.968],
                       [396.455,   5.986],
                       [381.811,   5.995],
                       [345.811,   5.995],
                       [324.455,   5.986],
                       [321.033,   5.976],
                       [317.865,   5.688],
                       [317.57 ,   5.544],
                       [316.186,   4.205],
                       [313.244,  -0.618],
                       [310.443,  -5.544],
                       [309.289,  -7.788],
                       [308.408,  -9.593],
                       [307.33 , -11.895],
                       [305.58 , -15.718]])


parser = argparse.ArgumentParser()
parser.add_argument("catalog", type=str, help="A catalog of objects to use to trim.")
parser.add_argument("-o", "--out", required=True, type=str, help="where to save the resulting file.")

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
rng = np.random.default_rng(91701)
rng.shuffle(idcs)

# Can't overample above the density we have.
target_density = np.min([density, args.matchdensity])
print(f"Sampling to density {target_density}...")

n_tot = int(args.matchdensity * sky_area)
keep_idcs = idcs[:n_tot]
data_tbl = data_tbl[keep_idcs]

print(f"Achieved density {len(data_tbl) / sky_area}.")

# Extracting the points as a 2d array
data_ra = np.array(data_tbl["RA"], dtype=float)
data_dec = np.array(data_tbl["DEC"], dtype=float)
data_points = np.vstack([data_ra, data_dec]).T

print("Checking NGC...")
p_ngc = mpPath(ngc_points)
in_ngc = p_ngc.contains_points(data_points)

print("Checking SGC...")
# Need to handle the rotation of the sgc, since it's disjoint
# when constraining angles to 0-360
to_rotate = data_ra < 90
data_points_rotate = np.array(data_points, copy=True)
data_points_rotate[to_rotate] += np.array([360, 0]) # Just add 360 to the lower points
p_sgc = mpPath(sgc_points)
in_sgc = p_sgc.contains_points(data_points_rotate)


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
