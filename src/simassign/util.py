# DESI imports
from desitarget.targetmask import desi_mask, obsconditions

# Non-DESI outside imports
from astropy.table import Table
import numpy as np


# Rotations for first through 15th passes of the sky of the DESI tiling
rots = np.array([[0, 0],
                 [2.56349, 1.70645],
                 [5.30251, 0.46050],
                 [2.77437, -1.24945],
                 [5.23554, 3.36665],
                 [2.54094, 4.69353],
                 [5.14642, 6.23672],
                 [-0.09047, 3.16649],
                 [0.11911, -2.83846],
                 [-2.51911, -1.80809],
                 [7.72129, 4.96644],
                 [-2.58180, 1.36959],
                 [2.68461, -4.42267],
                 [-0.03550, 6.49827],
                 [2.81603, 8.03110]])

new_rots = np.array([[-2.55045500e+00,  4.35673381e+00],
                     [-5.11839125e+00, -4.06454350e-01],
                     [-2.55045500e+00, -4.75454674e+00],
                     [-5.11839125e+00,  2.57172193e+00],
                     [ 5.22815667e+00, -2.48253543e+00],
                     [-5.11839125e+00, -3.36388339e+00],
                     [-1.71500000e-03, -6.13481438e+00],
                     [-5.11839125e+00,  5.64458274e+00],
                     [ 5.22815667e+00, -5.59955180e+00],
                     [-7.68632750e+00,  8.82433818e-01],
                     [ 7.72129000e+00, -7.94661891e-01],
                     [ 7.72129000e+00,  1.80247109e+00],
                     [-7.68632750e+00, -1.97322004e+00],
                     [ 2.59634667e+00, -7.54179212e+00],
                     [-2.55045500e+00,  7.89712104e+00],
                     [ 7.72129000e+00, -3.70042352e+00],
                     [-7.68632750e+00,  3.82615102e+00],
                     [-1.71500000e-03,  9.66675728e+00],
                     [ 1.02892262e+01,  2.78808610e-01],
                     [ 7.72129000e+00, -6.85089519e+00],
                     [-7.68632750e+00,  6.93243167e+00],
                     [ 5.22815667e+00, -8.96704624e+00],
                     [ 1.02892262e+01, -2.08251082e+00],
                     [-5.11839125e+00,  9.42078352e+00],
                     [ 1.02892262e+01,  3.26085565e+00],
                     [ 5.22815667e+00,  9.99467319e+00],
                     [ 1.02892262e+01, -4.95485261e+00],
                     [-2.55045500e+00,  1.12532961e+01],
                     [ 7.72129000e+00,  8.64451929e+00],
                     [ 2.59634667e+00,  1.14199273e+01],
                     [ 1.02892262e+01,  7.25385594e+00],
                     [-1.71500000e-03,  1.28269051e+01],
                     [ 7.72129000e+00, -1.03172001e+01],
                     [ 1.28571625e+01, -1.24485388e+00],
                     [ 1.28571625e+01,  1.66236756e+00],
                     [ 1.02892262e+01, -8.13978336e+00],
                     [ 1.28571625e+01, -3.37035974e+00],
                     [-7.68632750e+00,  1.09444460e+01],
                     [-5.11839125e+00,  1.28517842e+01],
                     [ 1.28571625e+01,  5.86319259e+00],
                     [ 1.28571625e+01, -6.20928170e+00],
                     [-2.55045500e+00,  1.42071727e+01],
                     [ 1.02892262e+01, -1.17078635e+01],
                     [ 1.28571625e+01, -9.42867152e+00],
                     [-7.68632750e+00,  1.44502723e+01],
                     [-5.11839125e+00,  1.55978360e+01],
                     [ 1.28571625e+01, -1.30985269e+01],
                     [-7.68632750e+00,  1.69884994e+01]])


def generate_random_objects(ramin, ramax, decmin, decmax, rng, density=1000):
    area = (ramax - ramin) * np.degrees((np.sin(np.radians(decmax)) - np.sin(np.radians(decmin))))
    n_obj = int(area * density) # 1000 / sq. deg * area

    ra = rng.uniform(ramin, ramax, size=n_obj)
    dec = rng.uniform(decmin, decmax, size=n_obj)

    return ra, dec

def radec_to_xyz(alpha, delta):
    # Converts to xyz cartesian from polar coords assuming unit radius
    alpha_r, delta_r = np.radians(alpha), np.radians(delta)
    x = np.cos(delta_r) * np.cos(alpha_r)
    y = np.cos(delta_r) * np.sin(alpha_r)
    z = np.sin(delta_r)
    return np.vstack([x, y, z])

def rotate_sphere(alpha, delta, ra, dec):
    # These two vectors define the rotation
    from_point = radec_to_xyz(0, 0)[:, 0]
    to_point = radec_to_xyz(alpha, delta)[:, 0]

    # Will rotate in cartesian and then convert back later.
    points_cart = radec_to_xyz(ra, dec)
    theta = np.arccos(from_point.dot(to_point)) # Should be in radians

    # Cross product to define the axis to rotate about
    k = np.cross(from_point, to_point)
    k /= np.linalg.norm(k)

    # Rodrigues' rotation formula
    K = np.array([[0, -k[2], k[1]], [0, 0, k[0]], [0, 0, 0]])
    K -= K.T
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    rotated_cart = R @ points_cart

    # Convert the cartesina back to the ra/dec
    dec_out = np.arcsin(rotated_cart[2, :])
    ra_out = np.arctan2(rotated_cart[1, :], rotated_cart[0, :])

    return np.degrees(ra_out), np.degrees(dec_out)

def fix_wraparound(ra):
    ra_out = np.copy(ra)
    ra_out[ra < 0] = ra[ra < 0] + 360
    return ra_out

def rotate_tiling(tiles_tbl, pass_num=1):
    n_desi = len(rots)
    if pass_num <= n_desi:
        rot = rots[pass_num - 1]
    else:
        rot = new_rots[(pass_num % n_desi) - 1]

    ra, dec = tiles_tbl["RA"], tiles_tbl["DEC"]

    if pass_num == 1:
        ra_new, dec_new = ra, dec
    else:
        ra_new, dec_new = rotate_sphere(rot[0], rot[1], ra, dec)
        ra_new = fix_wraparound(ra_new)

    #Generating a new tile table.
    footprint = Table(data={"PROGRAM": ["DARK"] * len(tiles_tbl),
                            "OBSCONDITIONS": [obsconditions.mask("DARK")] * len(tiles_tbl),
                            "PASS": [pass_num] * len(tiles_tbl)})
    footprint["IN_DESI"] = np.ones(len(tiles_tbl), dtype=bool)
    footprint["RA"] = ra_new
    footprint["DEC"] = dec_new
    # Including the pass number in the tile id ensures they're all unique.
    # There's about 4100 tiles in the base tiling of the sky, so we need
    # the pass number to exist in a high enough value that there's no
    # conflicts between pass numbers.
    footprint["TILEID"] = np.arange(len(tiles_tbl)) + pass_num * 10000
    return footprint
