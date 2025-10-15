# stdlib imports
from datetime import datetime
from pathlib import Path

# DESI imports
from desimodel.focalplane import get_tile_radius_deg
from desitarget.targetmask import desi_mask, obsconditions

# Non-DESI outside imports
from astropy.table import Table
import numpy as np

from simassign.mtl import deduplicate_mtl

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
    """
    Generate uniformly distributed, randomly positioned targets across a
    coordinate box.

    Parameters
    ----------
    ramin : float
        Minimum RA of the box to generate over.

    ramax : float
        Maximum RA of the box to generate over.

    decmin : float
        Minimum DEC of the box to generate over.

    decmax : float
        Maximum DEC of the box to generate over.

    rng : :class:`numpy.random.Generator`
        A random number generator to use for generating the random objects.

    density : int or float
        Uniform densty of targets per square degree to generate. Defaults to
        1000.

    Returns
    -------
    :class:`~numpy.array`
        Right ascension coordinates of the generated points.

    :class:`~numpy.array`
        Declination coordinates of the generated points.
    """
    area = (ramax - ramin) * np.degrees((np.sin(np.radians(decmax)) - np.sin(np.radians(decmin))))
    n_obj = int(area * density) # density * area

    ra = rng.uniform(ramin, ramax, size=n_obj)
    dec = rng.uniform(decmin, decmax, size=n_obj)

    return ra, dec

def radec_to_xyz(alpha, delta):
    """
    Converts a set of polar coordinate points assuming unit radius to 3D xyz
    cartesian coordinates.

    Parameters
    ----------
    alpha : :class:`~numpy.array`
        Angle around the sphere from the coordinate 0,0. In celestial
        coordinates, also known as right ascension.

    delta : :class:`~numpy.array`
        Angle above the horizon, in celestial
        coordinates, also known as declination.

    Returns
    -------
    :class:`~numpy.array`
        Two dimensional array containing the x, y, and z coordinates of the given
        points in the first, second and third rows respectively.
    """

    alpha_r, delta_r = np.radians(alpha), np.radians(delta)
    x = np.cos(delta_r) * np.cos(alpha_r)
    y = np.cos(delta_r) * np.sin(alpha_r)
    z = np.sin(delta_r)
    return np.vstack([x, y, z])

def rotate_sphere(alpha, delta, ra, dec):
    """
    Spherically rotate a set of RA and DEC points through the great circle
    defined by the offset angles alpha and delta.

    The angles alpha and delta define a point in spherical space. The
    points defined by RA and DEC are rotated by the rotation defined such
    that the point (0,0) is rotated into (alpha, delta)

    Parameters
    ----------
    alpha : :class:`~numpy.array`
        Right ascension coordinate defining the rotation vector, in degrees.

    delta : :class:`~numpy.array`
        Declination coordinate defining the rotation vector, in degrees.

    ra : :class:`~numpy.array`
        Right ascension coordinate of the points to be rotated, in degrees.

    dec : :class:`~numpy.array`
        Declination coordinate of the points to be rotated, in degrees.

    Returns
    -------
    :class:`~numpy.array`
        The new, rotated, right ascension points.

    :class:`~numpy.array`
        The new, rotated, declination points.
    """
    # These two vectors define the rotation
    from_point = radec_to_xyz(0, 0)[:, 0]
    to_point = radec_to_xyz(alpha, delta)[:, 0]

    # Will rotate in cartesian and then convert back later.
    points_cart = radec_to_xyz(ra, dec)
    theta = np.arccos(from_point.dot(to_point)) # Should be in radians

    # Cross product to define the axis to rotate about
    # (i.e. the cross product finds a vector normal to the plane defined
    # by the two vectors, and this is the axis of rotation such that rotations
    # occur in the defined plane)
    k = np.cross(from_point, to_point)
    k /= np.linalg.norm(k)

    # Rodrigues' rotation formula
    K = np.array([[0, -k[2], k[1]], [0, 0, k[0]], [0, 0, 0]])
    K -= K.T
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    rotated_cart = R @ points_cart

    # Convert the cartesian back to the ra/dec
    dec_out = np.arcsin(rotated_cart[2, :])
    ra_out = np.arctan2(rotated_cart[1, :], rotated_cart[0, :])

    return np.degrees(ra_out), np.degrees(dec_out)

def fix_wraparound(ra):
    """
    Fix right ascension (RA) values to correclty lie within 0 to 360 degrees.

    The output of :func:`rotate_tiling` may occasionally rotate tilings into negative RA,
    this helper function simply corrects those values.

    Parameters
    ----------
    ra : :class:`~numpy.array`
        Right ascension values which may or may not be negative.

    Returns
    -------
    :class:`~numpy.array`
        Corrected right ascension values such that all values lie between 0
        and 360.
    """
    ra_out = np.copy(ra) # Copy to avoid mutating inputs
    ra_out[ra < 0] = ra[ra < 0] + 360
    return ra_out

def rotate_tiling(tiles_tbl, pass_num=1):
    """
    Rotate the tiling defined by tiles_tbl by the rotation defined by pass_num.

    Parameters
    ----------
    tiles_tbl : :class:`~numpy.array` or :class:`~astropy.table.Table`
        A numpy rec array or astropy Table storing the tile definition.
        The datamodel is largely agnostic, but should include at minimum the
        tile centers as columns "RA" and "DEC".

    pass_num : int
        Which pass number will be defined by the rotated tiling. The pass number
        is used to determine which rotation point to use, for pass_num <=15
        the rotations are defined in DESI-0717, and define the original
        DESI main survey passes. For 15 < pass_num <= 63 the rotation is
        defined as one of the 48 new rotations determined by generate_new_rots.py.
        pass_num = 1 does no rotation, and is the default.

    Returns
    -------
    :class:`~astropy.table.Table`
        Tile table defining the new rotated tiling. The output tile table
        includes all columns necessary to run fiberassign. TILEIDs are determined
        as the tile's index in the array plus the pass_num * 10000.
    """
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
    # TODO copy obsconditions from input table.
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

def targets_in_tile(targs, tile_center):
    """
    Trim the targets table to only those targets that could potentially be observed
    by the tile centered at `tile_center`.

    Targets are trimmed using the DESI tile radius plus an extra 0.2 degree
    buffer to ensure no targets at the edge of the tile are trimmed. For
    speed targets are trimmed to the box that circumscribes the tile with this
    radius, rather than the exact circular tile geometry.

    Parameters
    ----------
    targs : :class:`~numpy.array` or :class:`~astropy.table.Table`
        A numpy rec array or astropy Table storing the target definition.
        The datamodel is largely agnostic, but should include at minimum the
        columns "RA" and "DEC" defining each target position. Any other
        columns are ignored.

    tile_center : (float, float)
        Tuple of RA and DEC defining a tile center.

    Returns
    -------
    :class:`~numpy.array` or :class:`~astropy.table.Table`
        Table of targets trimmed to those targets that lie within observing
        range of the tile. Return type will match that of the input type.
    """
    # Add a 0.2 buffer just in to avoid trimming targets at the edge of the focal plane.
    tile_rad = get_tile_radius_deg() + 0.2
    tile_ra, tile_dec = tile_center

    # Yes, this is a square and not a circle. But it trims the targets
    # enough, and it's going to be significantly faster.
    good_dec = np.abs(targs["DEC"] - tile_dec) <= tile_rad
    good_ra = np.abs(targs["RA"] - tile_ra) <= tile_rad

    return targs[good_ra & good_dec]

# Generate target files for each of the tiles and save them to
# outdir / pass_num. Also generate associated tile file.
def generate_target_files(targs, tiles, out_dir, pass_num=1, verbose=False, trunc=True):
    """
    Given a set of targets and a set of tiles, generate the necessary
    files on disk that encode which targets are accessible by each tile.

    Parameters
    ----------
    targs : :class:`~numpy.array` or :class:`~astropy.table.Table`
        A numpy rec array or astropy Table storing the target definition.
        The datamodel is largely agnostic, but should include at minimum the
        columns "RA" and "DEC" defining each target position.

    tiles : :class:`~numpy.array` or :class:`~astropy.table.Table`
        A numpy rec array or astropy Table storing the tile definition.
        The datamodel is largely agnostic, but should include at minimum the
        columns "RA" and "DEC" defining each tile center.

    out_dir : str or :class:`~pathlib.Path`
        The directory to save the tile and target files to.

    pass_num : int
        The integer corresponding to this pass number, used for generating
        a subdirectory in `out_dir` to group all associated files. Defaults to 1.

    verbose : bool
        Print verbosely when saving each file, defaults to False.

    trunc : bool
        If true, truncate each saved targets file to only be the targets
        corresponding to the given tile when saving that tile/target file
        combination. Otherwise save all targets to all tile files. Defaults to True.


    Returns
    -------
    list(str)
        List of all target file names.

    list(str)
        List of all tile file names.
    """
    if verbose: print(f"Passed {len(tiles)} to generate target files")
    save_loc = Path(out_dir) / f"pass-{pass_num}"

    # Make the director if it doesn't exist (very likely)
    save_loc.mkdir(parents=True, exist_ok=True)

    targ_files = []
    tile_files = []
    for tile in tiles:
        tileid = tile["TILEID"]
        if trunc:
            tile_targs = targets_in_tile(targs, (tile["RA"], tile["DEC"]))
        else:
            tile_targs = targs

        target_filename = save_loc / f"targets-{tileid}.fits"
        if verbose: print(f"Writing to {target_filename}")
        tile_targs.write(target_filename, overwrite=True)
        targ_files.append(str(target_filename))

        tile_filename = save_loc / f"tile-{tileid}.fits"
        if verbose: print(f"Writing to {tile_filename}")
        tile_tbl = Table(tile)
        tile_tbl.write(tile_filename, overwrite=True)
        tile_files.append(str(tile_filename))

    return targ_files, tile_files

def get_nobs_arr(mtl):
    """
    Given an MTL generate an array of the number of targets with $m <= N$ observations
    after $n$ passes, up to the total number $N$ passes.

    Parameters
    ----------
    mtl : :class:`~numpy.array` or :class:`~astropy.table.Table`
        A numpy rec array or astropy Table representing the MTL. It is
        necessary to have the columns TIMESTAMP, TARGETID and NUMOBS.

    Returns
    -------
    :class:`~numpy.array`
        Array storing the number of targets with the number of observations
        given by the 1st axis, at the pass number given by the position
        in the 0th axis. For example, the position obs_arr[6, 3] indicates
        how many targets have *exactly* 3 exposures after 6 passes.

    :class:`~numpy.array`
        Array storing the number of targets with *at least* the number of
        observations given by the 1st axis. For example, the position
        at_least_arr[6, 3] indicates
        how many targets have 3 *or more* exposures after 6 passes.
    """
    timestamps = np.array(mtl["TIMESTAMP"])
    ts = np.array([datetime.fromisoformat(x.decode()) for x in timestamps])
    unique_timestamps = np.unique(ts)

    # Timestamps correspond with when the MTL was created/updated
    # So we can loop over the timestamps to get information from each
    # fiberassign run.
    bins = np.arange(-0.5, len(unique_timestamps) - 0.4, 1) # "binning" for counting numbers of observations of targets.
    nobs = []
    for time in unique_timestamps:
        keep_rows = ts <= time
        # print(sum(keep_rows))

        trunc_mtl = deduplicate_mtl(mtl[keep_rows])
        h, _ = np.histogram(trunc_mtl["NUMOBS"], bins=bins)
        nobs.append(h)

    obs_arr = np.asarray(nobs)
    # Reverse to go max down to zero, then sum to get how many have at least that number exposures
    # i.e. at least 3 exposures should be the sum of n_3 and n_4. Since it's reversed this is true
    # since 4 will be the first element (not summed), the second is the sum of the first two (3 and 4)
    at_least_n = np.cumsum(obs_arr[:, ::-1], axis=1)[:, ::-1]

    return obs_arr, at_least_n
