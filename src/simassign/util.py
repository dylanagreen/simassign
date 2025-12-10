# stdlib imports
from pathlib import Path

# DESI imports
from desimodel.focalplane import get_tile_radius_deg
from desimodel.footprint import tiles2pix
from desitarget.targetmask import  obsconditions

# Non-DESI outside imports
from astropy.table import Table, unique
import healpy as hp
from matplotlib.patches import Path as mpPath
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
def generate_target_files(targs, tiles, out_dir, night=1, verbose=False, trunc=True):
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

    night : int
        The integer corresponding to the night of observation, used for generating
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
    save_loc = Path(out_dir) / f"night-{night}"

    # Make the director if it doesn't exist (very likely)
    save_loc.mkdir(parents=True, exist_ok=True)

    targ_files = []
    tile_files = []
    ntargs_on_tile = []
    for tile in tiles:
        tileid = tile["TILEID"]
        if trunc:
            tile_targs = targets_in_tile(targs, (tile["RA"], tile["DEC"]))
        else:
            tile_targs = targs

        target_filename = save_loc / f"targets-{tileid}.fits"
        if verbose: print(f"Writing {len(tile_targs)} to {target_filename}")
        tile_targs.write(target_filename, overwrite=True)
        targ_files.append(str(target_filename))

        tile_filename = save_loc / f"tile-{tileid}.fits"
        if verbose: print(f"Writing to {tile_filename}")
        tile_tbl = Table(tile)
        tile_tbl.write(tile_filename, overwrite=True)
        tile_files.append(str(tile_filename))

        ntargs_on_tile.append(len(tile_targs))

    return targ_files, tile_files, ntargs_on_tile

# TODO merge this with targ done, there is a lot of code duplication.
def get_nobs_arr(mtl, global_targs=None, global_timestamps=None):
    """
    Given an MTL generate an array of the number of targets with $m <= N$ observations
    after $n$ MTL updates, up to the total number $N$ MTL updates. Updates may
    correspond with the end of a pass or an observing night, depending on strategy
    used.

    Parameters
    ----------
    mtl : :class:`~numpy.array` or :class:`~astropy.table.Table`
        A numpy rec array or astropy Table representing the MTL. It is
        necessary to have the columns TIMESTAMP, TARGETID and NUMOBS.

    global_timestamps : :class:`~numpy.array`
        An array of global timestamps to use for generating the number of observations.
        I.e. return the number of observations at each timestamp in global_timestamps
        rather than at each timestamp in the input mtl. Optional, defaults to None,
        which uses the timestamps in the input mtl.

    Returns
    -------
    :class:`~numpy.array`
        Array storing the number of targets with the number of observations
        given by the 1st axis, at the update number given by the position
        in the 0th axis. For example, the position obs_arr[6, 3] indicates
        how many targets have *exactly* 3 exposures after 6 MTL updates.

    :class:`~numpy.array`
        Array storing the number of targets with *at least* the number of
        observations given by the 1st axis. For example, the position
        at_least_arr[6, 3] indicates
        how many targets have 3 *or more* exposures after 6 MTL updates.
    """
    timestamps = np.array(mtl["TIMESTAMP"], dtype=str)

    if global_timestamps is not None:
        unique_timestamps = np.unique(global_timestamps)
    else:
        unique_timestamps = np.unique(timestamps)

    if global_targs is not None:
        targs = global_targs
    else:
        good_targ = mtl["DESI_TARGET"] < 2**10 # Some other targets slip through sometimes...

        # if not split_subtype:
        targs = np.unique(mtl["DESI_TARGET"][~is_std & good_targ])

    # Timestamps correspond with when the MTL was created/updated
    # So we can loop over the timestamps to get information from each
    # fiberassign run.
    is_std = mtl["TARGET_STATE"] == "CALIB"

    ntargs = len(targs)
    nobs = len(unique_timestamps)
    nobs_arr = np.zeros((ntargs, nobs, nobs))

    ts_in_this_mtl = np.isin(unique_timestamps, np.unique(timestamps))
    these_timestamps = unique_timestamps[ts_in_this_mtl]
    for i, time in enumerate(these_timestamps):
        ts_idx = np.where(unique_timestamps == time)[0][0]
        keep_rows = timestamps <= time
        # print(sum(keep_rows))

        trunc_mtl = deduplicate_mtl(mtl[keep_rows & (~is_std)])

        for j, t in enumerate(targs):
            # if not split_subtype:
            this_targ = trunc_mtl["DESI_TARGET"] == t
            c = np.bincount(trunc_mtl["NUMOBS"][this_targ], minlength=nobs)
            nobs_arr[j, ts_idx:, :] = c
    # Reverse to go max down to zero, then sum to get how many have at least that number exposures
    # i.e. at least 3 exposures should be the sum of n_3 and n_4. Since it's reversed this is true
    # since 4 will be the first element (not summed), the second is the sum of the first two (3 and 4)
    at_least_n = np.cumsum(nobs_arr[:, :, ::-1], axis=2)[:, :, ::-1]

    return nobs_arr, at_least_n

def get_targ_done_arr(mtl, split_subtype=False, global_targs=None, global_timestamps=None, delta_stats=False,
                      tid_counts=None, full_nobs=False):
    """
    Given an MTL generate an array of the number of targets with $m <= N$ observations
    after $n$ MTL updates, up to the total number $N$ MTL updates. Updates may
    correspond with the end of a pass or an observing night, depending on strategy
    used.

    Parameters
    ----------
    mtl : :class:`~numpy.array` or :class:`~astropy.table.Table`
        A numpy rec array or astropy Table representing the MTL. It is
        necessary to have the columns TIMESTAMP, TARGETID and NUMOBS.

    split_subtype : bool
        Whether or not to split targets by their subtype. Defaults to False.

    global_targs : :class:`~numpy.array`
        An array of global targets to use for generating the number of done
        targets. I.e. return the number of done targets at each timestamp for the
        targets in global_targs rather than the local targets in the input mtl.
        Optional, defaults to None, which uses only the targets in the input mtl.
        If passed and split_subtype=True, global targs must be an array of strings
        of form "{target_bit}|{subtarget_bit}".

    global_timestamps : :class:`~numpy.array`
        An array of global timestamps to use for generating the number of observations.
        I.e. return the number of observations at each timestamp in global_timestamps
        rather than at each timestamp in the input mtl. Optional, defaults to None,
        which uses the timestamps in the input mtl.

    delta_stats : bool
        Whether we should collate statistics on what changed in each MTL update.
        There is a performance consequence to this. Defaults to False.

    tid_counts : :class:`~numpy.array` or :class:`~astropy.table.Table`
        A numpy rec array or astropy Table with columns "TARGETID" and "POSSIBLE".
        If provided, do not process any TARGETIDs whose POSSIBLE assignments
        are less than NUMOBS_INIT (the goal number of observations). Defaults to None,
        which means process all targets.

    full_nobs : bool
        If True, return the full array of number of observations per target
        per MTL update time. If False, return only the arrays per target of
        the number of targets that reached their goal number of exposures.
        The latter is significanlty faster, and adequate for most analyses.
        Defaults to False.

    Returns
    -------
    TODO write the return types.
    """
    timestamps = np.array(mtl["TIMESTAMP"], dtype=str)

    if global_timestamps is not None:
        unique_timestamps = np.unique(global_timestamps)
    else:
        unique_timestamps = np.unique(timestamps)

    if tid_counts is not None:
        tids_goals = mtl["TARGETID", "NUMOBS_INIT"]
        tids_goals = unique(tids_goals, keys="TARGETID") # Don't care to deduplicate, NUMOBS_INIT should never change
        # Some objects are never even able to be observed and aren't in tid_counts at all
        # This ensures that we only match on tids that are actually observable *at all*
        tids_goals = tids_goals[np.isin(tids_goals["TARGETID"], tid_counts["TARGETID"])]
        tids_in_mtl = np.isin(tid_counts["TARGETID"], tids_goals["TARGETID"])
        avail = tid_counts["POSSIBLE"][tids_in_mtl]

        # Since both tids_avail and tids_goals are sorted these are one to one
        can_get = avail >= tids_goals["NUMOBS_INIT"]
        keep_tids = tids_goals["TARGETID"][can_get]
        do_process = np.isin(mtl["TARGETID"], keep_tids)
    else:
        do_process = np.ones(len(mtl), dtype=bool)


    # This handles determining the unique targets in the file, especually
    # if we want to split them by subtype or not. It also excludes things
    # like standards.
    is_std = mtl["TARGET_STATE"] == "CALIB"
    if global_targs is not None:
        targs = global_targs
    else:
        good_targ = mtl["DESI_TARGET"] < 2**10 # Some other targets slip through sometimes...

        if not split_subtype:
            targs = np.unique(mtl["DESI_TARGET"][~is_std & good_targ])
        else:
            targ_bits = np.unique(mtl["DESI_TARGET"][~is_std & good_targ])
            # Get everything that is this target, take the first one, split on the pipe, first half is the name
            targ_names = [(mtl["TARGET_STATE"] == t)[0].split("|")[0] for t in targ_bits]
            targs = []
            for i, t in enumerate(targ_bits):
                if f"{targ_names[i]}_TARGET" in mtl.colnames:
                    subtargs = np.unique(mtl[f"{targ_names[i]}_TARGET"])
                    for s in subtargs:
                        targs.append(f"{t}|{s}")
                else:
                    targs.append(str(t))

    # Timestamps correspond with when the MTL was created/updated
    # So we can loop over the timestamps to get information from each
    # fiberassign run.
    nobs = len(unique_timestamps)
    ntargs = len(targs)
    nobs_arr = np.zeros((ntargs, nobs))
    num_each_targ = np.zeros(ntargs)

    # For speed we can iterate only over timestamps in this file, and
    # flood fill the value up to the next unique timestamp that is in this file
    # since the results won't change between those timestamps. I.e. if the
    # global timestamps are A, B, C, and only A and C are in this file,
    # then there is no MLT update at timestamp B. So uniquifying on timestamp
    # B will provide the same result as uniquifying on timestamp A.
    ts_in_this_mtl = np.isin(unique_timestamps, np.unique(timestamps))
    these_timestamps = unique_timestamps[ts_in_this_mtl]
    # print(f"{len(these_timestamps)} timestamps out of {len(unique_timestamps)} in this MTL.")
    if delta_stats:
        targets_obs = np.zeros(nobs)
        targets_obs_but_done = np.zeros(nobs)
    for i, time in enumerate(these_timestamps):
        ts_idx = np.where(unique_timestamps == time)[0][0]
        keep_rows = timestamps <= time

        trunc_mtl = deduplicate_mtl(mtl[keep_rows & (~is_std) & do_process])
        is_done = trunc_mtl["NUMOBS_MORE"] == 0
        for j, t in enumerate(targs):
            if not split_subtype:
                this_targ = trunc_mtl["DESI_TARGET"] == t
            else:
                targ_bit = int(t.split("|")[0])
                this_targ = trunc_mtl["DESI_TARGET"] == targ_bit

                # Don't even bother checking for the subbit because we
                # have nothing of this parent target if the sum is zero.
                # TODO throw a nice error if the {targ}_TARGET column doesn't exist
                # for that target type.
                if (np.sum(this_targ) > 0) and (len(t.split("|")) > 1):
                    sub_bit = int(t.split("|")[1])
                    name = trunc_mtl["TARGET_STATE"][this_targ][0]
                    name = name.split("|")[0]
                    this_targ = this_targ & (trunc_mtl[f"{name}_TARGET"] == sub_bit)

            # Should broadcast correctly. Flood fill all above this time to
            # the results for this timestamp. At next higher timestamp
            # everything above will be overwritten.
            nobs_arr[j, ts_idx:] = np.sum(is_done[this_targ])

            # This mostly used to determine fractional completeness.
            if i == 0:
                num_each_targ[j] = np.sum(this_targ)

            # At the zeroth timestamp everything was added so it'll
            # all be considered "observed" and we don't want that.
            if delta_stats and ts_idx != 0:
                updated_this_ts = trunc_mtl["TIMESTAMP"] == time
                targ_overobserved = trunc_mtl["NUMOBS"] > trunc_mtl["NUMOBS_INIT"]
                targets_obs[ts_idx] += np.sum(updated_this_ts & this_targ)
                targets_obs_but_done[ts_idx] += np.sum(updated_this_ts & this_targ & targ_overobserved)

    if delta_stats:
        return nobs_arr, num_each_targ, targets_obs, targets_obs_but_done
    return nobs_arr, num_each_targ

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

def check_in_survey_area(tbl, survey=None, trim_rad=0):
    # TODO docstring
    # Extracting the points as a 2d array
    data_ra = np.array(tbl["RA"], dtype=float)
    data_dec = np.array(tbl["DEC"], dtype=float)
    data_points = np.vstack([data_ra, data_dec]).T

    if survey is not None:
        if type(survey) == list: # Survey is split over multiple polygons, first axis indexes polygons.
            in_survey = np.zeros(len(tbl), dtype=bool)
            for i in range(len(survey)):
                survey_path = mpPath(survey[i])

                # Some polygon wrapped around so we need to wwrap some of the targets
                if np.any(survey[i][:, 0] > 360):
                    to_rotate = data_ra < np.max(survey[i][:, 0] - 360)
                    data_points_rotate = np.array(data_points, copy=True)
                    data_points_rotate[to_rotate] += np.array([360, 0]) # Just add 360 to the lower points
                    in_this = survey_path.contains_points(data_points_rotate, radius=trim_rad)

                else:
                    in_this = survey_path.contains_points(data_points, radius=trim_rad)

                print(np.sum(in_this), f"in survey {i}")
                in_survey = in_survey | in_this
        else:
            survey_path = mpPath(survey)
            in_survey = survey_path.contains_points(data_points, radius=trim_rad)
    else:
        # Radius has to be negative here due to the clockwise/counterclockwise
        # Directionality.
        print("Checking NGC...")
        p_ngc = mpPath(ngc_points)
        in_ngc = p_ngc.contains_points(data_points, radius=-trim_rad)

        print("Checking SGC...")
        # Need to handle the rotation of the sgc, since it's disjoint
        # when constraining angles to 0-360
        to_rotate = data_ra < 90
        data_points_rotate = np.array(data_points, copy=True)
        data_points_rotate[to_rotate] += np.array([360, 0]) # Just add 360 to the lower points
        p_sgc = mpPath(sgc_points)
        in_sgc = p_sgc.contains_points(data_points_rotate, radius=-trim_rad)

        in_survey = in_ngc | in_sgc

    return in_survey

def check_in_tile_area(tbl, tiles, nside=256):
    # TODO docstring
    hpx_tiles = tiles2pix(nside, tiles["TILEID", "RA", "DEC"], fact=2**9)

    ra = tbl["RA"]
    dec = tbl["DEC"]

    theta, phi = np.radians(90 - dec), np.radians(ra)
    hpx_targs = hp.ang2pix(nside, theta, phi, nest=True)

    return np.isin(hpx_targs, hpx_tiles)


def generate_stripe_tiles( delta=get_tile_radius_deg() * np.sqrt(2)):
    # TODO docstring

    # Generate the right ascension centers along the equator
    ra_centers_dec0 = np.arange(0, 360, delta)
    centers_radec = list(zip(ra_centers_dec0, np.zeros_like(ra_centers_dec0)))

    # Will use a numpy rec array to store this, so we can denote column
    # and row numbers. We will use those to offset tiles from each other slightly
    # sot that the tile gaps don't line up in stripes.
    centers = np.array(centers_radec, dtype=[("RA", "<f8"), ("DEC", "<f8")])
    centers = centers.view(np.recarray)
    centers = np.lib.recfunctions.append_fields(centers, "ROW", np.zeros_like(ra_centers_dec0))
    centers = np.lib.recfunctions.append_fields(centers, "COL", np.arange(len(ra_centers_dec0)))

    dec_deltas = np.arange(0, 90, delta)
    dec_deltas = dec_deltas[1:]

    # Generate all of the individual rows in the stripe tiling, iterating
    # by how much the declination changes at each row.
    rows = [centers]
    for i, d in enumerate(dec_deltas):
        new_centers = np.array(centers, copy=True)
        new_centers["DEC"] += d
        new_centers["ROW"] = i + 1
        rows.append(new_centers)

        new_centers = np.array(centers, copy=True)
        new_centers["DEC"] -= d
        new_centers["ROW"] = -(i + 1)
        rows.append(new_centers)

    final_centers = np.hstack(rows)

    # Dithering rows and columns relative to each other to avoid lining up
    # the tileg aps.
    dither_rows = (final_centers["ROW"] % 2) == 0
    final_centers["RA"][dither_rows] += 0.05

    dither_cols = (final_centers["COL"] % 2) == 0
    final_centers["DEC"][dither_cols] += 0.05

    return final_centers

def shift_stripes(num_shifts, tile_centers):
    # TODO docstring
    max_shift = np.sqrt(2)
    # Start and end are the max shift, which are the same base tiling, due to
    # symmetry. We will discard the end, but keep the beginning so that this
    # function returns all shifts from 0 up to num_shifts
    shifts = get_tile_radius_deg() * np.linspace(0, max_shift, num_shifts + 2)
    shifts = shifts[:-1]

    out_centers = []
    for s in shifts:
        cur_centers = np.array(tile_centers, copy=True)
        cur_centers["RA"] += s
        cur_centers["DEC"] += s
        out_centers.append(cur_centers)

    return out_centers

def target_mask_to_int(targetmask):
    # TODO docstring
    science_mask = 0
    for row in targetmask["desi_mask"]:
        science_mask += 2 ** (row[1])

    return science_mask