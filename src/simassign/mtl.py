# Code to handle the MTL, e.g. updating etc.
# TODO turn this into a formal docstring.

# Non-DESI outside imports
import numpy as np
from astropy.table import Table, join, vstack

# DESI imports
from desitarget.mtl import make_mtl


def update_mtl(mtl, tids_to_update, rng=np.random.default_rng(91701)):
    # By default we will use a dummy zcat for now to update.
    # TODO write custom update that doesn't use zcats or desitarget so we
    # can accurately handle a new target type.
    unique_mtl = mtl[::-1]
    _, ii = np.unique(unique_mtl["TARGETID"], return_index=True)
    unique_mtl = unique_mtl[ii]

    # Make a dummy zcat to use for updating, which will be done using the original make_mtl function.
    # dummy z_cat for update needs to have following cols:
    # Z, ZWARN, ZTILEID, TARGETID and NUMOBS
    # Z_QN, IS_QSO_QN, DELTACHI2
    zcat = Table()
    zcat["TARGETID"] = tids_to_update
    zcat["Z"] = rng.uniform(2.2, 3.2, len(tids_to_update)) # they're all lya qsos, to ensure reobservation
    zcat["ZWARN"] = np.zeros(len(tids_to_update), dtype=int) # They're all good redshifts
    zcat["ZTILEID"] = np.ones(len(tids_to_update), dtype=int) # Arbitrary tile number
    zcat["DELTACHI2"] = 50 * np.ones(len(tids_to_update), dtype=int)
    zcat["IS_QSO_QN"] = np.ones(len(tids_to_update), dtype=bool)
    zcat["Z_QN"] = zcat["Z"] + rng.uniform(-0.1, 0.1, len(tids_to_update))

    # Increment the observations by 1, match the previous MTL to determine the current
    # number of observations to get incremented.
    zcat = join(unique_mtl["TARGETID", "NUMOBS"], zcat)
    zcat["NUMOBS"] += 1

    # To update we need to compare against the deduplicated MTL to ensure we use the latest
    # details for each targetid
    mtl_updates = make_mtl(unique_mtl, "DARK", zcat=zcat, trimtozcat=True)

    # Append the updates to the full MTL, then sort it by TARGETID and TIMESTAMP
    # to put the updates in the right place.
    mtl = vstack([mtl, mtl_updates])
    mtl.sort(["TARGETID", "TIMESTAMP"])

    return mtl

def deduplicate_mtl(mtl):
    """
    Remove duplicates entries per TARGETID from the given MTL.

    Parameters
    ----------
    mtl : :class:`~numpy.array` or `~astropy.table.Table`
        A numpy rec array or astropy Table storing the MTL. The datamodel is
        largely agnostic, but should include at least the columns TARGETID and
        TIMESTAMP. This function assumes that the MTL is correctly sorted by
        TARGETID, and then TIMESTAMP, such that subsequent entries for the same
        TARGETID are later in time.

    Returns
    -------
    :class:`~numpy.array` or `~astropy.table.Table`
        MTL table with keeping only the most recent entry per TARGETID. Return
        type will match input MTL type.
    """
    # Flip to keep most recent element instead of first.
    trunc_mtl = mtl[::-1]
    _, ii = np.unique(trunc_mtl["TARGETID"], return_index=True)
    trunc_mtl = trunc_mtl[ii]
    return trunc_mtl
