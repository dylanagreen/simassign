# Code to handle the MTL, e.g. updating etc.
# TODO turn this into a formal docstring.

# Non-DESI outside imports
import numpy as np
from astropy.table import Table, join, vstack
import healpy as hp
import yaml

# DESI imports
from desitarget.mtl import make_mtl, get_utc_date
from desitarget.targets import encode_targetid

# stdlib imports
from importlib import resources
from multiprocessing import Pool
from pathlib import Path
import shutil

from simassign.logging import get_log
log = get_log()
mtl_cols = ['RA',
            'DEC',
            'PMRA',
            'PMDEC',
            'REF_EPOCH',
            'DESI_TARGET',
            'TARGETID',
            'BGS_TARGET',
            'MWS_TARGET',
            'NUMOBS_INIT',
            'PRIORITY_INIT',
            'PRIORITY',
            'SUBPRIORITY',
            'SCND_TARGET',
            'NUMOBS_MORE',
            'NUMOBS',
            'Z',
            'ZWARN',
            'ZTILEID',
            'Z_QN',
            'IS_QSO_QN',
            'DELTACHI2',
            'TARGET_STATE',
            'TIMESTAMP',
            'VERSION',
            'OBSCONDITIONS',
            'HEALPIX']


def update_mtl(mtl, tids_to_update, targetmask=None, timestamp=None, use_desitarget=False, verbose=False):
    """
    Update an MLT after taking an observation of `tids_to_update`.

    Parameters
    ----------
    mtl : :class:`~numpy.array` or :class:`~astropy.table.Table`
        A numpy rec array or astropy Table storing the MTL. This should have
        the full MTL data model.

    tids_to_update : :class:`~numpy.array`
        List of targetids to update in the MTL.

    timestamp : str
        Timestamp to associate with this MTL update, used for suvery simulation
        purposes. Defaults to None, which sets the timestamp to the time when
        this update code is actually run.

    use_desitarget : bool
        Whether or not to use `desitarget.make_mtl` to update MTL or to
        use the internal logic defined in the simassign targetmask.yaml
        file. Using `desitarget.make_mtl` restricts the update loop to use
        specifically the QSO target class of desitarget. Defaults to False.

   verbose : bool
        Verbosely log.details information on each update. Defaults to False.

    Returns
    -------
    :class:`~numpy.array` or :class:`~astropy.table.Table`
       Update MTL that contains all the original rows of the input MTL plus
       the additional rows defining the updated state of the targets defined
       in tids_to_update.
    """
    # Use desitarget = use a dummy zcat and use make_mtl to update the MTL
    if use_desitarget:
        # If we use desitarget we will use a dummy zcat for now to update.
        rng = np.random.default_rng() # Just used for dummy redshifts.

        # TODO change this to a dedpulicate_mtl call.
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
    else:
        # Make sure we only update on the latest version of the MTL.
        update_rows = np.isin(mtl["TARGETID"], tids_to_update)
        mtl_updates = deduplicate_mtl(mtl[update_rows])

        # Iterate the number of observations
        mtl_updates["NUMOBS"] += 1

        # Need to handle if we reobserved this extra times and don't want
        # num obs more to go negative
        nobs_more = mtl_updates["NUMOBS_MORE"]
        mtl_updates["NUMOBS_MORE"] = np.where(nobs_more > 0, nobs_more - 1, 0)

        if targetmask is None:
            targetmask = load_target_yaml("targetmask.yaml")

        was_unobs = mtl_updates["NUMOBS"] == 1
        is_complete = mtl_updates["NUMOBS_MORE"] == 0

        for target in targetmask["desi_mask"]:
            bit = 2**target[1]
            name = target[0]

            this_target = (mtl_updates["DESI_TARGET"] & bit) != 0

            if verbose:
                log.details(f"Update loop target {target} {np.any(this_target)} {np.any(was_unobs & this_target)}, {np.any(is_complete & this_target)}")

            # lazily assume that the target class is correct and we want more zgood.
            # Do this before is_complete so that is_complete overrides in the
            # case of 1 requested exposure.
            if "MORE_OBS1" in targetmask["priorities"]["desi_mask"][name].keys():
                pass
            else:
                mtl_updates["PRIORITY"][this_target & was_unobs] = targetmask["priorities"]["desi_mask"][name]["MORE_ZGOOD"]
                mtl_updates["TARGET_STATE"] = mtl_updates["TARGET_STATE"].astype("<U16") # So we don't truncate status.

                # TODO status from the target yaml instead of hard coded.
                mtl_updates["TARGET_STATE"][this_target & was_unobs] = f"{name}|MORE_ZGOOD"

            # Set priority for done targets.
            mtl_updates["PRIORITY"][this_target & is_complete] = targetmask["priorities"]["desi_mask"][name]["DONE"]
            mtl_updates["TARGET_STATE"][this_target & is_complete] = f"{name}|DONE"

        # This is important for storing the history of the MTL
        if timestamp is not None:
            mtl_updates["TIMESTAMP"] = timestamp
        else:
            mtl_updates["TIMESTAMP"] = get_utc_date("main")

        # We won't update the redshift or anything, it's not really relevant to
        # fiberassign anyway.

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
    mtl : :class:`~numpy.array` or :class:`~astropy.table.Table`
        A numpy rec array or astropy Table storing the MTL. The data model is
        largely agnostic, but should include at least the columns TARGETID and
        TIMESTAMP. This function assumes that the MTL is correctly sorted by
        TARGETID, and then TIMESTAMP, such that subsequent entries for the same
        TARGETID are later in time.

    Returns
    -------
    :class:`~numpy.array` or :class:`~astropy.table.Table`
        MTL table keeping only the most recent entry per TARGETID. Return
        type will match input MTL type.
    """
    # Flip to keep most recent element instead of first.
    trunc_mtl = mtl[::-1]
    _, ii = np.unique(trunc_mtl["TARGETID"], return_index=True)
    trunc_mtl = trunc_mtl[ii]
    return trunc_mtl

# TODO move this to io.
def load_target_yaml(fname):
    """
    Load the targeting yaml saved as `fname`.

    Parameters
    ----------
    fname : str
        Filename of the yaml file to load. File is searched for in the
        package data, so this should be a filename that exists in the
        simassign package.

    Returns
    -------
    Object
        Python object containing the data stored in `fname`.
    """
    floc = resources.files("simassign").joinpath(f"data/{fname}")
    with open(floc) as f:
        return yaml.safe_load(f)


def initialize_mtl(base_tbl, save_dir=None, stds_tbl=None, return_mtl_all=True,
                   as_dict=False, targetmask=None, nproc=1, start_id=0,
                   timestamp="2024-12-30T00:00:00+00:00"):
    """
    Initialize an MTL in a format readable by fiberassign contaning all
    the necessary columns for state tracking.

    This functoin generates an MTL from scratch to match the expected
    datamodel. The column values are updated as necessary to match
    the either the simassign targetmask.yaml or the input targetmask.

    Parameters
    ----------
    base_tbl : :class:`~numpy.array` or :class:`~astropy.table.Table`
        A numpy rec array or astropy Table storing the targets. At minimum
        this must include the RA and DEC coordinates of the targets. Other
        columns will be overwritten as necessary. If base_tbl includes
        a DESI_TARGET column, it will be used to set priorities and NUMOBS
        based on the targetmask.yaml. Otherwise if this column is not provided
        all targets will be assumed to be LAEs.

    save_dir : str or :class:`~pathlib.Path`, optional
        If given, where to save the produced MTL. MTLs are split by healpixel
        and saved in per healpixel sub-MTLs. Defaults to None, which does
        not save the MTL to disk.

    stds_catalog : str or :class:`~pathlib.Path`, optional
        If given, the a catalog that contains standard stars. Standard
        stars are appended to the input table before MTL generation. Only
        standard stars that lie in the healpix covering base_tbl are kept.
        Defaults to None.

    return_mtl_all : bool, optional
        Whether to return the initialized, concatenated mtl_all or not. In
        some cases when the MTL is large it is more efficient to only
        save them per healpix, and load them as necessary instead of storing
        the entire MTL in memory at once. Note: if save_dir is not provided
        and this is set to false, this function functionally does nothing
        except waste memory and time, since the MTL will not be saved
        nor returned. Defaults to True.

    as_dict : bool, optional
        Whether to return the mtl_all as a dict, with healpixel number as the key
        and the value as the individual MTL corresponding tot hat healpixel,
        or to stack the entire table into one. Defaults to False (stacking as a single table).

    targetmask : dict, optional
        Targetmask to use for target bits and target priorities where necessary. Defaults
        to None, which loads the default targetmask stored in the simassign pacakge.

    nproc : int, optional
        How many processes to use to initialize the mtl with the correct values.
        Currently does nothing, defaults to 1.

    start_id : int, optional
        Starting index of targets, for encoding unique targetids if we plan to
        join MTLs later in the survey. This ensures targets have unique targetis
        on join. Defaults to zero.

    timestamp : str, optional
        Timestamp to encode into this MTL, representing the starting date/time
        of the ledger. Defaults to "2024-12-30T00:00:00+00:00".

    Returns
    -------
    :class:`~numpy.array` or :class:`~astropy.table.Table` or dict
       MTL corresponding to the targets given in the `base_tbl`. Even when saving
       the MTL split across healpix (when `save_dir` is passed) this function
       still returns the full global MTL. Only returns when return_mtl_all is True.
    """
    # Minimum set of columns necessary for make_mtl:
    # TARGETID`, `DESI_TARGET`, `BGS_TARGET`, `MWS_TARGET`, `NUMOBS_INIT`, `PRIORITY_INIT`, `PRIORITY` (because we won't pass a zcat)
    # For targetids we will use the indices, they just need to be unique and non negative
    tbl = base_tbl.copy() # Don't want to mutate input
    idcs = np.arange(len(tbl)) + start_id
    obj_bits = 2**22
    objid = idcs % obj_bits
    brickid = idcs // obj_bits
    tbl["TARGETID"] = encode_targetid(objid, brickid, release=9010, mock=1)

    assert len(tbl) == len(np.unique(tbl["TARGETID"])), "Some non unique targetids"

    if "DESI_TARGET" not in tbl.colnames:
        log.details("DESI_TARGET not found... setting everything to LAEs")
        tbl["DESI_TARGET"] = 2**2 # Sets target bit all to LAE if it doesn't exist.

    # These two are unused but necessary to exist for mtl
    tbl["BGS_TARGET"] = 0
    tbl["MWS_TARGET"] = 0
    tbl["SCND_TARGET"] = 0

    # Most of these will be rewritten per target, they just need to exist
    # So that they span the entire table.
    tbl["NUMOBS_INIT"] = 4
    tbl["NUMOBS_MORE"] = 4
    tbl["NUMOBS"] = 0
    tbl["OBSCONDITIONS"] = 0
    tbl["TARGET_STATE"] = "UNOBS"
    tbl["PRIORITY_INIT"] = 3400
    tbl["PRIORITY"] = 3400

    rng = np.random.default_rng(100921)
    tbl["SUBPRIORITY"] = rng.uniform(size=len(tbl))

    nside = 64
    theta, phi = np.radians(90 - tbl["DEC"]), np.radians(tbl["RA"])
    hpx_data = hp.ang2pix(nside, theta, phi, nest=True)
    tbl["HEALPIX"] = hpx_data
    pixlist = np.unique(hpx_data)

    using_qso_target = "QSO_TARGET" in tbl.colnames
    using_lbg_target = "LBG_TARGET" in tbl.colnames
    if stds_tbl is not None:
        theta, phi = np.radians(90 - stds_tbl["DEC"]), np.radians(stds_tbl["RA"])
        hpx = hp.ang2pix(nside, theta, phi, nest=True)
        stds_tbl["HEALPIX"] = hpx
        keep_stds = np.isin(hpx, pixlist)
        keep_cols = tbl.colnames

        if using_qso_target:
            stds_tbl["QSO_TARGET"] = 0
        if using_lbg_target:
            stds_tbl["LBG_TARGET"] = 0
        tbl = vstack([stds_tbl[keep_stds][keep_cols], tbl])

    if targetmask is None:
        targetmask = load_target_yaml("targetmask.yaml")

    # Various afterburner / target parameters we don't care about too much
    # because we don't have actual spectra, just target positions,
    tbl["PMRA"] = 0.0
    tbl["PMDEC"] = 0.0
    tbl["REF_EPOCH"] = 0.0

    tbl["Z"] = -1.0
    tbl["ZWARN"] = -1
    tbl["ZTILEID"] = -1
    tbl["Z_QN"] = -1.0
    tbl["IS_QSO_QN"] = -1
    tbl["DELTACHI2"] = -1.0

    # TODO make this a parameter so the MTL can accurately reflect when
    # tbl_B targets were added.
    tbl["TIMESTAMP"] = timestamp
    tbl["VERSION"] = "0.0.1" # TODO make this the simassign version.

    for c in mtl_cols:
        assert c in tbl.colnames, f"{c} not found in cols!"

    # Set the target per target priority/numb obs values.
    for target in targetmask["desi_mask"]:
        bit = 2**target[1]
        name = target[0]
        this_target = (tbl["DESI_TARGET"] & bit) != 0

        log.details(f"Init Target {target} {np.sum(this_target)}")

        # Update to custom target type.
        tbl["TARGET_STATE"] = tbl["TARGET_STATE"].astype("<U15") # So we don't truncate status.
        tbl["TARGET_STATE"][this_target] = f"{name}|UNOBS"

        tbl["NUMOBS_INIT"][this_target] = targetmask["numobs"]["desi_mask"][name]
        tbl["NUMOBS_MORE"][this_target] = targetmask["numobs"]["desi_mask"][name]

        # Adjusting num obs for high z qsos.
        if using_qso_target:
            if name == "QSO":
                for qso_bit in range(len(targetmask["qso_mask"])):
                    this_qso = tbl["QSO_TARGET"] == 2 ** qso_bit
                    mult = targetmask["qso_mask"][qso_bit][-1]["numobs_mult"]
                    log.details(f"qso_bit {qso_bit}, mult {mult}, {np.sum(this_qso)}")
                    tbl["NUMOBS_INIT"][this_qso] *= mult
                    tbl["NUMOBS_MORE"][this_qso] *= mult

                # Adjusting num obs for high z qsos.
        if using_lbg_target:
            if name == "LBG":
                for lbg_bit in range(len(targetmask["lbg_mask"])):
                    this_lbg = tbl["LBG_TARGET"] == 2 ** lbg_bit
                    mult = targetmask["lbg_mask"][lbg_bit][-1]["numobs_mult"]
                    log.details(f"lbg_bit {lbg_bit}, mult {mult}, {np.sum(this_lbg)}")
                    tbl["NUMOBS_INIT"][this_lbg] *= mult
                    tbl["NUMOBS_MORE"][this_lbg] *= mult

        # Make sure the obsoncditions are set right
        obscond = target[-1]["obsconditions"]
        obscond = obscond.split("|")
        for oc in targetmask["obsconditions"]:
            if oc[0] in obscond: # 0 is the name, 1 is the bit
                tbl["OBSCONDITIONS"][this_target] += (2**oc[1])

        tbl["PRIORITY"][this_target] = targetmask["priorities"]["desi_mask"][name]["UNOBS"]
        tbl["PRIORITY_INIT"][this_target] = targetmask["priorities"]["desi_mask"][name]["UNOBS"]

    log.details(f"{len(pixlist)} HEALpix.")

    if save_dir is not None:
        base_dir = Path(save_dir)
        if (base_dir / "hp").exists(): shutil.rmtree((base_dir / "hp")) # Removes an old run in the same dir.
        hp_base = base_dir / "hp" / "main" / "dark"
        hp_base.mkdir(parents=True, exist_ok=True)

    if as_dict:
        tbl.sort(["HEALPIX", "TARGETID"]) # This will make the next line slightly quicker... I think.
        mtl_all = {}
        for hpx in pixlist:
            mtl_all[hpx] = tbl[tbl["HEALPIX"] == hpx]
            # mtl_all[hpx].sort("TARGETID")

            log.details(f"Generated {hpx}")
            fname =  f"mtl-dark-hp-{hpx}.ecsv"
            if save_dir is not None:
                mtl_all[hpx].write(hp_base / fname, overwrite=True) # Keep the original file extension.
                log.details(f"Saved to {str(hp_base / fname)}")

    # Want the global MTL sorted on TARGETID too.
    else:
        mtl_all = tbl
        mtl_all.sort("TARGETID")
        if save_dir is not None:
            mtl_all.write(base_dir / "targets.fits.gz", overwrite=True)

    # I suppose you could initialize without a save dir and then not return
    # it so it just gets lost to the ether but why would you do that?
    if return_mtl_all:
        return mtl_all
    return

def concatenate_mtls(mtl_a, mtl_b):
    """
    Cocnatenate two MTLs, ensuring the resulting file is sorted correctly.

    Parameters
    ----------
    mtl_a : :class:`~astropy.table.Table`
        One of two MTLs to concatenate.

    mtl_b : :class:`~astropy.table.Table`
        One of two MTLs to concatenate.

    Returns
    -------
    :class:`~astropy.table.Table` or dict
       A concatenated MTL containing the targets in both input MTLs and properly
       sorted by TARGETID and then TIMESTAMP.
    """
    joined = vstack([mtl_a, mtl_b])
    joined.sort(["TARGETID", "TIMESTAMP"])
    return joined