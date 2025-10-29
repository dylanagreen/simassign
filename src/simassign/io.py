# External imports
from astropy.table import Table, vstack
import numpy as np

# Stdlib imports
from multiprocessing import Pool

from simassign.mtl import deduplicate_mtl

def load_catalog(file_loc, box=None):
    """
    Load a catalog of targets located at file_loc, and return their right
    ascension (RA) and declination (DEC) coordinates.

    Parameters
    ----------
    file_loc : str or :class:`~pathlib.Path`
        Catalog file location

    box : :class:`~numpy.array`
        If not None, should be a four element array defining the corners of
        a box in RA, DEC space. The target catalog will be cut to the given
        box, and only objects that lie within that box are kept. Defaults
        to None (return all targets).

    Returns
    -------
    ra : :class:`~numpy.array`
        Right ascension of targets in the catalog,.

    dec : :class:`~numpy.array`
        Declination of all targets in catalog.

    """
    # TODO: support arbitrary cutting geometry.
    print(f"Loading catalog from... {file_loc}")
    # If box provided must be len four (two edges in RA/DEC each)
    if box is not None:
        assert len(box) == 4, "Must provide all four box edges!"

    tbl = Table.read(file_loc)

    # Actually does the cutting down, which we only need to do if
    # the box was provided.
    if box is not None:
        ra_min, ra_max, dec_min, dec_max = box
        in_ra = (tbl["RA"] >= (ra_min)) & (tbl["RA"] <= (ra_max))
        in_dec = (tbl["DEC"] >= (dec_min)) & (tbl["DEC"] <= (dec_max))

        tbl = tbl[in_ra & in_dec]

    return np.array(tbl["RA"]), np.array(tbl["DEC"])

def load_mtl(mtl_loc, deduplicate_on_load=False):
    # TODO docstirng?
    print(f"Loading {mtl_loc.name} with deduplicate_on_load={deduplicate_on_load}")
    tbl = Table.read(mtl_loc)
    if deduplicate_on_load:
        tbl = deduplicate_mtl(tbl)
    return tbl

def load_mtl_all(mtl_dir, as_dict=False, nproc=1, deduplicate_on_load=False):
    """
    Load all MTLs split by healpix stored in mtl_dir.

    This function iterates over the healpix subdirectories stored in mtl_dir
    and loads all MTL files stored there.

    NOTE: Assumes some defaults, namely that
    the MTLs are stored in top_dir / hp / main / dark and that all MTLs
    are stored as ecsv files.

    Parameters
    ----------
    mtl_dir : :class:`~pathlib.Path`
        Top level directory containing MTLs split by healpix.

    as_dict : bool
        Whether to return the mtl_all as a dict, with healpixel number as the key
        and the value as the individual MTL corresponding tot hat healpixel,
        or to stack the entire table into one. Defaults to False (stacking as a single table)

    nproc : int
        Number of processes to use to parallelize the loading. Defaults to 1

    deduplicate_on_load : bool
        Whether to deduplicate the table to get only the latest entries for all targets
        at loading time. Defaults to False.

    Returns
    -------
    :class:`~astropy.table.Table` or dict
        MTL table comprised of all the subtables stored in top_dir.
    """

    if as_dict:
        mtl_all = {}
    else:
        mtl_all = Table()

    locs = list(mtl_dir.glob("*.ecsv"))
    if len(locs) == 0:
        locs = list(mtl_dir.glob("*.fits"))

    assert len(locs) > 0, "No mtls found as either ecsv or fits!"

    # Helpixels are not z filled to the same digit length otherwies I'd use a regex to pull this out.
    pixlist = [loc.name.split("-")[-1].split(".")[0] for loc in locs]

    args = [(l, deduplicate_on_load) for l in locs]

    with Pool(nproc) as p:
        tbls = p.starmap(load_mtl, args)

    if as_dict:
        for i, temp_tbl in enumerate(tbls):
            hpx = pixlist[i]
            mtl_all[int(hpx)] = temp_tbl
    else:
        mtl_all = vstack(tbls)

    # Want the global MTL sorted on TARGETID too.
    if not as_dict:
        mtl_all.sort("TARGETID")

    return mtl_all