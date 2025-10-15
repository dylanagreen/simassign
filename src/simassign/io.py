# External imports
from astropy.table import Table, vstack
import numpy as np

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

def load_mtl_all(top_dir, verbose=False):
    """
    Load all MTLs split by healpix stored in top_dir.

    This function iterates over the healpix subdirectories stored in top_dir
    and loads all MTL files stored there.

    NOTE: Assumes some defaults, namely that
    the MTLs are stored in top_dir / hp / main / dark and that all MTLs
    are stored as fits files.

    Parameters
    ----------
    top_dir : :class:`~pathlib.Path`
        Top level directory containing MTLs split by healpix.

    verbose : bool
        Verbosely print when loading each individual MTL file.

    Returns
    -------
    :class:`~astropy.table.Table`
        MTL table comprised of all the subtables stored in top_dir.
    """
    hp_base = top_dir / "hp" / "main" / "dark"
    mtl_all = Table()
    tbls = []
    for mtl_loc in hp_base.glob("*.fits"):
        if verbose: print(f"Loading {mtl_loc.name}")
        temp_tbl = Table.read(mtl_loc)
        tbls.append(temp_tbl)
    mtl_all = vstack(tbls)
    return mtl_all