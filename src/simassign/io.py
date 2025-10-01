from astropy.table import Table
import numpy as np

def load_catalog(file_loc, box=None):
    print(f"Loading catalog from... {file_loc}")
    # IF box provided must be len four (two edges in RA/DEC each)
    if box is not None:
        assert len(box) == 4, "Must provide all four box edges!"

    tbl = Table.read(file_loc)

    if box is not None:
        ra_min, ra_max, dec_min, dec_max = box
        in_ra = (tbl["RA"] >= (ra_min)) & (tbl["RA"] <= (ra_max))
        in_dec = (tbl["DEC"] >= (dec_min)) & (tbl["DEC"] <= (dec_max))

        tbl = tbl[in_ra & in_dec]
    return np.array(tbl["RA"]), np.array(tbl["DEC"])