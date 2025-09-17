import unittest

from astropy.table import Table
import numpy as np

from simassign.mtl import *

class TestMTL(unittest.TestCase):
    def test_initialize_mtl(self):
        tbl = Table()
        # Could use random objects here but we want this to be reproducible,
        # and for this test the location of the objects is entirely irrelevant.
        nobj = 50
        tbl["RA"] = np.linspace(110, 120, nobj)
        tbl["DEC"] = np.linspace(10, 20, nobj)

        observed_mtl = initialize_mtl(tbl)

        expected_colnames = ["RA", "DEC", "PRIORITY", "SUBPRIORITY", "PRIORITY_INIT",
                             "TARGETID", "TARGET_STATE", "TIMESTAMP", "VERSION",
                             "OBSCONDITIONS", "NUMOBS", "NUMOBS_INIT", "NUMOBS_MORE",
                             "DESI_TARGET", "BGS_TARGET", "MWS_TARGET", "SCND_TARGET",
                             "Z", "Z_QN", "ZWARN", "IS_QSO_QN", "DELTACHI2",
                             "HEALPIX", "ZTILEID"]

        # Using sets because the order does not matter
        self.assertEqual(set(observed_mtl.colnames), set(expected_colnames))

        self.assertEqual(list(observed_mtl["DESI_TARGET"]), [2**2] * nobj)
        self.assertEqual(list(observed_mtl["MWS_TARGET"]), [0] * nobj)
        self.assertEqual(list(observed_mtl["BGS_TARGET"]), [0] * nobj)

        self.assertEqual(list(observed_mtl["PRIORITY"]), [3400] * nobj)

        self.assertEqual(list(observed_mtl["NUMOBS"]), [0] * nobj)
        self.assertEqual(list(observed_mtl["NUMOBS_INIT"]), [4] * nobj)
        self.assertEqual(list(observed_mtl["NUMOBS_MORE"]), [4] * nobj)

        # Subpriority is random but we should ensure that it's between zero and one.
        self.assertTrue(np.all(observed_mtl["SUBPRIORITY"] >= 0))
        self.assertTrue(np.all(observed_mtl["SUBPRIORITY"] <= 1))