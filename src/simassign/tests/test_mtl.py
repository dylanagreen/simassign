import time
import unittest

from astropy.table import Table
import numpy as np

from simassign.mtl import *

class TestMTL(unittest.TestCase):
    def setUp(self):
        # Need this for checking priorities etc.
        self.targetmask = load_target_yaml("targetmask.yaml")

    def test_initialize_mtl(self):
        tbl = Table()
        # Could use random objects here but we want this to be reproducible,
        # and for this test the location of the objects is entirely irrelevant.
        n_obj = 50
        tbl["RA"] = np.linspace(110, 120, n_obj)
        tbl["DEC"] = np.linspace(10, 20, n_obj)

        observed_mtl = initialize_mtl(tbl)

        expected_colnames = ["RA", "DEC", "PRIORITY", "SUBPRIORITY", "PRIORITY_INIT",
                             "TARGETID", "TARGET_STATE", "TIMESTAMP", "VERSION",
                             "OBSCONDITIONS", "NUMOBS", "NUMOBS_INIT", "NUMOBS_MORE",
                             "DESI_TARGET", "BGS_TARGET", "MWS_TARGET", "SCND_TARGET",
                             "Z", "Z_QN", "ZWARN", "IS_QSO_QN", "DELTACHI2",
                             "HEALPIX", "ZTILEID"]

        # Using sets because the order does not matter
        self.assertEqual(set(observed_mtl.colnames), set(expected_colnames))

        # TODO a better way to do this bit detecting?
        true_bit = self.targetmask["desi_mask"][0][1]
        np.testing.assert_array_equal(observed_mtl["DESI_TARGET"],
                                      2**true_bit * np.ones(n_obj, dtype=int))
        np.testing.assert_array_equal(observed_mtl["MWS_TARGET"],
                                      np.zeros(n_obj, dtype=int))
        np.testing.assert_array_equal(observed_mtl["BGS_TARGET"],
                                      np.zeros(n_obj, dtype=int))

        priority = self.targetmask["priorities"]["desi_mask"]["LAE"]["UNOBS"]
        np.testing.assert_array_equal(observed_mtl["PRIORITY"],
                                      priority * np.ones(n_obj, dtype=int))

        nobs_init = self.targetmask["numobs"]["desi_mask"]["LAE"]
        np.testing.assert_array_equal(observed_mtl["NUMOBS"],
                                      np.zeros(n_obj, dtype=int))
        np.testing.assert_array_equal(observed_mtl["NUMOBS_INIT"],
                                      nobs_init * np.ones(n_obj, dtype=int))
        np.testing.assert_array_equal(observed_mtl["NUMOBS_MORE"],
                                      nobs_init * np.ones(n_obj, dtype=int))

        # Subpriority is random but we should ensure that it's between zero and one.
        self.assertTrue(np.all(observed_mtl["SUBPRIORITY"] >= 0))
        self.assertTrue(np.all(observed_mtl["SUBPRIORITY"] <= 1))

    def test_single_update_mtl_nozcat(self):
        tbl = Table()
        n_obj = 50
        n_update = 20
        tbl["RA"] = np.linspace(110, 120, n_obj)
        tbl["DEC"] = np.linspace(10, 20, n_obj)

        init_mtl = initialize_mtl(tbl)
        tids_to_update = init_mtl["TARGETID"][0:n_update]
        time.sleep(1) # Need the timestamps to differ, this code is too fast otherwise.
        observed_mtl = update_mtl(init_mtl, tids_to_update=tids_to_update, use_desitarget=False)

        # There should be at least 50 + 20 rows (50 original, 20 updated)
        self.assertEqual(len(observed_mtl), n_obj + n_update)

        # Everything is an LAE for now.
        name = "LAE"
        for tid in tids_to_update:
            # Indices in this loop based on the assumption that update_mtl
            # correctly orders the mtl by targetid and then by timestamp
            # so that the later row is a later timestamp
            tid_tbl = observed_mtl[observed_mtl["TARGETID"] == tid]
            # Every updated targetid should have two rows in the new MTL.
            self.assertEqual(len(tid_tbl), 2)

            # NUMOBS should be increased by 1 in the updated row.
            # Should be equal to zero in the original row
            self.assertEqual(tid_tbl["NUMOBS"][0] + 1, tid_tbl["NUMOBS"][1])
            self.assertEqual(tid_tbl["NUMOBS"][0], 0)

            # NUMOBS_MORE should be decreased by one relative to previous.
            # Original should match the targetmask yaml.
            self.assertEqual(tid_tbl["NUMOBS_MORE"][0] - 1, tid_tbl["NUMOBS_MORE"][1])
            self.assertEqual(tid_tbl["NUMOBS_MORE"][0],
                             self.targetmask["numobs"]["desi_mask"][name]) # TODO derive target type from row.

            # The priority both before and after the update should match
            # what's expected in our custom targetmask.
            self.assertEqual(tid_tbl["PRIORITY"][0],
                             self.targetmask["priorities"]["desi_mask"][name]["UNOBS"])
            self.assertEqual(tid_tbl["PRIORITY"][1],
                             self.targetmask["priorities"]["desi_mask"][name]["MORE_ZGOOD"])

            # TODO more tests...