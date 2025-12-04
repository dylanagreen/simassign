from datetime import datetime
import os
from pathlib import Path
import time
import unittest

from astropy.table import Table
import numpy as np

from simassign.util import *

class TestMTLInits(unittest.TestCase):
    def setUp(self):
        self.datadir = Path(os.path.dirname(__file__)) /  "data"
        self.base_tiles = Table.read(self.datadir / "base_tiles.ecsv")

    def test_rotate_tiling(self):
        in_desi = np.zeros(len(self.base_tiles), dtype=bool)
        in_desi[:100] = True
        tiles_pass_0 = self.base_tiles[(self.base_tiles["PASS"] == 0)]
        tiles_pass_1 = self.base_tiles[(self.base_tiles["PASS"] == 1)]

        # Passes are zero indexed, but I designed rotations to be 1 indexed
        # for some reason. So need pass = 2 to get the first rotation.
        # Rotating pass 0 by the first rotation should give all centers
        # in pass 1
        observed = rotate_tiling(tiles_pass_0, 2)

        # Relatively generous relative tolerance because due to all sorts of
        # floating point conversions everywhere this isn't going to be exact.
        # For example, the DESI tiles file is rounded to three decimals.
        # As long it's close to a "good" degree of accuracy.
        np.testing.assert_allclose(tiles_pass_1["RA"], observed["RA"], rtol=1e-2)
        np.testing.assert_allclose(tiles_pass_1["DEC"], observed["DEC"], rtol=3e-2)