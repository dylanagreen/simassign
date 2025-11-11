#!/usr/bin/env python


# TODO proper docstring
import argparse
from datetime import datetime, timedelta
from multiprocessing import Pool

# Non-DESI Imports
import numpy as np
from astropy.table import Table, vstack, unique

from simassign.mtl import *

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out", required=True, type=str, help="where to save the mtl* and fba* output files.")
parser.add_argument("--laes", type=str, help="A catalog of laes to use for fiber assignment.")
parser.add_argument("--lbgs", type=str, help="A catalog of lbgs to use for fiber assignment.")
parser.add_argument("--qsos", type=str, help="A catalog of qsos to use for fiber assignment.")
parser.add_argument("--seed", required=False, type=int, default=91701, help="seed to use for randomness and reproducibility.")
args = parser.parse_args()

laes = Table.read(args.laes)
lbgs = Table.read(args.lbgs)

targetmask = load_target_yaml("targetmask.yaml")
print(f"Using targetmask {targetmask}")

targettypes = [row[2] for row in targetmask["desi_mask"]]
lae_index = targettypes.index("LAE")
lbg_index = targettypes.index("LBG")

lae_bit = targetmask["desi_mask"][lae_index][1]
print(f"Setting LAE DESI_TARGET to {lae_bit}")
laes["DESI_TARGET"] = 2 ** lae_bit
print(f"Num LAEs: {len(laes)}")

lbg_bit = targetmask["desi_mask"][lbg_index][1]
print(f"Setting LBG DESI_TARGET to {lbg_bit}")
lbgs["DESI_TARGET"] = 2 ** lbg_bit
print(f"Num LBGs: {len(lbgs)}")

if args.qsos is not None:
    qsos = Table.read(args.qsos)
    qso_index = targettypes.index("QSO")

    qso_bit = targetmask["desi_mask"][qso_index][1]
    print(f"Setting QSO DESI_TARGET to {qso_bit}")
    qsos["DESI_TARGET"] = 2 ** qso_bit

    qsos["QSO_TARGET"] = 1
    idcs = np.arange(len(qsos))
    rng = np.random.default_rng(args.seed)
    choice = rng.choice(idcs, replace=False, size=(len(idcs) // 2))
    qsos["QSO_TARGET"][choice] = 2 ** 1
    print(f"Num QSOs: {len(qsos)}")

    laes["QSO_TARGET"] = 0
    lbgs["QSO_TARGET"] = 0

    tbl = vstack([laes, lbgs, qsos])

else:
    tbl = vstack([laes, lbgs])

print(tbl)

tbl.write(args.out, overwrite=True)