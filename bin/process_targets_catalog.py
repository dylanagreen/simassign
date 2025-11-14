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
parser.add_argument("--qso_mask", action="store_true", help="include the QSO_TARGET column")
parser.add_argument("--lbg_mask", action="store_true", help="include the LBG_TARGET column")
parser.add_argument("--config", required=False, type=str, help="configuration yaml file with target parameters.")
parser.add_argument("--seed", required=False, type=int, default=91701, help="seed to use for randomness and reproducibility.")
args = parser.parse_args()

if args.config is not None:
    with open(args.config) as f:
        targetmask = yaml.safe_load(f)
else:
    targetmask = load_target_yaml("targetmask.yaml")
print(f"Using targetmask {targetmask}")

targettypes = [row[2] for row in targetmask["desi_mask"]]

lbg_index = targettypes.index("LBG")

all_tbls = []
if args.laes:
    laes = Table.read(args.laes)
    lae_index = targettypes.index("LAE")
    lae_bit = targetmask["desi_mask"][lae_index][1]
    print(f"Setting LAE DESI_TARGET to {lae_bit}")
    laes["DESI_TARGET"] = 2 ** lae_bit
    print(f"Num LAEs: {len(laes)}")

    all_tbls.append(laes)
if args.lbgs:
    lbgs = Table.read(args.lbgs)

    lbg_bit = targetmask["desi_mask"][lbg_index][1]
    print(f"Setting LBG DESI_TARGET to {lbg_bit}")
    lbgs["DESI_TARGET"] = 2 ** lbg_bit
    print(f"Num LBGs: {len(lbgs)}")

    all_tbls.append(lbgs)

if args.qsos is not None:
    qsos = Table.read(args.qsos)
    qso_index = targettypes.index("QSO")

    qso_bit = targetmask["desi_mask"][qso_index][1]
    print(f"Setting QSO DESI_TARGET to {qso_bit}")
    qsos["DESI_TARGET"] = 2 ** qso_bit

    if args.qso_mask:
        qsos["QSO_TARGET"] = 1
        idcs = np.arange(len(qsos))
        rng = np.random.default_rng(args.seed)
        choice = rng.choice(idcs, replace=False, size=(len(idcs) // 2))
        qsos["QSO_TARGET"][choice] = 2 ** 1

        laes["QSO_TARGET"] = 0
        lbgs["QSO_TARGET"] = 0

    print(f"Num QSOs: {len(qsos)}")

    all_tbls.append(qsos)
tbl = vstack(all_tbls)

if args.lbg_mask and args.lbgs:
    tbl["LBG_TARGET"] = 0
    is_lbg = tbl["DESI_TARGET"] == 2 ** lbg_bit
    n_lbgs = np.sum(is_lbg)
    idcs = np.where(is_lbg)[0]

    rng = np.random.default_rng(args.seed)
    # By default use the highest bit in the lbg mask, which is in essence
    # equivalent to filling out anything that doesn't get set to that bit.
    tbl["LBG_TARGET"][is_lbg] = 2 ** targetmask["lbg_mask"][-1][1]
    for lbg_subtarg in targetmask["lbg_mask"]:
        frac = eval(lbg_subtarg[-1]["frac"])

        choice = rng.choice(idcs, replace=False, size=int(n_lbgs * frac))
        print(f"{len(choice)} ({frac}) requested to be set to bit {lbg_subtarg[1]}")
        tbl["LBG_TARGET"][choice] = 2 ** lbg_subtarg[1]

        # Remove ones that were set.
        idcs = idcs[~np.isin(idcs, choice)]
        print(np.sum(tbl["LBG_TARGET"] == 2 ** lbg_subtarg[1]), "actually set.")

print(tbl)

tbl.write(args.out, overwrite=True)