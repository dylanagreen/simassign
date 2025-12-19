#!/usr/bin/env python

import argparse
from datetime import datetime, timedelta

# Non-DESI Imports
import numpy as np
from astropy.table import Table, vstack

parser = argparse.ArgumentParser()
parser.add_argument("exposures_a", type=str, help="earlier exposures file, representing the start of the survey.")
parser.add_argument("exposures_b", type=str, help="later exposure file to join to the earlier file.")
parser.add_argument("-o", "--out", required=True, type=str, help="where to save generated tile file.")
parser.add_argument("--fix_pass", required=False, action="store_true", help="fix pass numbers in exposures_b to have unique passes.")
parser.add_argument("--fix_tileids", required=False, action="store_true", help="fix tileids to be unique across the joined file")
parser.add_argument("--bstart", required=False, type=str, help="if passed, try and force exposures_b to start on bstart if possible. If bstart is before the end of exposures_a, then defaults to the day after as normal.")
args = parser.parse_args()

tbl_a = Table.read(args.exposures_a)
tbl_b = Table.read(args.exposures_b)

print(f"Len A {len(tbl_a)}")
print(f"Len B {len(tbl_b)}")

# These should be sorted by exposure date but they might not be
tbl_a.sort("TIMESTAMP")
tbl_b.sort("TIMESTAMP")

# The last date in tbl_a and start of tbl_b. We need to shift all of tbl_b
# to start after tbl_a, if that's not true already.
end_a = datetime.fromisoformat(tbl_a["TIMESTAMP"][-1])
start_b = datetime.fromisoformat(tbl_b["TIMESTAMP"][0])
print(f"{end_a = }")
print(f"{start_b = }")

if args.bstart:
    force_b = datetime.fromisoformat(args.bstart)
    if force_b > end_a:
        # Set the end of a to one day before the forced start of b.
        # end_a is only used to determine when b should start.
        end_a = force_b - timedelta(days=1)

delta_survey = end_a - start_b + timedelta(days=1)

ts = [datetime.fromisoformat(x) + delta_survey for x in tbl_b["TIMESTAMP"]]
tbl_b["TIMESTAMP"] = [x.isoformat() for x in ts]

ts_ymd = [x.strftime("%Y%m%d") for x in ts]
tbl_b["TIMESTAMP_YMD"] = ts_ymd

# Need to increase pass numbers and tileids in exposures_b if they conflict with exposures_a.
if args.fix_pass:
    max_pass_a = np.unique(tbl_a["PASS"])[-1]
    min_pass_b = np.unique(tbl_b["PASS"])[0]
    if min_pass_b <= max_pass_a:
        delta_pass = max_pass_a - min_pass_b + 1
        tbl_b["PASS"] += delta_pass

joined = vstack([tbl_a, tbl_b])

if args.fix_tileids:
    print(np.max(joined["PASS"]))
    for p in range(np.max(joined["PASS"]) + 1):
        this_pass = joined["PASS"] == p
        if np.sum(this_pass):
            tileids = np.arange(np.sum(this_pass)) + p * 10000
            print(np.sum(this_pass))
            print(p, np.max(tileids))
            joined["TILEID"][this_pass] = tileids

    assert len(np.unique(joined["TILEID"])) == len(joined), "Some non unique tileids!"

print(joined)
print(f"{len(joined)} exposures")

print(f"Writing output joined file to {args.out}")
joined.write(args.out, overwrite=True)

