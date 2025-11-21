#!/usr/bin/env python

import argparse
from datetime import datetime, timedelta

# Non-DESI Imports
import numpy as np
from astropy.table import Table, vstack

parser = argparse.ArgumentParser()
parser.add_argument("--exposures_a", required=True, type=str, help="earlier exposures file, representing the start of the survey.")
parser.add_argument("--exposures_b", required=True, type=str, help="later exposure file to join to the earlier file.")
parser.add_argument("-o", "--out", required=True, type=str, help="where to save generated tile file.")

args = parser.parse_args()

tbl_a = Table.read(args.exposures_a)
tbl_b = Table.read(args.exposures_b)

# These should be sorted by exposure date but they might not be
tbl_a.sort("TIMESTAMP")
tbl_b.sort("TIMESTAMP")

# The last date in tbl_a and start of tbl_b. We need to shift all of tbl_b
# to start after tbl_a, if that's not true already.
end_a = datetime.fromisoformat(tbl_a["TIMESTAMP"][-1])
start_b = datetime.fromisoformat(tbl_b["TIMESTAMP"][0])
print(f"{end_a = }")
print(f"{start_b = }")

delta_survey = end_a - start_b + timedelta(days=1)

ts = [datetime.fromisoformat(x) + delta_survey for x in tbl_b["TIMESTAMP"]]
tbl_b["TIMESTAMP"] = [x.isoformat() for x in ts]

ts_ymd = [x.strftime("%Y%m%d") for x in ts]
tbl_b["TIMESTAMP_YMD"] = ts_ymd

# Need to increase pass numbers and tileids in exposures_b if they comflict with exposures_a.
max_pass_a = np.unique(tbl_a["PASS"])[-1]
min_pass_b = np.unique(tbl_b["PASS"])[0]
if min_pass_b <= max_pass_a:
    delta_pass = max_pass_a - min_pass_b + 1
    tbl_b["PASS"] += delta_pass

joined = vstack([tbl_a, tbl_b])

print(joined)
print(f"{len(joined)} exposures")

print(f"Writing output joined file to {args.out}")
joined.write(args.out, overwrite=True)

