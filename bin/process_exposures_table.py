#!/usr/bin/env python

# stdlib imports
from pathlib import Path
import argparse
from datetime import datetime, timedelta

# Non-DESI Imports
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, join, unique
from astropy.time import Time


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out", required=True, type=str, help="where to save the output processed file.")
parser.add_argument("-e", "--exposures", type=str, help="A catalog of ordered exposures.")
parser.add_argument("-t", "--tiles", required=True, type=str, help="master tiling file to match to.")
args = parser.parse_args()

tbl = Table.read(args.exposures)
print(len(tbl), " exposures")

tiles_tbl = Table.read(args.tiles)

print(tiles_tbl)
joined = join(tbl, tiles_tbl, "TILEID")

times = Time(joined["MJD"], format="mjd")
times = times.to_datetime()

# The starting date for the MTL is late 2024, so we need the survey to start in 2025
# otherwise things get really weird.
if times[0] < datetime.strptime("20250101", "%Y%m%d"):
    times = times + timedelta(days=365)

# timestamps = [t.isoformat() for t in times]
timestamps = [t.strftime("%Y-%m-%dT%H:%M:%S%z")+"+00:00" for t in times] # This is the format string used by fiberassign....
timestamps_ymd = [t.strftime("%Y%m%d") for t in times]

joined["TIMESTAMP"] = timestamps
joined["TIMESTAMP_YMD"] = timestamps_ymd
joined.sort("TIMESTAMP")

# We want the unique per night, but we want the maximum SNR2FRAC per night,
# in the case that it is observed twice that night. We want to keep the
# max SNR2FRAC, because SNR2FRAC >=1  means its done, and we will use this
# to propogate the information to update_mtl or not.
joined.sort(keys=["TIMESTAMP_YMD", "SNR2FRAC"], reverse=True)
unique_per_night = unique(joined, keys=["TILEID", "TIMESTAMP_YMD"])
unique_per_night.sort("EXPID")
unique_per_night["TILEDONE"] = unique_per_night["SNR2FRAC"] >= 1

print(unique_per_night)
unique_per_night.write(args.out, overwrite=True)
