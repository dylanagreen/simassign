#!/usr/bin/env python

import argparse
from pathlib import Path

import shutil

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="top level directory of run to purge results from.")
args = parser.parse_args()

base_dir = Path(args.input)

if (base_dir / "hp").exists():
    print("Purging MTLS....")
    shutil.rmtree((base_dir / "hp"))
if (base_dir / "fba").exists():
    print("Purging Fiber assignment...")
    shutil.rmtree((base_dir / "fba"))

for night_details in base_dir.glob("night*"):
    print(f"Purging {night_details.name}")
    shutil.rmtree(night_details) # Purge the night details, i.e. target and tile files per night.

for tiles in base_dir.glob("tiles*fits"):
    print(f"Purging {tiles.name}")
    tiles.unlink() # Purge the remaining tile only files

for log in base_dir.glob("*log"):
    print(f"Purging {log.name}")
    log.unlink()

for processing in (base_dir / "processed").glob("*.npy"):
    print(f"Purging {processing.name}")
    processing.unlink()