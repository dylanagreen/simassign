#!/usr/bin/env python

# python generate_tiling_file.py --npass 30 -o /pscratch/sd/d/dylang/fiberassign/ --fourex --collapse

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
    shutil.rmtree((base_dir / "fba")) # Purge fiber assignment

for night_details in base_dir.glob("night*"):
    print(f"Purging {night_details.name}")
    shutil.rmtree(night_details) # Purge the night details, i.e. target and tile files.

for tiles in base_dir.glob("tiles*"):
    print(f"Purging {tiles.name}")
    tiles.unlink() # Purge the night details, i.e. target and tile files.