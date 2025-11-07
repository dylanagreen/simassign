#!/usr/bin/env python

# stdlib imports
# from pathlib import Path
import argparse
from multiprocessing import Pool
from pathlib import Path
import time

# DDSI imports
from desimodel.focalplane import get_tile_radius_deg

# Non-DESI Imports
from astropy.table import Table
import numpy as np
import yaml

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--outdir", required=True, type=str, help="where to save the output summary plots.")
parser.add_argument("-s", "--suffix", required=True, type=str, help="suffix to attach to file names.")
parser.add_argument("--targs", required=True, type=str, help="targets to process.")
parser.add_argument("--tiles", required=True, type=str, help="master tiling file to use for simulation.")
parser.add_argument("--config", required=True, type=str, help="configuration yaml file with run parameters.")
args = parser.parse_args()

out_dir = Path(args.outdir)

with open(args.config) as f:
    targetmask = yaml.safe_load(f)
print(f"Using targetmask {targetmask}")

tiles = Table.read(args.tiles)
keep_tiles = tiles["IN_DESI"] & (tiles["PROGRAM"] == "DARK")

targs = Table.read(args.targs)

targettypes = [row[2] for row in targetmask["desi_mask"]]
if "DESI_TARGET" not in targs.colnames:
    print("DESI_TARGET not found in input catalog, assuming all targets are LBGs")
    lbg_index = targettypes.index("LBG")
    targs["DESI_TARGET"] = 2 ** targetmask["desi_mask"][lbg_index][1]


targs["PRIORITY"] = 3500
for target in targetmask["desi_mask"]:
    bit = 2**target[1]
    name = target[0]

    this_target = (targs["DESI_TARGET"] & bit) != 0
    targs["PRIORITY"][this_target] = targetmask["priorities"]["desi_mask"][name]["UNOBS"]

# It's easier to manipulate this in array form, and the order of the table
# is never going to change.
targ_radec = np.vstack([targs["RA"], targs["DEC"]]).T

tile_rad = get_tile_radius_deg()
fudge = 0.1

fp_area = 8.63
fiber_area = 0.0015
p_one_fiber = fiber_area / fp_area
nfibers = 4100

def parallel_observe(center, verbose=False):
    if verbose: print(f"Dist to {center}")
    diff = targ_radec - center
    if verbose: print(diff.shape)
    keep_ra = np.abs(diff[:, 0]) <= tile_rad + fudge
    keep_dec = np.abs(diff[:, 1]) <= tile_rad + fudge

    if verbose: print(np.sum(keep_ra), np.sum(keep_dec), np.sum(keep_ra & keep_dec))

    # Get the euclidean distance to the center, then keep everything that's
    # within the radius. We return the indices of targets kept because we want
    # to update those indices in the glboal array.
    dist = np.linalg.norm(diff[keep_ra & keep_dec], axis=1)
    idcs = np.arange(len(targ_radec))[keep_ra & keep_dec]
    keep = dist <= tile_rad
    if verbose: print(f"{dist.shape[0]} targets kept for distance, {np.sum(keep)} in tile radius")

    idcs = idcs[keep] # Indices of everything in this tile.

    rng = np.random.default_rng()
    probs = rng.uniform(size=len(idcs))

    assigned = 0
    i = 0
    keep_from_idcs = []
    cur_prob = nfibers * p_one_fiber

    priorities = targs[keep_ra & keep_dec][keep]["PRIORITY"]
    idcs_sort = np.argsort(priorities)[::-1]
    # assert len(idcs_sort) == len(idcs)
    idcs = idcs[idcs_sort]

    for j, i in enumerate(idcs):
        if (assigned > nfibers): break

        if probs[j] < cur_prob:
            keep_from_idcs.append(i)
            cur_prob -= p_one_fiber
            assigned += 1

    return keep_from_idcs

targs["NUMOBS"] = 0
times = []
tot_tiles = 0
npasses = np.max(tiles["PASS"])
nobs_arr = np.zeros((len(targettypes), npasses + 1, npasses))

for passnum in np.unique(tiles["PASS"]):
    t_start = time.time()
    tiles_pass = tiles[keep_tiles & (tiles["PASS"] == passnum)]

    print(f"Pass {passnum}: {len(tiles_pass)} tiles")
    tot_tiles += len(tiles_pass)
    tile_radec = np.vstack([tiles_pass["RA"], tiles_pass["DEC"]]).T

    with Pool(32) as p:

        res = p.map(parallel_observe, tile_radec)
        all_idcs = np.concatenate(res)

    # Functionally nothing should appear on more than one tile, but better to be safe than sorry.
    idcs, counts = np.unique(all_idcs, return_counts = True)
    print(f"{len(idcs)} assigned ({len(idcs) / len(tiles_pass)} average per tile)")

    targs["NUMOBS"][idcs] += counts

    # Update Priorities
    # Update everything observed to max, finished ones are handled in the next line.
    for i, target in enumerate(targetmask["desi_mask"]):
        bit = 2**target[1]
        name = target[0]

        this_target = (targs["DESI_TARGET"] & bit) != 0
        this_target_update = this_target[idcs]
        print(f"Updating {np.sum(this_target)} {name}")
        targs["PRIORITY"][idcs[this_target_update]] = targetmask["priorities"]["desi_mask"][name]["MORE_ZGOOD"]

        # If it exceeds the target then it's done. Done is always 2.
        # TODO maybe we change the done priority such that if all targets are done
        # we prefer one over another? Thye'd require changing this.
        finished = targs["NUMOBS"][idcs] >= targetmask["numobs"]["desi_mask"][name]
        targs["PRIORITY"][idcs[finished & this_target_update]] = targetmask["priorities"]["desi_mask"][name]["DONE"]

        bincount = np.bincount(targs["NUMOBS"][this_target], minlength=npasses)
        nobs_arr[i, passnum, :] = bincount

    t_end = time.time()

    print(f"Pass {passnum} finished in {t_end - t_start} seconds...")
    times.append(t_end - t_start)

print(nobs_arr)
print(f"Total tiles: {tot_tiles}")
print(f"Average time per pass: {np.mean(times)}")

bincount = np.bincount(targs["NUMOBS"], minlength=npasses)
mean_exp = np.sum(bincount * np.arange(len(bincount))) / np.sum(bincount)
print(f"Mean coverage: {mean_exp}")

median_at = np.sum(bincount) // 2
median_exp = np.argmax(np.cumsum(bincount) >= median_at)
print(np.cumsum(bincount))
print(median_at)
print(f"Median coverage: {median_exp}")

done = np.zeros(nobs_arr.shape[:-1])
for i, target in enumerate(targetmask["desi_mask"]):
    bit = 2**target[1]
    name = target[0]

    this_target = (targs["DESI_TARGET"] & bit) != 0
    obs_targ = targetmask["numobs"]["desi_mask"][name]
    finished = targs["NUMOBS"][this_target] >= obs_targ
    print(f"Percent {name} complete: {np.sum(finished) / len(finished)}")

    nobs_arr[i, 0, 0] = np.sum(this_target)

    done[i, :] = np.sum(nobs_arr[i, :, obs_targ:], axis=1)

# done = np.sum(nobs_arr[:, :, 8:], axis=2)
fraction = done / nobs_arr[:, 0, 0][:, None]
print(done)
print(fraction)

np.save(out_dir / f"nobs_{args.suffix}_mc.npy", nobs_arr)
np.save(out_dir / f"done_{args.suffix}_mc.npy", done)
np.save(out_dir / f"fraction_{args.suffix}_mc.npy", fraction)

colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"] # Okabe and Ito colorbline friendly.
fig, ax = plt.subplots(figsize=(12, 5))

bins = np.arange(bincount.shape[-1] + 1) - 0.5
ax.stairs(bincount, bins, ec=colors[0])


ax.axvline(mean_exp, c=colors[0])
ax.axvline(median_exp, c=colors[0], ls="dashed")

ax.grid(alpha=0.5)
ax.set(xlim=(bins[0], bins[-1]), xlabel="Tile Coverage")

plt.savefig("plots/coverage_hist.jpg", bbox_inches="tight", dpi=256)
