#!/usr/bin/env python

# stdlib imports
# from pathlib import Path
import argparse
from multiprocessing import Pool
import time

# DDSI imports
from desimodel.focalplane import get_tile_radius_deg

# Non-DESI Imports
from astropy.table import Table
import numpy as np

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
# parser.add_argument("-o", "--out", required=True, type=str, help="where to save the output processed file.")
parser.add_argument("--targs", type=str, help="targets to process.")
parser.add_argument("--tiles", required=True, type=str, help="master tiling file to use for simulation.")
args = parser.parse_args()

tiles = Table.read(args.tiles)
keep_tiles = tiles["IN_DESI"] & (tiles["PROGRAM"] == "DARK")

targs = Table.read(args.targs)
targs["PRIORITY"] = 3500
# It's easier to manipulate this in array form, and the order of the table
# is never going to change.
targ_radec = np.vstack([targs["RA"], targs["DEC"]]).T

tile_rad = get_tile_radius_deg()
fudge = 0.1

fp_area = 8.63
fiber_area = 0.0015
p_one_fiber = fiber_area / fp_area
nfibers = 4000

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

    # passnum = tileid // 10000
    # in_tile = targs[f"PASS_{passnum}"] == tileid
    # idcs = np.arange(len(targs))[in_tile] # Indices of targets that are possible to observe in this tile.

    rng = np.random.default_rng()
    probs = rng.uniform(size=len(idcs))

    assigned = 0
    i = 0
    keep_from_idcs = []
    cur_prob = nfibers * p_one_fiber

    for j, i in enumerate(idcs):
        if (assigned > nfibers): break

        if probs[j] > cur_prob:
            keep_from_idcs.append(i)
            cur_prob -= p_one_fiber
            assigned += 1

    return keep_from_idcs


targs["NUMOBS"] = 0
times = []
tot_tiles = 0
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

        targs["NUMOBS"][idcs] += counts

        # Update Priorities
        # Update everything observed to max, finished ones are handled in the next line.
        targs["PRIORITY"][idcs] = 3550

        # update_bool = np.zeros(len(targs), dtype=bool)
        # update_bool[idcs] = True
        finished = targs["NUMOBS"] >= 8

        targs["PRIORITY"][finished] = 2

    t_end = time.time()

    print(f"Pass {passnum} finished in {t_end - t_start} seconds...")
    times.append(t_end - t_start)



bincount = np.bincount(targs["NUMOBS"])
print(f"Final bincount: {bincount}")
print(f"Total tiles: {tot_tiles}")
print(f"Average time per pass: {np.mean(times)}")

colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"] # Okabe and Ito colorbline friendly.
fig, ax = plt.subplots(figsize=(12, 5))

bins = np.arange(bincount.shape[-1] + 1) - 0.5
ax.stairs(bincount, bins, ec=colors[0])

mean_exp = np.sum(bincount * np.arange(len(bincount))) / np.sum(bincount)
print(f"Mean coverage: {mean_exp}")
ax.axvline(mean_exp, c=colors[0])

median_at = np.sum(bincount) // 2
median_exp = np.argmax(np.cumsum(bincount) >= median_at)
print(np.cumsum(bincount))
print(median_at)
print(f"Median coverage: {median_exp}")
ax.axvline(median_exp, c=colors[0], ls="dashed")

ax.grid(alpha=0.5)
ax.set(xlim=(bins[0], bins[-1]), xlabel="Tile Coverage")

plt.savefig("plots/coverage_hist.jpg", bbox_inches="tight", dpi=256)