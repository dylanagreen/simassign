#!/usr/bin/env python

# stdlib imports
import argparse
from datetime import datetime, timedelta

# DESI imports
from desimodel.focalplane import get_tile_radius_deg

# Other imports
from astropy.table import Table
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out", required=True, type=str, help="where to save the output simulation file.")
parser.add_argument("-t", "--tiles", required=True, type=str, help="master tiling file to simulate exposures for.")
parser.add_argument("--seed", required=False, type=int, default=91701, help="seed to use for randomness and reproducibility.")
parser.add_argument("--min_per_night", required=False, type=int, default=3, help="minimum tiles to observe per night.")
parser.add_argument("--max_per_night", required=False, type=int, default=25, help="maximum tiles to observe per night.")
parser.add_argument("--starttime", required=False, type=str, default="2025-01-01T00:00:00+00:00", help="starting timestamp for the first observing night.")
args = parser.parse_args()

# "/pscratch/sd/d/dylang/fiberassign/tiles/tiles-25pass-offset-tiles-500deg.ecsv"
tiles = Table.read(args.tiles)
rng = np.random.default_rng(args.seed)

# Buffer just exists so that we don't get tiles that are tangent.
buffer = 0.1
min_dist = 2 * get_tile_radius_deg() + buffer

start_time = datetime.fromisoformat(args.starttime)

cur_time = start_time

def get_tile_distances(tileid, tile_tbl):
    # TODO move to main package
    # Get the distances of all tiles to the tile given by tileid
    row = tile_tbl[tile_tbl["TILEID"] == tileid]
    tile_ra, tile_dec = row["RA"][0], row["DEC"][0]
    center = np.array([tile_ra, tile_dec])

    tile_centers = np.vstack([tile_tbl["RA"], tile_tbl["DEC"]]).T
    diff = tile_centers - center
    return np.linalg.norm(diff, axis=1)

# Use these two variables to determine distance from first tile of the night
# to all other tiles and determine which to "observe"
desi_tiles = tiles[tiles["IN_DESI"]]
tiles_radec = np.vstack([desi_tiles["RA"], desi_tiles["DEC"]]).T


first_tile = rng.choice(desi_tiles["TILEID"]) # Random place to start
unobserved_tileids = desi_tiles["TILEID"]
all_dists = get_tile_distances(first_tile, desi_tiles)

# We remove observed tiles from unobserved tiles so as long as there are
# things in that array we iterate.
while len(unobserved_tileids) > 0:
    print(f"Night {cur_time}, {len(unobserved_tileids)}")
    # Start by taking the closest tile to the starting tile.
    # This "simulates" the progression of the survey, attempting each night
    # to build up a contiguous area of observation.
    start_tile = unobserved_tileids[np.argmin(all_dists)]

    # Get distance to all other tiles from the starting tile.
    # Keep all tiles far enough away they don't overlap and that we haven't
    # observed yet.
    dist = get_tile_distances(start_tile, desi_tiles)
    can_observe = (dist > min_dist) & (np.isin(desi_tiles["TILEID"], unobserved_tileids))

    # Randomly pick which tiles to observe, as many as we can each night up to the tile per night
    # (minus 1 to account for the initial tile)
    tiles_tonight = int(rng.uniform(args.min_per_night, args.max_per_night + 1))
    n_observe = np.min([tiles_tonight - 1, np.sum(can_observe)])
    # will_observe = rng.choice(desi_tiles["TILEID"][can_observe], size=n_observe, replace=False)
    # will_observe = np.concatenate([np.atleast_1d(first_tile), will_observe])

    will_observe = [start_tile]
    while (len(will_observe) <= n_observe) and (np.sum(can_observe) > 0):
        # Sort the currently observable tiles by the distance to the first tile
        # and pick the closest. We need to cute any tileids we already
        # designated for observation or otherwise that one will always get picked.
        potential_tileids = desi_tiles["TILEID"]
        available = ~np.isin(potential_tileids, will_observe)
        potential_tileids = potential_tileids[can_observe & available]
        sort_by_distance = np.argsort(dist[can_observe & available])
        potential_tileids = potential_tileids[sort_by_distance]

        # Get the distance of all tiles to the newly added one. We will
        # then exclude any tiles that overlap the new one
        # so that can_observe is now all tiles that don'y overlap any tiles
        # in will_observe
        next_tile = potential_tileids[0]
        dist_to_new = get_tile_distances(next_tile, desi_tiles)

        will_observe.append(next_tile)
        can_observe = can_observe & (dist_to_new > min_dist)

    print(f"Observing tiles {len(will_observe)} {will_observe}")

    remove_from_unobs = ~np.isin(unobserved_tileids, will_observe)
    unobserved_tileids = unobserved_tileids[remove_from_unobs]
    all_dists = all_dists[remove_from_unobs]
    desi_tiles["TIMESTAMP"][np.isin(desi_tiles["TILEID"], will_observe)] = cur_time.isoformat()

    cur_time = cur_time + timedelta(days=1)

# Update the YMD timestamps used for parallelism
ts = [datetime.fromisoformat(x).strftime("%Y%m%d") for x in desi_tiles["TIMESTAMP"]]
desi_tiles["TIMESTAMP_YMD"] = ts
desi_tiles.sort("TIMESTAMP_YMD") # Sort by timestamp so its in exposure order

desi_tiles.write(args.out, overwrite=True)