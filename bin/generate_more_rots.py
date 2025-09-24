#!/usr/bin/env python

# Using the rotations defined in DESI-0717 interpolate a grid
# and generate new additional rotations as knots in the grid.
# TODO proper docstring

import numpy as np

# Rotations for first through 15th passes of the sky of the DESI tiling

rots = np.array([[0, 0],
                 [2.56349, 1.70645],
                 [5.30251, 0.46050],
                 [2.77437, -1.24945],
                 [5.23554, 3.36665],
                 [2.54094, 4.69353],
                 [5.14642, 6.23672],
                 [-0.09047, 3.16649],
                 [0.11911, -2.83846],
                 [-2.51911, -1.80809],
                 [7.72129, 4.96644],
                 [-2.58180, 1.36959],
                 [2.68461, -4.42267],
                 [-0.03550, 6.49827],
                 [2.81603, 8.03110]])

# Determine the rough vertical spacing of the grid by finding the
# vertical lines that best fit each set of vertical data.
# Array in the for loop is an initial "best guess"
ra_lines = []
for x in [-2.5, 0, 2.5, 5.2, 7.6]:
    close = (rots[:, 0] > (x - 0.2)) & (rots[:, 0] < (x + 0.2))
    ra_lines.append(np.mean(rots[close, 0]))

# Expand the grid of constant right ascension by adding an additionl
# line on either side of the currently defined ones.
deltas = np.diff(ra_lines)
ra_delta = np.mean(deltas)
ra_lines = np.concatenate([np.atleast_1d(ra_lines[0] - ra_delta), ra_lines, np.atleast_1d(ra_lines[-1] + ra_delta)])

# Best guess slopes and intercepts that bracket the current rotations, i.e.
# the grid line defining the rotation grid should fall between two
# slopes/intercepts of this bracketing.
intercepts = [-4, -1, 1, 5, 9, 11]
slopes = [-0.7] * len(intercepts)

# Actually fit each of the grid lines
fits = []
for i in range(len(slopes) - 1):
    m_l = slopes[i]
    b_l = intercepts[i]
    above_lower = rots[:, 1] > (rots[:, 0] * m_l + b_l)

    m_u = slopes[i + 1]
    b_u = intercepts[i + 1]
    below_upper = rots[:, 1] < (rots[:, 0] * m_u + b_u)

    fit_points = above_lower & below_upper

    X = np.hstack([rots[fit_points, 0][:, None], np.ones(np.sum(fit_points))[:, None]])
    a, *_ = np.linalg.lstsq(X, rots[fit_points, 1])
    fits.append(a)

fits = np.array(fits)

# Add one additional grid line above and below the previous ones using the mean slope
# and average change in intercept.
mean_slope = np.mean(fits[:, 0])
delta_intercept = np.mean(np.diff(fits[:, 1]))
fits_expanded = np.vstack([[mean_slope, fits[0, 1] - delta_intercept], fits, [mean_slope, fits[-1, 1] + delta_intercept]])

# Generate the knots of the grid, which is easy because the lines of constant
# RA are vertical, so the intersection point is just the diagonal line
# evaluated at that point.
knots = []
for ra_x in ra_lines:
    for fit in fits_expanded:
        knots.append([ra_x, fit[0] * ra_x + fit[1]])

knots = np.array(knots)

# We only want to keep the new rotations, so we iterate over the old ones
# and remove whichever knot is closest to that rotation from the
# generated knots.
keep = np.ones(knots.shape[0], dtype=bool)
for rot in rots:
    dists = np.linalg.norm(knots - rot, axis=1)
    keep[np.argmin(dists)] = False

# Finally we sort the new rotations by their linear distance from
# 0,0 which is a good analog for "how much to rotate by" (remember that
# technically these points are in ra, dec space). There's no
# real motivation to do this except for earlier tilings to have less of an
# offset than later tilings.
new_rots = knots[keep]
idcs_sort = np.argsort(np.linalg.norm(new_rots, axis=1))
new_rots = new_rots[idcs_sort]

print(new_rots)