import itertools
import numpy as np

import parameters as params


# COORDINATE TRANSFORMATION FUNCTIONS

def polar_to_cartesian(phi, r):
    """Convert polar coordinates to cartesian coordinates."""
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.array([x, y])


def cartesian_to_polar(x, y):
    """Convert cartesian coordinates to polar coordinates."""
    phi = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    return np.array([phi, r])


def cartesian_to_pixel(x, y):
    """Convert cartesian coordinates to pixel coordinates of world grid.
    
    Since `np.around` rounds values exactly halfway between rounded values
    to the nearest even value, the vertical grid edges are implicitly
    mapped to the incident pixel to the left or right with even 
    x-coordinate and the horizontal grid edges are mapped to the incident
    pixel above or below with even y-coordinate.
    """
    px = np.around(x / params.GRID_H)
    py = np.around(y / params.GRID_H)
    return np.array([px, py], dtype=int)


def pixel_to_cartesian(px, py):
    """Convert pixel coordinates of world grid to cartesian coordinates of pixel center."""
    return np.array([px, py]) * params.GRID_H


def polar_to_pixel(phi, r):
    """Convert polar coordinates to pixel coordinates of world grid."""
    x, y = polar_to_cartesian(phi, r)
    px, py = cartesian_to_pixel(x, y)
    return np.array([px, py], dtype=int)


# OTHER FUNCTIONS

def cartesian_product(*arrays):
    """Compute N-dimensional cartesian product."""
    # https://stackoverflow.com/a/11146645
    arrays = np.asarray(arrays)
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)
    # return np.array(list(itertools.product(*arrays)))


def setdiff2d(a, b):
    """Compute set difference between two 2D lists."""
    # https://stackoverflow.com/a/11903368
    return np.setdiff1d(
        a.copy().view([("x", a.dtype), ("y", a.dtype)]),
        b.copy().view([("x", b.dtype), ("y", b.dtype)]),
    ).view(a.dtype).reshape(-1, 2)
    # a = set(map(tuple, a))
    # b = set(map(tuple, b))
    # return np.array(list(a.difference(b)))


def is_in_range(val, range, mod=None):
    """Check whether value is in the given range while supporting modulos."""
    min, max = range
    if mod is not None:
        min = min % mod
        val = val % mod
        max = max % mod
        if min > max:
            return np.logical_or(val <= max, min <= val)
    return np.logical_and(min <= val, val <= max)


def intersect_functions(f1, f2, mode="left"):
    """Compute list of indices of the intersection points of two functions.
    
    The intersection points are computed by finding the zero-crossings of the
    difference of f1 and f2. Using "left" the point to the left of the
    intersection point is returned and using "right" the point to the right.
    """
    if mode == "left":
        offset = 0
    elif mode == "right":
        offset = 1
    else:
        print("WARNING: unknown mode for intersecting two functions")
    return np.nonzero(np.diff(np.sign(f1 - f2)))[0] + offset