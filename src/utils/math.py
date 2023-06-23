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
