"""
Inferring 3D Shape from 2D Images

This file contains implementations of various likelihood functions.

Created on Dec 2, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""
import numpy as np

# default values for various proposal parameters
MOVE_PART_VARIANCE = .005
MOVE_OBJECT_VARIANCE = 0.1
CHANGE_SIZE_VARIANCE = .005
CHANGE_VIEWPOINT_VARIANCE = 60.0 # in degrees

def change_viewpoint(h, params):
    """Propose a new hypothesis by changing viewpoint

    Args:
        h (I3DHypothesis): hypothesis
        params (dict): proposal parameters

    Returns:
        I3DHypothesis: new hypothesis
    """
    hp = h.copy()
    # we rotate viewpoint around z axis, keeping the distance to the origin fixed.
    # default viewpoint is (1.5, -1.5, 1.5)
    # add random angle
    change = np.random.randn() * np.sqrt(params['CHANGE_VIEWPOINT_VARIANCE'])
    for i, viewpoint in enumerate(hp.viewpoint):
        x = viewpoint[0]
        y = viewpoint[1]
        z = viewpoint[2]
        d = np.sqrt(x**2 + y**2)
        # calculate angle
        angle = ((180.0 * np.arctan2(y, x) / np.pi) + 360.0) % 360.0
        angle = (angle + change) % 360.0
        nx = d * np.cos(angle * np.pi / 180.0)
        ny = d * np.sin(angle * np.pi / 180.0)
        hp.viewpoint[i] = (nx, ny, z)

    return hp, 1.0, 1.0
