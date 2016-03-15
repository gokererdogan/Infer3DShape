"""
Inferring 3D Shape from 2D Images

This file contains implementations of various likelihood functions.

Created on Dec 2, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""
import numpy as np


def change_viewpoint_z(h, params):
    """Propose a new hypothesis by changing viewpoint around the z axis.

    Parameters:
        h (I3DHypothesis): hypothesis
        params (dict): proposal parameters.
            CHANGE_VIEWPOINT_VARIANCE specifies the inverse kappa for the von Mises distribution.

    Returns:
        I3DHypothesis: new hypothesis
    """
    hp = h.copy()
    # we rotate viewpoint around z axis, keeping the distance to the origin fixed.
    # convert from degrees to radians
    kappa = 1 / (params['CHANGE_VIEWPOINT_VARIANCE'] * np.pi**2 / 180**2)
    # add random angle
    change = np.random.vonmises(0.0, kappa) * 180.0 / np.pi
    for i in range(len(hp.viewpoint)):
        hp.viewpoint[i][1] += change

    return hp, 1.0, 1.0


def change_viewpoint(h, params):
    """Propose a new hypothesis by changing viewpoint.

    Viewpoint can move in any direction in 3D space. Note that we assume that viewpoint is simply a point on a 3D
    sphere of given radius from which we look at the object. In other words, one can think of viewpoint as the position
    of the camera oriented towards the object at the origin. We assume that the camera up always points in the z
    direction. This is taken care of in the `vision_forward_model` module.

    Parameters:
        h (I3DHypothesis): hypothesis
        params (dict): proposal parameters.
            CHANGE_VIEWPOINT_VARIANCE specifies the inverse kappa for the von Mises distribution.

    Returns:
        I3DHypothesis: new hypothesis
    """
    hp = h.copy()
    # convert from degrees to radians
    kappa = 1 / (params['CHANGE_VIEWPOINT_VARIANCE'] * np.pi**2 / 180**2)
    change = np.random.vonmises(0.0, kappa, 2) * 180.0 / np.pi
    for i in range(len(hp.viewpoint)):
        hp.viewpoint[i][1] += change[0]
        hp.viewpoint[i][2] += change[1]

    return hp, 1.0, 1.0

