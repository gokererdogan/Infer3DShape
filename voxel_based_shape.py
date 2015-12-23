"""
Inferring 3D Shape from 2D Images

This file contains the VoxelBasedShape hypothesis class.
This hypothesis splits the 3D space into voxels and represents object shape
as a list of occupied voxels. A representation with multiple levels of
resolution is created by allowing each voxel itself to be further split into
a set of voxels, i.e., the space is recursively split into voxels.

Created on Nov 16, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import numpy as np
import scipy.ndimage as spi
from copy import deepcopy

from i3d_hypothesis import *

FILL_PROBABILITY = 0.4

EMPTY_VOXEL = 0
FULL_VOXEL = 1
PARTIAL_VOXEL = 2

VOXELS_PER_AXIS = 4

WIDTH_X = .75
WIDTH_Y = .75
WIDTH_Z = .75


class Voxel(object):
    def __init__(self, origin, depth, status, subvoxels=None):
        self.origin = origin
        self.depth = depth
        self.status = status
        self.subvoxels = subvoxels

    @staticmethod
    def get_random_voxel(origin, depth):
        subvoxels = np.zeros((VOXELS_PER_AXIS, VOXELS_PER_AXIS, VOXELS_PER_AXIS), dtype=Voxel)
        # calculate subvoxel origins
        space_size_x = WIDTH_X / (VOXELS_PER_AXIS**depth)
        space_size_y = WIDTH_Y / (VOXELS_PER_AXIS**depth)
        space_size_z = WIDTH_Z / (VOXELS_PER_AXIS**depth)
        voxel_size_x = WIDTH_X / (VOXELS_PER_AXIS**(depth + 1))
        voxel_size_y = WIDTH_Y / (VOXELS_PER_AXIS**(depth + 1))
        voxel_size_z = WIDTH_Z / (VOXELS_PER_AXIS**(depth + 1))
        minx = origin[0] - (space_size_x / 2.0) + (voxel_size_x / 2.0)
        maxx = origin[0] + (space_size_x / 2.0) - (voxel_size_x / 2.0)
        miny = origin[1] - (space_size_y / 2.0) + (voxel_size_y / 2.0)
        maxy = origin[1] + (space_size_y / 2.0) - (voxel_size_y / 2.0)
        minz = origin[2] - (space_size_z / 2.0) + (voxel_size_z / 2.0)
        maxz = origin[2] + (space_size_z / 2.0) - (voxel_size_z / 2.0)
        voxel_pos_x = np.linspace(start=minx, stop=maxx, num=VOXELS_PER_AXIS)
        voxel_pos_y = np.linspace(start=miny, stop=maxy, num=VOXELS_PER_AXIS)
        voxel_pos_z = np.linspace(start=minz, stop=maxz, num=VOXELS_PER_AXIS)

        for i in range(VOXELS_PER_AXIS):
            for j in range(VOXELS_PER_AXIS):
                for k in range(VOXELS_PER_AXIS):
                    if np.random.rand() < FILL_PROBABILITY:
                        subvoxels[i, j, k] = Voxel.get_full_voxel((voxel_pos_x[i], voxel_pos_y[j], voxel_pos_z[k]),
                                                                  depth + 1)
                    else:
                        subvoxels[i, j, k] = Voxel.get_empty_voxel((voxel_pos_x[i], voxel_pos_y[j], voxel_pos_z[k]),
                                                                   depth + 1)

        return Voxel(origin=origin, depth=depth, status=PARTIAL_VOXEL, subvoxels=subvoxels)

    @staticmethod
    def get_full_voxel(origin, depth):
        return Voxel(origin=origin, depth=depth, status=FULL_VOXEL)

    @staticmethod
    def get_empty_voxel(origin, depth):
        return Voxel(origin=origin, depth=depth, status=EMPTY_VOXEL)

    def to_parts_positions(self):
        if self.status == EMPTY_VOXEL:
            return [], []

        if self.status == FULL_VOXEL:
            voxel_size_x = WIDTH_X / (VOXELS_PER_AXIS**self.depth)
            voxel_size_y = WIDTH_Y / (VOXELS_PER_AXIS**self.depth)
            voxel_size_z = WIDTH_Z / (VOXELS_PER_AXIS**self.depth)
            position = np.array(self.origin)
            size = np.array([voxel_size_x, voxel_size_y, voxel_size_z])
            return [position], [size]

        # partial voxels
        positions = []
        sizes = []
        for i in range(VOXELS_PER_AXIS):
            for j in range(VOXELS_PER_AXIS):
                for k in range(VOXELS_PER_AXIS):
                    p, s = self.subvoxels[i, j, k].to_parts_positions()
                    positions.extend(p)
                    sizes.extend(s)

        return positions, sizes

    def count_full_voxels(self):
        if self.status == FULL_VOXEL:
            return 1
        if self.status == EMPTY_VOXEL:
            return 0
        count = 0
        for i in range(VOXELS_PER_AXIS):
            for j in range(VOXELS_PER_AXIS):
                for k in range(VOXELS_PER_AXIS):
                    count += self.subvoxels[i, j, k].count_full_voxels()

        return count

    def copy(self):
        # NOTE that we assume that the depth of a voxel never changes.
        if self.status == FULL_VOXEL:
            return self.get_full_voxel(origin=deepcopy(self.origin), depth=self.depth)

        if self.status == EMPTY_VOXEL:
            return self.get_empty_voxel(origin=deepcopy(self.origin), depth=self.depth)

        subvoxels_copy = np.zeros((VOXELS_PER_AXIS, VOXELS_PER_AXIS, VOXELS_PER_AXIS), dtype=Voxel)
        for i in range(VOXELS_PER_AXIS):
            for j in range(VOXELS_PER_AXIS):
                for k in range(VOXELS_PER_AXIS):
                    subvoxels_copy[i, j, k] = self.subvoxels[i, j, k].copy()

        self_copy = Voxel(origin=deepcopy(self.origin), depth=self.depth, status=PARTIAL_VOXEL,
                          subvoxels=subvoxels_copy)
        return self_copy

    def __eq__(self, other):
        if self.status != other.status:
            return False

        if self.status == FULL_VOXEL or self.status == EMPTY_VOXEL:
            return self.origin == other.origin and self.depth == other.depth

        for i in range(VOXELS_PER_AXIS):
            for j in range(VOXELS_PER_AXIS):
                for k in range(VOXELS_PER_AXIS):
                    if self.subvoxels[i, j, k] != other.subvoxels[i, j, k]:
                        return False

        return True


class VoxelBasedShape(I3DHypothesis):
    """VoxelBasedShape class defines a 3D object by splitting the 3D space into voxels and
    representing an object as a list of occupied voxels.
    """
    def __init__(self, forward_model, viewpoint=None, params=None, voxel=None):
        I3DHypothesis.__init__(self, forward_model=forward_model, viewpoint=viewpoint, params=params)

        self.voxel = voxel
        if self.voxel is None:
            # initialize hypothesis randomly
            self.voxel = Voxel.get_random_voxel(origin=(0.0, 0.0, 0.0), depth=0)

    def _calculate_log_prior(self):
        return -np.log(self.voxel.count_full_voxels())

    def convert_to_positions_sizes(self):
        """Returns the positions of parts and their sizes.

        Used by VisionForwardModel for rendering.

        Returns:
            (list, list): positions and sizes of parts
        """
        return self.voxel.to_parts_positions()

    def copy(self):
        """Returns a (deep) copy of the instance
        """
        # NOTE that we are not copying params. This assumes that
        # parameters do not change from hypothesis to hypothesis.
        voxel_copy = self.voxel.copy()
        viewpoint_copy = deepcopy(self.viewpoint)
        self_copy = VoxelBasedShape(forward_model=self.forward_model, viewpoint=viewpoint_copy, voxel=voxel_copy,
                                    params=self.params)
        return self_copy

    def __str__(self):
        return self.voxel.__str__()

    def __repr__(self):
        return self.__str__()

    def __eq__(self, comp):
        return self.voxel == comp.voxel


# PROPOSAL FUNCTIONS
def voxel_based_shape_flip_voxel(self, h):
    hp = h.copy()
    i, j, k = np.random.randint(0, VOXELS_PER_AXIS, (3,))
    origin = hp.voxel.subvoxels[i, j, k].origin
    depth = hp.voxel.subvoxels[i, j, k].depth
    if hp.voxel.subvoxels[i, j, k].status == FULL_VOXEL:
        hp.voxel.subvoxels[i, j, k] = Voxel.get_empty_voxel(origin, depth)
    elif hp.voxel.subvoxels[i, j, k].status == EMPTY_VOXEL:
        hp.voxel.subvoxels[i, j, k] = Voxel.get_full_voxel(origin, depth)

    return hp, 1.0, 1.0

if __name__ == "__main__":
    import vision_forward_model as vfm
    import mcmclib.proposal
    import mcmclib.mh_sampler
    import i3d_proposal

    fwm = vfm.VisionForwardModel(render_size=(200, 200))
    h = VoxelBasedShape(forward_model=fwm, viewpoint=[(1.5, -1.5, 1.5)], params={'LL_VARIANCE': 0.001})

    moves = {'voxel_flip_voxel': voxel_based_shape_flip_voxel, 'change_viewpoint': i3d_proposal.change_viewpoint}

    params = {'CHANGE_VIEWPOINT_VARIANCE': 300.0}

    proposal = mcmclib.proposal.RandomMixtureProposal(moves, params)

    data = np.load('data/test1_single_view.npy')

    sampler = mcmclib.mh_sampler.MHSampler(h, data, proposal, burn_in=1000, sample_count=10, best_sample_count=10,
                                           thinning_period=2000, report_period=2000)

    run = sampler.sample()
