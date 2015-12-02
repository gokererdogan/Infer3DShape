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

from hypothesis import *

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
        # no need to copy anything if it is not a partial voxel
        # note that this assumes origin and depth never changes for an empty or filled voxel.
        if self.status == FULL_VOXEL or self.status == EMPTY_VOXEL:
            return self

        subvoxels_copy = np.zeros((VOXELS_PER_AXIS, VOXELS_PER_AXIS, VOXELS_PER_AXIS), dtype=Voxel)
        for i in range(VOXELS_PER_AXIS):
            for j in range(VOXELS_PER_AXIS):
                for k in range(VOXELS_PER_AXIS):
                    subvoxels_copy[i, j, k] = self.subvoxels[i, j, k].copy()

        self_copy = Voxel(origin=self.origin, depth=self.depth, status=PARTIAL_VOXEL, subvoxels=subvoxels_copy)
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

class VoxelBasedShape(Hypothesis):
    """
    VoxelBasedShape class defines a 3D object by splitting the 3D space into voxels and
    representing an object as a list of occupied voxels.
    """
    def __init__(self, viewpoint=None, voxel=None, params=None):
        self.params = params

        # this is the point from which we look at the object.
        # it is a list of 3-tuples, each 3-tuple specifying one viewpoint (x,y,z).
        # if this is not provided, camera_pos defined by forward_model is used.
        self.viewpoint = viewpoint

        self.voxel = voxel
        if self.voxel is None:
            # initialize hypothesis randomly
            self.voxel = Voxel.get_random_voxel(origin=(0.0, 0.0, 0.0), depth=0)

        Hypothesis.__init__(self)

    def prior(self):
        if self.p is None:
            self.p = 1.0 / self.voxel.count_full_voxels()
        return self.p

    def likelihood(self, data):
        if self.ll is None:
            self.ll = self._ll_pixel(data)
        return self.ll

    def convert_to_positions_sizes(self):
        """
        Returns the positions of parts and their sizes.
        Used by VisionForwardModel for rendering.
        :return: positions and sizes of parts
        """
        return self.voxel.to_parts_positions()

    def _ll_pixel_gaussian_filtered(self, data):
        img = self.params['forward_model'].render(self)
        img = spi.gaussian_filter(img, self.params['LL_FILTER_SIGMA'])
        ll = np.exp(-np.sum(np.square(img - data)) / (img.size * self.params['LL_VARIANCE']))
        return ll

    def _ll_pixel(self, data):
        img = self.params['forward_model'].render(self)
        ll = np.exp(-np.sum(np.square((img - data) / self.params['MAX_PIXEL_VALUE']))
                    / (img.size * 2 * self.params['LL_VARIANCE']))
        return ll

    def copy(self):
        """
        Returns a (deep) copy of the instance
        """
        # NOTE that we are not copying params. This assumes that
        # parameters do not change from hypothesis to hypothesis.
        self_copy = VoxelBasedShape(params=self.params)
        voxel_copy = self.voxel.copy()
        viewpoint_copy = deepcopy(self.viewpoint)
        self_copy = VoxelBasedShape(viewpoint=viewpoint_copy, voxel=voxel_copy, params=self.params)
        return self_copy

    def __str__(self):
        return self.voxel.__str__()

    def __repr__(self):
        return self.__str__()

    def __eq__(self, comp):
        return self.voxel == comp.voxel

    def __getstate__(self):
        # we cannot pickle VTKObjects, so get rid of them.
        del self.params['forward_model']
        return self.__dict__

class VoxelBasedShapeProposal(Proposal):
    def __init__(self, params, allow_viewpoint_update=False):
        Proposal.__init__(self)

        self.allow_viewpoint_update = allow_viewpoint_update
        self.params = params

    def propose(self, h, *args):
        # pick one move randomly
        if self.allow_viewpoint_update:
            i = np.random.randint(0, 2)
        else:
            i = np.random.randint(0, 1)

        if i == 0:
            info = "flip voxel"
            hp, q_hp_h, q_h_hp = self.flip_voxel(h)
        elif i == 1:
            info = "change viewpoint"
            hp, q_hp_h, q_h_hp = self.change_viewpoint(h)

        return info, hp, q_hp_h, q_h_hp

    def flip_voxel(self, h):
        hp = h.copy()
        i, j, k = np.random.randint(0, VOXELS_PER_AXIS, (3,))
        origin = hp.voxel.subvoxels[i, j, k].origin
        depth = hp.voxel.subvoxels[i, j, k].depth
        if hp.voxel.subvoxels[i, j, k].status == FULL_VOXEL:
            hp.voxel.subvoxels[i, j, k] = Voxel.get_empty_voxel(origin, depth)
        elif hp.voxel.subvoxels[i, j, k].status == EMPTY_VOXEL:
            hp.voxel.subvoxels[i, j, k] = Voxel.get_full_voxel(origin, depth)

        return hp, 1.0, 1.0

    def change_viewpoint(self, h):
        hp = h.copy()
        # we rotate viewpoint around z axis, keeping the distance to the origin fixed.
        # default viewpoint is (3.0, -3.0, 3.0)
        # add random angle
        change = np.random.randn() * np.sqrt(self.params['CHANGE_VIEWPOINT_VARIANCE'])
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

if __name__ == "__main__":
    import vision_forward_model as vfm
    import mcmc_sampler as mcmc
    fwm = vfm.VisionForwardModel(render_size=(200, 200))
    kernel_params = {'CHANGE_VIEWPOINT_VARIANCE': 60.0}
    kernel = VoxelBasedShapeProposal(params=kernel_params, allow_viewpoint_update=True)

    params = {'forward_model': fwm, 'MAX_PIXEL_VALUE': 175.0, 'LL_VARIANCE': 0.001}
    viewpoint = [(1.5, -1.5, 1.5)]
    # generate initial hypothesis shape randomly
    h = VoxelBasedShape(viewpoint=viewpoint, params=params)

    # read data (i.e., observed image) from disk
    obj_name = 'test2'
    # data = np.load('./data/stimuli20150624_144833/{0:s}.npy'.format(obj_name))
    data = np.load('./data/test2_single_view.npy')

    sampler = mcmc.MHSampler(h, data, kernel, 0, 10, 10, 1000, 1000)
    run = sampler.sample()
    """
    for i, sample in enumerate(run.best_samples.samples):
        fwm.save_render("results/voxel/{0:s}/b{1:d}.png".format(obj_name, i), sample)
    """
