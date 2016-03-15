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
from scipy.misc import comb
from copy import deepcopy

from i3d_hypothesis import *

FILL_PROBABILITY = 0.4

EMPTY_VOXEL = 0
FULL_VOXEL = 1
PARTIAL_VOXEL = 2

VOXELS_PER_AXIS = 3

WIDTH_X = 1.5
WIDTH_Y = 1.5
WIDTH_Z = 1.5


# helper functions for calculating prior probability
def get_seqs_with_sum(total, max_allowed=VOXELS_PER_AXIS**3, length=1000):
    """Returns all sequences of natural numbers in [0, max_allowed] with a given sum and number of elements.

    This function is used for calculating the possible number of children partial voxels each partial voxel at one level
    has. For example, if we want to calculate the number of possible voxel representations with k partial voxels at
    a particular level and T partial voxels in total, we need to find how many children partial voxels each of the k
    partial voxels can have. We can get all possible assignments of number of children voxels to k partial nodes with
    in total T partial voxels (not including the k voxels) with get_seqs_with_sum(T, length=k). For example, assume we
    have 2 voxels at one level and we know there should be in total 3 children partial voxels of these two nodes.
    Possible assignments are: [0, 3], [1, 2], [2, 1], [3, 0] where the first element is the number of children partial
    voxels of first voxel, and the second element is second voxel's number of chilren partial voxels.

    Parameters:
        total (int): the desired total of the sequence. must be positive.
        max_allowed (int): maximum number allowed in the sequence. must be positive.
        (int): Length of the sequence.

    Returns:
        list: a list of sequences with sum ``total``.
    """
    if length == 1:
        if total > max_allowed:
            return [None]
        else:
            return [[total]]
    seqs = []
    for i in range(0, min(total, max_allowed) + 1):
        seq = []
        for s in get_seqs_with_sum(total-i, max_allowed, length-1):
            if s is not None:
                ns = [i] + s
                seq.append(ns)
        seqs.extend(seq)
    return seqs


# Precalculated tree counts for speed
TREE_COUNTS = {(-1, 8, 0, np.inf): 2,
               (0, 8, 0, np.inf): 256,
               (1, 8, 0, np.inf): 262144.0,
               (2, 8, 0, np.inf): 385875968.0,
               (3, 8, 0, np.inf): 665719930880.0,
               (4, 8, 0, np.inf): 1256054595780608.0,
               (5, 8, 0, np.inf): 2.5102642110498079e+18,
               (6, 8, 0, np.inf): 5.2223155811036713e+21,
               (7, 8, 0, np.inf): 1.1190836043230874e+25,
               (8, 8, 0, np.inf): 2.4532153481140168e+28,
               (9, 8, 0, np.inf): 5.4755909543239472e+31,
               (10, 8, 0, np.inf): 6.7945621356541407e+34,
               (11, 8, 0, np.inf): 1.0707151551195507e+38,
               (12, 8, 0, np.inf): 1.8278400554429342e+41,
               (13, 8, 0, np.inf): 3.2357020581380588e+44,
               (14, 8, 0, np.inf): 5.815853901982639e+47,
               (15, 8, 0, np.inf): 1.0456261097839637e+51,
               (16, 8, 0, np.inf): 1.8524461622795484e+54}


def count_trees(partial_count, max_voxels=VOXELS_PER_AXIS**3, depth=0, max_depth=np.inf):
    """This function calculates the number of possible voxel representations with a given number of partial voxel count
    and maximum depth. This is necessary for calculating the prior probability of a hypothesis.

    Parameters:
        partial_count (int): Number of partial voxels in representation. NOTE that we do not count the root partial node
            here.
        max_voxels (int): Maximum number of voxels at one level (i.e., the branching factor of the tree)
        depth (int): The current depth of the tree. This is used internally. No need to supply a value.
        max_depth (int): Maximum depth allowed for representations.

    Returns:
        int: The number of possible trees (i.e., voxel representations) with a given number of partial voxel count and
        maximum depth.
    """
    # get from precalculated TREE_COUNTS
    key = (partial_count, max_voxels, depth, max_depth)
    if max_depth == np.inf:  # we don't care what current depth is
        key = (partial_count, max_voxels, 0, max_depth)
    if partial_count < (max_depth - depth):  # we don't care what max_depth is
        key = (partial_count, max_voxels, 0, np.inf)
    if key in TREE_COUNTS:
        return TREE_COUNTS[key]

    # calculate it from scratch
    if depth >= max_depth:
        return 0
    if partial_count == -1:
        # there are two trees with no P node.
        return 2
    if partial_count == 0:
        return 2**max_voxels
    count = 0
    for i in range(1, min(partial_count, max_voxels) + 1):
        count_i = 0
        seqs = get_seqs_with_sum(partial_count - i, max_allowed=max_voxels, length=i)
        for seq in seqs:
            if seq is not None:
                c = 1
                for s in seq:
                    c *= count_trees(partial_count=s, max_voxels=max_voxels, depth=depth+1, max_depth=max_depth)
                count_i += c
        count_i *= (2**(max_voxels-i) * comb(max_voxels, i))
        count += count_i
    # cache the results for future.
    TREE_COUNTS[(partial_count, max_voxels, depth, max_depth)] = count
    return count


class Voxel(object):
    def __init__(self, origin, depth, status, voxels_per_axis=VOXELS_PER_AXIS, size=(WIDTH_X, WIDTH_Y, WIDTH_Z),
                 subvoxels=None):
        self.origin = origin
        self.depth = depth
        self.status = status
        self.voxels_per_axis = voxels_per_axis
        self.size = size
        self.subvoxels = subvoxels

    @staticmethod
    def get_random_subvoxels(origin, depth, voxels_per_axis=VOXELS_PER_AXIS, size=(WIDTH_X, WIDTH_Y, WIDTH_Z)):
        subvoxels = np.zeros((voxels_per_axis, voxels_per_axis, voxels_per_axis), dtype=Voxel)
        # calculate subvoxel origins
        space_size_x = size[0] / (voxels_per_axis**depth)
        space_size_y = size[1] / (voxels_per_axis**depth)
        space_size_z = size[2] / (voxels_per_axis**depth)
        voxel_size_x = size[0] / (voxels_per_axis**(depth + 1))
        voxel_size_y = size[1] / (voxels_per_axis**(depth + 1))
        voxel_size_z = size[2] / (voxels_per_axis**(depth + 1))
        minx = origin[0] - (space_size_x / 2.0) + (voxel_size_x / 2.0)
        maxx = origin[0] + (space_size_x / 2.0) - (voxel_size_x / 2.0)
        miny = origin[1] - (space_size_y / 2.0) + (voxel_size_y / 2.0)
        maxy = origin[1] + (space_size_y / 2.0) - (voxel_size_y / 2.0)
        minz = origin[2] - (space_size_z / 2.0) + (voxel_size_z / 2.0)
        maxz = origin[2] + (space_size_z / 2.0) - (voxel_size_z / 2.0)
        voxel_pos_x = np.linspace(start=minx, stop=maxx, num=voxels_per_axis)
        voxel_pos_y = np.linspace(start=miny, stop=maxy, num=voxels_per_axis)
        voxel_pos_z = np.linspace(start=minz, stop=maxz, num=voxels_per_axis)

        for i in range(voxels_per_axis):
            for j in range(voxels_per_axis):
                for k in range(voxels_per_axis):
                    if np.random.rand() < FILL_PROBABILITY:
                        subvoxels[i, j, k] = Voxel.get_full_voxel((voxel_pos_x[i], voxel_pos_y[j], voxel_pos_z[k]),
                                                                  depth + 1, voxels_per_axis=voxels_per_axis,
                                                                  size=size)
                    else:
                        subvoxels[i, j, k] = Voxel.get_empty_voxel((voxel_pos_x[i], voxel_pos_y[j], voxel_pos_z[k]),
                                                                   depth + 1, voxels_per_axis=voxels_per_axis,
                                                                   size=size)

        return subvoxels

    @staticmethod
    def get_random_voxel(origin, depth, voxels_per_axis=VOXELS_PER_AXIS, size=(WIDTH_X, WIDTH_Y, WIDTH_Z)):
        subvoxels = Voxel.get_random_subvoxels(origin, depth, voxels_per_axis=voxels_per_axis, size=size)
        return Voxel(origin=origin, depth=depth, status=PARTIAL_VOXEL, subvoxels=subvoxels,
                     voxels_per_axis=voxels_per_axis, size=size)

    @staticmethod
    def get_full_voxel(origin, depth, voxels_per_axis=VOXELS_PER_AXIS, size=(WIDTH_X, WIDTH_Y, WIDTH_Z)):
        return Voxel(origin=origin, depth=depth, status=FULL_VOXEL, voxels_per_axis=voxels_per_axis, size=size)

    @staticmethod
    def get_empty_voxel(origin, depth, voxels_per_axis=VOXELS_PER_AXIS, size=(WIDTH_X, WIDTH_Y, WIDTH_Z)):
        return Voxel(origin=origin, depth=depth, status=EMPTY_VOXEL, voxels_per_axis=voxels_per_axis, size=size)

    def to_parts_positions(self):
        if self.status == EMPTY_VOXEL:
            return [], []

        if self.status == FULL_VOXEL:
            voxel_size_x = self.size[0] / (self.voxels_per_axis**self.depth)
            voxel_size_y = self.size[1] / (self.voxels_per_axis**self.depth)
            voxel_size_z = self.size[2] / (self.voxels_per_axis**self.depth)
            position = np.array(self.origin)
            size = np.array([voxel_size_x, voxel_size_y, voxel_size_z])
            return [position], [size]

        # partial voxels
        positions = []
        sizes = []
        for i in range(self.voxels_per_axis):
            for j in range(self.voxels_per_axis):
                for k in range(self.voxels_per_axis):
                    p, s = self.subvoxels[i, j, k].to_parts_positions()
                    positions.extend(p)
                    sizes.extend(s)

        return positions, sizes

    def get_voxels_by_status(self, status):
        """Returns a list of voxels having a given status.

        Args:
            status (int): FULL, EMPTY or PARTIAL

        Returns:
            list:  A list of voxels with given status.
        """
        voxels = []
        if status == self.status:
            voxels.append(self)

        # partial voxel
        if self.status == PARTIAL_VOXEL:
            for i in range(self.voxels_per_axis):
                for j in range(self.voxels_per_axis):
                    for k in range(self.voxels_per_axis):
                        voxels.extend(self.subvoxels[i, j, k].get_voxels_by_status(status))

        return voxels

    def count_voxels_by_status(self, status):
        count = 0
        if self.status == status:
            count = 1

        if self.status == PARTIAL_VOXEL:
            for i in range(self.voxels_per_axis):
                for j in range(self.voxels_per_axis):
                    for k in range(self.voxels_per_axis):
                        count += self.subvoxels[i, j, k].count_voxels_by_status(status)

        return count

    def calculate_depth(self):
        if self.status == FULL_VOXEL or self.status == EMPTY_VOXEL:
            return self.depth

        depths = []
        for i in range(self.voxels_per_axis):
            for j in range(self.voxels_per_axis):
                for k in range(self.voxels_per_axis):
                    depths.append(self.subvoxels[i, j, k].calculate_depth())

        return max(depths)

    def copy(self):
        if self.status == FULL_VOXEL:
            return self.get_full_voxel(origin=deepcopy(self.origin), depth=self.depth,
                                       voxels_per_axis=self.voxels_per_axis, size=self.size)

        if self.status == EMPTY_VOXEL:
            return self.get_empty_voxel(origin=deepcopy(self.origin), depth=self.depth,
                                        voxels_per_axis=self.voxels_per_axis, size=self.size)

        subvoxels_copy = np.zeros((self.voxels_per_axis, self.voxels_per_axis, self.voxels_per_axis), dtype=Voxel)
        for i in range(self.voxels_per_axis):
            for j in range(self.voxels_per_axis):
                for k in range(self.voxels_per_axis):
                    subvoxels_copy[i, j, k] = self.subvoxels[i, j, k].copy()

        self_copy = Voxel(origin=deepcopy(self.origin), depth=self.depth, status=PARTIAL_VOXEL,
                          subvoxels=subvoxels_copy, voxels_per_axis=self.voxels_per_axis, size=self.size)
        return self_copy

    def __eq__(self, other):
        if self.voxels_per_axis != other.voxels_per_axis:
            return False

        if self.status != other.status:
            return False

        if self.status == FULL_VOXEL or self.status == EMPTY_VOXEL:
            return self.origin == other.origin and self.depth == other.depth

        for i in range(self.voxels_per_axis):
            for j in range(self.voxels_per_axis):
                for k in range(self.voxels_per_axis):
                    if self.subvoxels[i, j, k] != other.subvoxels[i, j, k]:
                        return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        if self.status == PARTIAL_VOXEL:
            s = ""
            for j in range(self.voxels_per_axis):
                s += "|"
                for i in range(self.voxels_per_axis):
                    for k in range(self.voxels_per_axis):
                        s += " {0:d} ".format(self.subvoxels[i, j, k].status)
                    s += " | "
                s += "\n"
        else:
            s = "| {0:d} at {1:s}, d={2:d} |".format(self.status, self.origin, self.depth)

        return s


class VoxelBasedShape(I3DHypothesis):
    """VoxelBasedShape class defines a 3D object by splitting the 3D space into voxels and representing an object
    as a tree of occupied voxels.
    """
    def __init__(self, forward_model, viewpoint=None, params=None, voxel=None,
                 scale=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), voxels_per_axis=VOXELS_PER_AXIS,
                 size=(WIDTH_X, WIDTH_Y, WIDTH_Z)):
        I3DHypothesis.__init__(self, forward_model=forward_model, viewpoint=viewpoint, params=params)

        self.scale = np.array(scale)
        self.origin = np.array(origin)
        self.voxel = voxel
        if self.voxel is None:
            # initialize hypothesis randomly
            self.voxel = Voxel.get_random_voxel(origin=(0.0, 0.0, 0.0), depth=0, voxels_per_axis=voxels_per_axis,
                                                size=size)
        self.depth = self._calculate_depth()

    def _calculate_depth(self):
        """Calculate the depth of the representation.

        The depth of a hypothesis is the maximum number of times that a voxel is split into further subvoxels. This
        corresponds to the complexity (i.e., maximum resolution) of the representation.

        Returns
            int: depth of hypothesis
        """
        return self.voxel.calculate_depth()

    def update_depth(self):
        self.depth = self._calculate_depth()
        # prior depends on depth. So, update prior as well.
        self._log_p = self._calculate_log_prior()

    def _calculate_log_prior(self):
        # we want to impose a prior over the number of partial voxels.
        # p(H has |P| partial voxels) = 1 / k^|P| where |P| is the number of partial voxels in H
        # and k>1 is some constant we choose.
        # In order to calculate that prior, we need the number of H with a given number of partial voxels (because we
        # want the total probability mass for them to be proportional to 1 / k^|P|. We do that using the count_trees
        # function.
        # Then the probability of a H with |P| parts is 1 / (T_|P| k^|P|) where T_|P| is the number of H with |P|
        # partial voxels.
        p = self.voxel.count_voxels_by_status(PARTIAL_VOXEL) - 1
        v = self.voxel.voxels_per_axis**3
        return -np.log(count_trees(partial_count=p, max_voxels=v)) - ((p + 1) * np.log(2.0))

    def convert_to_positions_sizes(self):
        """Returns the positions of parts and their sizes.

        Used by VisionForwardModel for rendering.

        Returns:
            (list, list): positions and sizes of parts
        """
        positions, sizes = self.voxel.to_parts_positions()
        for i in range(len(positions)):
            positions[i] = (positions[i] * self.scale) + self.origin
            sizes[i] = sizes[i] * self.scale

        return positions, sizes

    def copy(self):
        """Returns a (deep) copy of the instance
        """
        # NOTE that we are not copying params. This assumes that
        # parameters do not change from hypothesis to hypothesis.
        voxel_copy = self.voxel.copy()
        viewpoint_copy = deepcopy(self.viewpoint)
        scale_copy = deepcopy(self.scale)
        origin_copy = deepcopy(self.origin)
        self_copy = VoxelBasedShape(forward_model=self.forward_model, viewpoint=viewpoint_copy, voxel=voxel_copy,
                                    scale=scale_copy, origin=origin_copy, params=self.params,
                                    voxels_per_axis=self.voxel.voxels_per_axis, size=self.voxel.size)
        return self_copy

    def __str__(self):
        return self.voxel.__str__()

    def __repr__(self):
        return self.__str__()

    def __eq__(self, comp):
        return np.sum(np.abs(self.scale - comp.scale)) < 1e-6 \
               and np.sum(np.abs(self.origin - comp.origin)) < 1e-6 \
               and self.voxel == comp.voxel

    def __ne__(self, other):
        return not self.__eq__(other)


class VoxelBasedShapeMaxD(VoxelBasedShape):
    """VoxelBasedShapeMaxD implements a maximum depth constraint on VoxelBasedShape hypothesis.
    """
    def __init__(self, forward_model, viewpoint=None, params=None, voxel=None, scale=(1.0, 1.0, 1.0),
                 origin=(0.0, 0.0, 0.0), voxels_per_axis=VOXELS_PER_AXIS, size=(WIDTH_X, WIDTH_Y, WIDTH_Z),
                 max_depth=3):
        VoxelBasedShape.__init__(self, forward_model=forward_model, viewpoint=viewpoint, params=params, voxel=voxel,
                                 scale=scale, origin=origin, voxels_per_axis=voxels_per_axis, size=size)

        self.max_depth = max_depth
        if self.depth > self.max_depth:
            raise ValueError("Initial voxel cannot have more than the maximum number of "
                             "levels {0:d}.".format(self.max_depth))

    def _calculate_log_prior(self):
        # we assume the same prior with VoxelBasedShape
        # don't count the root partial voxel (count_trees assumes this)
        p = self.voxel.count_voxels_by_status(PARTIAL_VOXEL) - 1
        # we don't want to count the trees every time we calculate the log prior. this is too costly for representations
        # with many voxels. we will use a rough estimate of how fast the number of trees grow with the number of
        # partial voxels to calculate the prior.
        # for VOXELS_PER_AXIS=2, the minimum increase in tree count = 128, maximum increase ~ 2232 (these are calculated
        # from count_trees function)
        # for VOXELS_PER_AXIS=3, minimum increase = 67108864, maximum increase ~ 4547675962
        if self.voxel.voxels_per_axis == 2:
            # return -((p + 1) * np.log(128)) - ((p + 1) * np.log(2.0))
            return -((p + 1) * np.log(2232)) - ((p + 1) * np.log(2.0))
        elif self.voxel.voxels_per_axis == 3:
            # return -((p + 1) * np.log(67108864))
            return -((p + 1) * np.log(4547675962)) - ((p + 1) * np.log(2.0))
        else:
            v = self.voxel.voxels_per_axis**3
            return -np.log(count_trees(partial_count=p, max_voxels=v, max_depth=self.max_depth)) - ((p+1) * np.log(2.0))

    def copy(self):
        voxel_copy = self.voxel.copy()
        scale_copy = deepcopy(self.scale)
        origin_copy = deepcopy(self.origin)
        viewpoint_copy = deepcopy(self.viewpoint)
        self_copy = VoxelBasedShapeMaxD(forward_model=self.forward_model, viewpoint=viewpoint_copy, voxel=voxel_copy,
                                        scale=scale_copy, origin=origin_copy, params=self.params,
                                        voxels_per_axis=self.voxel.voxels_per_axis, size=self.voxel.size,
                                        max_depth=self.max_depth)
        return self_copy


# PROPOSAL FUNCTIONS


def voxel_based_shape_flip_full_vs_empty(h, params):
    max_depth = np.inf
    if params is not None and 'MAX_DEPTH' in params.keys():
        max_depth = params['MAX_DEPTH']

    if h.depth > max_depth:
        raise ValueError("voxel_flip expects shape hypothesis with depth less than {0:d}.".format(max_depth))

    hp = h.copy()
    full_voxels = hp.voxel.get_voxels_by_status(FULL_VOXEL)
    empty_voxels = hp.voxel.get_voxels_by_status(EMPTY_VOXEL)

    choices = [0, 1]
    if len(full_voxels) == 0:
        choices.remove(0)

    if len(empty_voxels) == 0:
        choices.remove(1)

    if len(choices) == 0:
        raise RuntimeError("A voxel cannot consist of all partial voxels.")

    move_choice = np.random.choice(choices)
    q_hp_h = 1.0
    q_h_hp = 1.0
    if move_choice == 0:  # full to empty
        voxel = np.random.choice(full_voxels)
        voxel.status = EMPTY_VOXEL

        p_forward = 0.5
        # if all voxels were full, only possible move is full to empty.
        if len(empty_voxels) == 0:
            p_forward = 1.0
        q_hp_h = p_forward * (1.0 / len(full_voxels))

        p_backward = 0.5
        # if there was only a single full voxel, only possible backward move is empty to full
        if len(full_voxels) == 1:
            p_backward = 1.0
        q_h_hp = p_backward * (1.0 / (len(empty_voxels) + 1))

    elif move_choice == 1:  # empty to full
        voxel = np.random.choice(empty_voxels)
        voxel.status = FULL_VOXEL

        p_forward = 0.5
        # if all voxels were empty, only possible move is empty to full.
        if len(full_voxels) == 0:
            p_forward = 1.0
        q_hp_h = p_forward * (1.0 / len(empty_voxels))

        p_backward = 0.5
        # if there was only a single empty voxel, only possible backward move is full to empty
        if len(empty_voxels) == 1:
            p_backward = 1.0
        q_h_hp = p_backward * (1.0 / (len(full_voxels) + 1))

    return hp, q_hp_h, q_h_hp


def voxel_based_shape_flip_full_vs_partial(h, params):
    max_depth = np.inf
    if params is not None and 'MAX_DEPTH' in params.keys():
        max_depth = params['MAX_DEPTH']

    if h.depth > max_depth:
        raise ValueError("voxel_flip expects shape hypothesis with depth less than {0:d}.".format(max_depth))

    # for this move to be reversible, we need to constrain the full<->partial changes to partial voxels with no partial
    # children because when we are generating a new partial voxel we only generate a partial voxel with no partial
    # children.
    hp = h.copy()
    full_voxels = [v for v in hp.voxel.get_voxels_by_status(FULL_VOXEL) if v.depth < max_depth]
    # a partial voxel has no partial children if itself is the only partial voxel we find if we start traversing from
    # that voxel.
    partial_voxels = [v for v in hp.voxel.get_voxels_by_status(PARTIAL_VOXEL)
                      if v.count_voxels_by_status(PARTIAL_VOXEL) == 1]

    choices = [0, 1]
    if len(full_voxels) == 0:
        choices.remove(0)

    if len(partial_voxels) == 0:
        choices.remove(1)

    if len(choices) == 0:
        return hp, 1.0, 1.0

    move_choice = np.random.choice(choices)
    q_hp_h = 1.0
    q_h_hp = 1.0
    if move_choice == 0:  # full to partial
        voxel = np.random.choice(full_voxels)
        voxel.status = PARTIAL_VOXEL
        voxel.subvoxels = Voxel.get_random_subvoxels(voxel.origin, voxel.depth, voxels_per_axis=voxel.voxels_per_axis,
                                                     size=voxel.size)

        hp_full_voxels = [v for v in hp.voxel.get_voxels_by_status(FULL_VOXEL) if v.depth < max_depth]
        hp_partial_voxels = [v for v in hp.voxel.get_voxels_by_status(PARTIAL_VOXEL)
                             if v.count_voxels_by_status(PARTIAL_VOXEL) == 1]

        p_forward = 0.5
        # if all voxels were full, only possible move is full to partial.
        if len(partial_voxels) == 0:
            p_forward = 1.0
        # q(h -> hp) = p(full to partial) p(picking voxel) p(new partial voxel)
        q_hp_h = p_forward * (1.0 / len(full_voxels)) * (1.0 / (2**(voxel.voxels_per_axis**3)))

        p_backward = 0.5
        # if there is no full voxel in new hypothesis, only possible backward move is partial to full
        if len(hp_full_voxels) == 0:
            p_backward = 1.0
        # q(hp -> h) = p(partial to full) p(picking voxel)
        # Note that p(new full voxel) = 1, in contrast to p(new partial voxel) which is assumed
        # to be 1 / (number of voxels).
        q_h_hp = p_backward * (1.0 / len(hp_partial_voxels))

    elif move_choice == 1:  # partial to full
        voxel = np.random.choice(partial_voxels)
        voxel.status = FULL_VOXEL

        hp_full_voxels = [v for v in hp.voxel.get_voxels_by_status(FULL_VOXEL) if v.depth < max_depth]
        hp_partial_voxels = [v for v in hp.voxel.get_voxels_by_status(PARTIAL_VOXEL)
                             if v.count_voxels_by_status(PARTIAL_VOXEL) == 1]

        p_forward = 0.5
        # if all voxels were partial, only possible move is partial to full.
        if len(full_voxels) == 0:
            p_forward = 1.0
        q_hp_h = p_forward * (1.0 / len(partial_voxels))

        p_backward = 0.5
        # if there is no full partial voxel in new hypothesis, only possible backward move is full to partial
        if len(hp_partial_voxels) == 0:
            p_backward = 1.0
        q_h_hp = p_backward * (1.0 / len(hp_full_voxels)) * (1.0 / (2**(voxel.voxels_per_axis**3)))

    # depth of hypothesis might have changed. update it
    hp.update_depth()
    return hp, q_hp_h, q_h_hp


def voxel_based_shape_flip_empty_vs_partial(h, params):
    max_depth = np.inf
    if params is not None and 'MAX_DEPTH' in params.keys():
        max_depth = params['MAX_DEPTH']

    if h.depth > max_depth:
        raise ValueError("voxel_flip expects shape hypothesis with depth less than {0:d}.".format(max_depth))

    # for this move to be reversible, we need to constrain the full<->partial changes to partial voxels with no partial
    # children because when we are generating a new partial voxel we only generate a partial voxel with no partial
    # children.
    hp = h.copy()
    empty_voxels = [v for v in hp.voxel.get_voxels_by_status(EMPTY_VOXEL) if v.depth < max_depth]
    partial_voxels = [v for v in hp.voxel.get_voxels_by_status(PARTIAL_VOXEL)
                      if v.count_voxels_by_status(PARTIAL_VOXEL) == 1]

    choices = [0, 1]
    if len(empty_voxels) == 0:
        choices.remove(0)

    if len(partial_voxels) == 0:
        choices.remove(1)

    if len(choices) == 0:
        return hp, 1.0, 1.0

    move_choice = np.random.choice(choices)
    q_hp_h = 1.0
    q_h_hp = 1.0
    if move_choice == 0:  # empty to partial
        voxel = np.random.choice(empty_voxels)
        voxel.status = PARTIAL_VOXEL
        voxel.subvoxels = Voxel.get_random_subvoxels(voxel.origin, voxel.depth, voxels_per_axis=voxel.voxels_per_axis,
                                                     size=voxel.size)

        hp_empty_voxels = [v for v in hp.voxel.get_voxels_by_status(EMPTY_VOXEL) if v.depth < max_depth]
        hp_partial_voxels = [v for v in hp.voxel.get_voxels_by_status(PARTIAL_VOXEL)
                             if v.count_voxels_by_status(PARTIAL_VOXEL) == 1]

        p_forward = 0.5
        # if all voxels were empty, only possible move is empty to partial.
        if len(partial_voxels) == 0:
            p_forward = 1.0
        # q(h -> hp) = p(empty to partial) p(picking voxel) p(new partial voxel)
        q_hp_h = p_forward * (1.0 / len(empty_voxels)) * (1.0 / (2**(voxel.voxels_per_axis**3)))

        p_backward = 0.5
        # if there is no empty voxel in new hypothesis, only possible backward move is partial to empty
        if len(hp_empty_voxels) == 0:
            p_backward = 1.0
        # q(hp -> h) = p(partial to empty) p(picking voxel)
        # Note that p(new empty voxel) = 1, in contrast to p(new partial voxel) which is assumed
        # to be 1 / (number of voxels).
        q_h_hp = p_backward * (1.0 / len(hp_partial_voxels))

    elif move_choice == 1:  # partial to empty
        voxel = np.random.choice(partial_voxels)
        voxel.status = EMPTY_VOXEL

        hp_empty_voxels= [v for v in hp.voxel.get_voxels_by_status(EMPTY_VOXEL) if v.depth < max_depth]
        hp_partial_voxels = [v for v in hp.voxel.get_voxels_by_status(PARTIAL_VOXEL)
                             if v.count_voxels_by_status(PARTIAL_VOXEL) == 1]

        p_forward = 0.5
        # if all voxels were partial, only possible move is partial to empty.
        if len(empty_voxels) == 0:
            p_forward = 1.0
        q_hp_h = p_forward * (1.0 / len(partial_voxels))

        p_backward = 0.5
        # if there is not partial voxel in new hypothesis, only possible backward move is empty to partial
        if len(hp_partial_voxels) == 0:
            p_backward = 1.0
        q_h_hp = p_backward * (1.0 / len(hp_empty_voxels)) * (1.0 / (2**(voxel.voxels_per_axis**3)))

    # depth of hypothesis might have changed. update it
    hp.update_depth()
    return hp, q_hp_h, q_h_hp


def voxel_scale_space(h, params):
    """
    This proposal function scales the whole space.
    """
    hp = h.copy()
    scale_variance = params['SCALE_SPACE_VARIANCE']
    change = np.random.randn(3) * np.sqrt(scale_variance)
    if np.all((hp.scale + change) > 0.0):
        hp.scale += change
    # proposal is symmetric; hence, q(hp|h) = q(h|hp)
    return hp, 1.0, 1.0


if __name__ == "__main__":
    import vision_forward_model as vfm
    import mcmclib.proposal
    import mcmclib.mh_sampler
    import i3d_proposal

    max_depth = 2

    fwm = vfm.VisionForwardModel(render_size=(200, 200))
    h = VoxelBasedShapeMaxD(forward_model=fwm, viewpoint=[(np.sqrt(2.0), -np.sqrt(2.0), 2.0)], max_depth=max_depth,
                            params={'LL_VARIANCE': 0.0001})
    # h = VoxelBasedShape(forward_model=fwm, viewpoint=[(np.sqrt(2.0), -np.sqrt(2.0), 2.0)],
    #                    params={'LL_VARIANCE': 0.0001})

    moves = {'voxel_flip_full_vs_empty': voxel_based_shape_flip_full_vs_empty,
             'voxel_flip_partial_vs_full': voxel_based_shape_flip_full_vs_partial,
             'voxel_flip_partial_vs_empty': voxel_based_shape_flip_empty_vs_partial,
             'voxel_scale_space': voxel_scale_space,
             'change_viewpoint': i3d_proposal.change_viewpoint_z}

    params = {'SCALE_SPACE_VARIANCE': 0.0025, 'CHANGE_VIEWPOINT_VARIANCE': 30.0, 'MAX_DEPTH': max_depth}

    proposal = mcmclib.proposal.RandomMixtureProposal(moves, params)

    data = np.load('data/stimuli20150624_144833/o1_single_view.npy')
    # data = np.load('data/test1_single_view.npy')

    # choose sampler
    thinning_period = 200
    report_period = 500
    sampler_class = 'pt'
    if sampler_class == 'mh':
        import mcmclib.mh_sampler
        sampler = mcmclib.mh_sampler.MHSampler(h, data, proposal, burn_in=1000, sample_count=10, best_sample_count=10,
                                               thinning_period=thinning_period, report_period=report_period)
    elif sampler_class == 'pt':
        from mcmclib.parallel_tempering_sampler import ParallelTemperingSampler
        sampler = ParallelTemperingSampler(initial_hs=[h, h, h], data=data, proposals=[proposal, proposal, proposal],
                                           temperatures=[3.0, 1.5, 1.0], burn_in=1000, sample_count=10,
                                           best_sample_count=10, thinning_period=int(thinning_period / 3.0),
                                           report_period=int(thinning_period / 3.0))
    else:
        raise ValueError('Unknown sampler class')

    run = sampler.sample()
