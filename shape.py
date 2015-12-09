"""
Inferring 3D Shape from 2D Images

This file contains the Shape hypothesis class.
This hypothesis assumes objects are made up of rectangular prisms of arbitrary size
and position.

Created on Aug 27, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import numpy as np
from copy import deepcopy

import Infer3DShape.i3d_hypothesis as hyp

# probability of adding a new part used when generating objects randomly
ADD_PART_PROB = 0.6

class CuboidPrimitive(object):
    """
    CuboidPrimitive class defines a 3D rectangular prism used as a primitive
    in our Object class. A CuboidPrimitive is specified by its position and
    size.

    Attributes:
        position (3x1-ndarray): Position of primitive, sampled from ~ Unif(-0.5,0.5)
        size (3x1-ndarray): Size of primitive, sampled from ~ Unif(0,1)
    """
    def __init__(self, position=None, size=None):
        if position is None:
            # randomly pick position
            position = (np.random.rand(3) - 0.5)
        else:
            position = np.array(position)
            if np.any(np.abs(position) > 0.5):
                raise ValueError("Position must be in [-0.5, 0.5]")

        if size is None:
            # randomly pick size
            size = np.random.rand(3)
        else:
            size = np.array(size)
            if np.any(size > 1.0) or np.any(size < 0.0):
                raise ValueError("Size must be in [0.0, 1.0]")

        self.position = position
        self.size = size

    def __str__(self):
        s = "{0:20}{1:20}".format(np.array_str(self.position, precision=2), np.array_str(self.size, precision=2))
        return s

    def __repr__(self):
        return self.__str__()

    def __eq__(self, comp):
        if np.sum(np.abs(self.position - comp.position)) < 1e-6 and np.sum(np.abs(self.size - comp.size)) < 1e-6:
            return True
        return False

    def __sub__(self, other):
        return np.sum(np.abs(self.position - other.position)) + np.sum(np.abs(self.size - other.size))


class Shape(hyp.I3DHypothesis):
    """
    Shape class defines a 3D object. It consists of a number of shape primitives
    that specify a shape.
    """
    def __init__(self, forward_model, viewpoint=None, params=None, parts=None, part_count=None):
        hyp.I3DHypothesis.__init__(self, forward_model, viewpoint, params)

        self.parts = parts
        # generative process: add a new part until rand()>theta (add part prob.)
        # p(H|theta) = theta^|H| (1 - theta)
        if self.parts is None:
            # randomly generate a new shape
            self.parts = []
            if part_count is not None and part_count > 0:
                for i in range(part_count):
                    self.parts.append(CuboidPrimitive())
            else:
                self.parts = [CuboidPrimitive()]
                while np.random.rand() < ADD_PART_PROB:
                    self.parts.append(CuboidPrimitive())

    @staticmethod
    def from_narray(arr, forward_model, viewpoint=None, params=None):
        """
        Create a Shape object from a numpy array.
        arr contains the positions and sizes of each part.
        It is a vector of length (number of parts) * 6; however
        the array may in fact be larger and contain zeros.
        Therefore, objects with zero size are ignored.
        Format: part1_pos, part1_size, part2_pos, part2_size, ...
        """
        parts = []
        maxN = int(arr.size / 6.0)
        for i in range(maxN):
            pos = arr[(6 * i):((6 * i) + 3)]
            size = arr[((6 * i) + 3):((6 * i) + 6)]
            if np.all(size>0) and np.sum(size) > 1e-6:
                parts.append(CuboidPrimitive(position=pos, size=size))

        return Shape(forward_model=forward_model, parts=parts, viewpoint=viewpoint, params=params)

    def _calculate_log_prior(self):
        # assumes a uniform prob. dist. over add object probability,
        # position (in [-0.5,0.5]) and size (in [0,1])
        # p(H) = (1.0 / (part_count + 1)) * (1.0 / (part_count + 2.0))
        part_count = len(self.parts)
        return -np.log(part_count + 1) - np.log(part_count + 2.0)

    def convert_to_positions_sizes(self):
        """
        Returns the positions of parts and their sizes.
        Used by VisionForwardModel for rendering.
        :return: positions and sizes of parts
        """
        positions = []
        sizes = []
        for part in self.parts:
            positions.append(part.position)
            sizes.append(part.size)

        return positions, sizes

    def copy(self):
        """
        Returns a (deep) copy of the instance
        """
        # NOTE that we are not copying params. This assumes that
        # parameters do not change from hypothesis to hypothesis.
        self_copy = Shape(self.forward_model, params=self.params)
        parts_copy = deepcopy(self.parts)
        viewpoint_copy = deepcopy(self.viewpoint)
        self_copy.parts = parts_copy
        self_copy.viewpoint = viewpoint_copy
        return self_copy

    def __str__(self):
        s = "Id   Position            Size                \n"
        fmt = "{0:5}{1:40}\n"
        for i, part in enumerate(self.parts):
            s = s + fmt.format(str(i), str(part))
        return s

    def __repr__(self):
        return self.__str__()

    def __eq__(self, comp):
        if len(self.parts) != len(comp.parts):
            return False

        # indices of matched parts
        matched_parts = []
        for part in self.parts:
            try:
                # find my part in other object's parts
                i = comp.parts.index(part)
            except ValueError:
                # not found, return false
                return False

            # if found, but this part is already matched
            if i in matched_parts:
                return False
            # add matched part to list
            matched_parts.append(i)

        return True

    def distance(self, other):
        dist = 0.0
        remaining_parts = range(len(other.parts))
        for part in self.parts:
            # find the part in other that is closest to it
            dists = []
            for ri in remaining_parts:
                dists.append(part - other.parts[ri])

            if dists:
                mini = np.argmin(dists)
                dist = dist + dists[mini]
                remaining_parts.pop(mini)
            else:
                # 2.0 is for position difference. It is the on
                # average distance between two parts.
                dist = dist + 1.0 + np.sum(np.abs(part.size))

        for ri in remaining_parts:
            dist = dist + 1.0 + np.sum(np.abs(other.parts[ri].size))

        return dist

    def to_narray(self):
        """
        Converts object to numpy array of length 6 * (number of parts)
        Format: part1_pos, part1_size, part2_pos, part2_size, ...
        """
        arr = np.zeros((1, 6 * len(self.parts)))
        for i, p in enumerate(self.parts):
            arr[0, (6 * i):((6 * i) + 3)] = p.position
            arr[0, ((6 * i) + 3):((6 * i) + 6)] = p.size
        return arr

# Proposal functions
def shape_add_remove_part(h, params):
    max_part_count = np.inf
    if 'MAX_PART_COUNT' in params.keys():
        max_part_count = params['MAX_PART_COUNT']

    hp = h.copy()
    part_count = len(h.parts)

    if part_count > max_part_count:
        raise ValueError("add/remove part expects shape hypothesis with fewer than {0:s} parts.".format(max_part_count))

    if part_count == 0:
        raise ValueError("add/remove part expects shape hypothesis to have at least 1 part.")

    # we cannot add or remove parts if max_part_count is 1.
    if max_part_count == 1:
        return hp, 1.0, 1.0

    # we need to be careful about hypotheses with 1 or 2 parts
    # because we cannot apply remove move to a hypothesis with 1 parts
    # similarly we need to be careful with hypotheses with maxn or
    # maxn-1 parts
    if part_count == 1 or (part_count != max_part_count and np.random.rand() < .5):
        # add move
        new_part = CuboidPrimitive()
        hp.parts.append(new_part)

        # q(hp|h)
        q_hp_h = 0.5
        # if add is the only move possible
        if part_count == 1:
            q_hp_h = 1.0

        # q(h|hp)
        q_h_hp = 0.5 * (1.0 / (part_count + 1))
        #  if remove is the only possible reverse move
        if part_count == (max_part_count - 1):
            q_h_hp = 1.0 * (1.0 / (part_count + 1))
    else:
        # remove move
        remove_id = np.random.randint(0, part_count)
        hp.parts.pop(remove_id)

        q_h_hp = 0.5
        if part_count == 2:
            q_h_hp = 1.0

        q_hp_h = 0.5 * (1.0 / part_count)
        # if remove move is the only possible move
        if part_count == max_part_count:
            q_hp_h = 1.0 * (1.0 / part_count)

    return hp, q_hp_h, q_h_hp

def shape_move_part(h, params):
    hp = h.copy()
    part_count = len(h.parts)
    part_id = np.random.randint(0, part_count)
    hp.parts[part_id].position = (np.random.rand(3) - 0.5)
    # q(h|hp) = q(hp|h), that is why we simply set both q(.|.) to 1.
    return hp, 1.0, 1.0

def shape_move_part_local(h, params):
    hp = h.copy()
    part_count = len(h.parts)
    part_id = np.random.randint(0, part_count)
    change = np.random.randn(3) * np.sqrt(params['MOVE_PART_VARIANCE'])
    # if proposed position is not out of bounds ([-0.5, 0.5])
    if np.all((hp.parts[part_id].position + change) < 0.5) and np.all((hp.parts[part_id].position + change) > -0.5):
        hp.parts[part_id].position = hp.parts[part_id].position + change
    # proposal is symmetric; hence, q(hp|h) = q(h|hp)
    return hp, 1.0, 1.0

def shape_move_object(h, params):
    hp = h.copy()
    change = np.random.randn(3) * np.sqrt(params['MOVE_OBJECT_VARIANCE'])
    # if proposed position is out of bounds ([-0.5, 0.5])
    for part in hp.parts:
        if np.any((part.position + change) > 0.5) or np.any((part.position + change) < -0.5):
            return hp, 1.0, 1.0
    # if updated position is in bounds
    for part in hp.parts:
        part.position = part.position + change
    # proposal is symmetric; hence, q(hp|h) = q(h|hp)
    return hp, 1.0, 1.0

def shape_change_part_size(h, params):
    hp = h.copy()
    part_count = len(h.parts)
    part_id = np.random.randint(0, part_count)
    hp.parts[part_id].size = np.random.rand(3)
    return hp, 1.0, 1.0

def shape_change_part_size_local(h, params):
    hp = h.copy()
    part_count = len(h.parts)
    part_id = np.random.randint(0, part_count)
    change = np.random.randn(3) * np.sqrt(params['CHANGE_SIZE_VARIANCE'])
    # if proposed size is not out of bounds ([0, 1])
    if np.all((hp.parts[part_id].size + change) < 1.0) and np.all((hp.parts[part_id].size + change) > 0.0):
        hp.parts[part_id].size = hp.parts[part_id].size + change
    return hp, 1.0, 1.0

if __name__ == "__main__":
    import vision_forward_model as vfm
    import mcmclib.proposal
    import mcmclib.mh_sampler
    import i3d_proposal

    fwm = vfm.VisionForwardModel(render_size=(200, 200))
    h = Shape(forward_model=fwm, viewpoint=[(1.5, -1.5, 1.5)], params={'LL_VARIANCE': 0.001})

    moves = {'shape_add_remove_part': shape_add_remove_part, 'shape_move_part': shape_move_part,
             'shape_move_part_local': shape_move_part_local, 'shape_change_part_size': shape_change_part_size,
             'shape_change_part_size_local': shape_change_part_size_local, 'shape_move_object': shape_move_object,
             'change_viewpoint': i3d_proposal.change_viewpoint}

    params = {'MOVE_PART_VARIANCE': 0.01,
              'MOVE_OBJECT_VARIANCE': 0.01,
              'CHANGE_SIZE_VARIANCE': 0.01,
              'CHANGE_VIEWPOINT_VARIANCE': 300.0}

    proposal = mcmclib.proposal.RandomMixtureProposal(moves, params)

    data = np.load('data/test1_single_view.npy')

    sampler = mcmclib.mh_sampler.MHSampler(h, data, proposal, burn_in=1000, sample_count=10, best_sample_count=10,
                                           thinning_period=2000, report_period=2000)

    run = sampler.sample()
