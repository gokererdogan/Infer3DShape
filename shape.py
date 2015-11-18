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
import scipy.ndimage as spi
from copy import deepcopy

from hypothesis import *

ADD_PART_PROB = 0.6
# assuming that pixels ~ unif(0,1), expected variance of a pixel difference is 1/6
LL_VARIANCE = 0.001 # in squared pixel distance
MAX_PIXEL_VALUE = 175.0 # this is usually 256.0 but in our case because of the lighting in our renders, it is lower
LL_FILTER_SIGMA = 2.0
MOVE_PART_VARIANCE = .005
# variance of move object proposals. this proposal moves the whole object (all the parts).
MOVE_OBJECT_VARIANCE = 0.1
CHANGE_SIZE_VARIANCE = .005
# the default viewpoint(s) for an object
DEFAULT_VIEWPOINT = [(1.5, -1.5, 1.5)]
# in degrees
CHANGE_VIEWPOINT_VARIANCE = 60.0


class CuboidPrimitive:
    """
    CuboidPrimitive class defines a 3D rectangular prism used as a primitive
    in our Object class. A CuboidPrimitive is specified by its position and
    size.
    position (x,y,z) ~ Unif(-0.5,0.5)
    size (w,h,d) ~ Unif(0,1)
    """
    def __init__(self, position=None, size=None):
        self.position = position
        self.size = size

        if position is None:
            # randomly pick position
            self.position = (np.random.rand(3) - 0.5)
        if size is None:
            # randomly pick size
            self.size = np.random.rand(3)

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


class Shape(Hypothesis):
    """
    Shape class defines a 3D object. It consists of a number of shape primitives
    that specify a shape.
    """
    def __init__(self, forward_model, parts=None, viewpoint=None, part_count=None, params=None):
        self.parts = parts
        self.forward_model = forward_model

        self.params = params
        if self.params is None:
            self.params = {'ADD_PART_PROB': ADD_PART_PROB, 'LL_VARIANCE': LL_VARIANCE,
                           'MAX_PIXEL_VALUE': MAX_PIXEL_VALUE, 'LL_FILTER_SIGMA': LL_FILTER_SIGMA}

        # this is the point from which we look at the object.
        # it is a list of 3-tuples, each 3-tuple specifying one viewpoint (x,y,z).
        # if this is not provided, camera_pos defined by forward_model is used.
        self.viewpoint = viewpoint

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
                while np.random.rand() < self.params['ADD_PART_PROB']:
                    self.parts.append(CuboidPrimitive())

        Hypothesis.__init__(self)

    def prior(self):
        if self.p is None:
            # assumes a uniform prob. dist. over add object probability,
            # position (in [-0.5,0.5]) and size (in [0,1])
            part_count = len(self.parts)
            self.p = (1.0 / (part_count + 1)) * (1.0 / (part_count + 2.0))
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
        positions = []
        sizes = []
        for part in self.parts:
            positions.append(part.position)
            sizes.append(part.size)

        return positions, sizes

    def _ll_ground_truth(self, gt):
        return np.exp(-self.distance(gt))

    def _ll_pixel_gaussian_filtered(self, data):
        img = self.forward_model.render(self)
        img = spi.gaussian_filter(img, self.params['LL_FILTER_SIGMA'])
        ll = np.exp(-np.sum(np.square(img - data)) / (img.size * self.params['LL_VARIANCE']))
        return ll

    def _ll_pixel(self, data):
        img = self.forward_model.render(self)
        ll = np.exp(-np.sum(np.square((img - data) / self.params['MAX_PIXEL_VALUE']))
                    / (img.size * 2 * self.params['LL_VARIANCE']))
        return ll

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

    @staticmethod
    def from_narray(arr, forward_model):
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

        return Shape(forward_model=forward_model, parts=parts)

    def __getstate__(self):
        # we cannot pickle VTKObjects, so get rid of them.
        return {k: v for k, v in self.__dict__.iteritems() if k != 'forward_model'}

class ShapeProposal(Proposal):
    """
    ShapeProposal class implements the mixture kernel of the following moves
        add/remove part, move part, change part size
    """
    def __init__(self, allow_viewpoint_update=False, params=None):
        Proposal.__init__(self)

        self.allow_viewpoint_update = allow_viewpoint_update

        self.params = params
        if self.params is None:
            self.params = {'CHANGE_SIZE_VARIANCE': CHANGE_SIZE_VARIANCE, 'MOVE_PART_VARIANCE': MOVE_PART_VARIANCE,
                           'MOVE_OBJECT_VARIANCE': MOVE_OBJECT_VARIANCE,
                           'CHANGE_VIEWPOINT_VARIANCE': CHANGE_VIEWPOINT_VARIANCE}

    def propose(self, h, *args):
        # pick one move randomly
        if self.allow_viewpoint_update:
            i = np.random.randint(0, 7)
        else:
            i = np.random.randint(0, 6)

        if i == 0:
            info = "add/remove part"
            hp, q_hp_h, q_h_hp = self.add_remove_part(h)
        elif i == 1:
            info = "move part local"
            hp, q_hp_h, q_h_hp = self.move_part_local(h)
        elif i == 2:
            info = "move part"
            hp, q_hp_h, q_h_hp = self.move_part(h)
        elif i == 3:
            info = "change size local"
            hp, q_hp_h, q_h_hp = self.change_part_size_local(h)
        elif i == 4:
            info = "change size"
            hp, q_hp_h, q_h_hp = self.change_part_size(h)
        elif i == 5:
            info = "move object"
            hp, q_hp_h, q_h_hp = self.move_object(h)
        elif i == 6:
            info = "change viewpoint"
            hp, q_hp_h, q_h_hp = self.change_viewpoint(h)

        return info, hp, q_hp_h, q_h_hp

    def add_remove_part(self, h):
        hp = h.copy()
        part_count = len(h.parts)
        # we need to be careful about hypotheses with 1 or 2 parts
        # because we cannot apply remove move to a hypothesis with 1 parts
        if part_count == 1 or np.random.rand() < .5:
            # add move
            new_part = CuboidPrimitive()
            hp.parts.append(new_part)
            if part_count == 1:
                # q(hp|h)
                q_hp_h = 1.0
                # q(h|hp)
                q_h_hp = 0.5 * (1.0 / (part_count + 1))
            else:
                # prob. of picking the add move * prob. of picking x,y,z * prob. picking w,h,d
                q_hp_h = 0.5
                q_h_hp = 0.5 * (1.0 / (part_count + 1))
        else:
            # remove move
            remove_id = np.random.randint(0, part_count)
            hp.parts.pop(remove_id)
            if part_count == 2:
                q_hp_h = 0.5 * (1.0 / part_count)
                q_h_hp = 1.0
            else:
                q_hp_h = 0.5 * (1.0 / part_count)
                q_h_hp = 0.5

        return hp, q_hp_h, q_h_hp

    def move_part(self, h):
        hp = h.copy()
        part_count = len(h.parts)
        part_id = np.random.randint(0, part_count)
        hp.parts[part_id].position = (np.random.rand(3) - 0.5)
        # q(h|hp) = q(hp|h), that is why we simply set both q(.|.) to 1.
        return hp, 1.0, 1.0

    def move_part_local(self, h):
        hp = h.copy()
        part_count = len(h.parts)
        part_id = np.random.randint(0, part_count)
        change = np.random.randn(3) * np.sqrt(self.params['MOVE_PART_VARIANCE'])
        # if proposed position is not out of bounds ([-0.5, 0.5])
        if np.all((hp.parts[part_id].position + change) < 0.5) and np.all((hp.parts[part_id].position + change) > -0.5):
            hp.parts[part_id].position = hp.parts[part_id].position + change
        # proposal is symmetric; hence, q(hp|h) = q(h|hp)
        return hp, 1.0, 1.0

    def move_object(self, h):
        hp = h.copy()
        change = np.random.randn(3) * np.sqrt(self.params['MOVE_OBJECT_VARIANCE'])
        # if proposed position is out of bounds ([-0.5, 0.5])
        for part in hp.parts:
            if np.any((part.position + change) > 0.5) or np.any((part.position + change) < -0.5):
                return hp, 1.0, 1.0
        # if updated position is in bounds
        for part in hp.parts:
            part.position = part.position + change
        # proposal is symmetric; hence, q(hp|h) = q(h|hp)
        return hp, 1.0, 1.0

    def change_part_size(self, h):
        hp = h.copy()
        part_count = len(h.parts)
        part_id = np.random.randint(0, part_count)
        hp.parts[part_id].size = np.random.rand(3)
        return hp, 1.0, 1.0

    def change_part_size_local(self, h):
        hp = h.copy()
        part_count = len(h.parts)
        part_id = np.random.randint(0, part_count)
        change = np.random.randn(3) * np.sqrt(self.params['CHANGE_SIZE_VARIANCE'])
        # if proposed size is not out of bounds ([0, 1])
        if np.all((hp.parts[part_id].size + change) < 1.0) and np.all((hp.parts[part_id].size + change) > 0.0):
            hp.parts[part_id].size = hp.parts[part_id].size + change
        return hp, 1.0, 1.0

    def change_viewpoint(self, h):
        hp = h.copy()
        # we rotate viewpoint around z axis, keeping the distance to the origin fixed.
        # default viewpoint is (1.5, -1.5, 1.5)
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
    fwm = vfm.VisionForwardModel(render_size=(200, 200))
    # generate a test object
    # test 1
    parts = [CuboidPrimitive(np.array([0.0, 0.0, 0.0]), np.array([0.5, .75/2, .75/2])),
             CuboidPrimitive(np.array([.75/2, 0.0, 0.0]), np.array([.25, .25, .25]))]

    # test 2
    parts = [CuboidPrimitive(np.array([0.0, 0.0, 0.0]), np.array([0.5, .75/2, .75/2])),
             CuboidPrimitive(np.array([.45, 0.0, 0.0]), np.array([.4, .25, .25])),
             CuboidPrimitive(np.array([0.0, 0.0, 0.75/2]), np.array([0.25/2, 0.35/2, .75/2])),
             CuboidPrimitive(np.array([0.0, 0.2, 0.75/2]), np.array([.1, .45/2, .25/2]))]

    # test 3
    parts = [CuboidPrimitive(np.array([0.0, 0.0, 0.0]), np.array([0.2, 0.2, 0.45])),
             CuboidPrimitive(np.array([.35/2, 0.0, 0.10]), np.array([0.15, 0.15, 0.15])),
             CuboidPrimitive(np.array([0.3, 0.0, -0.45/2]), np.array([0.4, 0.2, 0.2])),
             CuboidPrimitive(np.array([0.85/2, 0.0, 0.05/2]), np.array([0.15, 0.15, 0.3])),
             CuboidPrimitive(np.array([-0.25, 0.0, 0.15]), np.array([0.3, 0.3, 0.3])),
             CuboidPrimitive(np.array([-0.25, -0.25, 0.2]), np.array([0.1, 0.2, 0.1]))]

    angle = np.random.randint(0, 360)
    print(angle)
    x = 1.5 * np.sqrt(2.0) * np.cos(angle / 180.0 * np.pi)
    y = 1.5 * np.sqrt(2.0) * np.sin(angle / 180.0 * np.pi)
    viewpoint = [(x, y, 1.5)]

    h = Shape(fwm, parts=parts, viewpoint=viewpoint)
    # fwm._view(h)
    img = fwm.render(h)
    np.save('./data/test3_single_view.npy', img)
    fwm.save_render('./data/test3_single_view.png', h)

    # generate shape randomly
    hr = Shape(fwm, viewpoint=viewpoint)
    # read data (i.e., observed image) from disk
    data = np.load('./data/test3_single_view.npy')
    # fwm._view(hr)

    print hr.likelihood(data)
    print h.likelihood(data)
