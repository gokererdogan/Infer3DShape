'''
Inferring 3D Shape from 2D Images

This file contains the Hypothesis and related classes.
Hypothesis class specifies a 3D scene consisting of objects.

Created on Aug 27, 2015

Goker Erdogan
https://github.com/gokererdogan/
'''

import numpy as np
import scipy.ndimage as spi
from copy import deepcopy
import vision_forward_model as vfm

ADD_OBJECT_PROB = 0.6
LL_VARIANCE = 25.0
LL_FILTER_SIGMA = 2.0
MOVE_PART_VARIANCE = .005
CHANGE_SIZE_VARIANCE = .005

class Hypothesis:
    """
    Hypothesis class is an abstract class that specifies the template
    for an MCMC hypothesis.
    """
    def __init__(self):
        """
        Hypothesis class constructor
        """
        # p: prior, ll: likelihood
        # we want to cache these values, therefore we initialize them to None
        # prior and ll methods should calculate these once and return p and ll
        self.p = None
        self.ll = None
        pass

    def prior(self):
        """
        Returns prior probability p(H) of the hypothesis
        """
        pass

    def likelihood(self, data):
        """
        Returns the likelihood of hypothesis given data, p(D|H)
        """
        pass

    def copy(self, data):
        """
        Returns a (deep) copy of the hypothesis. Used for generating
        new hypotheses based on itself.
        """
        pass

class Proposal:
    """
    Proposal class implements MCMC moves (i.e., proposals) on Hypothesis.
    This is an abstract class specifying the template for Proposal classes.
    Propose method is called by MCMCSampler to get the next proposed hypothesis.
    """
    def __init__(self):
        pass

    def propose(self, h, *args):
        """
        Proposes a new hypothesis based on h
        Returns (information string, new hypothesis, probability of move, probability of reverse move)
        args: optional additional parameters
        """
        pass



class CuboidPrimitive:
    """
    CuboidPrimitive class defines a 3D rectangular prism used as a primitive
    in our Object class. A CuboidPrimitive is specified by its position and
    size.
    position (x,y,z) ~ Unif(-1,1)
    size (w,h,d) ~ Unif(0,1)
    """
    def __init__(self, position=None, size=None):
        self.position = position
        self.size = size

        if position is None:
            # randomly pick position
            self.position = 2.0 * (np.random.rand(3) - 0.5)
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
    def __init__(self, forward_model, parts=None):
        self.parts = parts
        self.forward_model = forward_model

        # generative process: add a new part until rand()>theta (add part prob.)
        # p(H|theta) = theta^|H| (1 - theta)
        if self.parts is None:
            # randomly generate a new shape
            self.parts = [CuboidPrimitive()]
            while np.random.rand() < ADD_OBJECT_PROB:
                self.parts.append(CuboidPrimitive())

        Hypothesis.__init__(self) 

    def prior(self):
        if self.p is None:
            # assumes a uniform prob. dist. over add object probability,
            # position (in [-1,1]) and size (in [0,1])
            part_count = len(self.parts)
            self.p = (1.0 / (part_count + 1)) * (1.0 / (part_count + 2.0)) * ((1.0 / 8.0)**part_count)
        return self.p

    def likelihood(self, data):
        if self.ll is None:
            self.ll = self._ll_pixel(data)
            # self.ll = self._ll_ground_truth(data)
        return self.ll

    def _ll_ground_truth(self, gt):
        return np.exp(-self.distance(gt))

    def _ll_pixel_gaussian_filtered(self, data):
        img = self.forward_model.render(self)
        img = spi.gaussian_filter(img, LL_FILTER_SIGMA)
        ll = np.exp(-np.sum(np.square(img - data)) / (img.size * LL_VARIANCE))
        return ll

    def _ll_pixel(self, data):
        img = self.forward_model.render(self)
        ll = np.exp(-np.sum(np.square(img - data)) / (img.size * LL_VARIANCE))
        return ll

    def copy(self):
        """
        Returns a (deep) copy of the instance
        """
        self_copy = Shape(self.forward_model)
        parts_copy = deepcopy(self.parts)
        self_copy.parts = parts_copy
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
                dist = dist + 2.0 + np.sum(np.abs(part.size))

        for ri in remaining_parts:
            dist = dist + 2.0 + np.sum(np.abs(other.parts[ri].size))

        return dist


class ShapeProposal(Proposal):
    """
    ShapeProposal class implements the mixture kernel of the following moves
        add/remove part, move part, change part size
    """
    def __init__(self):
        Proposal.__init__(self)

    def propose(self, h, *args):
        # pick one move randomly
        i = np.random.randint(0, 5)
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
        else:
            info = "change size"
            hp, q_hp_h, q_h_hp = self.change_part_size(h)
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
                q_hp_h = 1.0 * (1.0 / 8.0)
                # q(h|hp)
                q_h_hp = 0.5 * (1.0 / (part_count + 1))
            else:
                # prob. of picking the add move * prob. of picking x,y,z * prob. picking w,h,d
                q_hp_h = 0.5 * (1.0 / 8.0)
                q_h_hp = 0.5 * (1.0 / (part_count + 1))
        else:
            # remove move
            remove_id = np.random.randint(0, part_count)
            hp.parts.pop(remove_id)
            if part_count == 2:
                q_hp_h = 0.5 * (1.0 / part_count)
                q_h_hp = 1.0 * (1.0 / 8.0)
            else:
                q_hp_h = 0.5 * (1.0 / part_count)
                q_h_hp = 0.5 * (1.0 / 8.0)

        return hp, q_hp_h, q_h_hp

    def move_part(self, h):
        hp = h.copy()
        part_count = len(h.parts)
        part_id = np.random.randint(0, part_count)
        hp.parts[part_id].position = 2.0 * (np.random.rand(3) - 0.5)
        # q(hp|h) = (1 / number of parts) * (1 / 8) (picking the new x,y,z)
        # q(h|hp) = q(hp|h), that is why we simple set both q(.|.) to 1.
        return hp, 1.0, 1.0

    def move_part_local(self, h):
        hp = h.copy()
        part_count = len(h.parts)
        part_id = np.random.randint(0, part_count)
        change = np.random.randn(3) * np.sqrt(MOVE_PART_VARIANCE)
        # if proposed position is not out of bounds ([-1, 1])
        if np.all((hp.parts[part_id].position + change) < 1.0) and np.all((hp.parts[part_id].position + change) > -1.0):
            hp.parts[part_id].position = hp.parts[part_id].position + change
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
        change = np.random.randn(3) * np.sqrt(CHANGE_SIZE_VARIANCE)
        # if proposed size is not out of bounds ([0, 1])
        if np.all((hp.parts[part_id].size + change) < 1.0) and np.all((hp.parts[part_id].size + change) > 0.0):
            hp.parts[part_id].size = hp.parts[part_id].size + change
        return hp, 1.0, 1.0


if __name__ == "__main__":
    fwm = vfm.VisionForwardModel()
    # generate a test object
    # test 1
    parts = [CuboidPrimitive(np.array([0.0, 0.0, 0.0]), np.array([1.0, .75, .75])),
             CuboidPrimitive(np.array([.75, 0.0, 0.0]), np.array([.5, .5, .5]))]

    # test 2
    parts = [CuboidPrimitive(np.array([0.0, 0.0, 0.0]), np.array([1.0, .75, .75])),
             CuboidPrimitive(np.array([.9, 0.0, 0.0]), np.array([.8, .5, .5])),
             CuboidPrimitive(np.array([0.0, 0.0, 0.75]), np.array([0.25, 0.35, .75])),
             CuboidPrimitive(np.array([0.0, 0.4, 0.75]), np.array([.2, .45, .25]))]

    # test 3
    parts = [CuboidPrimitive(np.array([0.0, 0.0, 0.0]), np.array([0.4, 0.4, 0.9])),
             CuboidPrimitive(np.array([.35, 0.0, 0.20]), np.array([0.3, 0.3, 0.3])),
             CuboidPrimitive(np.array([0.6, 0.0, -0.45]), np.array([0.8, 0.4, 0.4])),
             CuboidPrimitive(np.array([0.85, 0.0, 0.05]), np.array([0.3, 0.3, 0.6])),
             CuboidPrimitive(np.array([-0.5, 0.0, 0.3]), np.array([0.6, 0.6, 0.6])),
             CuboidPrimitive(np.array([-0.5, -0.5, 0.4]), np.array([0.2, 0.4, 0.2]))]

    h = Shape(fwm, parts)
    # fwm._view(h)
    img = fwm.render(h)
    np.save('test3.npy', img)
    fwm.save_render('test3.png', h)
    
    # generate shape randomly
    hr = Shape(fwm)
    # read data (i.e., observed image) from disk
    data = np.load('test3.npy')
    # fwm._view(hr)

    print hr.likelihood(data)
    print h.likelihood(data)
