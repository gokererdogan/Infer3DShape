"""
Inferring 3D Shape from 2D Images

This file contains the ShapeMaxN class which implements a
Shape object with maximum N parts. This is done in order to
assign a flat prior to all hypotheses. We don't want the
prior to favor objects with fewer parts.

Created on Sep 14, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

from hypothesis import *

class ShapeMaxN(Shape):
    """
    ShapeMaxN class defines a 3D object with maximum N parts.
    """
    def __init__(self, forward_model, maxn=6, parts=None):
        self.maxn = maxn

        # generative process: add a new part until rand()>theta (add part prob.)
        # p(H|theta) = theta^|H| (1 - theta)
        if parts is None:
            # randomly generate a new shape
            parts = [CuboidPrimitive()]
            while (np.random.rand() < ADD_OBJECT_PROB) and (len(parts) < self.maxn):
                parts.append(CuboidPrimitive())

        Shape.__init__(self, forward_model, parts)

    def prior(self):
        if self.p is None:
            # assumes a uniform prob. over all hypotheses (regardless
            # of number of parts)
            self.p = 1.0
        return self.p

    def copy(self):
        """
        Returns a (deep) copy of the instance
        """
        self_copy = ShapeMaxN(self.forward_model, self.maxn)
        parts_copy = deepcopy(self.parts)
        self_copy.parts = parts_copy
        return self_copy


class ShapeMaxNProposal(ShapeProposal):
    """
    ShapeMaxNProposal class implements the mixture kernel of the following moves
    for the ShapeMaxN class.
        add/remove part, move part, change part size
    """
    def __init__(self, allow_add_remove=True):
        # do we allow add/remove part move?
        self.allow_add_remove = allow_add_remove
        ShapeProposal.__init__(self)

    def propose(self, h, *args):
        # pick one move randomly
        if self.allow_add_remove:
            i = np.random.randint(0, 5)
        else:
            i = np.random.randint(1, 5)

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
        # similarly we need to be careful with hypotheses with maxn or
        # maxn-1 parts
        if part_count == 1 or (part_count != h.maxn and np.random.rand() < .5):
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
                # if remove is the only possible reverse move
                if part_count == (h.maxn - 1):
                    q_h_hp = 1.0 * (1.0 / (part_count + 1))
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
                # if remove move is the only possible move
                if part_count == h.maxn:
                    q_hp_h = 1.0 * (1.0 / part_count)

        return hp, q_hp_h, q_h_hp

if __name__ == "__main__":
    import vision_forward_model as vfm
    import mcmc_sampler as mcmc
    fwm = vfm.VisionForwardModel()
    kernel = ShapeMaxNProposal()

    max_part_count = 8
    # generate initial hypothesis shape randomly
    h = ShapeMaxN(fwm, max_part_count)

    # read data (i.e., observed image) from disk
    obj_name = 'o1_t2_rp_d2'
    data = np.load('./data/stimuli20150624_144833/{0:s}.npy'.format(obj_name))
    # data = np.load('./data/test2.npy')

    sampler = mcmc.MHSampler(h, data, kernel, 0, 10, 20, 20000, 2000)
    run = sampler.sample()
    print(run.best_samples.samples)
    print()
    print(run.best_samples.probs)
    run.save('{0:s}.pkl'.format(obj_name))

"""
    for i, sample in enumerate(run.samples.samples):
        fwm.save_render("samples_maxn/s{0:d}.png".format(i), sample)
    for i, sample in enumerate(run.best_samples.samples):
        fwm.save_render("samples_maxn/b{0:d}.png".format(i), sample)
"""
