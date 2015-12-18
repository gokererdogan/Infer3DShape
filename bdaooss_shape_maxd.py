"""
Inferring 3D Shape from 2D Images

This file contains the BDAoOSSShapeMaxD class that is BDAoOSSShape with a maximum depth
constraint.

Created on Oct 6, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""
import numpy as np
from copy import deepcopy

from bdaooss_shape import BDAoOSSShape


class BDAoOSSShapeMaxD(BDAoOSSShape):
    """Shape hypothesis class based on BDAoOSS Shape grammar. This class assumes a maximum depth to parse trees and
    uses a new prior distribution over hypotheses.

    Note that this class in fact does not enforce the maximum depth constraint. That is achieved by the proposal
    functions. However, this class defines a new prior over hypotheses; that is the main reason we want to constrain
    tree depth.
    """
    def __init__(self, forward_model, shape=None, viewpoint=None, params=None, max_depth=10):
        """Initialize BDAoOSSShapeMaxD instance

        Args:
            max_depth (int): Maximum depth allowed for trees

        Returns:
            BDAoOSSShapeMaxD
        """
        BDAoOSSShape.__init__(self, forward_model=forward_model, shape=shape, viewpoint=viewpoint, params=params)

        # this is actually not needed anywhere.
        self.max_depth = max_depth

    def _calculate_log_prior(self):
        """Prior for the BDAoOSShapeMaxD hypothesis.

        Returns:
            (float): log prior for the hypothesis.
        """
        # assume a uniform prior over number of parts. we can achieve this by letting the derivation prob. for all
        # trees to be equal. Note that we still have to keep the spatial model probability as there are many hypotheses
        # with the same number of parts but with different spatial models.
        # NOTE this prior will lead to chains that are potentially quite slow since the chain will spend an equal
        # amount of time sampling quite large trees as it does sampling small trees.
        return np.log(self.shape.spatial_model.probability())

    def copy(self):
        # NOTE that we are not copying params. This assumes that params do not change from
        # hypothesis to hypothesis.
        shape_copy = self.shape.copy()
        viewpoint_copy = deepcopy(self.viewpoint)
        return BDAoOSSShapeMaxD(forward_model=self.forward_model, shape=shape_copy, params=self.params,
                                max_depth=self.max_depth, viewpoint=viewpoint_copy)

if __name__ == "__main__":
    import vision_forward_model as vfm
    import mcmclib.proposal
    import mcmclib.mh_sampler
    import i3d_proposal
    import bdaooss_shape as bd

    fwm = vfm.VisionForwardModel(render_size=(200, 200))

    max_depth = 3
    h = BDAoOSSShapeMaxD(forward_model=fwm, viewpoint=[(1.5, -1.5, 1.5)], params={'LL_VARIANCE': 0.01},
                         max_depth=max_depth)

    moves = {'bdaooss_add_remove_part': bd.bdaooss_add_remove_part,
             'bdaooss_change_part_size': bd.bdaooss_change_part_size,
             'bdaooss_change_part_size_local': bd.bdaooss_change_part_size_local,
             'bdaooss_change_part_dock_face': bd.bdaooss_change_part_dock_face,
             'bdaooss_move_object': bd.bdaooss_move_object, 'change_viewpoint': i3d_proposal.change_viewpoint}

    params = {'MAX_DEPTH': max_depth,
              'MOVE_OBJECT_VARIANCE': 0.01,
              'CHANGE_SIZE_VARIANCE': 0.01,
              'CHANGE_VIEWPOINT_VARIANCE': 300.0}

    proposal = mcmclib.proposal.RandomMixtureProposal(moves, params)

    data = np.load('data/test1_single_view.npy')

    sampler = mcmclib.mh_sampler.MHSampler(h, data, proposal, burn_in=1000, sample_count=10, best_sample_count=10,
                                           thinning_period=2000, report_period=2000)

    run = sampler.sample()
