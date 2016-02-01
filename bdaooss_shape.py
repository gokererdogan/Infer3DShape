"""
Inferring 3D Shape from 2D Images

This file contains the shape class for BDAoOSS experiment.

Created on Oct 6, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""
import numpy as np
from copy import deepcopy

import BDAoOSS.bdaooss_grammar as bdaooss
import i3d_hypothesis as hyp


class BDAoOSSShape(hyp.I3DHypothesis):
    """
    This class implements a shape grammar hypothesis based on the true grammar for
    the objects in the experiment.
    """
    def __init__(self, forward_model, shape=None, viewpoint=None, params=None):
        """Initialize BDAoOSSShape instance

        Args:
            forward_model (VisionForwardModel): Instance of VisionForwardModel used for rendering
            viewpoint (list): Viewing orientation. Used only if viewpoint should also be inferred

        Returns:
            BDAoOSSShape
        """
        hyp.I3DHypothesis.__init__(self, forward_model=forward_model, viewpoint=viewpoint, params=params)

        self.shape = shape
        if self.shape is None:
            sm = bdaooss.BDAoOSSSpatialModel(size_prior=bdaooss.size_prior_infer3dshape)
            self.shape = bdaooss.BDAoOSSShapeState(spatial_model=sm)

    def _calculate_log_prior(self):
        """Prior for the hypothesis. We use the prior provided by BDAoOSSShapeState class.

        Returns:
            (float): log probability of the hypothesis
        """
        # prior = prob. of parse tree * prob. of spatial model
        # NOTE that instead of rational rules prior (which integrates out
        # production probabilities) we use the derivation prob. as prior.
        return np.log(self.shape.probability()) + np.log(self.shape.spatial_model.probability())

    def convert_to_positions_sizes(self):
        """
        Converts the BDAoOSS shape instance to lists of positions and sizes of parts.
        Used by VisionForwardModel for rendering.
        :return: positions and sizes
        """
        parts, positions, sizes, viewpoint = self.shape.convert_to_parts_positions()
        # get rid of really small parts (this shouldn't happen)
        positions = [p for p, s in zip(positions, sizes) if np.all(s>0.01)]
        sizes = [s for s in sizes if np.all(s>0.01)]
        return positions, sizes

    def copy(self):
        shape_copy = self.shape.copy()
        viewpoint_copy = deepcopy(self.viewpoint)
        # NOTE that we are not copying params. This assumes that params do not change from
        # hypothesis to hypothesis.
        return BDAoOSSShape(forward_model=self.forward_model, shape=shape_copy, params=self.params,
                            viewpoint=viewpoint_copy)

    def __str__(self):
        return str(self.shape)

    def __repr__(self):
        return repr(self.shape)

    def __eq__(self, comp):
        self_parts = self.shape.spatial_model.spatial_states.values()
        comp_parts = comp.shape.spatial_model.spatial_states.values()
        if len(self_parts) != len(comp_parts):
            return False

        # indices of matched parts
        matched_parts = []
        for part in self_parts:
            try:
                # find my part in other object's parts
                i = comp_parts.index(part)
            except ValueError:
                # not found, return false
                return False

            # if found, but this part is already matched
            if i in matched_parts:
                return False
            # add matched part to list
            matched_parts.append(i)

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getstate__(self):
        # we cannot pickle VTKObjects, so get rid of them.
        return {k: v for k, v in self.__dict__.iteritems() if k != 'forward_model'}


# PROPOSAL FUNCTIONS
def bdaooss_add_remove_part(h, params):
    max_depth = np.inf
    if params is not None and 'MAX_DEPTH' in params.keys():
        max_depth = params['MAX_DEPTH']

    hp = h.copy()
    tree = hp.shape.tree
    sm = hp.shape.spatial_model
    # NOTE: get_depth counts Null nodes too, so the depth returned will be 1 more than what we would expect,
    # BUT, because get_depth does not count the root level, we don't need to add 1 here.
    depth = hp.shape.get_depth()

    if depth > max_depth:
        raise ValueError("add/remove part expects shape hypothesis with depth less than {0:d}.".format(max_depth))

    # we cannot add or remove parts if max_depth is 1.
    if max_depth == 1:
        return hp, 1.0, 1.0

    # REMEMBER that our grammar allows P nodes to have at most 3 children.
    # we can add nodes under the P nodes that have less than 3 children
    # we can remove P nodes that have a single child Null
    depths = {}
    # find the set of parent nodes we can add to and the set of nodes we can remove
    add_nodes = []
    remove_nodes = []
    for node in tree.expand_tree(mode=bdaooss.Tree.WIDTH):
        if tree[node].tag.symbol == 'P':
            # if root, depth is 0
            if tree[node].bpointer is None:
                depths[node] = 0
            else:
                depths[node] = depths[tree[node].bpointer] + 1

            # possible parent node
            if depths[node] < (max_depth-1) and len(tree[node].fpointer) < bdaooss.MAX_CHILDREN:
                add_nodes.append(node)

            if depths[node] > 0 and len(tree[node].fpointer) == 1 \
                    and tree[tree[node].fpointer[0]].tag.symbol == 'Null':
                remove_nodes.append(node)

    choices = [0, 1]
    if len(add_nodes) == 0:
        choices.remove(0)
    if len(remove_nodes) == 0:
        choices.remove(1)

    # this should never happen
    if len(choices) == 0:
        raise KeyError("No nodes to add or remove.")

    move_choice = np.random.choice(choices)
    # ADD PART MOVE ----------------------------
    if move_choice == 0: # add part
        parent_node = np.random.choice(add_nodes)
        hp.shape.add_part(parent_node)

        # probability of proposing hp given h is
        #   prob. picking add move * picking the parent node * picking the docking face * picking size
        #   = 1/2 * 1/|add_nodes| * 1/|available_faces| * (1/1)^3
        # Note if there are no nodes to remove, i.e., add is the only possible move, above 1/2 should be 1.
        p_add = (1.0 / 2.0)
        parent_occupied_faces = sm.spatial_states[parent_node].occupied_faces
        available_face_count = bdaooss.FACE_COUNT - len(parent_occupied_faces) + 1
        if len(remove_nodes) == 0:
            p_add = 1.0
        p_hp_h = p_add * (1.0 / len(add_nodes)) * (1.0 / available_face_count)

        # probability of proposing h given hp is
        #   prob. picking remove move * picking the node to remove
        # Note that we do not know the number of nodes that can be removed in hp.
        # but we can calculate that from the number of remove_nodes in h.
        # if the node we added the new node to was itself in remove_nodes, the number of removable nodes
        # is the same in h and hp. But if the parent node was not in remove_nodes, then there is one more
        # removable node in hp than h.
        removable_count = len(remove_nodes)
        if parent_node not in remove_nodes:
            removable_count += 1

        # what about the prob. of picking remove move? This is in general 1/2. But if
        # there are no nodes to add to in hp, we can only pick the remove move, in which case
        # the prob. should be 1, not 1/2
        # how can we check that? if there was only a single add node in h that is at the maximum_depth and
        # has MAX_CHILDREN children (after adding the new node), there are no add nodes in hp.
        p_remove = (1.0 / 2.0)
        if len(add_nodes) == 1 and len(tree[add_nodes[0]].fpointer) == bdaooss.MAX_CHILDREN \
                and depths[add_nodes[0]] >= (max_depth - 2):
            p_remove = 1.0
        p_h_hp = p_remove * (1.0 / removable_count)

        return hp, p_hp_h, p_h_hp

    # REMOVE PART MOVE -----------------------------
    if move_choice == 1: # remove part
        # remove a part randomly
        node_to_remove = np.random.choice(remove_nodes)
        parent_node = tree[node_to_remove].bpointer
        hp.shape.remove_part(node_to_remove)

        # prob. of hp given h is
        #   prob. of picking the remove move * prob. of picking the node
        # Note if there are no add_nodes, remove move was the only available option, then prob. of
        # picking the remove move should be 1.0
        p_remove = (1.0 / 2.0)
        if len(add_nodes) == 0:
            p_remove = 1.0

        p_hp_h = p_remove * (1.0 / len(remove_nodes))

        # prob. of h given hp is
        #   prob. picking the add move * prob. of picking the parent node * prob. docking face * prob. of size
        # prob. of picking the add move is 1/2, but if hp has only a single node, we can't pick the remove move,
        # therefore, prob. of add move is 1 in that case.
        p_add = (1.0 / 2.0)
        if len(sm.spatial_states.keys()) == 1:
            p_add = 1.0
        # what is the number of add_nodes in hp? since we remove one node from h, hp will have one more add_node.
        # note this is only true if the parent node was not already in add_nodes in h.
        # also note, if the removed node was in add_nodes, we lose that one in hp.
        addable_count = len(add_nodes)
        if parent_node not in add_nodes:
            addable_count += 1
        if node_to_remove in add_nodes:
            addable_count -= 1
        # count the number of available faces in parent
        available_face_count = (bdaooss.FACE_COUNT - len(sm.spatial_states[parent_node].occupied_faces))

        p_h_hp = p_add * (1.0 / addable_count) * (1.0 / available_face_count)

        return hp, p_hp_h, p_h_hp


def bdaooss_change_part_size(h, params):
    hp = h.copy()
    sm = hp.shape.spatial_model
    nodes = sm.spatial_states.keys()
    if len(nodes) == 0:
        return hp, 1.0, 1.0
    node = np.random.choice(nodes)
    hp.shape.change_part_size(node)
    p_hp_h = 1.0
    p_h_hp = 1.0
    return hp, p_hp_h, p_h_hp


def bdaooss_change_part_size_local(h, params):
    hp = h.copy()
    sm = hp.shape.spatial_model
    nodes = sm.spatial_states.keys()
    if len(nodes) == 0:
        return hp, 1.0, 1.0
    node = np.random.choice(nodes)
    change = np.random.randn(3) * np.sqrt(params['CHANGE_SIZE_VARIANCE'])
    # if proposed size is not out of bounds ([0, 1])
    if np.all((sm.spatial_states[node].size + change) < 1.02) \
            and np.all((sm.spatial_states[node].size + change) > 0.02):
        sm.spatial_states[node].size += change
        sm.update(hp.shape.tree, hp.shape.grammar)

    return hp, 1.0, 1.0


def bdaooss_change_part_dock_face(h, params):
    hp = h.copy()
    sm = hp.shape.spatial_model
    nodes = sm.spatial_states.keys()
    # remove the root node and any node with no empty faces
    nodes = [node for node in nodes if sm.spatial_states[node].dock_face != bdaooss.NO_FACE]
    if len(nodes) == 0:
        return hp, 1.0, 1.0
    node = np.random.choice(nodes)
    hp.shape.change_part_dock_face(node)
    p_hp_h = 1.0
    p_h_hp = 1.0
    return hp, p_hp_h, p_h_hp


def bdaooss_move_object(h, params):
    hp = h.copy()
    sm = hp.shape.spatial_model
    change = np.random.randn(3) * np.sqrt(params['MOVE_OBJECT_VARIANCE'])
    # if proposed position is out of bounds ([-1.0, 1.0])
    for part in sm.spatial_states.values():
        if np.any((part.position + change) > 1.0) or np.any((part.position + change) < -1.0):
            return hp, 1.0, 1.0
    # if updated position is in bounds
    for part in sm.spatial_states.values():
        part.position += change
    # proposal is symmetric; hence, q(hp|h) = q(h|hp)
    return hp, 1.0, 1.0


if __name__ == "__main__":
    import vision_forward_model as vfm
    import mcmclib.proposal
    import i3d_proposal

    fwm = vfm.VisionForwardModel(render_size=(200, 200))
    h = BDAoOSSShape(forward_model=fwm, viewpoint=[(np.sqrt(2.0), -np.sqrt(2.0), 2.0)], params={'LL_VARIANCE': 0.0001})

    """
    moves = {'bdaooss_add_remove_part': bdaooss_add_remove_part, 'bdaooss_change_part_size': bdaooss_change_part_size,
             'bdaooss_change_part_size_local': bdaooss_change_part_size_local,
             'bdaooss_change_part_dock_face': bdaooss_change_part_dock_face,
             'bdaooss_move_object': bdaooss_move_object, 'change_viewpoint': i3d_proposal.change_viewpoint}
             """

    moves = {'bdaooss_add_remove_part': bdaooss_add_remove_part,
             'bdaooss_change_part_size_local': bdaooss_change_part_size_local,
             'bdaooss_change_part_dock_face': bdaooss_change_part_dock_face,
             'change_viewpoint': i3d_proposal.change_viewpoint}

    params = {'MOVE_OBJECT_VARIANCE': 0.00005,
              'CHANGE_SIZE_VARIANCE': 0.00005,
              'CHANGE_VIEWPOINT_VARIANCE': 30.0}

    proposal = mcmclib.proposal.RandomMixtureProposal(moves, params)

    # data = np.load('data/test1_single_view.npy')
    data = np.load('data/stimuli20150624_144833/o1_single_view.npy')

    # choose sampler
    thinning_period = 2000
    sampler_class = 'mh'
    if sampler_class == 'mh':
        import mcmclib.mh_sampler
        sampler = mcmclib.mh_sampler.MHSampler(h, data, proposal, burn_in=1000, sample_count=10, best_sample_count=10,
                                               thinning_period=thinning_period, report_period=thinning_period)
    elif sampler_class == 'pt':
        from mcmclib.parallel_tempering_sampler import ParallelTemperingSampler
        sampler = ParallelTemperingSampler(initial_hs=[h, h, h], data=data, proposals=[proposal, proposal, proposal],
                                           temperatures=[3.0, 1.5, 1.0], burn_in=1000, sample_count=10,
                                           best_sample_count=10, thinning_period=int(thinning_period / 3.0),
                                           report_period=int(thinning_period / 3.0))
    else:
        raise ValueError('Unknown sampler class')

    run = sampler.sample()
