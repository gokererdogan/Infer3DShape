"""
Inferring 3D Shape from 2D Images

This file contains the shape classes for BDAoOSS experiment.

Created on Oct 6, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""
import numpy as np
from copy import deepcopy

import BDAoOSS.bdaooss_grammar as bdaooss
import hypothesis as hyp

class BDAoOSSShape(hyp.Hypothesis):
    """
    This class implements a shape grammar hypothesis based on the true grammar for
    the objects in the experiment.
    """
    def __init__(self, forward_model, shape=None, viewpoint=None, params=None):
        """
        Initialize BDAoOSSShape instance
        :param forward_model: Instance of VisionForwardModel used for rendering
        :param viewpoint: Viewing orientation. Used only if viewpoint should also be inferred
        :return: Instance of BDAoOSSShape instance
        """
        self.forward_model = forward_model
        self.viewpoint = viewpoint

        self.shape = shape
        if self.shape is None:
            sm = bdaooss.BDAoOSSSpatialModel(size_prior=bdaooss.size_prior_infer3dshape)
            self.shape = bdaooss.BDAoOSSShapeState(spatial_model=sm)

        self.params = params
        if self.params is None:
            self.params = {'ADD_OBJECT_PROB': hyp.ADD_PART_PROB, 'LL_VARIANCE': hyp.LL_VARIANCE,
                           'MAX_PIXEL_VALUE': hyp.MAX_PIXEL_VALUE, 'LL_FILTER_SIGMA': hyp.LL_FILTER_SIGMA}

        hyp.Hypothesis.__init__(self)

    def prior(self):
        """
        Prior for the hypothesis. We use the prior provided by BDAoOSSShapeState class.
        :return:
        """
        if self.p is None:
            part_count = len(self.shape.spatial_model.spatial_states)
            # prior = prob. of parse tree * prob. of spatial model
            # NOTE that instead of rational rules prior (which integrates out
            # production probabilities) we use the derivation prob. as prior.
            # probability of a spatial state = prob. of size * prob. of position
            # since size is uniform in [0, 1] and position is uniform in [-0.5, 0.5], we get 1.0 for each part.
            self.p = self.shape.derivation_prob
        return self.p

    def likelihood(self, data):
        if self.ll is None:
            self.ll = self._ll_pixel(data)
        return self.ll

    def _ll_pixel(self, data):
        img = self.forward_model.render(self)
        ll = np.exp(-np.sum(np.square((img - data) / self.params['MAX_PIXEL_VALUE']))
                    / (img.size * 2 * self.params['LL_VARIANCE']))
        return ll

    def convert_to_positions_sizes(self):
        """
        Converts the BDAoOSS shape instance to lists of positions and sizes of parts.
        Used by VisionForwardModel for rendering.
        :return: positions and sizes
        """
        parts, positions, sizes, viewpoint = self.shape.convert_to_parts_positions()
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

    def __getstate__(self):
        # we cannot pickle VTKObjects, so get rid of them.
        return {k: v for k, v in self.__dict__.iteritems() if k != 'forward_model'}

class BDAoOSSShapeMaxD(BDAoOSSShape):
    """
    Shape hypothesis class based on BDAoOSS Shape grammar. This class assumes a maximum depth to parse trees and
    assigns a uniform distribution over hypotheses.
    """
    def __init__(self, forward_model, shape=None, viewpoint=None, max_depth=3, params=None):
        """
        Initialize BDAoOSSShapeMaxD instance
        :param forward_model: Instance of VisionForwardModel used for rendering
        :param maxD: Maximum depth allowed for trees
        :return: Instance of BDAoOSSShapeMaxD instance
        """
        self.max_depth = max_depth

        BDAoOSSShape.__init__(self, forward_model=forward_model, shape=shape, viewpoint=viewpoint, params=params)

    def prior(self):
        """
        Prior for the hypothesis. We assume it is uniform.
        :return: 1.0
        """
        return 1.0

    def copy(self):
        # NOTE that we are not copying params. This assumes that params do not change from
        # hypothesis to hypothesis.
        shape_copy = self.shape.copy()
        viewpoint_copy = deepcopy(self.viewpoint)
        return BDAoOSSShapeMaxD(forward_model=self.forward_model, shape=shape_copy, params=self.params,
                                max_depth=self.max_depth, viewpoint=viewpoint_copy)

class BDAoOSSShapeProposal(hyp.Proposal):
    """
    Proposal class that implements the mixture kernel of the following moves
        add/remove part, change docking face, change part size, move object, change viewpoint
    """
    def __init__(self, allow_viewpoint_update=False, params=None):
        hyp.Proposal.__init__(self)

        self.allow_viewpoint_update = allow_viewpoint_update

        self.params = params
        if self.params is None:
            self.params = {'CHANGE_SIZE_VARIANCE': hyp.CHANGE_SIZE_VARIANCE,
                           'MOVE_OBJECT_VARIANCE': hyp.MOVE_OBJECT_VARIANCE,
                           'CHANGE_VIEWPOINT_VARIANCE': hyp.CHANGE_VIEWPOINT_VARIANCE}

    def propose(self, h, *args):
        # pick one move randomly
        if self.allow_viewpoint_update:
            i = np.random.randint(0, 6)
        else:
            i = np.random.randint(0, 5)

        if i == 0:
            info = "add/remove part"
            hp, q_hp_h, q_h_hp = self.add_remove_part(h)
        elif i == 1:
            info = "change part size"
            hp, q_hp_h, q_h_hp = self.change_part_size(h)
        elif i == 2:
            info = "change part size local"
            hp, q_hp_h, q_h_hp = self.change_part_size_local(h)
        elif i == 3:
            info = "change part dock face"
            hp, q_hp_h, q_h_hp = self.change_part_dock_face(h)
        elif i == 4:
            info = "move object"
            hp, q_hp_h, q_h_hp = self.move_object(h)
        elif i == 5:
            info = "change viewpoint"
            hp, q_hp_h, q_h_hp = self.change_viewpoint(h)

        return info, hp, q_hp_h, q_h_hp

    def add_remove_part(self, h):
        # REMEMBER that our grammar allows P nodes to have at most 3 children.
        # we can add nodes under the P nodes that have less than 3 children
        # we can remove P nodes that have a single child Null
        hp = h.copy()
        tree = hp.shape.tree
        sm = hp.shape.spatial_model
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
                if len(tree[node].fpointer) < bdaooss.MAX_CHILDREN:
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
            # the prob. should be 1, not 1/2. However, note that this NEVER happens since we can
            # always add nodes to a tree.
            p_remove = (1.0 / 2.0)
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
            p_remove = (1.0 / 2.0)

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

    def change_part_size(self, h):
        hp = h.copy()
        sm = hp.shape.spatial_model
        nodes = hp.shape.spatial_model.spatial_states.keys()
        if len(nodes) == 0:
            return hp, 1.0, 1.0
        node = np.random.choice(nodes)
        hp.shape.change_part_size(node)
        p_hp_h = 1.0
        p_h_hp = 1.0
        return hp, p_hp_h, p_h_hp

    def change_part_size_local(self, h):
        hp = h.copy()
        sm = hp.shape.spatial_model
        nodes = hp.shape.spatial_model.spatial_states.keys()
        if len(nodes) == 0:
            return hp, 1.0, 1.0
        node = np.random.choice(nodes)
        change = np.random.randn(3) * np.sqrt(self.params['CHANGE_SIZE_VARIANCE'])
        # if proposed size is not out of bounds ([0, 1])
        if np.all((sm.spatial_states[node].size + change) < 1.0) \
                and np.all((sm.spatial_states[node].size + change) > 0.0):
            sm.spatial_states[node].size += change
            sm.update(hp.shape.tree, hp.shape.grammar)

        return hp, 1.0, 1.0

    def change_part_dock_face(self, h):
        hp = h.copy()
        sm = hp.shape.spatial_model
        nodes = hp.shape.spatial_model.spatial_states.keys()
        # remove the root node and any node with no empty faces
        nodes = [node for node in nodes if sm.spatial_states[node].dock_face != bdaooss.NO_FACE]
        if len(nodes) == 0:
            return hp, 1.0, 1.0
        node = np.random.choice(nodes)
        hp.shape.change_part_dock_face(node)
        p_hp_h = 1.0
        p_h_hp = 1.0
        return hp, p_hp_h, p_h_hp

    def move_object(self, h):
        hp = h.copy()
        sm = hp.shape.spatial_model
        change = np.random.randn(3) * np.sqrt(self.params['MOVE_OBJECT_VARIANCE'])
        # if proposed position is out of bounds ([-1, 1])
        for part in sm.spatial_states.values():
            if np.any((part.position + change) > 0.5) or np.any((part.position + change) < -0.5):
                return hp, 1.0, 1.0
        # if updated position is in bounds
        for part in sm.spatial_states.values():
            part.position = part.position + change
        # proposal is symmetric; hence, q(hp|h) = q(h|hp)
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

class BDAoOSSShapeMaxDProposal(BDAoOSSShapeProposal):
    """
    Proposal class that implements the mixture kernel of the following moves
        add/remove part, change docking face, change part size, move object,
        change viewpoint for BDAoOSSShapeMaxD class.
    """
    def __init__(self, allow_viewpoint_update=False, params=None):
        BDAoOSSShapeProposal.__init__(self, allow_viewpoint_update=allow_viewpoint_update, params=params)
        hyp.Proposal.__init__(self)

    def add_remove_part(self, h):
        # REMEMBER that we allow trees of depth D and our grammar allows P nodes
        # to have at most 3 children.
        # we can add nodes under the P nodes that have less than 3 children and are not at depth D
        # we can remove P nodes that have a single child Null
        hp = h.copy()
        tree = hp.shape.tree
        sm = hp.shape.spatial_model
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
                if depths[node] < (hp.max_depth-1) and len(tree[node].fpointer) < bdaooss.MAX_CHILDREN:
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
                    and depths[add_nodes[0]] >= (hp.max_depth - 2):
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

if __name__ == "__main__":
    import vision_forward_model as vfm
    import mcmc_sampler as mcmc
    fwm = vfm.VisionForwardModel(render_size=(200, 200))
    kernel = BDAoOSSShapeMaxDProposal(allow_viewpoint_update=True)

    # generate initial hypothesis shape randomly
    h = BDAoOSSShapeMaxD(fwm, max_depth=3, viewpoint=[(1.5, -1.5, 1.5)])

    # read data (i.e., observed image) from disk
    obj_name = 'test1'
    # data = np.load('./data/stimuli20150624_144833/{0:s}.npy'.format(obj_name))
    data = np.load('./data/test1_single_view.npy')

    sampler = mcmc.MHSampler(h, data, kernel, 0, 10, 20, 200, 400)
    run = sampler.sample()
    print(run.best_samples.samples)
    print()
    print(run.best_samples.probs)
    # run.save('bdaoossMax3_{0:s}.pkl'.format(obj_name))

    for i, sample in enumerate(run.samples.samples):
        fwm.save_render("results/bdaoossShapeMaxD/{0:s}/s{1:d}.png".format(obj_name, i), sample)
    for i, sample in enumerate(run.best_samples.samples):
        fwm.save_render("results/bdaoossShapeMaxD/{0:s}/b{1:d}.png".format(obj_name, i), sample)
