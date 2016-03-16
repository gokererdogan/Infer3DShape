"""
Inferring 3D Shape from 2D Images

Unit tests for bdaooss_shape module.

Created on Dec 4, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""
import sys

import treelib
import BDAoOSS.bdaooss_grammar as bd
from mcmclib.proposal import DeterministicMixtureProposal
from mcmclib.mh_sampler import MHSampler

from i3d_test_case import *
from Infer3DShape.bdaooss_shape import *
from Infer3DShape.bdaooss_shape_maxd import *


class BDAoOSSShapeTestHypothesis(BDAoOSSShape):
    def __init__(self, shape=None):
        BDAoOSSShape.__init__(self, forward_model=None, shape=shape)

    def _calculate_log_likelihood(self, data=None):
        return 0.0

    def copy(self):
        shape_copy = self.shape.copy()
        return BDAoOSSShapeTestHypothesis(shape=shape_copy)


class BDAoOSSShapeTest(I3DTestCase):
    def create_test_shape1(self):
        n1 = bd.ParseNode('P', 0)
        n2 = bd.ParseNode('Null', '')
        t = treelib.Tree()
        t.create_node(n1, identifier='n1')
        t.create_node(n2, identifier='n2', parent='n1')
        ss1 = bd.BDAoOSSSpatialState(size=np.array([0.2, 0.2, 0.2]), position=np.array([0.0, 0.0, 0.0]),
                                     dock_face=bd.NO_FACE, occupied_faces=[])
        sm = bd.BDAoOSSSpatialModel(spatial_states={'n1': ss1})
        s = bd.BDAoOSSShapeState(initial_tree=t, spatial_model=sm)
        h = BDAoOSSShape(forward_model=['old'], shape=s, viewpoint=[(1.0, 1.0, 1.0)], params={'x': 1.0})
        return h

    def create_test_shape2(self):
        n1 = bd.ParseNode('P', 3)
        n2 = bd.ParseNode('P', 0)
        n3 = bd.ParseNode('P', 0)
        n4 = bd.ParseNode('P', 0)
        n5 = bd.ParseNode('Null', '')
        n6 = bd.ParseNode('Null', '')
        n7 = bd.ParseNode('Null', '')
        t = treelib.Tree()
        t.create_node(n1, identifier='n1')
        t.create_node(n2, identifier='n2', parent='n1')
        t.create_node(n3, identifier='n3', parent='n1')
        t.create_node(n4, identifier='n4', parent='n1')
        t.create_node(n5, identifier='n5', parent='n2')
        t.create_node(n6, identifier='n6', parent='n3')
        t.create_node(n7, identifier='n7', parent='n4')
        ss1 = bd.BDAoOSSSpatialState(size=np.array([0.5, 0.5, 0.5]), position=np.array([0.0, 0.0, 0.0]),
                                     dock_face=bd.NO_FACE, occupied_faces=[1, 2, 3])
        ss2 = bd.BDAoOSSSpatialState(size=np.array([0.3, 0.3, 0.4]), position=np.array([-0.4, 0.0, 0.0]),
                                     dock_face=1, occupied_faces=[0])
        ss3 = bd.BDAoOSSSpatialState(size=np.array([0.2, 0.3, 0.3]), position=np.array([0.0, 0.4, 0.0]),
                                     dock_face=2, occupied_faces=[3])
        ss4 = bd.BDAoOSSSpatialState(size=np.array([0.4, 0.1, 0.3]), position=np.array([0.0, -0.3, 0.0]),
                                     dock_face=3, occupied_faces=[2])

        sm = bd.BDAoOSSSpatialModel(spatial_states={'n1': ss1, 'n2': ss2, 'n3': ss3, 'n4': ss4})
        s = bd.BDAoOSSShapeState(initial_tree=t, spatial_model=sm)
        h = BDAoOSSShape(forward_model=['old'], shape=s, viewpoint=[(1.0, 1.0, 1.0)], params={'x': 1.0})
        return h

    def setUp(self):
        self.h1 = self.create_test_shape1()
        self.h2 = self.create_test_shape2()

    def tearDown(self):
        self.h1 = None
        self.h2 = None

    def test_bdaooss_prior(self):
        self.assertAlmostEqual(self.h1.log_prior(), np.log(1.0 / 4.0))
        h, p1, p2 = bdaooss_add_remove_part(self.h1, params=None)
        self.assertAlmostEqual(h.log_prior(), np.log(1.0 / (4.0 * 4.0 * 6.0)))
        self.assertAlmostEqual(self.h2.log_prior(), np.log((1.0 / 4.0)**4 * (1.0 / 20.0)))

    def test_bdaooss_convert_to_parts_positions(self):
        pos, size = self.h1.convert_to_positions_sizes()
        self.assertNumpyArrayListEqual(pos, [np.array([0.0, 0.0, 0.0])])
        self.assertNumpyArrayListEqual(size, [np.array([0.2, 0.2, 0.2])])
        pos, size = self.h2.convert_to_positions_sizes()
        self.assertNumpyArrayListEqual(pos, [np.array([0.0, 0.0, 0.0]), np.array([-0.4, 0.0, 0.0]),
                                             np.array([0.0, 0.4, 0.0]), np.array([0.0, -0.3, 0.0])])
        self.assertNumpyArrayListEqual(size, [np.array([0.5, 0.5, 0.5]), np.array([0.3, 0.3, 0.4]),
                                              np.array([0.2, 0.3, 0.3]), np.array([0.4, 0.1, 0.3])])

    def test_bdaooss_copy(self):
        h = self.h1.copy()
        self.assertEqual(h, self.h1)
        # viewpoint should be copied
        h.viewpoint = [tuple(np.random.rand(3))]
        self.assertNotEqual(h.viewpoint[0], self.h1.viewpoint[0])
        # forward model should not be copied
        h.forward_model[0] = 'new'
        self.assertListEqual(h.forward_model, self.h1.forward_model)
        # params should not be copied
        h.params['x'] = 2.0
        self.assertDictEqual(h.params, self.h1.params)
        # parts should be copied
        h.shape.spatial_model.spatial_states.values()[0].position = [-1.0, -1.0, -1.0]
        self.assertNotEqual(h, self.h1)
        del h.shape.spatial_model.spatial_states[h.shape.spatial_model.spatial_states.keys()[0]]
        self.assertNotEqual(h, self.h1)

    def test_bdaooss_eq(self):
        h1_copy = self.h1.copy()
        self.assertEqual(h1_copy, self.h1)
        h1_copy.shape.spatial_model.spatial_states.values()[0].size = [-0.1, -0.1, -0.1]
        self.assertNotEqual(h1_copy, self.h1)
        h2_copy = self.h2.copy()
        self.assertEqual(h2_copy, self.h2)
        h2_copy.shape.spatial_model.spatial_states.values()[0].size = [-0.1, -0.1, -0.1]
        self.assertNotEqual(h2_copy, self.h2)

    def test_bdaooss_maxd_prior(self):
        # BDAoOSShapeMaxD has the same prior as BDAoOSShape
        h1 = BDAoOSSShapeMaxD(None, shape=self.h1.shape)
        self.assertAlmostEqual(h1.log_prior(), np.log(1.0 / 4.0))
        h2 = BDAoOSSShapeMaxD(None, shape=self.h2.shape)
        self.assertAlmostEqual(h2.log_prior(), np.log((1.0 / 4.0)**4 * (1.0 / 20.0)))

    def test_bdaooss_add_remove_part(self):
        # test with no max_depth limit

        # only possible move is add
        hp, p_hp_h, p_h_hp = bdaooss_add_remove_part(self.h1, None)
        self.assertNotEqual(self.h1, hp)
        self.assertEqual(len(hp.shape.spatial_model.spatial_states), 2)
        self.assertAlmostEqual(p_hp_h, (1.0 / 6.0))
        self.assertAlmostEqual(p_h_hp, (1.0 / 2.0))
        self.assertEqual(len(hp.shape.spatial_model.spatial_states['n1'].occupied_faces), 1)

        # both add and remove are possible
        hp, p_hp_h, p_h_hp = bdaooss_add_remove_part(self.h2, None)
        self.assertNotEqual(self.h2, hp)
        part_count = len(hp.shape.spatial_model.spatial_states)
        self.assertIn(part_count, [3, 5])
        if part_count == 3:
            # remove move
            self.assertAlmostEqual(p_hp_h, ((1.0 / 2.0) * (1.0 / 3.0)))
            self.assertAlmostEqual(p_h_hp, ((1.0 / 2.0) * (1.0 / 3.0) * (1.0 / 4.0)))
            self.assertEqual(len(hp.shape.spatial_model.spatial_states['n1'].occupied_faces), 2)
        else:
            # add move
            self.assertAlmostEqual(p_hp_h, ((1.0 / 2.0) * (1.0 / 3.0) * (1.0 / 5.0)))
            self.assertAlmostEqual(p_h_hp, ((1.0 / 2.0) * (1.0 / 3.0)))
            l = [len(hp.shape.spatial_model.spatial_states['n2'].occupied_faces),
                 len(hp.shape.spatial_model.spatial_states['n3'].occupied_faces),
                 len(hp.shape.spatial_model.spatial_states['n4'].occupied_faces)]
            self.assertItemsEqual(l, [1, 1, 2])

        # test with max_depth limit
        self.assertRaises(ValueError, bdaooss_add_remove_part, self.h2, {'MAX_DEPTH': 1})

        hp, p_hp_h, p_h_hp = bdaooss_add_remove_part(self.h1, {'MAX_DEPTH': 1})
        self.assertEqual(hp, self.h1)
        self.assertEqual(p_h_hp, 1.0)
        self.assertEqual(p_hp_h, 1.0)

        # only possible move should be remove
        hp, p_hp_h, p_h_hp = bdaooss_add_remove_part(self.h2, {'MAX_DEPTH': 2})
        self.assertNotEqual(hp, self.h2)
        self.assertEqual(len(hp.shape.spatial_model.spatial_states), 3)
        self.assertAlmostEqual(p_hp_h, (1.0 / 3.0))
        self.assertAlmostEqual(p_h_hp, (1.0 / 2.0) * (1.0 / 4.0))
        self.assertEqual(len(hp.shape.spatial_model.spatial_states['n1'].occupied_faces), 2)

    @unittest.skipIf('--nosampling' in sys.argv, "Sampling tests are turned off.")
    def test_bdaooss_add_remove_sample(self):
        # test if add_remove samples correctly
        # sample from the prior using add_remove and see if we get each tree with its expected probability.
        # here we constrain depth to 1; so, there are 4 possible trees: P, P->P, P->[P, P], P->[P, P, P].
        # p({P}) = 1/4, p({P->P}) = 1/16, p({P->{P, P}) = 1/64, p({P->{P,P,P}) = 1/256
        # therefore, we expect to see P 64/85 of the time, P->P 16/85 of the time, P->[P, P] 4/85 of the time, and
        # P->[P, P, P] 1/85 of the time.
        h = BDAoOSSShapeTestHypothesis(shape=self.h1.shape)
        p = DeterministicMixtureProposal(moves={'add_remove': bdaooss_add_remove_part}, params={'MAX_DEPTH': 2})
        sampler = MHSampler(initial_h=h, proposal=p, data=None, burn_in=0, sample_count=50000, best_sample_count=1,
                            thinning_period=1, report_period=5000)
        run = sampler.sample()

        k = np.array([len(s.shape.spatial_model.spatial_states) for s in run.samples.samples])

        self.assertAlmostEqual(np.mean(k == 1), 64/85.0, places=1)
        self.assertAlmostEqual(np.mean(k == 2), 16/85.0, places=1)
        self.assertAlmostEqual(np.mean(k == 3), 4/85.0, places=1)
        self.assertAlmostEqual(np.mean(k == 4), 1/85.0, places=1)

    def test_bdaooss_change_part_size(self):
        hp, p_hp_h, p_h_hp = bdaooss_change_part_size(self.h1, None)
        self.assertNotEqual(hp, self.h1)
        self.assertNumpyArrayNotEqual(hp.shape.spatial_model.spatial_states['n1'].size,
                                      self.h1.shape.spatial_model.spatial_states['n1'].size)
        self.assertNumpyArrayEqual(hp.shape.spatial_model.spatial_states['n1'].position,
                                   self.h1.shape.spatial_model.spatial_states['n1'].position)

        hp, p_hp_h, p_h_hp = bdaooss_change_part_size(self.h2, None)
        self.assertEqual(p_hp_h, 1.0)
        self.assertEqual(p_h_hp, 1.0)
        hp_ss = hp.shape.spatial_model.spatial_states
        h2_ss = self.h2.shape.spatial_model.spatial_states
        # find the node that changed size
        nodes = ['n1', 'n2', 'n3', 'n4']
        changed_node = None
        for node in nodes:
            if np.any(hp_ss[node].size != h2_ss[node].size):
                changed_node = node
                break
        self.assertIsNotNone(changed_node)
        if changed_node != 'n1':
            self.assertNumpyArrayNotEqual(hp_ss[changed_node].position, h2_ss[changed_node].position)
        else:
            self.assertNumpyArrayEqual(hp_ss[changed_node].position, h2_ss[changed_node].position)
        nodes.remove(changed_node)
        for node in nodes:
            self.assertNumpyArrayEqual(hp_ss[node].size, h2_ss[node].size)
            if changed_node == 'n1':
                self.assertNumpyArrayNotEqual(hp_ss[node].position, h2_ss[node].position)
            else:
                self.assertNumpyArrayEqual(hp_ss[node].position, h2_ss[node].position)

    def test_bdaooss_change_part_size_local(self):
        # NOTE, with a very low probability, these tests might fail when the size change exceeds the size bound [0, 1]
        hp, p_hp_h, p_h_hp = bdaooss_change_part_size_local(self.h1, {'CHANGE_SIZE_VARIANCE': 0.001})
        self.assertNotEqual(hp, self.h1)
        self.assertNumpyArrayNotEqual(hp.shape.spatial_model.spatial_states['n1'].size,
                                      self.h1.shape.spatial_model.spatial_states['n1'].size)
        self.assertNumpyArrayEqual(hp.shape.spatial_model.spatial_states['n1'].position,
                                   self.h1.shape.spatial_model.spatial_states['n1'].position)

        hp, p_hp_h, p_h_hp = bdaooss_change_part_size_local(self.h2, {'CHANGE_SIZE_VARIANCE': 0.001})
        self.assertEqual(p_hp_h, 1.0)
        self.assertEqual(p_h_hp, 1.0)
        hp_ss = hp.shape.spatial_model.spatial_states
        h2_ss = self.h2.shape.spatial_model.spatial_states
        # find the node that changed size
        nodes = ['n1', 'n2', 'n3', 'n4']
        changed_node = None
        for node in nodes:
            if np.any(hp_ss[node].size != h2_ss[node].size):
                changed_node = node
                break
        self.assertIsNotNone(changed_node)
        if changed_node != 'n1':
            self.assertNumpyArrayNotEqual(hp_ss[changed_node].position, h2_ss[changed_node].position)
        else:
            self.assertNumpyArrayEqual(hp_ss[changed_node].position, h2_ss[changed_node].position)
        nodes.remove(changed_node)
        for node in nodes:
            self.assertNumpyArrayEqual(hp_ss[node].size, h2_ss[node].size)
            if changed_node == 'n1':
                self.assertNumpyArrayNotEqual(hp_ss[node].position, h2_ss[node].position)
            else:
                self.assertNumpyArrayEqual(hp_ss[node].position, h2_ss[node].position)

    def test_bdaooss_change_part_dock_face(self):
        hp, p_hp_h, p_h_hp = bdaooss_change_part_dock_face(self.h1, None)
        self.assertEqual(hp, self.h1)
        self.assertEqual(p_hp_h, 1.0)
        self.assertEqual(p_h_hp, 1.0)

        hp, p_hp_h, p_h_hp = bdaooss_change_part_dock_face(self.h2, None)
        hp_ss = hp.shape.spatial_model.spatial_states
        nodes = ['n2', 'n3', 'n4']
        old_dock_faces = {'n2': 1, 'n3': 2, 'n4': 3}
        changed_node = None
        for node in nodes:
            if hp_ss[node].dock_face != old_dock_faces[node]:
                changed_node = node
                break
        self.assertIsNotNone(changed_node)
        self.assertNotIn(old_dock_faces[changed_node], hp_ss['n1'].occupied_faces)
        self.assertIn(hp_ss[changed_node].dock_face, hp_ss['n1'].occupied_faces)
        nodes.remove(changed_node)
        for node in nodes:
            self.assertIn(old_dock_faces[node], hp_ss['n1'].occupied_faces)

    def test_bdaooss_move_object(self):
        # NOTE, with a very low probability, these tests might fail when the position change exceeds the position
        # bound [-1, 1]
        hp, p_hp_h, p_h_hp = bdaooss_move_object(self.h1, {'MOVE_OBJECT_VARIANCE': 0.001})
        self.assertNotEqual(hp, self.h1)
        self.assertNumpyArrayNotEqual(hp.shape.spatial_model.spatial_states['n1'].position,
                                      self.h1.shape.spatial_model.spatial_states['n1'].position)
        self.assertNumpyArrayEqual(hp.shape.spatial_model.spatial_states['n1'].size,
                                   self.h1.shape.spatial_model.spatial_states['n1'].size)
        self.assertEqual(p_hp_h, 1.0)
        self.assertEqual(p_h_hp, 1.0)

        hp, p_hp_h, p_h_hp = bdaooss_move_object(self.h2, {'MOVE_OBJECT_VARIANCE': 0.001})
        self.assertNotEqual(hp, self.h2)
        hp_ss = hp.shape.spatial_model.spatial_states
        h2_ss = self.h2.shape.spatial_model.spatial_states
        change = hp_ss['n1'].position - h2_ss['n1'].position
        self.assertNumpyArrayNotEqual(change, [0.0, 0.0, 0.0])
        self.assertNumpyArrayEqual(hp_ss['n2'].position, h2_ss['n2'].position + change)
        self.assertNumpyArrayEqual(hp_ss['n3'].position, h2_ss['n3'].position + change)
        self.assertNumpyArrayEqual(hp_ss['n4'].position, h2_ss['n4'].position + change)
        self.assertNumpyArrayEqual(hp_ss['n1'].size, h2_ss['n1'].size)
        self.assertNumpyArrayEqual(hp_ss['n2'].size, h2_ss['n2'].size)
        self.assertNumpyArrayEqual(hp_ss['n3'].size, h2_ss['n3'].size)
        self.assertNumpyArrayEqual(hp_ss['n4'].size, h2_ss['n4'].size)
