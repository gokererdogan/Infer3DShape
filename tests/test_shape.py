"""
Inferring 3D Shape from 2D Images

Unit tests for shape module.

Created on Dec 2, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import unittest

from mcmclib.proposal import DeterministicMixtureProposal
from mcmclib.mh_sampler import MHSampler

from Infer3DShape.shape import *
from Infer3DShape.shape_maxn import *


class ShapeTestHypothesis(ShapeMaxN):
    def __init__(self, maxn=3, parts=None):
        ShapeMaxN.__init__(self, forward_model=None, maxn=maxn, parts=parts)

    def _calculate_log_likelihood(self, data=None):
        return 0.0

    def _calculate_log_prior(self):
        return np.log(1.0 / len(self.parts))

    def copy(self):
        self_copy = ShapeTestHypothesis(maxn=self.maxn)
        parts_copy = deepcopy(self.parts)
        self_copy.parts = parts_copy
        return self_copy


class ShapeTest(unittest.TestCase):
    def assertNumpyArrayEqual(self, arr1, arr2):
        if np.any(arr1 != arr2):
            raise AssertionError("Numpy arrays are not equal: {0:s} - {1:s}".format(arr1, arr2))

    def assertNumpyArrayNotEqual(self, arr1, arr2):
        if np.all(arr1 == arr2):
            raise AssertionError("Numpy arrays are equal: {0:s} - {1:s}".format(arr1, arr2))

    def test_cuboid_primitive(self):
        self.assertRaises(ValueError, CuboidPrimitive, position=(-2.0, 0.0, 0.0), size=(0.2, 0.2, 0.2))
        self.assertRaises(ValueError, CuboidPrimitive, position=(0.0, 0.0, 0.0), size=(-0.2, 0.2, 0.2))
        pos = (0.0, 0.0, 0.0)
        size = (0.2, 0.2, 0.2)
        p1 = CuboidPrimitive(position=pos, size=size)
        p2 = CuboidPrimitive(position=pos, size=size)
        p3 = CuboidPrimitive()
        self.assertTrue(np.all(p1.position == (0.0, 0.0, 0.0)))
        self.assertTrue(np.all(p1.size == (0.2, 0.2, 0.2)))
        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3, msg="This test might fail randomly; don't worry about it.")
        # updating p2 should not change p1
        p2.position[0] = 0.1
        self.assertNumpyArrayNotEqual(p1.position, p2.position)
        p2.size[2] = 0.5
        self.assertNumpyArrayNotEqual(p1.size, p2.size)

    def test_shape_from_narray(self):
        pos1 = [0.0, 0.0, 0.0]
        pos2 = [0.1, -0.1, 0.2]
        size1 = [0.2, 0.3, 0.4]
        size2 = [0.2, 0.3, 0.1]
        arr = np.array(pos1 + size1 + pos2 + size2)
        s = Shape.from_narray(arr, forward_model=None)
        self.assertEqual(s.parts[0], CuboidPrimitive(pos1, size1))
        self.assertEqual(s.parts[1], CuboidPrimitive(pos2, size2))
        size2 = [0.0, 0.0, 0.0]
        arr = np.array(pos1 + size1 + pos2 + size2)
        s = Shape.from_narray(arr, forward_model=None)
        self.assertEqual(s.parts[0], CuboidPrimitive(pos1, size1))
        self.assertEqual(len(s.parts), 1)

    def test_shape_prior(self):
        pos1 = [0.0, 0.0, 0.0]
        pos2 = [0.1, -0.1, 0.2]
        size1 = [0.2, 0.3, 0.4]
        size2 = [0.2, 0.3, 0.1]
        part1 = CuboidPrimitive(pos1, size1)
        part2 = CuboidPrimitive(pos2, size2)
        s = Shape(forward_model=None, parts=[part1, part2])
        # Prior(H) = (1 / (part_count + 1)(part_count + 2))
        self.assertAlmostEqual(np.exp(s.log_prior()), (1. / (3. * 4.)))
        del s.parts[0]
        # Remember that once prior and likelihood are calculated, they are never calculated again.
        # You need to do it if you change the object
        self.assertAlmostEqual(np.exp(s._calculate_log_prior()), (1. / (2. * 3.)))

    def test_shape_convert_to_parts_positions(self):
        pos1 = [0.0, 0.0, 0.0]
        pos2 = [0.1, -0.1, 0.2]
        size1 = [0.2, 0.3, 0.4]
        size2 = [0.2, 0.3, 0.1]
        part1 = CuboidPrimitive(pos1, size1)
        part2 = CuboidPrimitive(pos2, size2)
        s = Shape(forward_model=None, parts=[part1, part2])
        pos, size = s.convert_to_positions_sizes()
        self.assertNumpyArrayEqual(pos1, pos[0])
        self.assertNumpyArrayEqual(pos2, pos[1])
        self.assertNumpyArrayEqual(size1, size[0])
        self.assertNumpyArrayEqual(size2, size[1])

    def test_shape_copy(self):
        pos1 = [0.0, 0.0, 0.0]
        pos2 = [0.1, -0.1, 0.2]
        size1 = [0.2, 0.3, 0.4]
        size2 = [0.2, 0.3, 0.1]
        part1 = CuboidPrimitive(pos1, size1)
        part2 = CuboidPrimitive(pos2, size2)
        s1 = Shape(forward_model=['old'], parts=[part1, part2], viewpoint=[(1.0, 1.0, 1.0)], params={'x': 1.0})
        s2 = s1.copy()
        self.assertEqual(s1, s2)
        # viewpoint should be copied
        s2.viewpoint = [tuple(np.random.rand(3))]
        self.assertNotEqual(s1.viewpoint[0], s2.viewpoint[0])
        # forward model should not be copied
        s2.forward_model[0] = 'new'
        self.assertListEqual(s1.forward_model, s2.forward_model)
        # params should not be copied
        s2.params['x'] = 2.0
        self.assertDictEqual(s1.params, s2.params)
        # parts should be copied
        s2.parts[0].position = [-1.0, -1.0, -1.0]
        self.assertNotEqual(s1, s2)
        del s2.parts[0]
        self.assertNotEqual(s1, s2)

    def test_shape_equal(self):
        s1 = Shape(None)
        s2 = Shape(None)
        self.assertNotEqual(s1, s2)
        pos1 = [0.0, 0.0, 0.0]
        pos2 = [0.1, -0.1, 0.2]
        size1 = [0.2, 0.3, 0.4]
        size2 = [0.2, 0.3, 0.1]
        part1 = CuboidPrimitive(pos1, size1)
        part1_c = CuboidPrimitive(pos1, size1)
        part2 = CuboidPrimitive(pos2, size2)
        part2_c = CuboidPrimitive(pos2, size2)
        s1 = Shape(forward_model=['old'], parts=[part1, part2], viewpoint=[(1.0, 1.0, 1.0)], params={'x': 1.0})
        s2 = Shape(forward_model=['old'], parts=[part2_c, part1_c], viewpoint=[(1.0, 1.0, 1.0)], params={'x': 1.0})
        self.assertEqual(s1, s2)
        s1.forward_model = None
        self.assertEqual(s1, s2)
        s1.viewpoint = None
        self.assertEqual(s1, s2)
        s1.params = None
        self.assertEqual(s1, s2)
        s1.parts[0].position = np.array([-1.0, -2.0, -3.0])
        self.assertNotEqual(s1, s2)
        del s1.parts[0]
        self.assertNotEqual(s1, s2)

    def test_shape_to_narray(self):
        pos1 = [0.0, 0.0, 0.0]
        pos2 = [0.1, -0.1, 0.2]
        size1 = [0.2, 0.3, 0.4]
        size2 = [0.2, 0.3, 0.1]
        part1 = CuboidPrimitive(pos1, size1)
        part2 = CuboidPrimitive(pos2, size2)
        s = Shape(forward_model=['old'], parts=[part1, part2], viewpoint=[(1.0, 1.0, 1.0)], params={'x': 1.0})
        arr = s.to_narray()
        self.assertNumpyArrayEqual(arr, np.array(pos1 + size1 + pos2 + size2))
        s.parts[0].position = np.array([1.0, 1.0, 1.0])
        arr = s.to_narray()
        self.assertNumpyArrayEqual(arr, np.array([1.0, 1.0, 1.0] + size1 + pos2 + size2))
        del s.parts[0]
        arr = s.to_narray()
        self.assertNumpyArrayEqual(arr, np.array(pos2 + size2))

    def test_shape_add_remove_part(self):
        s = Shape(forward_model=['old'], parts=[], viewpoint=[(1.0, 1.0, 1.0)], params={'x': 1.0})
        self.assertRaises(ValueError, shape_add_remove_part, s, {})
        pos1 = [0.0, 0.0, 0.0]
        size1 = [0.2, 0.3, 0.4]
        part1 = CuboidPrimitive(pos1, size1)
        # object with one part
        s = Shape(forward_model=['old'], parts=[part1], viewpoint=[(1.0, 1.0, 1.0)], params={'x': 1.0})
        # only possible move is add part
        sp, p_hp_h, p_h_hp = shape_add_remove_part(s, {})
        self.assertNotEqual(s, sp)
        self.assertEqual(len(sp.parts), 2)
        self.assertAlmostEqual(p_hp_h, 1.0 / 2.0)
        self.assertAlmostEqual(p_h_hp, 1.0 / 4.0)
        # random object
        s = Shape(forward_model=None)
        part_count = len(s.parts)
        sp, q_hp_h, q_h_hp = shape_add_remove_part(s, {})
        new_part_count = len(sp.parts)
        self.assertIn(new_part_count, [part_count - 1, part_count + 1])
        self.assertNotEqual(s, sp)
        if new_part_count > part_count:  # add move
            p_hp_h = 0.5 * (1.0 / new_part_count)
            p_h_hp = 0.5 * (1.0 / new_part_count)
            if part_count == 1:
                p_hp_h = 1.0 * (1.0 / new_part_count)
        else:
            p_hp_h = 0.5 * (1.0 / part_count)
            p_h_hp = 0.5 * (1.0 / part_count)
            if part_count == 2:
                p_h_hp = 1.0 * (1.0 / part_count)

        self.assertAlmostEqual(q_hp_h, p_hp_h)
        self.assertAlmostEqual(q_h_hp, p_h_hp)

        # test if add_remove_part traverses the sample space correctly
        # look at how much probability mass objects with a certain number of parts get
        maxn = 3
        s = ShapeTestHypothesis(parts=[CuboidPrimitive()], maxn=3)
        sample_count = 40000
        p = DeterministicMixtureProposal(moves={'add_remove': shape_add_remove_part}, params={'MAX_PART_COUNT': maxn})
        sampler = MHSampler(initial_h=s, data=None, proposal=p, burn_in=5000, sample_count=sample_count,
                            best_sample_count=1, thinning_period=1, report_period=4000)
        run = sampler.sample()
        # count how many times we sampled each part_count
        k = np.array([len(sample.parts) for sample in run.samples.samples])
        # since the prior p(h) = 1 / part_count and part_count is 1,2 or 3, we expect 6/11 of all samples to have a
        # single part, 3/11 to have 2 parts, and 2/11 to have 3 parts.
        self.assertAlmostEqual(np.mean(k == 1), 6.0 / 11.0, places=1)
        self.assertAlmostEqual(np.mean(k == 2), 3.0 / 11.0, places=1)
        self.assertAlmostEqual(np.mean(k == 3), 2.0 / 11.0, places=1)

    def test_shape_add_remove_part_max_limit(self):
        s = Shape(forward_model=['old'], parts=[], viewpoint=[(1.0, 1.0, 1.0)], params={'x': 1.0})
        self.assertRaises(ValueError, shape_add_remove_part, s, {})

        pos1 = [0.0, 0.0, 0.0]
        size1 = [0.2, 0.3, 0.4]
        part1 = CuboidPrimitive(pos1, size1)
        # object with one part
        s = Shape(forward_model=['old'], parts=[part1], viewpoint=[(1.0, 1.0, 1.0)], params={'x': 1.0})

        self.assertRaises(ValueError, shape_add_remove_part, s, {'MAX_PART_COUNT': 0})

        # if max_part_count = 1, returns the same object
        sp, p_hp_h, p_h_hp = shape_add_remove_part(s, {'MAX_PART_COUNT': 1})
        self.assertEqual(s, sp)
        self.assertAlmostEqual(p_hp_h, 1.0)
        self.assertAlmostEqual(p_h_hp, 1.0)

        # only possible move is add part, and max_part_count is 2
        sp, p_hp_h, p_h_hp = shape_add_remove_part(s, {'MAX_PART_COUNT': 2})
        self.assertNotEqual(s, sp)
        self.assertEqual(len(sp.parts), 2)
        self.assertAlmostEqual(p_hp_h, 1.0 / 2.0)
        self.assertAlmostEqual(p_h_hp, 1.0 / 2.0)

        s.parts.append(CuboidPrimitive())
        # only possible move is remove part
        sp, p_hp_h, p_h_hp = shape_add_remove_part(s, {'MAX_PART_COUNT': 2})
        self.assertNotEqual(s, sp)
        self.assertEqual(len(sp.parts), 1)
        self.assertAlmostEqual(p_hp_h, 1.0 / 2.0)
        self.assertAlmostEqual(p_h_hp, 1.0 / 2.0)

    def test_shape_move_part(self):
        s = Shape(None)
        sp, q1, q2 = shape_move_part(s, None)
        self.assertNotEqual(s, sp)
        self.assertEqual(q1, q2)

    def test_shape_move_part_local(self):
        s = Shape(None)
        sp, q1, q2 = shape_move_part_local(s, {'MOVE_PART_VARIANCE': 0.001})
        self.assertNotEqual(s, sp)
        self.assertEqual(q1, q2)

        pos1 = [0.0, 0.0, 0.0]
        size1 = [0.2, 0.3, 0.4]
        part1 = CuboidPrimitive(pos1, size1)
        part1.position[0] = 100.0
        # object with one part
        s = Shape(forward_model=['old'], parts=[part1], viewpoint=[(1.0, 1.0, 1.0)], params={'x': 1.0})
        sp, q1, q2 = shape_move_part_local(s, {'MOVE_PART_VARIANCE': 0.001})
        # should not move any parts because they are out of bounds
        self.assertEqual(s, sp)
        self.assertEqual(q1, q2)

    def test_shape_change_part_size(self):
        s = Shape(None)
        sp, q1, q2 = shape_change_part_size(s, None)
        self.assertNotEqual(s, sp)
        self.assertEqual(q1, q2)

    def test_shape_change_part_size_local(self):
        s = Shape(None)
        sp, q1, q2 = shape_change_part_size_local(s, {'CHANGE_SIZE_VARIANCE': 0.001})
        self.assertNotEqual(s, sp)
        self.assertEqual(q1, q2)

        pos1 = [0.0, 0.0, 0.0]
        size1 = [0.2, 0.3, 0.4]
        part1 = CuboidPrimitive(pos1, size1)
        part1.size[0] = 100.0
        # object with one part
        s = Shape(forward_model=['old'], parts=[part1], viewpoint=[(1.0, 1.0, 1.0)], params={'x': 1.0})
        sp, q1, q2 = shape_change_part_size_local(s, {'CHANGE_SIZE_VARIANCE': 0.001})
        # should change size because they are out of bounds
        self.assertEqual(s, sp)
        self.assertEqual(q1, q2)

    def test_shape_move_object(self):
        s = Shape(None)
        sp, q1, q2 = shape_move_object(s, {'MOVE_OBJECT_VARIANCE': 0.001})
        self.assertNotEqual(s, sp)
        self.assertEqual(q1, q2)

        pos1 = [0.0, 0.0, 0.0]
        size1 = [0.2, 0.3, 0.4]
        part1 = CuboidPrimitive(pos1, size1)
        part1.position[0] = 100.0
        # object with one part
        s = Shape(forward_model=['old'], parts=[part1], viewpoint=[(1.0, 1.0, 1.0)], params={'x': 1.0})
        sp, q1, q2 = shape_move_object(s, {'MOVE_OBJECT_VARIANCE': 0.001})
        # should not move any parts because they are out of bounds
        self.assertEqual(s, sp)
        self.assertEqual(q1, q2)

    def test_shape_maxn(self):
        s = ShapeMaxN(forward_model=['old'], maxn=10)
        self.assertLessEqual(len(s.parts), 10)
