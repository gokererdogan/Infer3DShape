"""
Inferring 3D Shape from 2D Images

Unit tests for voxel_based_shape module.

Created on Dec 24, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import pdb
import unittest

from mcmclib.proposal import RandomMixtureProposal
from mcmclib.mh_sampler import MHSampler

from Infer3DShape.voxel_based_shape import *


class VoxelBasedShapeTestHypothesis(VoxelBasedShapeMaxD):
    def __init__(self, voxel=None, max_depth=2):
        VoxelBasedShapeMaxD.__init__(self, forward_model=None, voxel=voxel, max_depth=max_depth)

    def _calculate_log_likelihood(self, data=None):
        return 0.0

    def _calculate_log_prior(self):
        p = self.voxel.count_voxels_by_status(PARTIAL_VOXEL) - 1
        v = self.voxel.voxels_per_axis**3
        return -np.log(count_trees(partial_count=p, max_voxels=v, max_depth=self.max_depth))

    def copy(self):
        voxel_copy = self.voxel.copy()
        self_copy = VoxelBasedShapeTestHypothesis(voxel=voxel_copy, max_depth=self.max_depth)
        return self_copy

"""
Most of these tests assume that space is [-0.75, 0.75] x [-0.75, 0.75] x [-0.75, 0.75], and there are 2 voxels per axis.
"""


class VoxelBasedShapeTest(unittest.TestCase):
    def assertAlmostIn(self, val, arr):
        found = False
        for comp in arr:
            try:
                self.assertAlmostEqual(val, comp)
            except AssertionError:
                pass
            else:
                found = True
                break
        if not found:
            raise AssertionError("Value cannot be found in array.")

    def assertNumpyArrayEqual(self, arr1, arr2):
        if np.any(arr1 != arr2):
            raise AssertionError("Numpy arrays are not equal: {0:s} - {1:s}".format(arr1, arr2))

    def assertNumpyArrayNotEqual(self, arr1, arr2):
        if np.all(arr1 == arr2):
            raise AssertionError("Numpy arrays are equal: {0:s} - {1:s}".format(arr1, arr2))

    def assertNumpyArrayListEqual(self, l1, l2):
        if len(l1) != len(l2):
            raise AssertionError("Lists {0:s} and {1:s} does not have the same number of elements.".format(l1, l2))
        for i1 in l1:
            found = False
            for i2 in l2:
                if np.all(i1 == i2):
                    found = True
                    break
            if not found:
                raise AssertionError("Item {0:s} cannot be found in list {1:s}".format(i1, l2))

    def create_test_voxel1(self):
        h = Voxel.get_random_voxel(origin=[0.0, 0.0, 0.0], depth=0, voxels_per_axis=2, size=(1.5, 1.5, 1.5))
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    h.subvoxels[x, y, z].status = EMPTY_VOXEL
        return h

    def create_test_voxel2(self):
        h = Voxel.get_random_voxel(origin=[0.0, 0.0, 0.0], depth=0, voxels_per_axis=2, size=(1.5, 1.5, 1.5))
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    h.subvoxels[x, y, z].status = EMPTY_VOXEL
        h.subvoxels[0, 0, 0].status = FULL_VOXEL
        h.subvoxels[1, 1, 1].status = FULL_VOXEL
        return h

    def create_test_voxel3(self):
        h = Voxel.get_random_voxel(origin=[0.0, 0.0, 0.0], depth=0, voxels_per_axis=2, size=(1.5, 1.5, 1.5))
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    h.subvoxels[x, y, z].status = EMPTY_VOXEL
        h.subvoxels[1, 1, 1].status = FULL_VOXEL
        sv = Voxel.get_random_voxel(h.subvoxels[0, 0, 0].origin, 1, voxels_per_axis=2)
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    sv.subvoxels[x, y, z].status = EMPTY_VOXEL
        sv.subvoxels[0, 0, 0].status = FULL_VOXEL
        h.subvoxels[0, 0, 0] = sv
        return h

    def setUp(self):
        self.v1 = self.create_test_voxel1()
        self.v2 = self.create_test_voxel2()
        self.v3 = self.create_test_voxel3()
        self.s1 = VoxelBasedShape(forward_model=None, voxel=self.v1)
        self.s2 = VoxelBasedShape(forward_model=None, voxel=self.v2)
        self.s3 = VoxelBasedShape(forward_model=None, voxel=self.v3)

    def tearDown(self):
        self.v1 = None
        self.v2 = None
        self.v3 = None
        self.s1 = None
        self.s2 = None
        self.s3 = None

    def test_voxel_get_random_subvoxel(self):
        # assumes the whole space is [-0.75, 0.75] x [-0.75, 0.75] x [-0.75, 0.75]
        # assumes 2 voxels per axis
        voxel_xyz = [-0.375, 0.375]
        sv = Voxel.get_random_subvoxels([0.0, 0.0, 0.0], 0, voxels_per_axis=2, size=(1.5, 1.5, 1.5))
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    self.assertNumpyArrayEqual(np.array([voxel_xyz[x], voxel_xyz[y], voxel_xyz[z]]), sv[x, y, z].origin)
                    self.assertIn(sv[x, y, z].status, [EMPTY_VOXEL, FULL_VOXEL])

    def test_voxel_to_parts_positions(self):
        # voxel is empty
        p, s = self.v1.to_parts_positions()
        self.assertListEqual(p, [])
        self.assertListEqual(s, [])
        # voxel has only two full subvoxels
        p, s = self.v2.to_parts_positions()
        self.assertNumpyArrayListEqual(p, [[-0.375, -0.375, -0.375], [0.375, 0.375, 0.375]])
        self.assertNumpyArrayListEqual(s, [[0.75, 0.75, 0.75], [0.75, 0.75, 0.75]])
        # voxel has one full voxel depth 1 and one at depth 2.
        p, s = self.v3.to_parts_positions()
        self.assertNumpyArrayListEqual(p, [[-0.5625, -0.5625, -0.5625], [0.375, 0.375, 0.375]])
        self.assertNumpyArrayListEqual(s, [[0.375, 0.375, 0.375], [0.75, 0.75, 0.75]])

    def test_voxel_get_voxels_by_status(self):
        # v1
        l = self.v1.get_voxels_by_status(EMPTY_VOXEL)
        self.assertEqual(len(l), 8)
        l = self.v1.get_voxels_by_status(FULL_VOXEL)
        self.assertEqual(len(l), 0)
        l = self.v1.get_voxels_by_status(PARTIAL_VOXEL)
        self.assertEqual(len(l), 1)
        self.assertEqual(l[0], self.v1)
        # v2
        l = self.v2.get_voxels_by_status(EMPTY_VOXEL)
        self.assertEqual(len(l), 6)
        l = self.v2.get_voxels_by_status(FULL_VOXEL)
        self.assertEqual(len(l), 2)
        self.assertListEqual(l, [self.v2.subvoxels[0, 0, 0], self.v2.subvoxels[1, 1, 1]])
        l = self.v2.get_voxels_by_status(PARTIAL_VOXEL)
        self.assertEqual(len(l), 1)
        self.assertEqual(l[0], self.v2)
        # v3
        l = self.v3.get_voxels_by_status(EMPTY_VOXEL)
        self.assertEqual(len(l), 13)
        l = self.v3.get_voxels_by_status(FULL_VOXEL)
        self.assertEqual(len(l), 2)
        self.assertListEqual(l, [self.v3.subvoxels[0, 0, 0].subvoxels[0, 0, 0], self.v3.subvoxels[1, 1, 1]])
        l = self.v3.get_voxels_by_status(PARTIAL_VOXEL)
        self.assertEqual(len(l), 2)
        self.assertListEqual(l, [self.v3, self.v3.subvoxels[0, 0, 0]])

    def test_voxel_count_voxels_by_status(self):
        # v1
        c = self.v1.count_voxels_by_status(EMPTY_VOXEL)
        self.assertEqual(c, 8)
        c = self.v1.count_voxels_by_status(FULL_VOXEL)
        self.assertEqual(c, 0)
        c = self.v1.count_voxels_by_status(PARTIAL_VOXEL)
        self.assertEqual(c, 1)
        # v2
        c = self.v2.count_voxels_by_status(EMPTY_VOXEL)
        self.assertEqual(c, 6)
        c = self.v2.count_voxels_by_status(FULL_VOXEL)
        self.assertEqual(c, 2)
        c = self.v2.count_voxels_by_status(PARTIAL_VOXEL)
        self.assertEqual(c, 1)
        # v3
        c = self.v3.count_voxels_by_status(EMPTY_VOXEL)
        self.assertEqual(c, 13)
        c = self.v3.count_voxels_by_status(FULL_VOXEL)
        self.assertEqual(c, 2)
        c = self.v3.count_voxels_by_status(PARTIAL_VOXEL)
        self.assertEqual(c, 2)

    def test_voxel_calculate_depth(self):
        self.assertEqual(self.v1.calculate_depth(), 1)
        self.assertEqual(self.v2.calculate_depth(), 1)
        self.assertEqual(self.v3.calculate_depth(), 2)

    def test_voxel_copy(self):
        # v1
        copy = self.v1.copy()
        self.assertNumpyArrayEqual(copy.origin, self.v1.origin)
        self.assertEqual(copy.depth, self.v1.depth)
        self.assertEqual(copy.status, self.v1.status)
        # changing copy should not change original
        copy.status = -1
        self.assertNotEqual(copy.status, self.v1.status)
        copy.depth = 2
        self.assertNotEqual(copy.depth, self.v1.depth)
        copy.origin[1] = -1.0
        self.assertNumpyArrayNotEqual(copy.origin, self.v1.origin)
        # v3
        copy = self.v3.copy()
        self.assertIsNot(copy.subvoxels[0, 0, 0], self.v3.subvoxels[0, 0, 0])
        self.assertEqual(copy.subvoxels[0, 0, 0], self.v3.subvoxels[0, 0, 0])
        copy.subvoxels[0, 0, 0].subvoxels[0, 0, 0].status = EMPTY_VOXEL
        self.assertNotEqual(copy.subvoxels[0, 0, 0], self.v3.subvoxels[0, 0, 0])

    def test_voxel_eq(self):
        v = Voxel.get_random_voxel([0.0, 0.0, 0.0], 0, voxels_per_axis=2, size=(1.5, 1.5, 1.5))
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    v.subvoxels[x, y, z].status = EMPTY_VOXEL

        self.assertEqual(self.v1, v)
        self.assertNotEqual(self.v2, v)
        self.assertNotEqual(self.v3, v)

        v.subvoxels[0, 0, 0].status = FULL_VOXEL
        v.subvoxels[1, 1, 1].status = FULL_VOXEL
        self.assertNotEqual(self.v1, v)
        self.assertEqual(self.v2, v)
        self.assertNotEqual(self.v3, v)

        v.subvoxels[0, 0, 0] = self.v3.subvoxels[0, 0, 0]
        self.assertNotEqual(self.v1, v)
        self.assertNotEqual(self.v2, v)
        self.assertEqual(self.v3, v)

    def test_voxel_based_shape_prior_and_update_depth(self):
        self.assertEqual(self.s1.depth, 1)
        self.assertAlmostEqual(self.s1.log_prior(), -np.log(256.0) - np.log(2.0))
        self.s1.voxel.subvoxels[0, 0, 0] = Voxel.get_random_voxel(self.s1.voxel.subvoxels[0, 0, 0].origin, 1)
        self.s1.update_depth()
        self.assertEqual(self.s1.depth, 2)
        self.assertAlmostEqual(self.s1.log_prior(), -np.log(262144.0) - (2 * np.log(2.0)))

        self.assertEqual(self.s2.depth, 1)
        self.assertAlmostEqual(self.s2.log_prior(), -np.log(256.0) - np.log(2.0))
        self.s2.voxel.subvoxels[0, 0, 0] = Voxel.get_random_voxel(self.s2.voxel.subvoxels[0, 0, 0].origin, 1)
        self.s2.update_depth()
        self.assertEqual(self.s2.depth, 2)
        self.assertAlmostEqual(self.s2.log_prior(), -np.log(262144.0) - 2 * np.log(2.0))

        self.assertEqual(self.s3.depth, 2)
        self.assertAlmostEqual(self.s3.log_prior(), -np.log(262144.0) - 2 * np.log(2.0))
        self.s3.voxel.subvoxels[0, 0, 0] = Voxel.get_full_voxel(self.s3.voxel.subvoxels[0, 0, 0].origin, 1)
        self.s3.update_depth()
        self.assertEqual(self.s3.depth, 1)
        self.assertAlmostEqual(self.s3.log_prior(), -np.log(256.0) - np.log(2.0))

    def test_voxel_based_shape_prior_and_update_depth_maxd(self):
        self.assertRaises(ValueError, VoxelBasedShapeMaxD, forward_model=None, voxel=self.v3, max_depth=1)

        s = VoxelBasedShapeMaxD(forward_model=None, voxel=self.v1, max_depth=2)
        self.assertEqual(s.depth, 1)
        self.assertAlmostEqual(s.log_prior(), -np.log(2232) - np.log(2.0))
        s.voxel.subvoxels[0, 0, 0] = Voxel.get_random_voxel(s.voxel.subvoxels[0, 0, 0].origin, 1)
        s.update_depth()
        self.assertEqual(s.depth, 2)
        self.assertAlmostEqual(s.log_prior(), -(2 * np.log(2232)) - (2 * np.log(2.0)))

        s = VoxelBasedShapeMaxD(forward_model=None, voxel=self.v2, max_depth=2)
        self.assertEqual(s.depth, 1)
        self.assertAlmostEqual(s.log_prior(), -(1 * np.log(2232)) - (1 * np.log(2.0)))
        s.voxel.subvoxels[0, 0, 0] = Voxel.get_random_voxel(s.voxel.subvoxels[0, 0, 0].origin, 1)
        s.update_depth()
        self.assertEqual(s.depth, 2)
        self.assertAlmostEqual(s.log_prior(), -(2 * np.log(2232)) - (2 * np.log(2.0)))

        s = VoxelBasedShapeMaxD(forward_model=None, voxel=self.v3, max_depth=2)
        self.assertEqual(s.depth, 2)
        self.assertAlmostEqual(s.log_prior(), -(2 * np.log(2232)) - (2 * np.log(2.0)))
        s.voxel.subvoxels[0, 0, 1] = Voxel.get_random_voxel(s.voxel.subvoxels[0, 0, 1].origin, 1)
        s.update_depth()
        self.assertEqual(s.depth, 2)
        self.assertAlmostEqual(s.log_prior(), -(3 * np.log(2232)) - (3 * np.log(2.0)))

    def test_voxel_based_shape_copy(self):
        # voxel.copy is already tested above. just test viewpoint, scale and origin copying.
        self.s3.viewpoint = [np.array([0.0, 0.0, 0.0])]
        self.s3.origin = np.array([0.0, 0.0, 0.0])
        self.s3.scale = np.array([1.0, 1.0, 1.0])
        copy = self.s3.copy()
        self.assertNumpyArrayListEqual(self.s3.viewpoint, copy.viewpoint)
        self.assertNumpyArrayEqual(self.s3.origin, copy.origin)
        self.assertNumpyArrayEqual(self.s3.scale, copy.scale)
        copy.viewpoint[0][0] = -1.0
        self.assertNumpyArrayNotEqual(self.s3.viewpoint[0], copy.viewpoint[0])
        copy.origin[0] = -1.0
        self.assertNumpyArrayNotEqual(self.s3.origin, copy.origin)
        copy.scale[0] = 2.0
        self.assertNumpyArrayNotEqual(self.s3.scale, copy.scale)
        self.assertEqual(self.s3.voxel, copy.voxel)
        self.assertNotEqual(self.s3, copy)

    def test_voxel_flip_full_vs_empty(self):
        # only empty to full possible
        hp, q_hp_h, q_h_hp = voxel_based_shape_flip_full_vs_empty(self.s1, None)
        self.assertEqual(hp.voxel.count_voxels_by_status(FULL_VOXEL), 1)
        self.assertAlmostEqual(q_hp_h, 1.0 / 8.0)
        self.assertAlmostEqual(q_h_hp, 1.0 / 2.0)
        self.assertEqual(hp.depth, 1)

        # both moves are possible
        hp, q_hp_h, q_h_hp = voxel_based_shape_flip_full_vs_empty(self.s2, None)
        self.assertEqual(hp.depth, 1)
        c = hp.voxel.count_voxels_by_status(FULL_VOXEL)
        self.assertIn(c, [1, 3])
        if c == 1:  # full to empty
            self.assertAlmostEqual(q_hp_h, 1.0 / 4.0)
            self.assertAlmostEqual(q_h_hp, 1.0 / 14.0)
        elif c == 3:  # empty to full
            self.assertAlmostEqual(q_hp_h, 1.0 / 12.0)
            self.assertAlmostEqual(q_h_hp, 1.0 / 6.0)

        # both moves are possible
        hp, q_hp_h, q_h_hp = voxel_based_shape_flip_full_vs_empty(self.s3, None)
        self.assertEqual(hp.depth, 2)
        c = hp.voxel.count_voxels_by_status(FULL_VOXEL)
        self.assertIn(c, [1, 3])
        if c == 1:  # full to empty
            self.assertAlmostEqual(q_hp_h, 1.0 / 4.0)
            self.assertAlmostEqual(q_h_hp, 1.0 / 28.0)
        elif c == 2:  # empty to full
            self.assertAlmostEqual(q_hp_h, 1.0 / 13.0)
            self.assertAlmostEqual(q_h_hp, 1.0 / 6.0)

        # check depth limit
        params = {'MAX_DEPTH': 1}
        self.assertRaises(ValueError, voxel_based_shape_flip_full_vs_empty, self.s3, params)

    def test_voxel_flip_full_vs_partial(self):
        # REMEMBER that the root voxel (the whole space) is also a PARTIAL_VOXEL.
        # only partial to full is possible
        hp, q_hp_h, q_h_hp = voxel_based_shape_flip_full_vs_partial(self.s1, None)
        self.assertEqual(hp.depth, 0)
        self.assertEqual(hp.voxel.count_voxels_by_status(FULL_VOXEL), 1)
        self.assertAlmostEqual(q_hp_h, 1.0 / 1.0)
        self.assertAlmostEqual(q_h_hp, 1.0 / 256.0)

        # both moves are possible
        hp, q_hp_h, q_h_hp = voxel_based_shape_flip_full_vs_partial(self.s2, None)
        self.assertIn(hp.depth, [0, 2])
        c = hp.voxel.count_voxels_by_status(PARTIAL_VOXEL)
        # we can convert the root partial_voxel to a full_voxel, in which case we are left with 0 partial_voxels.
        self.assertIn(c, [0, 2])
        if c == 2:  # full to partial
            self.assertAlmostEqual(q_hp_h, 1.0 / (2 * 2 * 256))
            self.assertAlmostEqual(q_h_hp, 1.0 / 2.0)
        elif c == 0:
            self.assertAlmostEqual(q_hp_h, 1.0 / 2.0)
            self.assertAlmostEqual(q_h_hp, 1.0 / 256.0)

        # both moves are possible
        hp, q_hp_h, q_h_hp = voxel_based_shape_flip_full_vs_partial(self.s3, None)
        self.assertIn(hp.depth, [1, 2, 3])
        # NOTE we cannot pick the root partial voxel because it has partial children.
        c = hp.voxel.count_voxels_by_status(PARTIAL_VOXEL)
        self.assertIn(c, [1, 3])
        if c == 3:  # full to partial
            self.assertAlmostEqual(q_hp_h, 1.0 / (2 * 2 * 256))
            # if we picked the full voxel at depth 1, we have 2 partial voxels to pick from
            # elif we picked the full voxel at depth 2, we have only 1 partial node to pick.
            self.assertAlmostIn(q_h_hp, [1.0 / 2.0, 1.0 / 4.0])
        else:  # partial to full
            self.assertAlmostEqual(q_hp_h, 1.0 / 2.0)
            self.assertAlmostEqual(q_h_hp, 1.0 / (2 * 2 * 256))

    def test_voxel_flip_full_vs_partial_max_depth(self):
        # check depth limit
        params = {'MAX_DEPTH': 1}
        self.assertRaises(ValueError, voxel_based_shape_flip_full_vs_partial, self.s3, params)

        # REMEMBER that the root voxel (the whole space) is also a PARTIAL_VOXEL.
        # only partial to full is possible
        hp, q_hp_h, q_h_hp = voxel_based_shape_flip_full_vs_partial(self.s1, {'MAX_DEPTH': 1})
        self.assertEqual(hp.depth, 0)
        self.assertEqual(hp.voxel.count_voxels_by_status(FULL_VOXEL), 1)
        self.assertAlmostEqual(q_hp_h, 1.0 / 1.0)
        self.assertAlmostEqual(q_h_hp, 1.0 / 256.0)

        # only partial to full is possible
        hp, q_hp_h, q_h_hp = voxel_based_shape_flip_full_vs_partial(self.s2, {'MAX_DEPTH': 1})
        self.assertEqual(hp.depth, 0)
        self.assertEqual(hp.voxel.count_voxels_by_status(FULL_VOXEL), 1)
        self.assertAlmostEqual(q_hp_h, 1.0 / 1.0)
        self.assertAlmostEqual(q_h_hp, 1.0 / 256.0)

        # both moves are possible
        hp, q_hp_h, q_h_hp = voxel_based_shape_flip_full_vs_partial(self.s3, {'MAX_DEPTH': 2})
        self.assertIn(hp.depth, [1, 2])
        # NOTE we cannot pick the root partial voxel because it has partial children.
        c = hp.voxel.count_voxels_by_status(PARTIAL_VOXEL)
        self.assertIn(c, [1, 3])
        if c == 3:  # full to partial
            self.assertAlmostEqual(q_hp_h, 1.0 / (2 * 256))
            self.assertAlmostEqual(q_h_hp, 1.0 / 2.0)
        else:  # partial to full
            self.assertAlmostEqual(q_hp_h, 1.0 / 2.0)
            self.assertAlmostEqual(q_h_hp, 1.0 / (2 * 2 * 256))

    def test_voxel_flip_empty_vs_partial(self):
        # REMEMBER that the root voxel (the whole space) is also a PARTIAL_VOXEL.
        # both moves are possible
        hp, q_hp_h, q_h_hp = voxel_based_shape_flip_empty_vs_partial(self.s1, None)
        self.assertIn(hp.depth, [0, 2])
        c = hp.voxel.count_voxels_by_status(PARTIAL_VOXEL)
        self.assertIn(c, [0, 2])
        if c == 2:  # empty to partial
            self.assertAlmostEqual(q_hp_h, 1.0 / (2 * 8 * 256))
            self.assertAlmostEqual(q_h_hp, 1.0 / 2.0)
        elif c == 0:  # partial to empty
            self.assertAlmostEqual(q_hp_h, 1.0 / 2.0)
            self.assertAlmostEqual(q_h_hp, 1.0 / 256.0)

        # both moves are possible
        hp, q_hp_h, q_h_hp = voxel_based_shape_flip_empty_vs_partial(self.s2, None)
        self.assertIn(hp.depth, [0, 2])
        c = hp.voxel.count_voxels_by_status(PARTIAL_VOXEL)
        self.assertIn(c, [0, 2])
        if c == 2:  # empty to partial
            self.assertAlmostEqual(q_hp_h, 1.0 / (2 * 6 * 256))
            self.assertAlmostEqual(q_h_hp, 1.0 / 2.0)
        elif c == 0:  # partial to empty
            self.assertAlmostEqual(q_hp_h, 1.0 / 2.0)
            self.assertAlmostEqual(q_h_hp, 1.0 / 256.0)

        # both moves are possible
        hp, q_hp_h, q_h_hp = voxel_based_shape_flip_empty_vs_partial(self.s3, None)
        self.assertIn(hp.depth, [1, 2, 3])
        c = hp.voxel.count_voxels_by_status(PARTIAL_VOXEL)
        self.assertIn(c, [1, 3])
        if c == 3:  # empty to partial
            self.assertAlmostEqual(q_hp_h, 1.0 / (2 * 13 * 256))
            self.assertAlmostIn(q_h_hp, [1.0 / 2.0, 1.0 / 4.0])
        elif c == 1:
            self.assertAlmostEqual(q_hp_h, 1.0 / 2.0)
            self.assertAlmostEqual(q_h_hp, 1.0 / (2 * 7 * 256))

    def test_voxel_flip_empty_vs_partial_max_depth(self):
        # REMEMBER that the root voxel (the whole space) is also a PARTIAL_VOXEL.
        # only partial to empty is possible
        hp, q_hp_h, q_h_hp = voxel_based_shape_flip_empty_vs_partial(self.s1, {'MAX_DEPTH': 1})
        self.assertEqual(hp.depth, 0)
        c = hp.voxel.count_voxels_by_status(PARTIAL_VOXEL)
        self.assertEqual(c, 0)
        self.assertAlmostEqual(q_hp_h, 1.0 / 1.0)
        self.assertAlmostEqual(q_h_hp, 1.0 / 256.0)

        # only partial to empty is possible
        hp, q_hp_h, q_h_hp = voxel_based_shape_flip_empty_vs_partial(self.s2, {'MAX_DEPTH': 1})
        self.assertEqual(hp.depth, 0)
        c = hp.voxel.count_voxels_by_status(PARTIAL_VOXEL)
        self.assertEqual(c, 0)
        self.assertAlmostEqual(q_hp_h, 1.0 / 1.0)
        self.assertAlmostEqual(q_h_hp, 1.0 / 256.0)

        # both moves are possible
        hp, q_hp_h, q_h_hp = voxel_based_shape_flip_empty_vs_partial(self.s3, {'MAX_DEPTH': 2})
        self.assertIn(hp.depth, [1, 2])
        c = hp.voxel.count_voxels_by_status(PARTIAL_VOXEL)
        self.assertIn(c, [1, 3])
        if c == 3:  # empty to partial
            self.assertAlmostEqual(q_hp_h, 1.0 / (2 * 6 * 256))
            self.assertAlmostEqual(q_h_hp, 1.0 / 4.0)
        elif c == 1:  # partial to empty
            self.assertAlmostEqual(q_hp_h, 1.0 / 2.0)
            self.assertAlmostEqual(q_h_hp, 1.0 / (2 * 7 * 256))

        # check depth limit
        params = {'MAX_DEPTH': 1}
        self.assertRaises(ValueError, voxel_based_shape_flip_empty_vs_partial, self.s3, params)

    def test_voxel_scale_space(self):
        hp, q_hp_h, q_h_hp = voxel_scale_space(self.s1, {'SCALE_SPACE_VARIANCE': .01})
        self.assertAlmostEqual(q_h_hp, 1.0)
        self.assertAlmostEqual(q_hp_h, 1.0)
        self.assertTrue(np.all(hp.scale > 0.0))
        if np.all(hp.scale == self.s1.scale):  # if updated scale was out out bounds
            self.assertEqual(hp, self.s1)
        else:
            self.assertNotEqual(hp, self.s1)

    def test_voxel_sample_prior(self):
        # test if sampling from the prior produces a set of samples with expected frequency statistics.
        # for the prior defined in VoxelBasedShapeMaxD, we would expect to see an equal number of samples for each
        # number of partial voxels. Here we constrain depth to 2, so a tree can have at least 0 and at most 9 partial
        # voxels. Therefore, we would expect to see each number of partial voxels 0.1 of the time.
        h = VoxelBasedShapeTestHypothesis(voxel=self.v1, max_depth=2)
        moves = {'flip_empty_vs_partial': voxel_based_shape_flip_empty_vs_partial,
                 'flip_full_vs_partial': voxel_based_shape_flip_full_vs_partial,
                 'flip_full_vs_empty': voxel_based_shape_flip_full_vs_empty}
        proposal = RandomMixtureProposal(moves=moves, params={'MAX_DEPTH': 2})
        sampler = MHSampler(initial_h=h, data=None, proposal=proposal, burn_in=0, thinning_period=1,
                            sample_count=50000, best_sample_count=1, report_period=5000)

        run = sampler.sample()
        partial_voxel_counts = np.array([s.voxel.count_voxels_by_status(PARTIAL_VOXEL) for s in run.samples.samples])

        for i in range(10):
            self.assertAlmostEqual(np.mean(partial_voxel_counts == i), 1.0 / 10.0, places=1)
