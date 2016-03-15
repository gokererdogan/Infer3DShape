"""
Inferring 3D Shape from 2D Images

Unit tests for paperclip_shape module.

Created on Mar 13, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""
import numpy as np

from mcmclib.proposal import DeterministicMixtureProposal
from mcmclib.mh_sampler import MHSampler

from i3d_test_case import *
from Infer3DShape.paperclip_shape import *
from Infer3DShape.paperclip_shape import _get_random_vector_along


class PaperClipTestHypothesis(PaperClipShape):
    def __init__(self, joint_positions=None, mid_segment_id=2):
        PaperClipShape.__init__(self, forward_model=None, joint_positions=joint_positions,
                                params={'JOINT_VARIANCE': 0.2}, mid_segment_id=mid_segment_id,
                                min_joints=2, max_joints=4)

    def _calculate_log_likelihood(self, data=None):
        return 0.0

    def _calculate_log_prior(self):
        return -np.log(self.joint_count-1)

    def copy(self):
        joint_positions_copy = [jp.copy() for jp in self.joint_positions]
        self_copy = PaperClipTestHypothesis(joint_positions=joint_positions_copy, mid_segment_id=self.mid_segment_id)
        return self_copy


class PaperClipTest(I3DTestCase):
    def test_paperclip_init(self):
        # random object
        # JOINT_VARIANCE must be provided
        self.assertRaises(ValueError, PaperClipShape, forward_model=None)

        s = PaperClipShape(forward_model=None, params={'JOINT_VARIANCE': 0.3})
        self.assertEqual(s.max_joints, 6)
        self.assertEqual(s.min_joints, 6)
        self.assertEqual(s.mid_segment_id, 2)
        self.assertNumpyArrayEqual(s.joint_positions[s.mid_segment_id] + s.joint_positions[s.mid_segment_id+1], 0.0)
        self.assertNumpyArrayEqual(s.joint_positions[s.mid_segment_id][1:], 0.0)
        self.assertNumpyArrayEqual(s.joint_positions[s.mid_segment_id+1][1:], 0.0)

        for i in range(s.joint_count-1):
            self.assertGreater(s._get_segment_length(i), MIN_SEGMENT_LENGTH)

        # random object
        s = PaperClipShape(forward_model=None, max_joints=9, min_joints=-4, mid_segment_id=12,
                           params={'JOINT_VARIANCE': 0.3})
        self.assertEqual(s.max_joints, 6)
        self.assertEqual(s.min_joints, 6)
        self.assertEqual(s.mid_segment_id, 2)
        self.assertNumpyArrayEqual(s.joint_positions[s.mid_segment_id] + s.joint_positions[s.mid_segment_id+1], 0.0)

        self.assertNumpyArrayEqual(s.joint_positions[s.mid_segment_id][1:], 0.0)
        self.assertNumpyArrayEqual(s.joint_positions[s.mid_segment_id+1][1:], 0.0)

        for i in range(s.joint_count-1):
            self.assertGreater(s._get_segment_length(i), MIN_SEGMENT_LENGTH)

        # specify shape
        self.assertRaises(ValueError, PaperClipShape, forward_model=None, joint_positions=[np.array((0.0, 0.0, 0.0))],
                          max_joints=4, min_joints=5)

        self.assertRaises(ValueError, PaperClipShape, forward_model=None, joint_positions=[np.array((0.0, 0.0, 0.0))],
                          max_joints=np.inf, min_joints=5)

        self.assertRaises(ValueError, PaperClipShape, forward_model=None, joint_positions=[np.array((0.0, 0.0, 0.0))],
                          max_joints=6, min_joints=2, mid_segment_id=-1)

        self.assertRaises(ValueError, PaperClipShape, forward_model=None, joint_positions=[np.array((0.0, 0.0, 0.0))],
                          max_joints=6, min_joints=2, mid_segment_id=6)

        self.assertRaises(ValueError, PaperClipShape, forward_model=None, joint_positions=[np.array((0.0, 0.0, 0.0))],
                          max_joints=6, min_joints=2, mid_segment_id=6)

        self.assertRaises(ValueError, PaperClipShape, forward_model=None, joint_positions=[np.array((0.1, 0.0, 0.0)),
                                                                                 np.array((0.2, 0.0, 0.0))],
                          max_joints=6, min_joints=2, mid_segment_id=0)

        self.assertRaises(ValueError, PaperClipShape, forward_model=None, joint_positions=[np.array((0.1, 0.1, 0.1)),
                                                                                 np.array((-0.1, -0.1, -0.1))],
                          max_joints=6, min_joints=2, mid_segment_id=0)

        self.assertRaises(ValueError, PaperClipShape, forward_model=None, max_joints=6, min_joints=2, mid_segment_id=0,
                          joint_positions=[np.array((MIN_SEGMENT_LENGTH/4.0, 0.0, 0.0)),
                                           np.array((-MIN_SEGMENT_LENGTH/4.0, 0.0, 0.0))])

    def test_paperclip_get_segment_length(self):
        s = PaperClipShape(forward_model=None, joint_positions=[np.array((0.5, 0.0, 0.0)), np.array((0.2, 0.0, 0.0)),
                                                      np.array((-0.2, 0.0, 0.0)), np.array((-0.7, 0.0, 0.0))],
                           mid_segment_id=1, min_joints=2, max_joints=10)

        self.assertRaises(ValueError, s._get_segment_length, segment_id=-1)
        self.assertRaises(ValueError, s._get_segment_length, segment_id=3)
        self.assertAlmostEqual(s._get_segment_length(0), 0.3)
        self.assertAlmostEqual(s._get_segment_length(1), 0.4)
        self.assertAlmostEqual(s._get_segment_length(2), 0.5)

    def test_paperclip_get_joint_angle(self):
        s = PaperClipShape(forward_model=None, joint_positions=[np.array((0.5, 0.3, 0.0)), np.array((0.2, 0.0, 0.0)),
                                                      np.array((-0.2, 0.0, 0.0)), np.array((-0.2, 0.4, 0.0))],
                           mid_segment_id=1, min_joints=2, max_joints=10)

        self.assertRaises(ValueError, s._get_joint_angle, joint_id=0)
        self.assertRaises(ValueError, s._get_joint_angle, joint_id=3)
        self.assertAlmostEqual(s._get_joint_angle(1), 45.0)
        self.assertAlmostEqual(s._get_joint_angle(2), 90.0)

    def test_paperclip_change_segment_length(self):
        jp = [np.array((0.5, 0.3, 0.0)), np.array((0.2, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)),
              np.array((-0.2, 0.4, 0.0))]

        s = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=2, max_joints=10)

        self.assertRaises(ValueError, s.change_segment_length, segment_id=-1, change_ratio=0.5)
        self.assertRaises(ValueError, s.change_segment_length, segment_id=3, change_ratio=0.5)
        self.assertRaises(ValueError, s.change_segment_length, segment_id=0, change_ratio=-2.0)

        # try to make segment smaller than allowed minimum length
        change = (MIN_SEGMENT_LENGTH / 0.4) - 1.0 - 0.1
        s.change_segment_length(1, change, update_children=False)
        self.assertNumpyArrayListEqual(s.joint_positions, jp)

        change = (MIN_SEGMENT_LENGTH / (0.3 * np.sqrt(2.0))) - 1.0 - 0.1
        s.change_segment_length(0, change, update_children=False)
        self.assertNumpyArrayListEqual(s.joint_positions, jp)

        change = (MIN_SEGMENT_LENGTH / 0.4) - 1.0 - 0.1
        s.change_segment_length(2, change, update_children=False)
        self.assertNumpyArrayListEqual(s.joint_positions, jp)

        # these should be allowed
        s.change_segment_length(0, 0.5, update_children=False)
        self.assertNumpyArrayEqual(s.joint_positions[0], (0.65, 0.45, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[1], jp[1])
        self.assertNumpyArrayEqual(s.joint_positions[2], jp[2])
        self.assertNumpyArrayEqual(s.joint_positions[3], jp[3])

        s.change_segment_length(2, -0.25, update_children=False)
        self.assertNumpyArrayEqual(s.joint_positions[0], (0.65, 0.45, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[1], jp[1])
        self.assertNumpyArrayEqual(s.joint_positions[2], jp[2])
        self.assertNumpyArrayEqual(s.joint_positions[3], (-0.2, 0.3, 0.0))

        s.change_segment_length(1, 0.2, update_children=False)
        self.assertNumpyArrayEqual(s.joint_positions[0], (0.65, 0.45, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[1], (0.24, 0.0, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[2], (-0.24, 0.0, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[3], (-0.2, 0.3, 0.0))

        jp = [np.array((0.5, 0.3, 0.0)), np.array((0.2, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)),
              np.array((-0.2, 0.4, 0.0))]

        s = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=2, max_joints=10)

        s.change_segment_length(0, 0.5, update_children=True)
        self.assertNumpyArrayEqual(s.joint_positions[0], (0.65, 0.45, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[1], jp[1])
        self.assertNumpyArrayEqual(s.joint_positions[2], jp[2])
        self.assertNumpyArrayEqual(s.joint_positions[3], jp[3])

        s.change_segment_length(2, -0.25, update_children=True)
        self.assertNumpyArrayEqual(s.joint_positions[0], (0.65, 0.45, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[1], jp[1])
        self.assertNumpyArrayEqual(s.joint_positions[2], jp[2])
        self.assertNumpyArrayEqual(s.joint_positions[3], (-0.2, 0.3, 0.0))

        s.change_segment_length(1, 0.2, update_children=True)
        self.assertNumpyArrayEqual(s.joint_positions[0], (0.69, 0.45, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[1], (0.24, 0.0, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[2], (-0.24, 0.0, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[3], (-0.24, 0.3, 0.0))

        jp = [np.array((-0.5, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)), np.array((0.2, 0.0, 0.0)),
              np.array((0.7, 0.0, 0.0))]

        s = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=2, max_joints=10)

        # this should not be allowed because a neighboring segment becomes shorter than allowed.
        change = (0.3 - MIN_SEGMENT_LENGTH) / 0.4 + 0.1
        s.change_segment_length(1, change, update_children=False)
        self.assertNumpyArrayListEqual(s.joint_positions, jp)

    def test_paperclip_move_joint(self):
        jp = [np.array((0.5, 0.3, 0.0)), np.array((0.2, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)),
              np.array((-0.2, 0.4, 0.0))]

        s = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=2, max_joints=10)

        self.assertRaises(ValueError, s.move_joint, joint_id=-1, change=np.array((0.0, 0.0, 0.0)))
        self.assertRaises(ValueError, s.move_joint, joint_id=4, change=np.array((0.0, 0.0, 0.0)))

        s.move_joint(0, np.array((0.1, 0.1, 0.0)), update_children=False)
        self.assertNumpyArrayEqual(s.joint_positions[0], (0.6, 0.4, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[1], jp[1])
        self.assertNumpyArrayEqual(s.joint_positions[2], jp[2])
        self.assertNumpyArrayEqual(s.joint_positions[3], jp[3])

        s.move_joint(0, np.array((-0.1, -0.1, 0.0)), update_children=True)
        self.assertNumpyArrayEqual(s.joint_positions[0], (0.5, 0.3, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[1], jp[1])
        self.assertNumpyArrayEqual(s.joint_positions[2], jp[2])
        self.assertNumpyArrayEqual(s.joint_positions[3], jp[3])

        # this shouldn't work
        s.move_joint(1, np.array((0.1, 0.1, 0.0)), update_children=False)
        self.assertNumpyArrayEqual(s.joint_positions[0], jp[0])
        self.assertNumpyArrayEqual(s.joint_positions[1], jp[1])
        self.assertNumpyArrayEqual(s.joint_positions[2], jp[2])
        self.assertNumpyArrayEqual(s.joint_positions[3], jp[3])

        # this should work
        s.move_joint(1, np.array((0.1, 0.0, 0.0)), update_children=False)
        self.assertNumpyArrayEqual(s.joint_positions[0], jp[0])
        self.assertNumpyArrayEqual(s.joint_positions[1], (0.3, 0.0, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[2], (-0.3, 0.0, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[3], jp[3])

        s.move_joint(1, np.array((0.1, 0.0, 0.0)), update_children=True)
        self.assertNumpyArrayEqual(s.joint_positions[0], (0.6, 0.3, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[1], (0.4, 0.0, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[2], (-0.4, 0.0, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[3], (-0.3, 0.4, 0.0))

    def test_paperclip_can_move_joint(self):
        jp = [np.array((0.5, 0.3, 0.0)), np.array((0.2, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)),
              np.array((-0.2, 0.4, 0.0))]

        s = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=2, max_joints=10)

        self.assertTrue(s._can_joint_move(0, np.array((0.05, 0.05, 0.0)), update_children=False))
        self.assertTrue(s._can_joint_move(0, np.array((-0.05, -0.05, 0.0)), update_children=True))
        # this shouldn't work
        self.assertFalse(s._can_joint_move(1, np.array((0.1, 0.1, 0.0)), update_children=False))
        # this should work
        self.assertTrue(s._can_joint_move(1, np.array((0.1, 0.0, 0.0)), update_children=False))

        jp = [np.array((-0.5, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)), np.array((0.2, 0.0, 0.0)),
              np.array((0.7, 0.0, 0.0))]

        s = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=2, max_joints=10)

        # this should not be allowed because a neighboring segment becomes shorter than allowed.
        change = np.array((-((0.3 - MIN_SEGMENT_LENGTH) + 0.1), 0.0, 0.0))
        self.assertFalse(s._can_joint_move(1, change, update_children=False))
        self.assertTrue(s._can_joint_move(1, change, update_children=True))

        # this should not be allowed because a neighboring segment becomes shorter than allowed.
        change = np.array((((0.5 - MIN_SEGMENT_LENGTH) + 0.1), 0.0, 0.0))
        self.assertFalse(s._can_joint_move(2, change, update_children=False))
        self.assertTrue(s._can_joint_move(2, change, update_children=True))

    def test_paperclip_move_single_joint(self):
        jp = [np.array((0.5, 0.3, 0.0)), np.array((0.2, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)),
              np.array((-0.2, 0.4, 0.0))]

        s = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=2, max_joints=10)

        s._move_single_joint(0, (0.1, -0.1, 0.2), update_children=False)
        self.assertNumpyArrayEqual(s.joint_positions[0], (0.6, 0.2, 0.2))
        self.assertNumpyArrayEqual(s.joint_positions[1], jp[1])
        self.assertNumpyArrayEqual(s.joint_positions[2], jp[2])
        self.assertNumpyArrayEqual(s.joint_positions[3], jp[3])

        s._move_single_joint(0, (-0.1, 0.1, -0.2), update_children=True)
        self.assertNumpyArrayEqual(s.joint_positions[0], (0.5, 0.3, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[1], jp[1])
        self.assertNumpyArrayEqual(s.joint_positions[2], jp[2])
        self.assertNumpyArrayEqual(s.joint_positions[3], jp[3])

        s._move_single_joint(1, (0.1, 0.0, 0.0), update_children=False)
        self.assertNumpyArrayEqual(s.joint_positions[0], jp[0])
        self.assertNumpyArrayEqual(s.joint_positions[1], (0.3, 0.0, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[2], jp[2])
        self.assertNumpyArrayEqual(s.joint_positions[3], jp[3])

        s._move_single_joint(1, (-0.1, 0.0, 0.0), update_children=True)
        self.assertNumpyArrayEqual(s.joint_positions[0], (0.4, 0.3, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[1], (0.2, 0.0, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[2], jp[2])
        self.assertNumpyArrayEqual(s.joint_positions[3], jp[3])

        s._move_single_joint(2, (-0.1, 0.0, 0.0), update_children=False)
        self.assertNumpyArrayEqual(s.joint_positions[0], (0.4, 0.3, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[1], jp[1])
        self.assertNumpyArrayEqual(s.joint_positions[2], (-0.3, 0.0, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[3], jp[3])

        s._move_single_joint(2, (0.1, 0.0, 0.0), update_children=True)
        self.assertNumpyArrayEqual(s.joint_positions[0], (0.4, 0.3, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[1], jp[1])
        self.assertNumpyArrayEqual(s.joint_positions[2], (-0.2, 0.0, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[3], (-0.1, 0.4, 0.0))

        s._move_single_joint(3, (0.1, -0.1, 0.2), update_children=False)
        self.assertNumpyArrayEqual(s.joint_positions[0], (0.4, 0.3, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[1], jp[1])
        self.assertNumpyArrayEqual(s.joint_positions[2], jp[2])
        self.assertNumpyArrayEqual(s.joint_positions[3], (0.0, 0.3, 0.2))

        s._move_single_joint(3, (-0.1, 0.1, -0.2), update_children=True)
        self.assertNumpyArrayEqual(s.joint_positions[0], (0.4, 0.3, 0.0))
        self.assertNumpyArrayEqual(s.joint_positions[1], jp[1])
        self.assertNumpyArrayEqual(s.joint_positions[2], jp[2])
        self.assertNumpyArrayEqual(s.joint_positions[3], (-0.1, 0.4, 0.0))

    def test_paperclip_copy(self):
        jp = [np.array((0.5, 0.3, 0.0)), np.array((0.2, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)),
              np.array((-0.2, 0.4, 0.0))]

        s = PaperClipShape(forward_model=None, viewpoint=[np.array((2.0, 45.0, 45.0)), np.array((2.0, -120.0, 110.0))],
                           joint_positions=jp, mid_segment_id=1, min_joints=2, max_joints=10)

        sc = s.copy()
        self.assertEqual(s.min_joints, sc.min_joints)
        self.assertEqual(s.max_joints, sc.max_joints)
        self.assertEqual(s.mid_segment_id, sc.mid_segment_id)
        self.assertEqual(s.joint_count, sc.joint_count)
        self.assertNumpyArrayEqual(s.joint_positions[0], sc.joint_positions[0])
        self.assertNumpyArrayEqual(s.joint_positions[1], sc.joint_positions[1])
        self.assertNumpyArrayEqual(s.joint_positions[2], sc.joint_positions[2])
        self.assertNumpyArrayEqual(s.joint_positions[3], sc.joint_positions[3])
        self.assertNumpyArrayListEqual(s.viewpoint, sc.viewpoint)

        sc.joint_positions[0][0] = -10.0
        self.assertNotEqual(s.joint_positions[0][0], sc.joint_positions[0][0])

        sc.viewpoint[0][0] = 10.0
        self.assertNotEqual(s.viewpoint[0][0], sc.viewpoint[0][0])

    def test_paperclip_eq(self):
        jp1 = [np.array((0.5, 0.3, 0.0)), np.array((0.2, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)),
              np.array((-0.2, 0.4, 0.0))]
        jp2 = [np.array((0.5, 0.3, 0.0)), np.array((0.2, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0))]
        jp3 = [np.array((-0.5, -0.3, 0.0)), np.array((-0.2, 0.0, 0.0)), np.array((0.2, 0.0, 0.0))]
        jp4 = [np.array((0.2, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)), np.array((-0.5, -0.3, 0.0))]

        s1 = PaperClipShape(forward_model=None, joint_positions=jp1, mid_segment_id=1, min_joints=2, max_joints=10)
        s2 = PaperClipShape(forward_model=None, joint_positions=jp2, mid_segment_id=1, min_joints=2, max_joints=10)
        s3 = PaperClipShape(forward_model=None, joint_positions=jp3, mid_segment_id=1, min_joints=2, max_joints=10)
        s4 = PaperClipShape(forward_model=None, joint_positions=jp4, mid_segment_id=0, min_joints=2, max_joints=10)

        self.assertNotEqual(s1, s2)
        self.assertEqual(s1, s1)
        self.assertEqual(s2, s2)
        self.assertEqual(s2, s3)
        self.assertEqual(s2, s4)
        self.assertEqual(s3, s4)

    def test_paperclip_can_add_joint(self):
        jp = [np.array((0.5, 0.3, 0.0)), np.array((0.2, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)),
              np.array((-0.2, 0.4, 0.0))]

        s = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=2, max_joints=4)
        self.assertFalse(s.can_add_joint())

        s = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=2, max_joints=5)
        self.assertTrue(s.can_add_joint())

    def test_paperclip_add_joint(self):
        jp = [np.array((0.5, 0.3, 0.0)), np.array((0.2, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)),
              np.array((-0.2, 0.4, 0.0))]

        s = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=2, max_joints=4)
        self.assertRaises(ValueError, s.add_joint, start_joint_id=0, new_joint_position=None)

        s = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=2, max_joints=5)
        self.assertRaises(ValueError, s.add_joint, start_joint_id=-2, new_joint_position=None)
        self.assertRaises(ValueError, s.add_joint, start_joint_id=0, new_joint_position=None)
        self.assertRaises(ValueError, s.add_joint, start_joint_id=1, new_joint_position=None)
        self.assertRaises(ValueError, s.add_joint, start_joint_id=2, new_joint_position=None)
        self.assertRaises(ValueError, s.add_joint, start_joint_id=4, new_joint_position=None)

        jp = [np.array((0.6, 0.0, 0.0)), np.array((0.2, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)),
              np.array((-0.2, 0.4, 0.0))]

        s = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=2, max_joints=6)
        self.assertRaises(ValueError, s.add_joint, start_joint_id=3,
                          new_joint_position=(-0.2+MIN_SEGMENT_LENGTH/2, 0.4, 0.0))

        s.add_joint(start_joint_id=-1, new_joint_position=(0.6+(2*MIN_SEGMENT_LENGTH), 0.0, 0.0))
        self.assertEqual(s.joint_count, 5)
        self.assertEqual(s.mid_segment_id, 2)
        self.assertNumpyArrayEqual(s.joint_positions[0], (0.6+(2*MIN_SEGMENT_LENGTH), 0.0, 0.0))

        s.add_joint(start_joint_id=4, new_joint_position=(-0.2, 0.4+(2*MIN_SEGMENT_LENGTH), 0.0))
        self.assertEqual(s.joint_count, 6)
        self.assertEqual(s.mid_segment_id, 2)
        self.assertNumpyArrayEqual(s.joint_positions[5], (-0.2, 0.4+(2*MIN_SEGMENT_LENGTH), 0.0))

    def test_paperclip_get_add_choices(self):
        jp = [np.array((0.2+(3*MIN_SEGMENT_LENGTH), 0.0, 0.0)), np.array((0.2, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)),
              np.array((-0.2, MIN_SEGMENT_LENGTH, 0.0))]

        s = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=2, max_joints=8)
        choices = s.get_add_joint_choices()
        self.assertListEqual(choices, [-1, 3])

    def test_paperclip_can_remove_joint(self):
        jp = [np.array((0.5, 0.0, 0.0)), np.array((0.2, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)),
              np.array((-0.2, 0.4, 0.0))]

        s = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=4, max_joints=8)
        self.assertFalse(s.can_remove_joint())
        s.min_joints = 2
        self.assertTrue(s.can_remove_joint())

    def test_paperclip_remove_joint(self):
        jp = [np.array((0.5, 0.0, 0.0)), np.array((0.2, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)),
              np.array((-0.2, 0.4, 0.0))]

        jp_copy = [j.copy() for j in jp]

        s = PaperClipShape(forward_model=None, joint_positions=jp_copy, mid_segment_id=1, min_joints=4, max_joints=8)

        self.assertRaises(ValueError, s.remove_joint, joint_id=0)

        s.min_joints = 2
        self.assertRaises(ValueError, s.remove_joint, joint_id=-1)
        self.assertRaises(ValueError, s.remove_joint, joint_id=4)
        self.assertRaises(ValueError, s.remove_joint, joint_id=1)
        self.assertRaises(ValueError, s.remove_joint, joint_id=2)

        s.remove_joint(joint_id=0)
        self.assertEqual(s.joint_count, 3)
        self.assertEqual(s.mid_segment_id, 0)
        self.assertNumpyArrayEqual(s.joint_positions[0], jp[1])
        self.assertNumpyArrayEqual(s.joint_positions[1], jp[2])
        self.assertNumpyArrayEqual(s.joint_positions[2], jp[3])

        s.remove_joint(joint_id=2)
        self.assertEqual(s.joint_count, 2)
        self.assertEqual(s.mid_segment_id, 0)
        self.assertNumpyArrayEqual(s.joint_positions[0], jp[1])
        self.assertNumpyArrayEqual(s.joint_positions[1], jp[2])

    def test_paperclip_get_remove_choices(self):
        jp = [np.array((0.5, 0.0, 0.0)), np.array((0.2, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)),
                  np.array((-0.2, 0.4, 0.0))]

        s = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=4, max_joints=8)

        choices = s.get_remove_joint_choices()
        self.assertListEqual(choices, [0, 3])

    def test_paperclip_rotate_midsegment(self):
        jp = [np.array((0.5, 0.0, 0.0)), np.array((0.2, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)),
              np.array((-0.6, 0.0, 0.0))]

        h = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=2, max_joints=8)
        h.rotate_midsegment(np.array((0.0, 0.0, 1.0)), 90.0)
        self.assertNumpyArrayEqual(h.joint_positions[0], (0.0, -0.5, 0.0))
        self.assertNumpyArrayEqual(h.joint_positions[1], jp[1])
        self.assertNumpyArrayEqual(h.joint_positions[2], jp[2])
        self.assertNumpyArrayEqual(h.joint_positions[3], (0.0, 0.6, 0.0))

    def test_paperclip_get_random_vector_along(self):
        z = np.random.randn(3)
        for i in range(1000):
            v = _get_random_vector_along(z)
            self.assertGreater(geom_3d.angle_between_vectors(v, z), 30.0)
            self.assertAlmostEqual(np.linalg.norm(v), 1.0)

    def test_paperclip_add_remove_joint_move(self):
        jp = [np.array((0.6, 0.0, 0.0)), np.array((0.2, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)),
              np.array((-0.2, 0.5, 0.0))]

        h = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=4, max_joints=4)
        hp, q_hp_h, q_h_hp = paperclip_shape_add_remove_joint(h, None)
        self.assertEqual(h, hp)
        self.assertAlmostEqual(q_h_hp, 1.0)
        self.assertAlmostEqual(q_hp_h, 1.0)

        # only add part possible
        h = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=4, max_joints=8)
        hp, q_hp_h, q_h_hp = paperclip_shape_add_remove_joint(h, params={'MAX_NEW_SEGMENT_LENGTH': 0.4})
        self.assertEqual(hp.joint_count, 5)
        self.assertAlmostEqual(q_h_hp, 1.0 / (2.0 * 2.0))
        self.assertAlmostEqual(q_hp_h, 1.0 / (1.0 * 2.0))

        # only add part possible and for return only remove move possible
        h = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=4, max_joints=5)
        hp, q_hp_h, q_h_hp = paperclip_shape_add_remove_joint(h, params={'MAX_NEW_SEGMENT_LENGTH': 0.4})
        self.assertEqual(hp.joint_count, 5)
        self.assertAlmostEqual(q_h_hp, 1.0 / (1.0 * 2.0))
        self.assertAlmostEqual(q_hp_h, 1.0 / (1.0 * 2.0))

        # only remove part possible
        h = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=2, max_joints=4)
        hp, q_hp_h, q_h_hp = paperclip_shape_add_remove_joint(h, params={'MAX_NEW_SEGMENT_LENGTH': 0.4})
        self.assertEqual(hp.joint_count, 3)
        self.assertAlmostEqual(q_h_hp, 1.0 / (2.0 * 2.0))
        self.assertAlmostEqual(q_hp_h, 1.0 / (1.0 * 2.0))

        # only remove part possible and for return only add move possible
        h = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=3, max_joints=4)
        hp, q_hp_h, q_h_hp = paperclip_shape_add_remove_joint(h, params={'MAX_NEW_SEGMENT_LENGTH': 0.4})
        self.assertEqual(hp.joint_count, 3)
        self.assertAlmostEqual(q_h_hp, 1.0 / (1.0 * 2.0))
        self.assertAlmostEqual(q_hp_h, 1.0 / (1.0 * 2.0))

        # both add and remove moves possible
        h = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=2, max_joints=8)
        hp, q_hp_h, q_h_hp = paperclip_shape_add_remove_joint(h, params={'MAX_NEW_SEGMENT_LENGTH': 0.4})
        if hp.joint_count > h.joint_count:
            self.assertEqual(hp.joint_count, 5)
            self.assertAlmostEqual(q_h_hp, 1.0 / (2.0 * 2.0))
            self.assertAlmostEqual(q_hp_h, 1.0 / (2.0 * 2.0))
        else:
            self.assertEqual(hp.joint_count, 3)
            self.assertAlmostEqual(q_h_hp, 1.0 / (2.0 * 2.0))
            self.assertAlmostEqual(q_hp_h, 1.0 / (2.0 * 2.0))

    def test_paperclip_move_joint_move(self):
        jp = [np.array((0.2+MIN_SEGMENT_LENGTH, 0.0, 0.0)), np.array((0.2, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)),
              np.array((-0.2, 3*MIN_SEGMENT_LENGTH, 0.0))]

        h = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=2, max_joints=8)
        hp, q_hp_h, q_h_hp = paperclip_shape_move_joint(h, params={'MOVE_JOINT_VARIANCE': 0.1})
        self.assertAlmostEqual(q_h_hp, 1.0)
        self.assertAlmostEqual(q_hp_h, 1.0)
        # at most one joint position should change (unless it is a midsegment joint)
        max_changed_joints = 1
        changed_joints = 0
        for i in range(h.joint_count):
            if not np.allclose(h.joint_positions[i], hp.joint_positions[i]):
                changed_joints += 1
                if i == h.mid_segment_id:
                    max_changed_joints += 1
        self.assertLessEqual(changed_joints, max_changed_joints)

    def test_paperclip_move_branch_move(self):
        jp = [np.array((0.2+MIN_SEGMENT_LENGTH, 0.0, 0.0)), np.array((0.2, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)),
              np.array((-0.2, 3*MIN_SEGMENT_LENGTH, 0.0))]

        h = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=2, max_joints=8)
        hp, q_hp_h, q_h_hp = paperclip_shape_move_branch(h, params={'MOVE_JOINT_VARIANCE': 0.1})
        self.assertAlmostEqual(q_h_hp, 1.0)
        self.assertAlmostEqual(q_hp_h, 1.0)

        # either 0, 1 or 4 joints should change position
        midsegment_changed = False
        changed_joints = 0
        for i in range(h.joint_count):
            if not np.allclose(h.joint_positions[i], hp.joint_positions[i]):
                changed_joints += 1
                if i == h.mid_segment_id:
                    midsegment_changed = True
        if not midsegment_changed:
            self.assertLessEqual(changed_joints, 1)
        else:
            self.assertEqual(changed_joints, 4)

    def test_paperclip_change_segment_length_move(self):
        jp = [np.array((0.2+MIN_SEGMENT_LENGTH, 0.0, 0.0)), np.array((0.2, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)),
              np.array((-0.2, 3*MIN_SEGMENT_LENGTH, 0.0))]

        h = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=2, max_joints=8)
        self.assertRaises(ValueError, paperclip_shape_change_segment_length, h=h,
                          params={'MAX_SEGMENT_LENGTH_CHANGE': 10.0})

        hp, q_hp_h, q_h_hp = paperclip_shape_change_segment_length(h, params={'MAX_SEGMENT_LENGTH_CHANGE': 0.5})
        self.assertAlmostEqual(q_h_hp, 1.0)
        self.assertAlmostEqual(q_hp_h, 1.0)

        # either 0, 1 or 3 segments should change length
        midsegment_changed = False
        changed_segments = 0
        for i in range(h.joint_count-1):
            if not np.isclose(h._get_segment_length(i), hp._get_segment_length(i)):
                changed_segments += 1
                if i == h.mid_segment_id:
                    midsegment_changed = True
        if not midsegment_changed:
            self.assertLessEqual(changed_segments, 1)
        else:
            self.assertEqual(changed_segments, 3)

    def test_paperclip_change_branch_length_move(self):
        jp = [np.array((0.2+MIN_SEGMENT_LENGTH, 0.0, 0.0)), np.array((0.2, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)),
              np.array((-0.2, 3*MIN_SEGMENT_LENGTH, 0.0))]

        h = PaperClipShape(forward_model=None, joint_positions=jp, mid_segment_id=1, min_joints=2, max_joints=8)
        self.assertRaises(ValueError, paperclip_shape_change_branch_length, h=h,
                          params={'MAX_SEGMENT_LENGTH_CHANGE': 10.0})

        hp, q_hp_h, q_h_hp = paperclip_shape_change_branch_length(h, params={'MAX_SEGMENT_LENGTH_CHANGE': 0.5})
        self.assertAlmostEqual(q_h_hp, 1.0)
        self.assertAlmostEqual(q_hp_h, 1.0)

        # either 0, 1 segments should change length
        changed_segments = 0
        for i in range(h.joint_count-1):
            if not np.isclose(h._get_segment_length(i), hp._get_segment_length(i)):
                changed_segments += 1

        self.assertLessEqual(changed_segments, 1)

    def test_paperclip_rotate_midsegment_move(self):
        h = PaperClipShape(forward_model=None, viewpoint=[np.array((1.0, 0.0, 90.0))], params={'JOINT_VARIANCE': 0.3})
        hp, q_hp_h, q_h_hp = paperclip_shape_rotate_midsegment(h, params={'ROTATE_MIDSEGMENT_VARIANCE': 100.0})
        self.assertNumpyArrayEqual(h.joint_positions[h.mid_segment_id], hp.joint_positions[hp.mid_segment_id])
        self.assertNumpyArrayEqual(h.joint_positions[h.mid_segment_id+1], hp.joint_positions[hp.mid_segment_id+1])
        # we can't really do anymore than testing whether the new viewpoint is different. In order to test if rotated
        # the right amount, we need the rotation axis which cannot be recovered from any information available in h and
        # hp
        self.assertNumpyArrayNotEqual(geom_3d.spherical_to_cartesian(h.viewpoint[0]),
                                      geom_3d.spherical_to_cartesian(hp.viewpoint[0]))

    def test_paperclip_add_remove_joint_sample(self):
        # test if add_remove_joint traverses the sample space correctly
        # look at how much probability mass objects with a certain number of parts get
        h = PaperClipTestHypothesis(joint_positions=[np.array((-0.4, 0.0, 0.0)), np.array((0.4, 0.0, 0.0))],
                                    mid_segment_id=0)
        h.min_joints = 2
        h.max_joints = 4
        sample_count = 40000
        p = DeterministicMixtureProposal(moves={'add_remove': paperclip_shape_add_remove_joint},
                                         params={'MAX_NEW_SEGMENT_LENGTH': 0.6})
        sampler = MHSampler(initial_h=h, data=None, proposal=p, burn_in=5000, sample_count=sample_count,
                            best_sample_count=1, thinning_period=1, report_period=4000)
        run = sampler.sample()
        # count how many times we sampled each joint
        k = np.array([sample.joint_count for sample in run.samples.samples])
        # since the prior is uniform and we min_joints=5, max_joints=6,
        # we expect to see an equal number of samples with 5, 6, or 7 joints.
        self.assertAlmostEqual(np.mean(k == 2), 1.0 / 3.0, places=1)
        self.assertAlmostEqual(np.mean(k == 3), 1.0 / 3.0, places=1)
        self.assertAlmostEqual(np.mean(k == 4), 1.0 / 3.0, places=1)

