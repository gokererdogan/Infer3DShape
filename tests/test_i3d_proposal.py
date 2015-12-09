"""
Inferring 3D Shape from 2D Images

Unit tests for i3d_proposal module.

Created on Dec 2, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import unittest

from Infer3DShape.i3d_proposal import *
import Infer3DShape.i3d_hypothesis as hyp

class I3DProposalTest(unittest.TestCase):
    def test_change_viewpoint(self):
        h = hyp.I3DHypothesis(forward_model=None, viewpoint=[(1.0, 1.0, 1.0), (2.0, 2.0, 1.5)], params=None)
        hp, p0, p1 = change_viewpoint(h, {'CHANGE_VIEWPOINT_VARIANCE': 60.0})
        self.assertAlmostEqual(p0, 1.0)
        self.assertAlmostEqual(p1, 1.0)
        vp = hp.viewpoint[0]
        self.assertAlmostEqual(vp[2], 1.0)
        self.assertAlmostEqual(vp[0]**2 + vp[1]**2, 2.0)
        vp = hp.viewpoint[1]
        self.assertAlmostEqual(vp[2], 1.5)
        self.assertAlmostEqual(vp[0]**2 + vp[1]**2, 8.0)

