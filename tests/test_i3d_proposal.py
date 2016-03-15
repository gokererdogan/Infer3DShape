"""
Inferring 3D Shape from 2D Images

Unit tests for i3d_proposal module.

Created on Dec 2, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""
import Infer3DShape.i3d_hypothesis as hyp
import Infer3DShape.geometry_3d as geom_3d

from i3d_test_case import *
from Infer3DShape.i3d_proposal import *


class I3DProposalTest(I3DTestCase):
    def test_change_viewpoint_z(self):
        vp1 = np.array((1.0, 0.0, 0.0))
        vp2 = np.array((2.0, 150.0, 0.0))
        h = hyp.I3DHypothesis(forward_model=None, viewpoint=[vp1, vp2], params=None)

        for i in range(1000):
            hp, p0, p1 = change_viewpoint_z(h, {'CHANGE_VIEWPOINT_VARIANCE': np.square(1.0 / 180.0 * np.pi)})
            self.assertAlmostEqual(p0, 1.0)
            self.assertAlmostEqual(p1, 1.0)

            vp = hp.viewpoint[0]
            self.assertAlmostEqual(vp[0], 1.0)
            # because the variance is 1.0, we would expect the angle between vectors to be small
            self.assertTrue(geom_3d.angle_between_vectors(geom_3d.spherical_to_cartesian(vp),
                                                          geom_3d.spherical_to_cartesian(vp1)) < 5.0)

            vp = hp.viewpoint[1]
            self.assertAlmostEqual(vp[0], 2.0)
            self.assertTrue(geom_3d.angle_between_vectors(geom_3d.spherical_to_cartesian(vp),
                                                          geom_3d.spherical_to_cartesian(vp2)) < 5.0)

    def test_change_viewpoint(self):
        vp1 = np.array((1.0, 0.0, 0.0))
        vp2 = np.array((2.0, 150.0, 0.0))
        h = hyp.I3DHypothesis(forward_model=None, viewpoint=[vp1, vp2], params=None)

        for i in range(1000):
            hp, p0, p1 = change_viewpoint(h, {'CHANGE_VIEWPOINT_VARIANCE': np.square(5.0 / 180.0 * np.pi)})
            self.assertAlmostEqual(p0, 1.0)
            self.assertAlmostEqual(p1, 1.0)

            vp = hp.viewpoint[0]
            self.assertAlmostEqual(vp[0], 1.0)
            # because the variance is 1.0, we would expect the angle between vectors to be small
            self.assertTrue(geom_3d.angle_between_vectors(geom_3d.spherical_to_cartesian(vp),
                                                          geom_3d.spherical_to_cartesian(vp1)) < 20.0)

            vp = hp.viewpoint[1]
            self.assertAlmostEqual(vp[0], 2.0)
            self.assertTrue(geom_3d.angle_between_vectors(geom_3d.spherical_to_cartesian(vp),
                                                          geom_3d.spherical_to_cartesian(vp2)) < 20.0)
