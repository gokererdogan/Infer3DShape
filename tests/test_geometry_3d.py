"""
Inferring 3D Shape from 2D Images

Unit tests for geometry_3d module.

Created on Mar 11, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""

import numpy as np

from i3d_test_case import *
from Infer3DShape.geometry_3d import *


class I3DLikelihoodTest(I3DTestCase):
    def test_cartesian_to_spherical(self):
        self.assertTrue(np.allclose(cartesian_to_spherical((0.0, 0.0, 0.0)), (0.0, 0.0, 0.0)))
        self.assertTrue(np.allclose(cartesian_to_spherical((1.0, 0.0, 0.0)), (1.0, 0.0, 90.0)))
        self.assertTrue(np.allclose(cartesian_to_spherical((-1.0, 0.0, 0.0)), (1.0, 180.0, 90.0)))
        self.assertTrue(np.allclose(cartesian_to_spherical((0.0, 0.0, 1.0)), (1.0, 0.0, 0.0)))
        self.assertTrue(np.allclose(cartesian_to_spherical((0.0, 0.0, -1.0)), (1.0, 0.0, 180.0)))
        self.assertTrue(np.allclose(cartesian_to_spherical((1.0, 1.0, np.sqrt(2.0))), (2.0, 45.0, 45.0)))
        self.assertTrue(np.allclose(cartesian_to_spherical((1.0, -1.0, np.sqrt(2.0))), (2.0, -45.0, 45.0)))
        self.assertTrue(np.allclose(cartesian_to_spherical((-1.0, -1.0, np.sqrt(2.0))), (2.0, -135.0, 45.0)))
        self.assertTrue(np.allclose(cartesian_to_spherical((-1.0, 1.0, np.sqrt(2.0))), (2.0, 135.0, 45.0)))
        self.assertTrue(np.allclose(cartesian_to_spherical((1.0, 1.0, -np.sqrt(2.0))), (2.0, 45.0, 135.0)))
        self.assertTrue(np.allclose(cartesian_to_spherical((1.0, -1.0, -np.sqrt(2.0))), (2.0, -45.0, 135.0)))
        self.assertTrue(np.allclose(cartesian_to_spherical((-1.0, -1.0, -np.sqrt(2.0))), (2.0, -135.0, 135.0)))
        self.assertTrue(np.allclose(cartesian_to_spherical((-1.0, 1.0, -np.sqrt(2.0))), (2.0, 135.0, 135.0)))

    def test_spherical_to_cartesian(self):
        self.assertTrue(np.allclose((0.0, 0.0, 0.0), spherical_to_cartesian((0.0, 0.0, 0.0))))
        self.assertTrue(np.allclose((1.0, 0.0, 0.0), spherical_to_cartesian((1.0, 0.0, 90.0))))
        self.assertTrue(np.allclose((-1.0, 0.0, 0.0), spherical_to_cartesian((1.0, 180.0, 90.0))))
        self.assertTrue(np.allclose((0.0, 0.0, 1.0), spherical_to_cartesian((1.0, 0.0, 0.0))))
        self.assertTrue(np.allclose((0.0, 0.0, -1.0), spherical_to_cartesian((1.0, 0.0, 180.0))))
        self.assertTrue(np.allclose((1.0, 1.0, np.sqrt(2.0)), spherical_to_cartesian((2.0, 45.0, 45.0))))
        self.assertTrue(np.allclose((1.0, -1.0, np.sqrt(2.0)), spherical_to_cartesian((2.0, -45.0, 45.0))))
        self.assertTrue(np.allclose((-1.0, -1.0, np.sqrt(2.0)), spherical_to_cartesian((2.0, -135.0, 45.0))))
        self.assertTrue(np.allclose((-1.0, 1.0, np.sqrt(2.0)), spherical_to_cartesian((2.0, 135.0, 45.0))))
        self.assertTrue(np.allclose((1.0, 1.0, -np.sqrt(2.0)), spherical_to_cartesian((2.0, 45.0, 135.0))))
        self.assertTrue(np.allclose((1.0, -1.0, -np.sqrt(2.0)), spherical_to_cartesian((2.0, -45.0, 135.0))))
        self.assertTrue(np.allclose((-1.0, -1.0, -np.sqrt(2.0)), spherical_to_cartesian((2.0, -135.0, 135.0))))
        self.assertTrue(np.allclose((-1.0, 1.0, -np.sqrt(2.0)), spherical_to_cartesian((2.0, 135.0, 135.0))))
        self.assertTrue(np.allclose((0.0, 0.0, 1.0), spherical_to_cartesian((1.0, 20.0, 0.0))))
        self.assertTrue(np.allclose((0.0, 0.0, 1.0), spherical_to_cartesian((1.0, 120.0, 0.0))))
        self.assertTrue(np.allclose((0.0, 0.0, 1.0), spherical_to_cartesian((1.0, 220.0, 0.0))))

        self.assertTrue(np.allclose((0.0, 0.0, 0.0), spherical_to_cartesian((0.0, 360.0, 0.0))))
        self.assertTrue(np.allclose((1.0, 0.0, 0.0), spherical_to_cartesian((1.0, 0.0, 450.0))))
        self.assertTrue(np.allclose((-1.0, 0.0, 0.0), spherical_to_cartesian((1.0, 540.0, 90.0))))
        self.assertTrue(np.allclose((0.0, 0.0, 1.0), spherical_to_cartesian((1.0, 0.0, -360.0))))
        self.assertTrue(np.allclose((0.0, 0.0, -1.0), spherical_to_cartesian((1.0, 0.0, -180.0))))
        self.assertTrue(np.allclose((1.0, 1.0, np.sqrt(2.0)), spherical_to_cartesian((2.0, -315.0, 45.0))))
        self.assertTrue(np.allclose((1.0, -1.0, np.sqrt(2.0)), spherical_to_cartesian((2.0, 315.0, 45.0))))
        self.assertTrue(np.allclose((-1.0, -1.0, np.sqrt(2.0)), spherical_to_cartesian((2.0, -135.0, 405.0))))
        self.assertTrue(np.allclose((-1.0, 1.0, np.sqrt(2.0)), spherical_to_cartesian((2.0, -225.0, 45.0))))
        self.assertTrue(np.allclose((1.0, 1.0, -np.sqrt(2.0)), spherical_to_cartesian((2.0, 405.0, 135.0))))
        self.assertTrue(np.allclose((1.0, -1.0, -np.sqrt(2.0)), spherical_to_cartesian((2.0, -45.0, -225.0))))
        self.assertTrue(np.allclose((-1.0, -1.0, -np.sqrt(2.0)), spherical_to_cartesian((2.0, -135.0, 495.0))))
        self.assertTrue(np.allclose((-1.0, 1.0, -np.sqrt(2.0)), spherical_to_cartesian((2.0, 495.0, 135.0))))
        self.assertTrue(np.allclose((0.0, 0.0, 1.0), spherical_to_cartesian((1.0, 20.0, 7200.0))))
        self.assertTrue(np.allclose((0.0, 0.0, 1.0), spherical_to_cartesian((1.0, 3720.0, 0.0))))
        self.assertTrue(np.allclose((0.0, 0.0, 1.0), spherical_to_cartesian((1.0, 220.0, -3600.0))))

    def test_rotate_vector_by_vector(self):
        v = rotate_vector_by_vector(np.array((0.0, 0.0, 0.0)), np.array((0.0, 0.0, 0.0)), np.array((0.0, 0.0, 0.0)))
        self.assertTrue(np.allclose(v, (0.0, 0.0, 0.0)))

        v = rotate_vector_by_vector(np.array((1.0, 0.0, 0.0)), np.array((0.0, 0.0, 0.0)), np.array((0.0, 0.0, 0.0)))
        self.assertTrue(np.allclose(v, (1.0, 0.0, 0.0)))

        v = rotate_vector_by_vector(np.array((1.0, 0.0, 0.0)), np.array((0.0, 0.0, 0.0)), np.array((1.0, 0.0, 0.0)))
        self.assertTrue(np.allclose(v, (1.0, 0.0, 0.0)))

        v = rotate_vector_by_vector(np.array((1.0, 0.0, 0.0)), np.array((1.0, 0.0, 0.0)), np.array((1.0, 0.0, 0.0)))
        self.assertTrue(np.allclose(v, (1.0, 0.0, 0.0)))

        v = rotate_vector_by_vector(np.array((1.0, 0.0, 0.0)), np.array((1.0, 0.0, 0.0)), np.array((2.0, 0.0, 0.0)))
        self.assertTrue(np.allclose(v, (1.0, 0.0, 0.0)))

        v = rotate_vector_by_vector(np.array((1.0, 0.0, 0.0)), np.array((1.0, 0.0, 0.0)), np.array((-1.0, 0.0, 0.0)))
        self.assertTrue(np.allclose(v, (-1.0, 0.0, 0.0)))

        v = rotate_vector_by_vector(np.array((1.0, 0.0, 0.0)), np.array((1.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0)))
        self.assertTrue(np.allclose(v, (0.0, 1.0, 0.0)))

        v = rotate_vector_by_vector(np.array((1.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0)), np.array((0.0, -1.0, 0.0)))
        self.assertTrue(np.allclose(v, (-1.0, 0.0, 0.0)))

        v = rotate_vector_by_vector(np.array((1.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0)), np.array((0.0, 0.0, 1.0)))
        self.assertTrue(np.allclose(v, (1.0, 0.0, 0.0)))

        v = rotate_vector_by_vector(np.array((1.0, 1.0, 0.0)), np.array((0.0, 1.0, 0.0)), np.array((0.0, 0.0, 1.0)))
        self.assertTrue(np.allclose(v, (1.0, 0.0, 1.0)))

    def test_rotate_axis_angle(self):
        v = rotate_axis_angle(np.array((0.0, 0.0, 0.0)), np.array((0.0, 0.0, 1.0)), 0.0)
        self.assertTrue(np.allclose(v, (0.0, 0.0, 0.0)))

        v = rotate_axis_angle(np.array((0.0, 0.0, 0.0)), np.array((0.0, 0.0, 1.0)), 40.0)
        self.assertTrue(np.allclose(v, (0.0, 0.0, 0.0)))

        v = rotate_axis_angle(np.array((0.0, 0.0, 0.0)), np.array((0.0, 0.0, 0.0)), 0.0)
        self.assertTrue(np.allclose(v, (0.0, 0.0, 0.0)))

        v = rotate_axis_angle(np.array((0.0, 0.0, 0.0)), np.array((0.0, 0.0, 0.0)), -320.0)
        self.assertTrue(np.allclose(v, (0.0, 0.0, 0.0)))

        v = rotate_axis_angle(np.array((1.0, 1.0, 1.0)), np.array((1.0, 1.0, 1.0)), 40.0)
        self.assertTrue(np.allclose(v, (1.0, 1.0, 1.0)))

        v = rotate_axis_angle(np.array((1.0, 1.0, 1.0)), np.array((-1.0, -1.0, -1.0)), 340.0)
        self.assertTrue(np.allclose(v, (1.0, 1.0, 1.0)))

        v = rotate_axis_angle(np.array((1.0, 1.0, 1.0)), np.array((0.0, 0.0, 1.0)), 90.0)
        self.assertTrue(np.allclose(v, (-1.0, 1.0, 1.0)))

        v = rotate_axis_angle(np.array((1.0, 1.0, 1.0)), np.array((0.0, 0.0, 1.0)), -90.0)
        self.assertTrue(np.allclose(v, (1.0, -1.0, 1.0)))

        v = rotate_axis_angle(np.array((1.0, 1.0, 1.0)), np.array((0.0, 0.0, -1.0)), -90.0)
        self.assertTrue(np.allclose(v, (-1.0, 1.0, 1.0)))

        v = rotate_axis_angle(np.array((1.0, 1.0, 1.0)), np.array((0.0, 0.0, -1.0)), 180.0)
        self.assertTrue(np.allclose(v, (-1.0, -1.0, 1.0)))

        v = rotate_axis_angle(np.array((0.0, 0.0, 1.0)), np.array((1.0, 0.0, 0.0)), 90.0)
        self.assertTrue(np.allclose(v, (0.0, -1.0, 0.0)))

        v = rotate_axis_angle(np.array((0.0, -1.0, 0.0)), np.array((0.0, 0.0, 1.0)), -90.0)
        self.assertTrue(np.allclose(v, (-1.0, 0.0, 0.0)))

    def test_vectors_to_axis_angle(self):
        v, a = vectors_to_axis_angle(np.array((0.0, 0.0, 0.0)), np.array((0.0, 0.0, 0.0)))
        self.assertTrue(np.allclose(v, (0.0, 0.0, 1.0)))
        self.assertAlmostEqual(a, 0.0)

        v, a = vectors_to_axis_angle(np.array((1.0, 0.0, 0.0)), np.array((0.0, 0.0, 0.0)))
        self.assertTrue(np.allclose(v, (0.0, 0.0, 1.0)))
        self.assertAlmostEqual(a, 0.0)

        v, a = vectors_to_axis_angle(np.array((1.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0)))
        self.assertTrue(np.allclose(v, (0.0, 0.0, 1.0)))
        self.assertAlmostEqual(a, 90.0)

        v, a = vectors_to_axis_angle(np.array((0.0, 1.0, 0.0)), np.array((1.0, 0.0, 0.0)))
        self.assertTrue(np.allclose(v, (0.0, 0.0, -1.0)))
        self.assertAlmostEqual(a, 90.0)

        v, a = vectors_to_axis_angle(np.array((1.0, 0.0, 0.0)), np.array((1.0, 0.0, 0.0)))
        self.assertTrue(np.allclose(v, (0.0, 0.0, 1.0)))
        self.assertAlmostEqual(a, 00.0)

        v, a = vectors_to_axis_angle(np.array((1.0, 0.0, 0.0)), np.array((-1.0, 0.0, 0.0)))
        self.assertTrue(np.allclose(v, (0.0, 0.0, 1.0)))
        self.assertAlmostEqual(a, 180.0)

        v, a = vectors_to_axis_angle(np.array((-1.0, 0.0, 0.0)), np.array((1.0, 0.0, 0.0)))
        self.assertTrue(np.allclose(v, (0.0, 0.0, 1.0)))
        self.assertAlmostEqual(a, 180.0)

        v, a = vectors_to_axis_angle(np.array((0.0, 1.0, 0.0)), np.array((0.0, 0.0, 1.0)))
        self.assertTrue(np.allclose(v, (1.0, 0.0, 0.0)))
        self.assertAlmostEqual(a, 90.0)

        v, a = vectors_to_axis_angle(np.array((0.0, 0.0, 1.0)), np.array((0.0, 1.0, 0.0)))
        self.assertTrue(np.allclose(v, (-1.0, 0.0, 0.0)))
        self.assertAlmostEqual(a, 90.0)

        v, a = vectors_to_axis_angle(np.array((0.0, 1.0, 0.0)), np.array((0.0, 0.0, -1.0)))
        self.assertTrue(np.allclose(v, (-1.0, 0.0, 0.0)))
        self.assertAlmostEqual(a, 90.0)

        v, a = vectors_to_axis_angle(np.array((1.0, 0.0, 0.0)), np.array((0.0, 0.0, 1.0)))
        self.assertTrue(np.allclose(v, (0.0, -1.0, 0.0)))
        self.assertAlmostEqual(a, 90.0)

        v, a = vectors_to_axis_angle(np.array((0.0, 0.0, 1.0)), np.array((1.0, 0.0, 0.0)))
        self.assertTrue(np.allclose(v, (0.0, 1.0, 0.0)))
        self.assertAlmostEqual(a, 90.0)

        v, a = vectors_to_axis_angle(np.array((1.0, 1.0, 0.0)), np.array((1.0, 1.0, np.sqrt(6.0))))
        self.assertTrue(np.allclose(v, (0.5*np.sqrt(2.0), -0.5*np.sqrt(2.0), 0.0)))
        self.assertAlmostEqual(a, 60.0)

    def test_angle_between_vectors(self):
        self.assertAlmostEqual(angle_between_vectors(np.array((0.0, 0.0, 0.0)), np.array((0.0, 0.0, 0.0))), 0.0)
        self.assertAlmostEqual(angle_between_vectors(np.array((0.0, 0.0, 1.0)), np.array((0.0, 0.0, 0.0))), 0.0)
        self.assertAlmostEqual(angle_between_vectors(np.array((0.0, 0.0, 1.0)), np.array((0.0, 0.0, 1.0))), 0.0)
        self.assertAlmostEqual(angle_between_vectors(np.array((0.0, 0.0, 1.0)), np.array((0.0, 0.0, 10.0))), 0.0)
        self.assertAlmostEqual(angle_between_vectors(np.array((0.0, 0.0, 1.0)), np.array((0.0, 0.0, -1.0))), 180.0)
        self.assertAlmostEqual(angle_between_vectors(np.array((0.0, 0.0, 1.0)), np.array((0.0, 0.0, -10.0))), 180.0)
        self.assertAlmostEqual(angle_between_vectors(np.array((0.0, 0.0, 1.0)), np.array((0.0, 1.0, 0.0))), 90.0)
        self.assertAlmostEqual(angle_between_vectors(np.array((0.0, 0.0, 1.0)), np.array((1.0, 0.0, 0.0))), 90.0)
        self.assertAlmostEqual(angle_between_vectors(np.array((0.0, 0.0, 1.0)), np.array((0.0, -1.0, 0.0))), 90.0)
        self.assertAlmostEqual(angle_between_vectors(np.array((0.0, 0.0, 1.0)), np.array((-1.0, 0.0, 0.0))), 90.0)
        self.assertAlmostEqual(angle_between_vectors(np.array((0.0, 0.0, -1.0)), np.array((0.0, -1.0, 0.0))), 90.0)
        self.assertAlmostEqual(angle_between_vectors(np.array((0.0, 0.0, -1.0)), np.array((-1.0, 0.0, 0.0))), 90.0)
        self.assertAlmostEqual(angle_between_vectors(np.array((1.0, 1.0, 0.0)),
                                                     np.array((1.0, 1.0, np.sqrt(2.0)))), 45.0)
        self.assertAlmostEqual(angle_between_vectors(np.array((1.0, 1.0, np.sqrt(2.0))),
                                                     np.array((1.0, 1.0, 0.0))), 45.0)
        self.assertAlmostEqual(angle_between_vectors(np.array((1.0, 1.0, 0.0)),
                                                     np.array((1.0, 1.0, np.sqrt(6.0)))), 60.0)
