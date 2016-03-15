"""
Inferring 3D Shape from 2D Images

Unit tests for i3d_likelihood module.

Created on Dec 2, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import numpy as np

import Infer3DShape.vision_forward_model as vfm
import Infer3DShape.shape as shape

from i3d_test_case import *
from Infer3DShape.i3d_likelihood import *


class I3DLikelihoodTest(I3DTestCase):
    def setUp(self):
        self.fwm = vfm.VisionForwardModel(render_size=(50, 50))
        self.s = shape.Shape(forward_model=self.fwm, parts=[])

    def tearDown(self):
        del self.fwm
        del self.s

    def test_log_ll_pixel(self):
        d0 = np.zeros((3, 50, 50))
        d1 = np.ones((3, 50, 50))
        ll = log_likelihood_pixel(self.s, d0, 1.0, 1.0)
        self.assertAlmostEqual(ll, 0.0)
        ll = log_likelihood_pixel(self.s, d1, 1.0, 1.0)
        self.assertAlmostEqual(ll, -0.5)
        ll = log_likelihood_pixel(self.s, d1, 2.0, 1.0)
        self.assertAlmostEqual(ll, -0.125)
        ll = log_likelihood_pixel(self.s, d1, 1.0, 2.0)
        self.assertAlmostEqual(ll, -0.25)
        ll = log_likelihood_pixel(self.s, d1, 2.0, 2.0)
        self.assertAlmostEqual(ll, -0.0625)

    def test_log_ll_pixel_gaussian_filtered(self):
        d0 = np.zeros((3, 50, 50))
        d1 = np.ones((3, 50, 50))
        # since both inputs are constant, gaussian filtering should not change anything
        ll = log_likelihood_pixel_gaussian_filtered(self.s, d0, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(ll, 0.0)
        ll = log_likelihood_pixel_gaussian_filtered(self.s, d1, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(ll, -0.5)
        ll = log_likelihood_pixel_gaussian_filtered(self.s, d1, 2.0, 1.0, 1.0)
        self.assertAlmostEqual(ll, -0.125)
        ll = log_likelihood_pixel_gaussian_filtered(self.s, d1, 1.0, 2.0, 2.0)
        self.assertAlmostEqual(ll, -0.25)
        ll = log_likelihood_pixel_gaussian_filtered(self.s, d1, 2.0, 2.0, 4.0)
        self.assertAlmostEqual(ll, -0.0625)

