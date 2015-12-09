"""
Inferring 3D Shape from 2D Images

Unit tests for i3d_likelihood module.

Created on Dec 2, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import numpy as np
import unittest

from Infer3DShape.i3d_likelihood import *

class I3DLikelihoodTest(unittest.TestCase):
    def test_log_ll_pixel(self):
        d0 = np.zeros((2, 50, 50))
        d1 = np.ones((2, 50, 50))
        ll = log_likelihood_pixel(d0, d0, 1.0, 1.0)
        self.assertAlmostEqual(ll, 0.0)
        ll = log_likelihood_pixel(d0, d1, 1.0, 1.0)
        self.assertAlmostEqual(ll, -0.5)
        ll = log_likelihood_pixel(d0, d1, 2.0, 1.0)
        self.assertAlmostEqual(ll, -0.125)
        ll = log_likelihood_pixel(d0, d1, 1.0, 2.0)
        self.assertAlmostEqual(ll, -0.25)
        ll = log_likelihood_pixel(d0, d1, 2.0, 2.0)
        self.assertAlmostEqual(ll, -0.0625)

    def test_log_ll_pixel_gaussian_filtered(self):
        d0 = np.zeros((2, 50, 50))
        d1 = np.ones((2, 50, 50))
        # since both inputs are constant, gaussian filtering should not change anything
        ll = log_likelihood_pixel_gaussian_filtered(d0, d0, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(ll, 0.0)
        ll = log_likelihood_pixel_gaussian_filtered(d0, d1, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(ll, -0.5)
        ll = log_likelihood_pixel_gaussian_filtered(d0, d1, 2.0, 1.0, 1.0)
        self.assertAlmostEqual(ll, -0.125)
        ll = log_likelihood_pixel_gaussian_filtered(d0, d1, 1.0, 2.0, 2.0)
        self.assertAlmostEqual(ll, -0.25)
        ll = log_likelihood_pixel_gaussian_filtered(d0, d1, 2.0, 2.0, 4.0)
        self.assertAlmostEqual(ll, -0.0625)

