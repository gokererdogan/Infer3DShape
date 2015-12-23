"""
Inferring 3D Shape from 2D Images

Unit tests for vision_forward_model module.

Created on Dec 4, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import os

import numpy as np
import scipy.ndimage as spi
import unittest

from Infer3DShape.vision_forward_model import *
from Infer3DShape.shape import Shape, CuboidPrimitive


class VisionForwardModelTest(unittest.TestCase):
    def test_render(self):
        fwm = VisionForwardModel(render_size=(200, 200),
                                 camera_pos=[(2.0, 2.0, 1.5), (-2.0, -2.0, 1.5), (2.0, 0.5, 1.5)])
        # blank image
        s = Shape(forward_model=fwm, parts=[])
        r = fwm.render(s)
        self.assertTupleEqual(r.shape, (3, 200, 200))
        self.assertAlmostEqual(np.sum(r), 0.0)
        # test object
        part1 = CuboidPrimitive([0.0, 0.0, 0.0], [0.3, 0.2, 0.4])
        part2 = CuboidPrimitive([-0.3, 0.1, 0.2], [0.1, 0.6, 0.1])
        s = Shape(forward_model=fwm, parts=[part1, part2])
        r = fwm.render(s)
        correct = np.load('test_vision_forward_model_image.npy')
        self.assertAlmostEqual(np.sum(np.abs(r - correct)), 0.0)
        # test changing viewpoint
        s.viewpoint = [(2.0, 2.0, 1.5)]
        r = fwm.render(s)
        self.assertAlmostEqual(np.sum(np.abs(r - correct[0])), 0.0)
        s.viewpoint = [(-2.0, -2.0, 1.5)]
        r = fwm.render(s)
        self.assertAlmostEqual(np.sum(np.abs(r - correct[1])), 0.0)
        s.viewpoint = [(2.0, 0.5, 1.5)]
        r = fwm.render(s)
        self.assertAlmostEqual(np.sum(np.abs(r - correct[2])), 0.0)

    def test_save_render(self):
        fwm = VisionForwardModel(render_size=(200, 200),
                                 camera_pos=[(2.0, 2.0, 1.5), (-2.0, -2.0, 1.5), (2.0, 0.5, 1.5)])
        # test object
        part1 = CuboidPrimitive([0.0, 0.0, 0.0], [0.3, 0.2, 0.4])
        part2 = CuboidPrimitive([-0.3, 0.1, 0.2], [0.1, 0.6, 0.1])
        s = Shape(forward_model=fwm, parts=[part1, part2])
        fwm.save_render('r.png', s)
        r = spi.imread('r_0.png')
        correct = spi.imread('test_vision_forward_model_0.png')
        self.assertAlmostEqual(np.sum(np.abs(r - correct)), 0.0)
        r = spi.imread('r_1.png')
        correct = spi.imread('test_vision_forward_model_1.png')
        self.assertAlmostEqual(np.sum(np.abs(r - correct)), 0.0)
        r = spi.imread('r_2.png')
        correct = spi.imread('test_vision_forward_model_2.png')
        self.assertAlmostEqual(np.sum(np.abs(r - correct)), 0.0)
        os.remove('r_0.png')
        os.remove('r_1.png')
        os.remove('r_2.png')

