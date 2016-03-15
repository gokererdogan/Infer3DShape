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

from Infer3DShape.shape import Shape, CuboidPrimitive
from Infer3DShape.paperclip_shape import PaperClipShape

from i3d_test_case import *
from Infer3DShape.vision_forward_model import *

# There is a problem with offscreen rendering in VTK. If we create two forward models with offscreen rendering in the
# same instance, we get segmentation faults when rendering. In order circumvent this problem for the moment, I have
# split the testing routines with no custom lighting and custom lighting present.


class VisionForwardModelTestNoLighting(unittest.TestCase):
    def setUp(self):
        dist = np.sqrt(8.0)
        self.camera_pos = []
        for phi in range(0, 181, 45):
            for theta in range(0, 360, 45):
                self.camera_pos.append((dist, theta, phi))

        self.fwm_no_lighting = VisionForwardModel(render_size=(200, 200), custom_lighting=False,
                                                  camera_pos=self.camera_pos, offscreen_rendering=True)

    def tearDown(self):
        del self.fwm_no_lighting
        self.camera_pos = None

    def test_render_blank(self):
        # blank image
        s = Shape(forward_model=None, parts=[])
        r = self.fwm_no_lighting.render(s)
        self.assertTupleEqual(r.shape, (40, 200, 200))
        self.assertAlmostEqual(np.sum(r), 0.0)

    def test_render_cube_no_lighting(self):
        # test object
        part1 = CuboidPrimitive([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        s = Shape(forward_model=None, parts=[part1])
        r = self.fwm_no_lighting.render(s)
        correct = np.load('test_images/cube_no_lighting.npy')
        self.assertAlmostEqual(np.sum(np.abs(r - correct)), 0.0)

        # test changing viewpoint
        for i, pos in enumerate(self.camera_pos):
            s.viewpoint = [pos]
            r = self.fwm_no_lighting.render(s)
            self.assertAlmostEqual(np.sum(np.abs(r - correct[i])), 0.0)

    def test_render_cube_object_no_lighting(self):
        # test object
        part1 = CuboidPrimitive([0.0, 0.0, 0.0], [0.8, 1.0, 0.4])
        part2 = CuboidPrimitive([0.6, 0.0, 0.0], [0.4, 0.4, 0.3])
        part3 = CuboidPrimitive([0.0, 0.0, 0.50], [0.2, 0.6, 0.6])
        s = Shape(forward_model=None, parts=[part1, part2, part3])
        r = self.fwm_no_lighting.render(s)
        correct = np.load('test_images/cube_object_no_lighting.npy')
        self.assertAlmostEqual(np.sum(np.abs(r - correct)), 0.0)

    def test_render_tube_object_no_lighting(self):
        # test object
        s = PaperClipShape(forward_model=None, params={'JOINT_VARIANCE': 0.2})
        s.joint_positions=[(.6, 0.0, 0.0), (0.0, .6, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, .6)]

        r = self.fwm_no_lighting.render(s)
        correct = np.load('test_images/tube_object_no_lighting.npy')
        self.assertAlmostEqual(np.sum(np.abs(r - correct)), 0.0)

    def test_calculate_camera_up(self):
        pos = (1.0, 0.0, 0.0)
        self.assertTrue(np.allclose(self.fwm_no_lighting._calculate_camera_up(pos), (-1.0, 0.0, 0.0)))

        pos = (1.0, 45.0, 0.0)
        self.assertTrue(np.allclose(self.fwm_no_lighting._calculate_camera_up(pos),
                                    (-0.5*np.sqrt(2.0), -0.5*np.sqrt(2.0), 0.0)))

        pos = (1.0, 90.0, 0.0)
        self.assertTrue(np.allclose(self.fwm_no_lighting._calculate_camera_up(pos),
                                    (0.0, -1.0, 0.0)))

        pos = (1.0, 0.0, 45.0)
        self.assertTrue(np.allclose(self.fwm_no_lighting._calculate_camera_up(pos),
                                    (-0.5*np.sqrt(2.0), 0.0, 0.5*np.sqrt(2.0))))

        pos = (1.0, 135.0, 90.0)
        self.assertTrue(np.allclose(self.fwm_no_lighting._calculate_camera_up(pos),
                                    (0.0, 0.0, 1.0)))

        pos = (1.0, 45.0, 135.0)
        self.assertTrue(np.allclose(self.fwm_no_lighting._calculate_camera_up(pos),
                                    (0.5, 0.5, 0.5*np.sqrt(2.0))))

        pos = (1.0, 0.0, 180.0)
        self.assertTrue(np.allclose(self.fwm_no_lighting._calculate_camera_up(pos),
                                    (1.0, 0.0, 0.0)))

        pos = (1.0, 225.0, 180.0)
        self.assertTrue(np.allclose(self.fwm_no_lighting._calculate_camera_up(pos),
                                    (-0.5*np.sqrt(2.0), -0.5*np.sqrt(2.0), 0.0)))


class VisionForwardModelTestCustomLighting(unittest.TestCase):
    def setUp(self):
        dist = np.sqrt(8.0)
        self.camera_pos = []
        for phi in range(0, 181, 45):
            for theta in range(0, 360, 45):
                self.camera_pos.append((dist, theta, phi))

        self.fwm_custom_lighting = VisionForwardModel(render_size=(200, 200), custom_lighting=True,
                                                      camera_pos=self.camera_pos, offscreen_rendering=True)

    def tearDown(self):
        del self.fwm_custom_lighting
        self.camera_pos = None

    def test_render_blank(self):
        s = Shape(forward_model=None, parts=[])
        r = self.fwm_custom_lighting.render(s)
        self.assertTupleEqual(r.shape, (40, 200, 200))
        self.assertAlmostEqual(np.sum(r), 0.0)

    def test_render_cube_custom_lighting(self):
        # test object
        part1 = CuboidPrimitive([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        s = Shape(forward_model=None, parts=[part1])
        r = self.fwm_custom_lighting.render(s)
        correct = np.load('test_images/cube_custom_lighting.npy')
        self.assertAlmostEqual(np.sum(np.abs(r - correct)), 0.0)

        # maximum intensity value for custom lighting should be around 174-175. This value is used in our likelihood
        # calculations; therefore, it is important to make sure that it does not change.
        self.assertTrue(175.0 > np.max(r) > 174.0)

        # test changing viewpoint
        for i, pos in enumerate(self.camera_pos):
            s.viewpoint = [pos]
            r = self.fwm_custom_lighting.render(s)
            self.assertAlmostEqual(np.sum(np.abs(r - correct[i])), 0.0)

    def test_render_cube_object_custom_lighting(self):
        # test object
        part1 = CuboidPrimitive([0.0, 0.0, 0.0], [0.8, 1.0, 0.4])
        part2 = CuboidPrimitive([0.6, 0.0, 0.0], [0.4, 0.4, 0.3])
        part3 = CuboidPrimitive([0.0, 0.0, 0.50], [0.2, 0.6, 0.6])
        s = Shape(forward_model=None, parts=[part1, part2, part3])
        r = self.fwm_custom_lighting.render(s)
        correct = np.load('test_images/cube_object_custom_lighting.npy')
        self.assertAlmostEqual(np.sum(np.abs(r - correct)), 0.0)

    def test_render_tube_object_custom_lighting(self):
        # test object
        s = PaperClipShape(forward_model=None, params={'JOINT_VARIANCE': 0.2})
        s.joint_positions=[(.6, 0.0, 0.0), (0.0, .6, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, .6)]

        r = self.fwm_custom_lighting.render(s)
        correct = np.load('test_images/tube_object_custom_lighting.npy')
        self.assertAlmostEqual(np.sum(np.abs(r - correct)), 0.0)

    def test_save_render(self):
        # test object
        part1 = CuboidPrimitive([0.0, 0.0, 0.0], [0.8, 1.0, 0.4])
        part2 = CuboidPrimitive([0.6, 0.0, 0.0], [0.4, 0.4, 0.3])
        part3 = CuboidPrimitive([0.0, 0.0, 0.50], [0.2, 0.6, 0.6])
        s = Shape(forward_model=None, parts=[part1, part2, part3])
        self.fwm_custom_lighting.save_render('test_images/r.png', s)
        for i in range(len(self.fwm_custom_lighting.camera_pos)):
            r = spi.imread('test_images/r_{0:d}.png'.format(i))
            correct = spi.imread('test_images/cube_object_custom_lighting_{0:d}.png'.format(i))
            self.assertAlmostEqual(np.sum(np.abs(r - correct)), 0.0)
            os.remove('test_images/r_{0:d}.png'.format(i))

