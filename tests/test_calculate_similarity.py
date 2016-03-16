"""
Inferring 3D Shape from 2D Images

Unit tests for calculate_similarity module.

Created on Mar 14, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""
import numpy as np

from i3d_test_case import *
from Infer3DShape.shape import Shape, CuboidPrimitive
import Infer3DShape.vision_forward_model as i3d_vfm

from Infer3DShape.similarity.calculate_similarity import calculate_probability_image_given_hypothesis, \
    calculate_similarity_image_given_image


class CalculateSimilarityTest(I3DTestCase):
    def test_calculate_probability_image_given_hypothesis(self):
        fwm = i3d_vfm.VisionForwardModel(render_size=(200, 200), offscreen_rendering=True, custom_lighting=False)
        # create a cube, this will be our sample
        s = Shape(forward_model=fwm, viewpoint=[np.array([np.sqrt(8.0), 0.0, 90.0])],
                  params={'MAX_PIXEL_VALUE': 255.0, 'LL_VARIANCE': 0.1},
                  parts=[CuboidPrimitive(position=[0.0, 0.0, 0.0], size=[1.0, 1.0, 1.0])])
        # observed image
        image = np.load('test_images/test_image.npy')

        # calculate logp(image|sample)
        logp, logp_max = calculate_probability_image_given_hypothesis(image, s, viewpoint_samples=180)
        # these are pre-calculated. NOTE these could change if VisionForwardModel changes.
        self.assertAlmostEqual(logp, -0.38698888158103095)
        self.assertAlmostEqual(logp_max, 0.0)

        # calculate logp(blank image|sample)
        image = np.zeros((1, 200, 200))
        logp, logp_max = calculate_probability_image_given_hypothesis(image, s, viewpoint_samples=180)
        self.assertAlmostEqual(logp, -1.7362902909245306)
        self.assertAlmostEqual(logp_max, -1.2966692431630578)

    def test_calculate_similarity_image_given_image(self):
        fwm = i3d_vfm.VisionForwardModel(render_size=(200, 200), offscreen_rendering=True, custom_lighting=False)
        # create two samples
        s1 = Shape(forward_model=fwm, viewpoint=[np.array([np.sqrt(8.0), 0.0, 90.0])],
                   params={'MAX_PIXEL_VALUE': 255.0, 'LL_VARIANCE': 0.1},
                   parts=[CuboidPrimitive(position=[0.0, 0.0, 0.0], size=[1.0, 1.0, 1.0])])

        s2 = Shape(forward_model=fwm, viewpoint=[np.array([np.sqrt(8.0), 0.0, 90.0])],
                   params={'MAX_PIXEL_VALUE': 255.0, 'LL_VARIANCE': 0.1},
                   parts=[CuboidPrimitive(position=[0.0, 0.0, 0.0], size=[.5, .5, .5])])

        # observed image
        image = np.load('test_images/test_image.npy')

        # calculate similarity using two samples
        logp_avg, logp_wavg, logp_best, logp_wbest = calculate_similarity_image_given_image(image, [s1, s2],
                                                                                            [np.log(2.0), np.log(1.0)])
        # these are pre-calculated. NOTE these could change if VisionForwardModel changes.
        self.assertAlmostEqual(logp_avg, np.log(0.42176727900195082))
        self.assertAlmostEqual(logp_wavg, np.log(0.50754440113060817))
        self.assertAlmostEqual(logp_best, np.log(0.58575908347843986))
        self.assertAlmostEqual(logp_wbest, np.log(0.72383938898562661))

        # calculate similarity from one sample
        logp_avg, logp_wavg, logp_best, logp_wbest = calculate_similarity_image_given_image(image, [s1], [np.log(2.0)])

        # these are pre-calculated. NOTE these could change if VisionForwardModel changes.
        self.assertAlmostEqual(logp_avg, -0.38698888158103095)
        self.assertAlmostEqual(logp_wavg, -0.38698888158103095)
        self.assertAlmostEqual(logp_best, 0.0)
        self.assertAlmostEqual(logp_wbest, 0.0)



