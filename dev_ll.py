"""
Inferring 3D Shape from 2D Images

This script is for developing different likelihood models for Shape class.

Created on Sep 2, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import hypothesis as hyp
import vision_forward_model as vfm
import numpy as np

fwm = vfm.VisionForwardModel()

# test object 1
print('Test object 1')

parts = [hyp.CuboidPrimitive(np.array([0.0, 0.0, 0.0]), np.array([1.0, .75, .75])),
         hyp.CuboidPrimitive(np.array([.75, 0.0, 0.0]), np.array([.5, .5, .5]))]

gt = hyp.Shape(fwm, parts)
data = np.load('./data/test1.npy')

print("prior: {0:f}, ll: {1:f}, post: {2:f}".format(gt.prior(), gt.likelihood(data), gt.prior() * gt.likelihood(data)))

parts = [hyp.CuboidPrimitive(np.array([0.19, 0.0, 0.0]), np.array([1.38, .75, .75]))]

h1 = hyp.Shape(fwm, parts)
h1_data = fwm.render(h1)
# fwm.save_render("1_1.png", h1)

print("prior: {0:f}, ll: {1:f}, post: {2:f}".format(h1.prior(), h1.likelihood(data), h1.prior() * h1.likelihood(data)))

parts = [hyp.CuboidPrimitive(np.array([0.1, 0.0, 0.0]), np.array([1.2, .75, .75])),
         hyp.CuboidPrimitive(np.array([.8, 0.0, 0.0]), np.array([.6, .55, .5]))]

h2 = hyp.Shape(fwm, parts)
h2_data = fwm.render(h2)
# fwm.save_render("1_2.png", h2)

print("prior: {0:f}, ll: {1:f}, post: {2:f}".format(h2.prior(), h2.likelihood(data), h2.prior() * h2.likelihood(data)))


# test object 2
print("\nTest object 2")
parts = [hyp.CuboidPrimitive(np.array([0.0, 0.0, 0.0]), np.array([1.0, .75, .75])),
         hyp.CuboidPrimitive(np.array([.9, 0.0, 0.0]), np.array([.8, .5, .5])),
         hyp.CuboidPrimitive(np.array([0.0, 0.0, 0.75]), np.array([0.25, 0.35, 0.75])),
         hyp.CuboidPrimitive(np.array([0.0, 0.4, 0.75]), np.array([.2, .45, .25]))]

gt2 = hyp.Shape(fwm, parts)
data2 = np.load('./data/test2.npy')

print("prior: {0:f}, ll: {1:f}, post: {2:f}".format(gt2.prior(), gt2.likelihood(data2), gt2.prior() * gt2.likelihood(data2)))

parts = [hyp.CuboidPrimitive(np.array([0.0, 0.0, 0.0]), np.array([1.0, .75, .75])),
         hyp.CuboidPrimitive(np.array([.9, 0.0, 0.0]), np.array([.8, .5, .5])),
         hyp.CuboidPrimitive(np.array([0.0, 0.0, 0.75]), np.array([0.25, 0.35, 0.75]))]

h2_1 = hyp.Shape(fwm, parts)
h2_1_data = fwm.render(h2_1)
# fwm.save_render("2_1.png", h2_1)

print("prior: {0:f}, ll: {1:f}, post: {2:f}".format(h2_1.prior(), h2_1.likelihood(data2), h2_1.prior() * h2_1.likelihood(data2)))


parts = [hyp.CuboidPrimitive(np.array([0.0, 0.0, 0.0]), np.array([1.0, .75, .75])),
         hyp.CuboidPrimitive(np.array([.9, 0.0, 0.0]), np.array([.8, .5, .5])),
         hyp.CuboidPrimitive(np.array([0.0, 0.0, 0.75]), np.array([0.25, 0.35, 0.75])),
         hyp.CuboidPrimitive(np.array([0.0, 0.3, 0.75]), np.array([.15, .25, .55]))]

h2_2 = hyp.Shape(fwm, parts)
h2_2_data = fwm.render(h2_2)
# fwm.save_render("2_2.png", h2_2)

print("prior: {0:f}, ll: {1:f}, post: {2:f}".format(h2_2.prior(), h2_2.likelihood(data2), h2_2.prior() * h2_2.likelihood(data2)))

'''
import skimage as ski
import skimage.feature as skif
import skimage.draw as skid

import skimage.io as skiio
'''
# daisy
# hog
# register translation
