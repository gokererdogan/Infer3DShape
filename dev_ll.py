"""
Inferring 3D Shape from 2D Images

This script is for developing different likelihood models for Shape class.

Created on Sep 2, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import numpy as np

import vision_forward_model as vfm
import shape


LL_VARIANCE = 0.001

fwm = vfm.VisionForwardModel(render_size=(200, 200))
params = {'LL_VARIANCE': LL_VARIANCE}

# test object 1
print('Test object 1')

view_angle = 262.0
viewpoint_x = 1.5 * np.sqrt(2.0) * np.cos(view_angle * np.pi / 180.0)
viewpoint_y = 1.5 * np.sqrt(2.0) * np.sin(view_angle * np.pi / 180.0)
viewpoint_z = 1.5
viewpoint = [(viewpoint_x, viewpoint_y, viewpoint_z)]

parts = [shape.CuboidPrimitive(np.array([0.0, 0.0, 0.0]), np.array([0.5, .375, .375])),
         shape.CuboidPrimitive(np.array([.375, 0.0, 0.0]), np.array([.25, .25, .25]))]

gt = shape.Shape(forward_model=fwm, parts=parts, viewpoint=viewpoint, params=params)
data = np.load('./data/test1_single_view.npy')

print("prior: {0:f}, ll: {1:f}, post: {2:f}".format(gt.log_prior(), gt.log_likelihood(data),
                                                    gt.log_prior() + gt.log_likelihood(data)))


parts = [shape.CuboidPrimitive(np.array([0.095, 0.0, 0.0]), np.array([0.69, .375, .375]))]

h1 = shape.Shape(forward_model=fwm, parts=parts, viewpoint=viewpoint, params=params)
h1_data = fwm.render(h1)
# fwm.save_render("1_1.png", h1)

print("prior: {0:f}, ll: {1:f}, post: {2:f}".format(h1.log_prior(), h1.log_likelihood(data),
                                                    h1.log_prior() + h1.log_likelihood(data)))


parts = [shape.CuboidPrimitive(np.array([0.05, 0.0, 0.0]), np.array([0.6, .375, .375])),
         shape.CuboidPrimitive(np.array([.4, 0.0, 0.0]), np.array([.3, .275, .25]))]

h2 = shape.Shape(forward_model=fwm, parts=parts, viewpoint=viewpoint, params=params)
h2_data = fwm.render(h2)
# fwm.save_render("1_2.png", h2)

print("prior: {0:f}, ll: {1:f}, post: {2:f}".format(h2.log_prior(), h2.log_likelihood(data),
                                                    h2.log_prior() + h2.log_likelihood(data)))

"""
# test object 2
print("\nTest object 2")
parts = [hyp.CuboidPrimitive(np.array([0.0, 0.0, 0.0]), np.array([1.0, .75, .75])),
         hyp.CuboidPrimitive(np.array([.9, 0.0, 0.0]), np.array([.8, .5, .5])),
         hyp.CuboidPrimitive(np.array([0.0, 0.0, 0.75]), np.array([0.25, 0.35, 0.75])),
         hyp.CuboidPrimitive(np.array([0.0, 0.4, 0.75]), np.array([.2, .45, .25]))]

gt2 = hyp.Shape(fwm, parts)
data2 = np.load('./data/test2.npy')

print("prior: {0:e}, ll: {1:e}, post: {2:e}".format(gt2.prior(), gt2.likelihood(data2), gt2.prior() * gt2.likelihood(data2)))

parts = [hyp.CuboidPrimitive(np.array([0.0, 0.0, 0.0]), np.array([1.0, .75, .75])),
         hyp.CuboidPrimitive(np.array([.9, 0.0, 0.0]), np.array([.8, .5, .5])),
         hyp.CuboidPrimitive(np.array([0.0, 0.0, 0.75]), np.array([0.25, 0.35, 0.75]))]

h2_1 = hyp.Shape(fwm, parts)
h2_1_data = fwm.render(h2_1)
# fwm.save_render("2_1.png", h2_1)

print("prior: {0:e}, ll: {1:e}, post: {2:e}".format(h2_1.prior(), h2_1.likelihood(data2), h2_1.prior() * h2_1.likelihood(data2)))


parts = [hyp.CuboidPrimitive(np.array([0.0, 0.0, 0.0]), np.array([1.0, .75, .75])),
         hyp.CuboidPrimitive(np.array([.9, 0.0, 0.0]), np.array([.8, .5, .5])),
         hyp.CuboidPrimitive(np.array([0.0, 0.0, 0.75]), np.array([0.25, 0.35, 0.75])),
         hyp.CuboidPrimitive(np.array([0.0, 0.3, 0.75]), np.array([.15, .25, .55]))]

h2_2 = hyp.Shape(fwm, parts)
h2_2_data = fwm.render(h2_2)
# fwm.save_render("2_2.png", h2_2)

print("prior: {0:e}, ll: {1:e}, post: {2:e}".format(h2_2.prior(), h2_2.likelihood(data2), h2_2.prior() * h2_2.likelihood(data2)))
"""
'''
import skimage as ski
import skimage.feature as skif
import skimage.draw as skid

import skimage.io as skiio
'''
# daisy
# hog
# register translation


# 3 Dec. 2015 -----------------------------------------------------------#
# WHAT SHOULD LL_VARIANCE BE?
# I have been struggling to strike a balance between the prior and likelihood.
# So far, this was mainly a trial and error based procedure.
# Here, I would like to look at how one can calculate what the maximum variance value should be if one wants to ensure
# that more complex objects have higher posterior.
# Here is the problem. Imagine we have an object with a single part and this gets 1-d of the image right, i.e.,
# difference between the predicted and observed / image size = d. Note that since we assume pixels take values in [0, 1]
# d=1.0 means the two images are as far as they can be, and d=0.0 means two are exactly the same.
# now, imagine that we do know the right hypothesis has in fact two parts and adding a new part decreases likelihood k
# fold. What should the LL variance be to ensure that a 2-part object with a higher posterior exists?
# to put it in another way, if 1-part object is d percent wrong, assuming we can get d down to 0.0 by adding a new part,
# can this new 2-part object has a posterior at least as large as the 1-part object? Can the increase in the image
# match compensate for the decrease in prior?
# Here is how figure this out. Let d denote how wrong 1-part object is. Let k denote how many times prior decreases when
# we add a new part, and let s denote the variance of the likelihood. Finally, let dd denote the increase in match when
# we add the new part.
# then, p(1-part) = prior * exp(-d^2/2s). p(2-part) = (prior / k) * exp(-(d - dd)^2/2s).
# If you equate these two and a little bit of algebra, you get
#   dd = d - np.sqrt(d^2 - 2slog(k))  for d in [0,1]
# What this tells us that if d^2 < 2*s*log(k), there is no way we can add a new part and still keep the posterior same.
# It also tells us that as we get closer to true image, we demand more and more increase (dd) to justify adding a new
# part.
# For example, if k=2.0 and s=0.01, for d < 0.117, you can never compensate for the decrease in prior. In other words,
# if there is a 1-part object with d < 0.117, there is no 2-part object that is better (higher posterior). Therefore,
# you probably shouldn't set s=0.01
# however, note all this is not so straightforward. in addition to ensuring that true hypothesis has the highest
# posterior, you need to make sure that the chain can travel around the space as freely as possible. but, as you
# decrease s, you are making this harder and harder because even small increases in d matter quite a lot. hence, you
# will not be able to accept that slightly more wrong but crucial hypothesis that leads to higher posterior regions.
"""
k = 2.0
s = 0.0001 # this seems like a decent value
mind = np.sqrt(2*s*np.log(k))
d = np.linspace(mind, 1.0)
dd = d - np.sqrt(d**2 - (2*s*np.log(k)))
"""
