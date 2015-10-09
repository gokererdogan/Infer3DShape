"""
Inferring 3D Shape from 2D Images

This file contains the script for generating a dataset
of objects. This dataset is used for training the
neural network models in this folder.

Created on Sep 15, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import Infer3DShape.hypothesis as hyp
import Infer3DShape.vision_forward_model as vfm
import numpy as np

max_part_count = 8
object_per_part_count = 100

data_folder = "./data"
img_folder = "./data/png"
save_img = False
img_size = 100 * 100 * 5

fwm = vfm.VisionForwardModel()

x = np.zeros(((max_part_count * object_per_part_count), img_size))
y = np.zeros(((max_part_count * object_per_part_count), max_part_count * 6))
for p in range(1, max_part_count + 1):
    print(p)
    for o in range(object_per_part_count):
        i = ((p - 1) * object_per_part_count) + o
        h = hyp.Shape(forward_model=fwm, part_count=p)
        img = fwm.render(h)
        x[i, :] = img.flatten()
        y[i, 0:(p * 6)] = h.to_narray()
        if save_img:
            fwm.save_render("{0:s}/{1:d}.png".format(img_folder, i), h)

np.save("{0:s}/x.npy".format(data_folder), x)
np.save("{0:s}/y.npy".format(data_folder), y)



