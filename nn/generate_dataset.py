"""
Inferring 3D Shape from 2D Images

This file contains the script for generating a dataset
of objects. This dataset is used for training the
neural network models in this folder.

Created on Sep 15, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import numpy as np

import Infer3DShape.hypothesis as hyp
import Infer3DShape.vision_forward_model as vfm
import gmllib.helpers as hlp


def generate_object_fixed_part_count_and_size(part_count=5):
    s = hyp.Shape(forward_model=None, part_count=part_count)
    # sort parts according to x position
    x = np.zeros(part_count)
    for i, part in enumerate(s.parts):
        x[i] = part.position[0]
        part.size = np.ones(3) * 0.6
    sort_ix = np.argsort(x)
    # get rid of size information (it is constant)
    y = np.zeros(part_count * 3)
    for i, ix in enumerate(sort_ix):
        y[(i * 3):((i + 1) * 3)] = s.parts[ix].position
    return s, y

if __name__ == "__main__":
    part_count = 2
    object_count = 20000
    max_pixel_value = 175.0

    data_folder = "./data"
    img_folder = "./data/png"
    save_img = False
    img_size = (50, 50)

    # use a small image from a single viewpoint
    fwm = vfm.VisionForwardModel(render_size=img_size, camera_pos=[(3.0, -3.0, 3.0)])

    x = np.zeros((object_count, img_size[0] * img_size[1]))
    y = np.zeros((object_count, part_count * 3))
    for i in range(object_count):
        hlp.progress_bar(current=i+1, max=object_count, label='Generating object...')
        h, h_y = generate_object_fixed_part_count_and_size(part_count=part_count)
        img = fwm.render(h)
        # normalize to -1, 1
        img = ((img / max_pixel_value) - 0.5) * 2
        x[i, :] = img.flatten()
        y[i, :] = h_y
        if save_img:
            fwm.save_render("{0:s}/{1:d}.png".format(img_folder, i), h)

    tx = x[0:16000]
    ty = y[0:16000]
    sx = x[16000:]
    sy = y[16000:]
    np.save("{0:s}/2_part_fixed_size/train_x.npy".format(data_folder), tx)
    np.save("{0:s}/2_part_fixed_size/train_y.npy".format(data_folder), ty)
    np.save("{0:s}/2_part_fixed_size/test_x.npy".format(data_folder), sx)
    np.save("{0:s}/2_part_fixed_size/test_y.npy".format(data_folder), sy)


