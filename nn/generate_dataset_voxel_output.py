"""
Inferring 3D Shape from 2D Images

This file contains the script for generating a dataset
of 2D to 3D inference problem. Here the output is a
binary voxel-based representation of the 3D space where a
1 represents an object in that voxel and 0 otherwise.
This dataset is created with the hope that this output
coding will be easier for neural networks to learn.

Created on Oct 26, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import numpy as np

import Infer3DShape.hypothesis as hyp
import Infer3DShape.vision_forward_model as vfm
import gmllib.helpers as hlp

# 3 voxels per axis
VOXEL_PER_AXIS = 3
voxel_boundaries = np.linspace(-1.0, 1.0, VOXEL_PER_AXIS + 1)
voxel_pos = np.array([-2.0/3.0, 0.0, 2.0/3.0])

def get_voxel_id(axis, value):
    ix = np.argwhere(value > voxel_boundaries)
    # voxel id is the last index where value > voxel_boundaries
    # because argwhere returns array of arrays we need to index
    # two times
    return ix[-1][0]


def generate_dataset_sample(part_count=5):
    s = hyp.Shape(forward_model=None, part_count=part_count)
    y = np.zeros((VOXEL_PER_AXIS, VOXEL_PER_AXIS, VOXEL_PER_AXIS), dtype=bool)
    for part in s.parts:
        ix = get_voxel_id('x', part.position[0])
        iy = get_voxel_id('y', part.position[1])
        iz = get_voxel_id('z', part.position[2])
        y[ix, iy, iz] = True
    return s, y.flatten()

def voxel_output_to_shape(y):
    """
    Converts the voxel representation to Shape instance.
    :param y:
    :return:
    """
    y = np.reshape(y, (3, 3, 3))

    ix = y.nonzero()
    ix = np.array(ix)
    ix = ix.transpose()
    n = ix.shape[0]
    parts = []
    for i in range(n):
        x = ix[i, 0]
        y = ix[i, 1]
        z = ix[i, 2]
        part = hyp.CuboidPrimitive(position=np.array([voxel_pos[x], voxel_pos[y], voxel_pos[z]]),
                                   size=np.array([2.0/3.0, 2.0/3.0, 2.0/3.0]))
        parts.append(part)

    s = hyp.Shape(forward_model=None, parts=parts)
    return s

if __name__ == "__main__":
    object_count = 20000
    max_pixel_value = 175.0

    data_folder = "./data"
    img_folder = "./data/png"
    save_img = False
    img_size = (50, 50)

    # use a small image from a single viewpoint
    fwm = vfm.VisionForwardModel(render_size=img_size, camera_pos=[(3.0, -3.0, 3.0)])

    x = np.zeros((object_count, img_size[0] * img_size[1]))
    y = np.zeros((object_count, VOXEL_PER_AXIS ** 3))
    for i in range(object_count):
        hlp.progress_bar(current=i+1, max=object_count, label='Generating object...')
        h, h_y = generate_dataset_sample(part_count=np.random.randint(1, 10))
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
    np.save("{0:s}/random_part_random_size_voxel_output/train_x.npy".format(data_folder), tx)
    np.save("{0:s}/random_part_random_size_voxel_output/train_y.npy".format(data_folder), ty)
    np.save("{0:s}/random_part_random_size_voxel_output/test_x.npy".format(data_folder), sx)
    np.save("{0:s}/random_part_random_size_voxel_output/test_y.npy".format(data_folder), sy)


