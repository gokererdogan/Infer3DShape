"""
Toy dataset for testing if a NN can learn to map from
images to a voxel-based representation in 2D.

Goker Erdogan
29 Oct. 2015
https://gitub.com/gokererdogan
"""


import numpy as np
import itertools as iter

# width/height of input vector
d = 30
# max. number of objects in input
k = 8
# max. object width/height
object_wh = 20
# voxels per axis
voxels_per_axis = 3
# voxel width/height
voxel_wh = d / voxels_per_axis
# number of samples
N = 10000

x = np.zeros((N, d, d))
y = np.zeros((N, voxels_per_axis * voxels_per_axis))

for n in range(N):
    obj_count = np.random.binomial(k, .5)
    # generate input image
    for i in range(obj_count):
        ix = np.random.randint(0, d)
        iy = np.random.randint(0, d)
        w = np.random.randint(1, object_wh)
        h = np.random.randint(1, object_wh)
        x[n, ix:(ix+w), iy:(iy+h)] = 1.0
    # calculate output voxel representation
    for i in range(voxels_per_axis):
        sx = i * voxel_wh
        ex = (i + 1) * voxel_wh
        for j in range(voxels_per_axis):
            sy = j * voxel_wh
            ey = (j + 1) * voxel_wh
            y[n, (i * voxels_per_axis) + j] = np.mean(x[n, sx:ex, sy:ey])

# center x
x -= 0.5

# split into train and test
tx = x[0:8000]
sx = x[8000:10000]
ty = y[0:8000]
sy = y[8000:10000]

data_folder = "./data"

np.save("{0:s}/toy_voxel_output/train_x.npy".format(data_folder), tx)
np.save("{0:s}/toy_voxel_output/train_y.npy".format(data_folder), ty)
np.save("{0:s}/toy_voxel_output/test_x.npy".format(data_folder), sx)
np.save("{0:s}/toy_voxel_output/test_y.npy".format(data_folder), sy)
