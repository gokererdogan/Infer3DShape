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

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adagrad
from keras.callbacks import EarlyStopping

np.random.seed(1234)

# reshape data for 2D Convolution layer
tx = np.reshape(tx, (8000, 1, d, d))
sx = np.reshape(sx, (2000, 1, d, d))

model = Sequential()
model.add(Convolution2D(input_shape=(1, d, d), nb_filter=32, nb_col=3, nb_row=3, border_mode='full'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(output_dim=512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim=(voxels_per_axis * voxels_per_axis)))
model.add(Activation('sigmoid'))

"""
model = Sequential()
model.add(Flatten(input_shape=(d, d)))
model.add(Dense(input_dim=(d * d), output_dim=128))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
# model.add(Dense(input_dim=256, output_dim=128))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
model.add(Dense(input_dim=128, output_dim=voxels_per_axis * voxels_per_axis))
model.add(Activation('sigmoid'))
"""

# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
ada = Adagrad()
model.compile(loss='binary_crossentropy', optimizer=ada)

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(tx, ty, validation_split=0.1, nb_epoch=50, batch_size=128, callbacks=[early_stopping], verbose=True)
score = model.evaluate(sx, sy)
print(score)
pred = model.predict(sx)

