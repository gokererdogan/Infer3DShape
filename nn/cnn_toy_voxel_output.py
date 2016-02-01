"""
Convolutional NN for predicting shape for voxel output toy dataset.
The output is the voxel based representation, not
the positions/sizes of objects.

Goker Erdogan
16 Nov. 2015
https://gitub.com/gokererdogan
"""

import numpy as np

import gmllib.dataset as dataset

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adagrad
from keras.callbacks import EarlyStopping

np.random.seed(1234)

# width/height of input vector
d = 30
# voxels per axis
voxels_per_axis = 3

# load dataset
ds = dataset.DataSet.load_from_path('toy voxel output', folder='data/toy_voxel_output')
tx = ds.train.x
sx = ds.test.x
ty = ds.train.y
sy = ds.test.y

# reshape data for 2D Convolution layer
tx = np.reshape(tx, (8000, 1, d, d))
sx = np.reshape(sx, (2000, 1, d, d))

"""
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

# MLP implementation
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

# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
ada = Adagrad()
model.compile(loss='binary_crossentropy', optimizer=ada)

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(tx, ty, validation_split=0.1, nb_epoch=50, batch_size=128, callbacks=[early_stopping], verbose=True)
score = model.evaluate(sx, sy)
print(score)
pred = model.predict(sx)

