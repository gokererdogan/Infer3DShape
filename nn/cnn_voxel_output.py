"""
Convolutional NN for predicting shape
The output is the voxel based representation, not
the positions/sizes of objects.

Goker Erdogan
26 Oct. 2015
https://gitub.com/gokererdogan
"""

import numpy as np

import gmllib.dataset as dataset

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adagrad, Adadelta
from keras.callbacks import EarlyStopping

np.random.seed(1234)

VOXEL_PER_AXIS = 3
# load dataset
ds = dataset.DataSet.load_from_path('voxel output', folder='data/random_part_random_size_voxel_output')
# reshape data for 2D Convolution layer
train_x = np.reshape(ds.train.x, (16000, 1, 50, 50))
test_x = np.reshape(ds.test.x, (4000, 1, 50, 50))

model = Sequential()
model.add(Convolution2D(input_shape=(1, 50, 50), nb_filter=32, nb_col=3, nb_row=3, border_mode='full'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(output_dim=512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim=(VOXEL_PER_AXIS ** 3)))
# model.add(Activation('sigmoid'))

# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# ada = Adagrad()
opt = Adadelta()
model.compile(loss='mean_squared_error', optimizer=opt)

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(train_x, ds.train.y, validation_split=0.1, nb_epoch=20, batch_size=128, callbacks=[early_stopping], verbose=True)
score = model.evaluate(test_x, ds.test.y)
print(score)
pred = model.predict(test_x)


