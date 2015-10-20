"""
Convolutional NN for predicting shape

Goker Erdogan
19 Oct. 2015
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

# load dataset
ds = dataset.DataSet.load_from_path('5 part, fixed size', folder='data/5_part_fixed_size')
# reshape data for 2D Convolution layer
train_x = np.reshape(ds.train.x, (16000, 1, 50, 50))
test_x = np.reshape(ds.test.x, (4000, 1, 50, 50))

model = Sequential()
model.add(Convolution2D(input_shape=(1, 50, 50), nb_filter=32, nb_col=5, nb_row=5, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filter=32, nb_col=5, nb_row=5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(output_dim=256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim=(5*3)))

# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# ada = Adagrad()
opt = Adadelta()
model.compile(loss='mean_squared_error', optimizer=opt)

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(train_x, ds.train.y, validation_split=0.2, nb_epoch=20, batch_size=128, callbacks=[early_stopping], verbose=True)
score = model.evaluate(test_x, ds.test.y)
print(score)
pred = model.predict(test_x)


