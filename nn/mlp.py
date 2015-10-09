"""
Multilayer perceptron for predicting shape

Goker Erdogan
22 Sep. 2015
https://gitub.com/gokererdogan
"""

import numpy as np
np.random.seed(1234)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adagrad
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(Dense(100*100*5, 1000, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(1000, 1000, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(1000, 8*6, init='uniform'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# ada = Adagrad()
model.compile(loss='mean_squared_error', optimizer=sgd)

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x_train, y_train, validation_split=0.2, nb_epoch=20, batch_size=20, callbacks=[early_stopping], verbose=True)
score = model.evaluate(x_test, y_test, batch_size=20)
print(score)
pred = model.predict(x_test)

"""
import hypothesis as hyp
fwm = hyp.vfm.VisionForwardModel()

s = hyp.Shape.from_narray(pred[0], fwm)
fwm.save_render('pred.png', s)
"""
