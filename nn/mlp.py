"""
Multilayer perceptron for predicting shape

Goker Erdogan
22 Sep. 2015
https://gitub.com/gokererdogan
"""

import numpy as np
import gmllib.dataset as dataset


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adagrad
from keras.callbacks import EarlyStopping

np.random.seed(1234)

model = Sequential()
model.add(Dense(input_dim=(50*50), output_dim=1000))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(input_dim=1000, output_dim=(5*3)))

# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
ada = Adagrad()
model.compile(loss='mean_squared_error', optimizer=ada)

# load dataset
ds = dataset.DataSet.load_from_path('5 part, fixed size', folder='data/5_part_fixed_size')

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(ds.train.x, ds.train.y, validation_split=0.2, nb_epoch=20, batch_size=128, callbacks=[early_stopping], verbose=True)
score = model.evaluate(ds.test.x, ds.test.y)
print(score)
pred = model.predict(ds.test.x)

"""
import hypothesis as hyp
fwm = hyp.vfm.VisionForwardModel()

s = hyp.Shape.from_narray(pred[0], fwm)
fwm.save_render('pred.png', s)
"""
