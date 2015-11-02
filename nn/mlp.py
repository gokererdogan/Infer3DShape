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

input_size = 50 * 50
part_count = 1
output_size = part_count * 3

model = Sequential()
model.add(Dense(input_dim=input_size, output_dim=512))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(input_dim=512, output_dim=output_size))

# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
ada = Adagrad()
model.compile(loss='mean_squared_error', optimizer=ada)

# load dataset
ds = dataset.DataSet.load_from_path('1 part, fixed size', folder='data/1_part_fixed_size')

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
