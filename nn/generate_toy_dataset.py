"""
Toy dataset for testing if a NN can extract positions
of objects in its input.

Goker Erdogan
20 Oct. 2015
https://gitub.com/gokererdogan
"""


import numpy as np
import itertools as iter

# length of input vector
d = 10
# number of objects in input
k = 2
# positions of objects in input
pos = list(iter.combinations(range(d), k))
# input matrix (we add one unit for bias)
x = np.zeros((len(pos), d+1))
for r, ix in enumerate(pos):
    x[r, ix] = 1.0
    x[r, d] = 1.0 # bias input is 1.0
# output matrix is just the positions
y = np.array(pos)
# normalize input and output
x = (x * 2.0) - 1.0
y = (y - ((d - 1) / 2.0)) / ((d - 1) / 2.0)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adagrad

np.random.seed(1234)

model = Sequential()
model.add(Dense(input_dim=(d+1), output_dim=256))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(input_dim=256, output_dim=128))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(input_dim=128, output_dim=k))

# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
ada = Adagrad()
model.compile(loss='mean_squared_error', optimizer=ada)

model.fit(x, y, nb_epoch=40, verbose=True)
score = model.evaluate(x, y)
print(score)
pred = model.predict(x)



