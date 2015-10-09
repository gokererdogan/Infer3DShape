import numpy as np

# load data
x = np.load('data/x.npy')
y = np.load('data/y.npy')
# shuffle and split data
p = np.random.permutation(800)
x = x[p, :]
y = y[p, :]
x_train = x[0:700, :]
y_train = y[0:700, :]
x_test = x[701:800, :]
y_test = y[701:800, :]
