import numpy as np


# -------------------------------------------------------------
# GRADIENT DESCENT MODES

def batch(net, X, y):
    net._propagate(X, y)  # train whole dataset in one, order inconsequential


def minibatch(net, X, y):  # train dataset in segments determined by batch_size
    X, y = shuffle(X, y)  # random element

    index = 0
    for b in range(X.shape[0] // net.batch_size):
        b_X = X[index:index + net.batch_size, :]
        b_y = y[index:index + net.batch_size, :]
        net._propagate(b_X, b_y)
        index += net.batch_size
    if X.shape[0] % net.batch_size > 0:  # last batch fragment if necessary
        b_X = X[index:, :]
        b_y = y[index:, :]
        net._propagate(b_X, b_y)


def stochastic(net, X, y):  # train each sample in turn
    X, y = shuffle(X, y)  # random element

    for i in range(X.shape[0]):
        net._propagate(X[i:i + 1, :], y[i:i + 1, :])  # i:i+1 keeps dimensionality without flattening into 1D vector


# -------------------------------------------------------------
# DATA SHUFFLING
def shuffle(X, y):
    Xy = np.concatenate((X, y), axis=1)  # combine X & y
    np.random.shuffle(Xy)  # shuffle both identically
    return Xy[:, :-y.shape[1]], Xy[:, -y.shape[1]:]  # re-separate

