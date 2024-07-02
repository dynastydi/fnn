import numpy as np


# -------------------------------------------------------------
# LOSS FUNCTIONS
def cross_entropy(y, pred):

    return pred - y

def mse(y, pred):
    return 2 * (pred - y) / y.shape[1]
