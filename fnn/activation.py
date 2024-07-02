import numpy as np


# -------------------------------------------------------------
# ACTIVATION FUNCTIONS

def linear(data):
    return data

def sigmoid(data):
    data = np.clip(data, -500, 500)  # exponential is prone to overflow - clip values.
    return 1 / (1 + np.exp(- data))


def relu(data):
    return data * (data > 0)


def tanh(data):
    return np.tanh(data)


# -------------------------------------------------------------
# ACTIVATION FUNCTION DERIVATIVES
def sigmoid_prime(data):
    return data * (1 - data)


def relu_prime(data):
    return 1 * (data > 0)


def tanh_prime(data):
    return 1 - (np.tanh(data)) ** 2




