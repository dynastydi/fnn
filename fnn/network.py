import numpy as np

from fnn import activation, gradient, loss

# -------------------------------------------------------------
# NETWORK CLASS
class Net:
    # INITIALISATION
    def __init__(self, shape, activations, learning_rate: float, mode: str, loss_function: str, batch_size: int = None):

        assert len(shape) == len(activations) + 1  # ensure dimensions match up
        for a in activations:
            assert a.lower() in ['sigmoid', 'relu', 'tanh']  # ensure activations exist

        if mode.lower() == 'minibatch':
            assert batch_size is not None
            self.batch_size = batch_size
        else:
            assert mode.lower() in ['batch', 'stochastic']
            assert batch_size is None  # tells user batch_size won't work in other modes.

        assert loss_function.lower() in ['hinge', 'hinge_square', 'cross_entropy', 'mse']  # Check loss function

        # These variables can be reassigned between training cycles if appropriate for scheduling etc.
        # mode & activations are evaluated during each call of network propagation.
        self.learning_rate = learning_rate
        self.mode = mode
        self.loss_function = loss_function
        self.activations = activations

        # THESE VARIABLES CAN NOT BE REASSIGNED WITHOUT RE-INSTANTIATING WEIGHTS
        self.shape = shape
        self.ln = len(shape) - 1  # number of layers (excl output)

        # instantiate weights between - 0.1 & 0.1 & bias
        self.W = [(np.random.random((shape[i], shape[i + 1])) - 0.5) / 100 for i in range(self.ln)]
        self.b = [(np.full((1, shape[i + 1]), 0.001)) for i in range(self.ln)]

    # -------------------------------------------------------------
    # Train/Test functions
    def train(self, X, y):  # run selected gradient descent mode
        model = getattr(gradient, self.mode)
        model(self, X, y)

    def test(self, X, y):  # evaluate current state against fresh data
        a = X
        for i in range(self.ln):
            z = a.dot(self.W[i]) + self.b[i]
            f = getattr(activation, self.activations[i].lower())
            a = f(z)

        if y.shape[1] == 1:
            r = sum((a > 0.5) == y) / X.shape[0]
        else:
            r = sum(np.argmax(a, axis=1) == np.argmax(y, axis=1)) / X.shape[0]
        return r

    # -------------------------------------------------------------
    def refresh(self):
        # re-instantiate weights between - 0.1 & 0.1
        self.W = [(np.random.random((self.shape[i], self.shape[i + 1])) - 0.5) / 100 for i in range(self.ln)]
        self.b = [(np.full((1, self.shape[i + 1]), 0.001)) for i in range(self.ln)]

    # -------------------------------------------------------------
    # Propagation function
    def _propagate(self, X, y):

        m = X.shape[0]
        a = X
        A = [a]
        Z = []

        # Forward Pass
        for i in range(self.ln):
            z = a.dot(self.W[i]) + self.b[i]
            f = getattr(activation, self.activations[i].lower())
            a = f(z)
            Z.append(z)
            A.append(a)

        # Back propagation for last layer
        loss_func = getattr(loss, self.loss_function.lower())
        dZ = loss_func(y, A[-1])  # calculate errors
        dW = A[-2].T.dot(dZ) / m  # calculate weight adjustment
        db = sum(dZ) / m
        # apply adjustment (clip in case of overflow - ReLU is prone at this stage.)
        self.W[-1] -= self.learning_rate * np.clip(dW, -10,10)
        self.b[-1] -= self.learning_rate * db

        # Back propagate through remaining layers
        for i in reversed(range(self.ln - 1)):
            a_func_prime = getattr(activation, self.activations[i + 1].lower() + "_prime")
            dZ = dZ.dot(self.W[i + 1].T) * a_func_prime(Z[i])
            db = sum(dZ) / m
            dW = A[i].T.dot(dZ) / m

            self.W[i] -= self.learning_rate * np.clip(dW, -10, 10)
            self.b[i] -= self.learning_rate * db

