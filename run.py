from fnn import network
import numpy as np
import sys
import time

# quick demo with a toy dataset. 

def main():
    file = np.genfromtxt('wdbc.csv', delimiter=',', dtype=str)

    y = np.array(([[x] for x in file[:, 1] == 'M']), ndmin=2).astype(float)
    X = file[:, 2:].astype(float)

    t_y = y[-64:, :]
    t_X = X[-64:, :]

    y = y[:-64, :]
    X = X[:-64, :]

    net = network.Net(
        shape=(30, 8, 1),
        activations=('relu', 'tanh'),
        learning_rate=0.0001,
        mode='minibatch',  # gradient descent mode
        loss_function="cross_entropy",
        batch_size=50,
    )

    epoch = 1000

    t = time.time()
    for i in range(epoch):
        net.train(X, y)
        r = net.test(t_X, t_y) * 100
        sys.stdout.write(f'\riter {i + 1} / {epoch}: {round(r[0], 3)} % accuracy.')
    print("\nTraining time: " + str(time.time() - t) + " seconds")


main()
