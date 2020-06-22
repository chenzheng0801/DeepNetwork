import numpy as np
from numba import jit


class FullConnection:
    """
        Full connection layer
        shape = (input_size * output_size)
    """
    def __init__(self, shape, activator, derivative):
        input_size = shape[0]
        output_size = shape[1]
        l = -np.sqrt(6.0/(output_size+input_size))
        r = np.sqrt(6.0/(output_size+input_size))

        self.weight = np.random.uniform(l, r, (input_size, output_size))
        self.b = np.random.uniform(l, r, (1, output_size))
        self.activator = activator
        self.derivative = derivative

    # @jit(forceobj=True)
    def forward(self, input_vec):
        output = np.add(np.dot(input_vec, self.weight), self.b)
        self.input_vec = input_vec
        return self.activator(output)

    # @jit(forceobj=True)
    def backward(self, delta, rate):
        res_delta = self.derivative(self.input_vec)
        res_delta = res_delta * np.dot(delta, self.weight.T)

        batch = delta.shape[0]
        batch_weight = np.einsum("Bi,Bj->Bij", self.input_vec, delta)
        self.b += rate * (np.sum(delta, axis=0, keepdims=True) / batch)
        self.weight += rate * (np.sum(batch_weight, axis=0) / batch)
        return res_delta
