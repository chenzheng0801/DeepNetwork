import numpy as np


class FullConnection(object):
    def __init__(self, input_size, output_size, activator, derivative, input_vec=None):
        self.activator = activator
        self.input_size = input_size
        self.output_size = output_size
        self.derivative = derivative
        l = -np.sqrt(6.0/(output_size+input_size))
        r = np.sqrt(6.0/(output_size+input_size))
        # l = -0.8
        # r = 0.8
        self.weight = np.random.uniform(l, r, (input_size, output_size))
        self.b = np.random.uniform(l, r, (1, output_size))
        self.input = input_vec

    def foward(self, input_vec):
        output = np.add(np.dot(input_vec, self.weight), self.b)
        output = self.activator(output)
        self.input = input_vec
        return output

    def backward(self, delta, derivative, learning_rate, batch):
        ret_delta = derivative(self.input) * np.dot(delta, self.weight.T)

        batch_weight = np.einsum("Bi,Bj->Bij", self.input, delta)
        self.b += learning_rate * (np.sum(delta, axis=0, keepdims=True) / batch)
        self.weight += learning_rate * (np.sum(batch_weight, axis=0) / batch)
        return ret_delta

