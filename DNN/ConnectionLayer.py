import numpy as np

class FullConnection():
    def __init__(self, input_size, output_size, activator, input_vec=None):
        self.activator = activator
        self.input_size = input_size
        self.output_size = output_size
        self.weight = np.random.uniform(-0.05, 0.05, (output_size, input_size))
        self.b = np.random.uniform(-0.05, 0.05, (output_size,1))
        self.delta_weight = 0.0
        self.delta_b = 0.0
        self.input = input_vec

    def set_zero(self):
        self.delta_weight = 0.0
        self.delta_b = 0.0

    def foward(self, input_vec):
        output = np.add(np.dot(self.weight, input_vec), self.b)
        output = self.activator(output)
        self.input = input_vec
        return output

    def backward(self, delta, drivative):
        ret_delta = drivative(self.input) * np.dot(self.weight.T, delta)
        self.delta_b += delta
        self.delta_weight += np.dot(delta, self.input.T)
        return ret_delta

    def update(self, learning_rate):
        self.weight += learning_rate * self.delta_weight
        self.b += learning_rate * self.delta_b