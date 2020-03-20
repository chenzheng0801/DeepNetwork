import numpy as np
from ConnectionLayer import FullConnection;

def sigmoid(weight_inp):
    return 1.0/(1.0 + np.exp(-weight_inp))


def derivative(output):
    return output * (1 - output)


class DNNnet():
    def __init__(self, layers):
        layer_count = len(layers)
        self.layers = []
        for i in range(1,layer_count):
            conn = FullConnection(layers[i-1], layers[i], sigmoid)
            self.layers.append(conn)

    def predict(self, sample):
        output = sample
        for layer in self.layers:
            output = layer.foward(output)
        return output

    def train_one_example(self, example, label, learning_rate):
        output = self.predict(example)
        loss_value = np.sqrt(np.sum(np.square(output - label)))
        print("loss value %f" % loss_value)
        self.calc_gradient(output, label)
        self.update_weight(learning_rate)

    def calc_gradient(self, output, label):
        delta = (label-output) * derivative(output)
        for layer in self.layers[::-1]:
            delta = layer.backward(delta, derivative)
        return delta

    def train(self, labels, data_set, rate, epoch, num):
        for i in range(epoch):
            for d in range(num):
                self.train_one_example(data_set[d],
                                       labels[d], rate)

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)