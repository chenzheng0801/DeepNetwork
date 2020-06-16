import numpy as np
from numba import jit

from ConvLayer import Convolution
from PoolingLayer import MaxPooling
from ConnectionLayer import FullConnection
from ConvertLayer import ShapeTransform


def random_shuffle_in_union(a, b):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


class CNNet:
    def __init__(self, loss_func_derivative, output_function):
        self.loss_func_drt = loss_func_derivative
        self.output_layer = output_function
        self.layers = []
        self.toltal_iteration = 0

    def add_layer(self, shape, activator=None,
                  derivative=None, strides=(2, 2),
                  padding=(0, 0), layer="FullConnection"):
        conn = None
        if layer == "FullConnection":
            conn = FullConnection(shape, activator, derivative)
        elif layer == "MaxPooling":
            conn = MaxPooling(shape, strides, padding)
        elif layer == "Convolution":
            conn = Convolution(shape, strides, padding)
        elif layer == "ShapeTransform":
            conn = ShapeTransform(shape)
        if conn:
            self.layers.append(conn)

    # @jit(forceobj=True)
    def predict(self, samples):
        output_batch = samples
        # i = 0
        for layer in self.layers:
            output_batch = layer.forward(output_batch)
            # print("Layer:%d"%i+"  mean:"+str(np.median(output)) +"  std:" + str(np.var(output)) )
            # i += 1
        output_batch = self.output_layer(output_batch)
        return output_batch

    @jit(forceobj=True)
    def back_gradient(self, output_batch, label_batch, rate):
        delta = self.loss_func_drt(label_batch, output_batch) \
                * self.layers[-1].derivative(output_batch)
        delta = -1 * delta
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            delta = layer.backward(delta, rate)
        return delta

    @jit(forceobj=True)
    def train(self, labels, data_set, rate, epoch, num, batch):
        for _ in range(epoch):
            step = 0
            tol_loss = 0
            while step + batch < num:
                train_samples = data_set[step: step + batch]
                train_labels = labels[step: step + batch]
                tol_loss += self.update_weight(train_samples, train_labels, rate)
                step += batch

            train_samples = data_set[step: num]
            train_labels = labels[step: num]
            tol_loss += self.update_weight(train_samples, train_labels, rate)
            avg_loss = tol_loss / num
            random_shuffle_in_union(train_samples, train_labels)
            self.toltal_iteration += 1
            print("epoch %d , avg_loss = %f" % (self.toltal_iteration, avg_loss))

    def update_weight(self, train_samples, train_labels, rate):
        output_batch = self.predict(train_samples)
        self.back_gradient(output_batch, train_labels, rate)
        loss_batch = np.einsum("Bi,Bi->B", train_labels, np.log(output_batch))
        return -np.sum(loss_batch)

