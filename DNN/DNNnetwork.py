from ConnectionLayer import FullConnection, BatchNorm
import numpy as np
from numba import jit

def identity_derivative(output):
    return 1


def random_shuffle_in_union(a, b):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


class DNNnet():
    def __init__(self, loss_func_derivative, output_function):
        self.loss_func_drt = loss_func_derivative
        self.output_layer = output_function
        self.layers = []
        self.toltal_iteration = 0

    def add_layer(self, input_size, output_size, activator, derivative, layer="fullconnection"):
        if layer == "fullconnection":
            conn = FullConnection(input_size, output_size, activator, derivative)
        elif layer == "batchnorm":
            conn = BatchNorm(input_size, output_size, activator, derivative)
        self.layers.append(conn)

    def predict(self, samples):
        output_batch = samples
        # i = 0
        for layer in self.layers:
            output_batch = layer.foward(output_batch)
            # print("Layer:%d"%i+"  mean:"+str(np.median(output)) +"  std:" + str(np.var(output)) )
            # i += 1
        output_batch = self.output_layer(output_batch)
        return output_batch

    def do_inference(self, sample):
        output = sample
        for layer in self.layers:
            output = layer.do_inference(output)

        output = self.output_layer(output)
        return output

    def batch_gradient(self, output_batch, label_batch, rate, batch, momentum=0):
        delta = self.loss_func_drt(label_batch, output_batch)\
                * self.layers[-1].derivative(output_batch)
        delta = -1*delta

        for i in range(len(self.layers)-1, -1, -1):
            layer = self.layers[i]
            if i > 0:
                delta = layer.backward(delta, self.layers[i-1].derivative, rate, batch)
            else:
                delta = layer.backward(delta, identity_derivative, rate, batch)
        return delta

    def momentum_gradient(self, output_batch, label_batch, rate, batch, momentum):
        delta = self.loss_func_drt(label_batch, output_batch)\
                * self.layers[-1].derivative(output_batch)
        delta = -1*delta

        for i in range(len(self.layers)-1, -1, -1):
            layer = self.layers[i]
            if i > 0:
                delta = layer.momentum_backward(delta, self.layers[i-1].derivative,
                                                rate, batch, momentum)
            else:
                delta = layer.momentum_backward(delta, identity_derivative,
                                                rate, batch, momentum)
        return delta

    @jit(forceobj=True)
    def train(self, labels, data_set, rate, epoch, num,
              batch, optimizer="sgd", momentum=0):
        if optimizer == "sgd":
            cal_gradient = self.batch_gradient
        elif optimizer == "sgd_momentum":
            cal_gradient = self.momentum_gradient
        else:
            cal_gradient = self.batch_gradient

        for _ in range(epoch):
            step = 0
            tol_loss = 0
            while step + batch < num:
                train_samples = data_set[step: step + batch]
                train_labels = labels[step: step + batch]
                tol_loss += self.train_with_batch(train_samples, train_labels, rate,
                                                  batch, momentum, cal_gradient)
                step += batch

            train_samples = data_set[step: num]
            train_labels = labels[step: num]
            tol_loss += self.train_with_batch(train_samples, train_labels, rate,
                                              num - step, momentum, cal_gradient)
            avg_loss = tol_loss / num
            random_shuffle_in_union(train_samples, train_labels)
            self.toltal_iteration += 1
            print("epoch %d , avg_loss = %f" % (self.toltal_iteration, avg_loss))

    def train_with_batch(self, train_samples, train_labels, rate, batch, momentum, cal_gradient):
        output_batch = self.predict(train_samples)
        cal_gradient(output_batch, train_labels, rate, batch, momentum)
        loss_batch = np.einsum("Bi,Bi->B", train_labels, np.log(output_batch))
        return -np.sum(loss_batch)
