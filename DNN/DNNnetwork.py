from ConnectionLayer import FullConnection
import numpy as np

def identity_derivative(output):
    return 1


class DNNnet():
    def __init__(self, loss_func_derivative, output_function):
        self.loss_func_drt = loss_func_derivative
        self.output_layer = output_function
        self.layers = []

    def add_layer(self, input_size, output_size, activator, derivative):
        conn = FullConnection(input_size, output_size, activator, derivative)
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

    def calc_gradient(self, output_batch, label_batch, rate, batch):
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

    def train(self, labels, data_set, rate, epoch, num, batch):
        for _ in range(epoch):
            step = 0
            tol_loss = 0
            while step + batch < num:
                train_samples = data_set[step: step + batch]
                train_labels = labels[step: step + batch]
                tol_loss += self.train_with_batch(train_samples, train_labels, rate, batch)
                step += batch

            train_samples = data_set[step: num]
            train_labels = labels[step: num]
            tol_loss += self.train_with_batch(train_samples, train_labels, rate, num - step)
            avg_loss = tol_loss / num
            print("epoch %d , avg_loss = %f" % (_, avg_loss))

    def train_with_batch(self, train_samples, train_labels, rate, batch):
        output_batch = self.predict(train_samples)
        self.calc_gradient(output_batch, train_labels, rate, batch)
        loss_batch = np.einsum("Bi,Bi->B", output_batch, np.log(output_batch))
        return -np.sum(loss_batch)
