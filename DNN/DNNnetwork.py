from ConnectionLayer import FullConnection


def identity_derivative(output):
    return 1


class DNNnet():
    def __init__(self, loss_func_derivative):
        self.loss_func_drt = loss_func_derivative
        self.layers = []

    def add_layer(self, input_size, output_size, activator, derivative):
        conn = FullConnection(input_size, output_size, activator, derivative)
        self.layers.append(conn)

    def predict(self, sample):
        output = sample
        for layer in self.layers:
            output = layer.foward(output)
        return output

    def train_one_example(self, example, label, learning_rate):
        output = self.predict(example)
        # loss_value = np.sqrt(np.sum(np.square(output - label)))
        # print("loss value %f" % loss_value)
        self.calc_gradient(output, label)
        self.update_weight(learning_rate)

    def calc_gradient(self, output, label):
        delta = self.loss_func_drt(label, output) * self.layers[-1].derivative(output)
        delta = -1*delta

        for i in range(len(self.layers)-1, -1, -1):
            layer = self.layers[i]
            if i > 0:
                delta = layer.backward(delta, self.layers[i-1].derivative)
            else:
                delta = layer.backward(delta, identity_derivative)
                print("fasaa")
        return delta

    def train(self, labels, data_set, rate, epoch, num):
        for i in range(epoch):
            for d in range(num):
                self.train_one_example(data_set[d],
                                       labels[d], rate)

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)
