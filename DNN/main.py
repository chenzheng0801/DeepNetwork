import os
import struct
import numpy as np
import matplotlib.pyplot as plt

from DNNnetwork import DNNnet


def sigmoid(weight_inp):
    return 1.0/(1.0 + np.exp(-weight_inp))


def sigmoid_derivative(output):
    return output * (1 - output)


def linear(weight_inp):
    return weight_inp


def linear_derivative(output):
    return 1


def MSE_derivative(label, output):
    return output-label


def norm(label):
    label_vec = []
    label_value = label
    for i in range(10):
        if i == label_value:
            label_vec.append(1)
        else:
            label_vec.append(0.0)
    return label_vec


def load_mnist(path, kind="train"):
    """Load mnist file from path"""
    images_path = os.path.join(path, "%s-images.idx3-ubyte" % kind)
    labels_path = os.path.join(path, "%s-labels.idx1-ubyte" % kind)

    with open(images_path, "rb") as images_operator:
        magic, num, rows, cols = struct.unpack(">IIII", images_operator.read(16))
        images = np.fromfile(images_operator, dtype = np.uint8).reshape(num, rows*cols, 1)

    with open(labels_path, "rb") as labels_operator:
        magic, n = struct.unpack(">II", labels_operator.read(8))
        labels = np.fromfile(labels_operator, dtype = np.uint8)

    return images, labels, num


def get_result(inv):
    max_value = 0.0
    max_index = 0
    for i in range(len(inv)):
        if inv[i] >= max_value:
            max_value = inv[i]
            max_index = i

    return max_index


def evaluate(network, test_datas, test_labels, n):
    error = 0
    total = n

    for i in range(n):
        label = get_result(network.predict((test_datas[i])))
        if label != test_labels[i]:
            error += 1
    return float(error) / float(total)


def check_gradient(network, inp_vec, out_vec, dim):
    predict_value = network.predict(inp_vec)
    gradient = network.calc_gradient(predict_value, out_vec)
    grad_now = -gradient[0]

    epsilon = 0.0000000001
    epsilon_vec = np.zeros((dim, 1), dtype="float64")
    epsilon_vec[0] = epsilon

    inp_vec = inp_vec + epsilon_vec
    delta1 = network.predict(inp_vec)
    loss_value1 = np.sum(np.square(delta1 - out_vec))
    inp_vec = inp_vec - 2*epsilon_vec
    delta2 = network.predict(inp_vec)
    loss_value2 = np.sum(np.square(delta2 - out_vec))

    grad_target = (loss_value1 - loss_value2)/(4*epsilon)

    print("expected gradient %.13f, real gradient %.13f" %(grad_target, grad_now))


def build_net(layers):
    net = DNNnet(MSE_derivative)
    layer_count = len(layers)
    for i in range(1, layer_count):
        # net.add_layer(layers[i - 1], layers[i], sigmoid, sigmoid_derivative)
        if i < layer_count-1:
            net.add_layer(layers[i - 1], layers[i], sigmoid, sigmoid_derivative)
        else:
            net.add_layer(layers[i - 1], layers[i], linear, linear_derivative)
    return net


if __name__ == "__main__":
    data_path = os.path.abspath(os.path.dirname(__file__))
    images, labels, num = load_mnist(data_path)
    labels_vec = []
    for i in range(num):
        labels_vec.append(np.array(norm(labels[i])).reshape(10, 1))

    rate = 0.0013
    epoch = 14
    network = build_net([784, 30, 10])

    # check_gradient(network, images[0], labels_vec[0], 784)

    train_images = images #[images[0]]
    train_labels = labels_vec #[labels_vec[0]]
    train_num = num #1
    network.train(train_labels, train_images, rate, epoch, train_num)

    test_datas, test_labels, test_n = load_mnist(data_path, "t10k")
    error_ratio = evaluate(network, test_datas, test_labels, test_n )

    print("error ratio %f" % error_ratio)
