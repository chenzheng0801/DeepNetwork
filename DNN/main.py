import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from DNNnetwork import DNNnet


def sigmoid(weight_inp):
    return 1.0/(1.0 + np.exp(-weight_inp))


def sigmoid_derivative(output):
    return output * (1 - output)


def identity_derivative(output):
    return 1


def identity(inp):
    return inp


def soft_max(inp):
    exp_inp = np.exp(inp)
    return exp_inp/(np.sum(exp_inp, axis=-1, keepdims=True))


def soft_max_derivative(label, output):
    return output - label


def MSE_derivative(label, output):
    return output - label


def norm(label, dim):
    label_vec = []
    label_value = label
    for i in range(dim):
        if i == label_value:
            label_vec.append(1)
        else:
            label_vec.append(0)
    return label_vec


def load_mnist(path, kind="train"):
    """Load mnist file from path"""
    images_path = os.path.join(path, "%s-images.idx3-ubyte" % kind)
    labels_path = os.path.join(path, "%s-labels.idx1-ubyte" % kind)

    with open(images_path, "rb") as images_operator:
        magic, num, rows, cols = struct.unpack(">IIII", images_operator.read(16))
        images = np.fromfile(images_operator, dtype=np.uint8).reshape(num, rows*cols)

    with open(labels_path, "rb") as labels_operator:
        magic, n = struct.unpack(">II", labels_operator.read(8))
        labels = np.fromfile(labels_operator, dtype=np.uint8)

    return images, labels, num


def get_result(inv):
    max_value = 0.0
    max_index = 0
    for i in range(len(inv)):
        if inv[i] >= max_value:
            max_value = inv[i]
            max_index = i

    return max_index


@jit(forceobj=True)
def evaluate(network, test_datas, test_labels, n):
    error = 0
    total = n

    for i in range(n):
        label = get_result(network.do_inference((test_datas[i]))[0])
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

    print("expected gradient %.13f, real gradient %.13f" % (grad_target, grad_now))


def build_net(layers):
    net = DNNnet(soft_max_derivative, soft_max)
    layer_count = len(layers)
    for i in range(1, layer_count):
        if i < layer_count-1:
            net.add_layer(layers[i - 1], layers[i], sigmoid, sigmoid_derivative)
        else:
            net.add_layer(layers[i - 1], layers[i], identity, identity_derivative)
    return net


def build_BatchNet(layers):
    net = DNNnet(soft_max_derivative, soft_max)
    layer_count = len(layers)
    for i in range(1, layer_count):
        if i < layer_count-1:
            net.add_layer(layers[i - 1], layers[i], sigmoid, sigmoid_derivative, layer="batchnorm")
        else:
            net.add_layer(layers[i - 1], layers[i], identity, identity_derivative, layer="batchnorm")
    return net


if __name__ == "__main__":
    data_path = os.path.abspath(os.path.dirname(__file__))
    images, train_labels, num = load_mnist(data_path)
    labels_vecs = []
    for i in range(num):
        labels_vecs.append(np.array(norm(train_labels[i], dim=10)))

    # network = build_net([784, 40, 40, 40, 10])
    network = build_BatchNet([784, 40, 40, 40, 10])
    # check_gradient(network, images[0], labels_vec[0], 784)

    train_images = images #[images[0]]
    train_labels_vecs = np.array(labels_vecs) #[labels_vec[0]]
    train_num = num

    rate = 0.1
    epoch = 8
    network.train(train_labels_vecs, train_images, rate, epoch, train_num, batch=32)

    # rate = 0.01
    # epoch = 8
    # network.train(train_labels_vecs, train_images, rate, epoch, train_num, batch=32)
    # rate = 0.0059
    # epoch = 4
    # network.train(train_labels_vecs, train_images, rate, epoch, train_num, batch=32)
    # rate = 0.003
    # epoch = 8
    # network.train(train_labels_vecs, train_images, rate, epoch, train_num, batch=32)
    # rate = 0.1
    # epoch = 6
    # momentum = 0.4
    # network.train(train_labels_vecs, train_images, rate, epoch, train_num,
    #               batch=256, optimizer="sgd_momentum", momentum=momentum)
    #
    # rate = 0.02
    # epoch = 6
    # momentum = 0.9
    # network.train(train_labels_vecs, train_images, rate, epoch, train_num,
    #               batch=1024, optimizer="sgd_momentum", momentum=momentum)
    # rate = 0.01
    # epoch = 8
    # momentum = 0.9
    # network.train(train_labels_vecs, train_images, rate, epoch, train_num,
    #               batch=2048, optimizer="sgd_momentum", momentum=momentum)

    test_datas, test_labels, test_n = load_mnist(data_path, "t10k")
    error_ratio = evaluate(network, test_datas, test_labels, test_n)
    print("error ratio on test data set %f" % error_ratio)

    error_ratio = evaluate(network, train_images, train_labels, train_num)
    print("error ratio on train data set %f" % error_ratio)
