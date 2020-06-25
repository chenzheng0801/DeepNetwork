import os
import struct
import numpy as np

from CNNetwork import CNNet


def relu(x):
    return (1 * (x > 0)) * x


def relu_derivative(x):
    return (1 * (x > 0))


def softmax(inp):
    exp_inp = np.exp(inp)
    return exp_inp/(np.sum(exp_inp, axis=-1, keepdims=True))


def softmax_derivative(label, output):
    return output - label


def identity_derivative(output):
    return 1


def identity(inp):
    return inp


def load_mnist(path, kind="train"):
    """Load mnist file from path"""
    images_path = os.path.join(path, "%s-images.idx3-ubyte" % kind)
    labels_path = os.path.join(path, "%s-labels.idx1-ubyte" % kind)

    with open(images_path, "rb") as images_operator:
        magic, num, rows, cols = struct.unpack(">IIII", images_operator.read(16))
        images = np.fromfile(images_operator, dtype=np.uint8).reshape(num, rows, cols, 1)

    with open(labels_path, "rb") as labels_operator:
        magic, n = struct.unpack(">II", labels_operator.read(8))
        labels = np.fromfile(labels_operator, dtype=np.uint8)

    return images, labels, num


def norm(label, dim):
    label_vec = []
    label_value = label
    for i in range(dim):
        if i == label_value:
            label_vec.append(1)
        else:
            label_vec.append(0)
    return label_vec


def build_CNNet():
    net = CNNet(softmax_derivative, softmax)

    # output image shape=[28*28*6]
    net.add_layer(shape=(3, 3, 1, 6), strides=1,
                  padding=1, layer="Convolution")

    # output image shape=[14*14*6]
    net.add_layer(shape=(2, 2, 6), strides=2,
                  padding=0, layer="MaxPooling")

    # output image shape=[10*10*16]
    net.add_layer(shape=(5, 5, 6, 16), strides=1,
                  padding=0, layer="Convolution")

    # output image shape=[5*5*16]
    net.add_layer(shape=(2, 2, 16), strides=2,
                  padding=0, layer="MaxPooling")

    # output dimension = 400
    net.add_layer(shape=400, layer="ShapeTransform")

    net.add_layer(shape=(400, 30), activator=relu,
                  derivative=relu_derivative, layer="FullConnection")

    net.add_layer(shape=(30, 30), activator=relu,
                  derivative=relu_derivative, layer="FullConnection")
    net.add_layer(shape=(30, 10), activator=identity,
                  derivative=identity_derivative, layer="FullConnection")
    return net


if __name__ == "__main__":
    bath_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(bath_path, "DNN")

    train_images, train_labels, num = load_mnist(data_path)
    labels_vecs = []
    for i in range(num):
        labels_vecs.append(np.array(norm(train_labels[i], dim=10)))

    train_labels = np.array(labels_vecs) #[labels_vec[0]]
    train_num = num

    network = build_CNNet()
    rate = 0.0001
    epoch = 20
    network.train(train_labels, train_images, rate, epoch, train_num, batch=50)
