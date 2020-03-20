import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from DNNnetwork import DNNnet


def norm(label):
    label_vec = []
    label_value = label
    for i in range(10):
        if i == label_value:
            label_vec.append(0.9)
        else:
            label_vec.append(0.1)
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


if __name__ == "__main__":
    data_path = os.path.abspath(os.path.dirname(__file__))
    images, labels, n = load_mnist(data_path)
    labels_vec = []
    for i in range(n):
        labels_vec.append(np.array(norm(labels[i])).reshape(10, 1))

    test_datas, test_labels, test_n = load_mnist(data_path, "t10k")
    rate = 0.3
    epoch = 4
    network = DNNnet([784, 40, 40, 10])
    network.train(labels_vec, images, rate, epoch, n)
    error_ratio = evaluate(network, test_datas, test_labels, test_n )
    print("error ratio %f" % error_ratio)