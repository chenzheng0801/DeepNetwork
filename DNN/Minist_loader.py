import os
import struct
import numpy as np
import matplotlib.pyplot as plt

def load_mnist(path, kind="train"):
    """Load mnist file from path"""
    images_path = os.path.join(path, "%s-images.idx3-ubyte" % kind)
    labels_path = os.path.join(path, "%s-labels.idx1-ubyte" % kind)

    with open(images_path, "rb") as images_operator:
        magic, num, rows, cols = struct.unpack(">IIII", images_operator.read(16))
        images = np.fromfile(images_operator, dtype = np.uint8).reshape(num, rows*cols)

    with open(labels_path, "rb") as labels_operator:
        magic, n = struct.unpack(">II", labels_operator.read(8))
        labels = np.fromfile(labels_operator, dtype = np.uint8)

    return images, labels

if __name__ == "__main__":
    images, labels = load_mnist(os.path.abspath(os.path.dirname(__file__)), "t10k")

    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        sharey=True, )

    ax[0].imshow(images[0].reshape(28, 28), cmap='Greys', interpolation='nearest')
    ax[1].imshow(images[1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.tight_layout()
    plt.show()
    print(labels[0], labels[1])