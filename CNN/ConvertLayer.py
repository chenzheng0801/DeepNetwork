import numpy as np


class ShapeTransform:
    def __init__(self, dim):
        self.vec_dim = dim

    def forward(self, img):
        self.img_shape = img.shape
        batch = img.shape[0]
        return img.reshape(batch, self.vec_dim)

    def backward(self, delta, rate=0.0):
        return delta.reshape(self.img_shape)
