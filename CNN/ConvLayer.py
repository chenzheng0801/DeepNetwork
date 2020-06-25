import numpy as np
from numba import jit
from numpy.lib.stride_tricks import as_strided

class Convolution:
    """
    filter_shape = (height * width * channel * next_channel)
    stride: height_stride and width_stride is same
    padding = height_padding and width_padding is same
    """
    def __init__(self, filter_shape, stride, padding):
        self.filter_shape = filter_shape
        self.stride = stride
        self.padding = padding

        l = -0.001
        r = -l
        self.filters = np.random.uniform(l, r, filter_shape)
        self.bias = np.random.uniform(l, r, (1, filter_shape[3]))

    def im2col_as_strided(self, X, flt_h, flt_w, stride):
        batch, height, width, channel = X.shape
        new_h = (height - flt_h) // stride + 1
        new_w = (width - flt_w) // stride + 1
        shape = (batch, new_h, new_w, flt_h, flt_w, channel)
        strides = (X.strides[0],) + (stride*X.strides[1],) + (stride*X.strides[2],) + X.strides[1:]
        A = as_strided(X, shape=shape, strides=strides)
        return A

    def get_im2col_indcies(self, x_shape, f_h, f_w, stride=1):
        N, H, W, C = x_shape
        new_height = (H - f_h) // stride + 1
        new_width = (W - f_w) // stride + 1
        row_i = stride * np.repeat(np.arange(new_height), new_width)
        row_j = np.repeat(np.arange(f_h), f_w)
        row = np.add.outer(
            row_i.reshape(new_height, new_width),
            row_j.reshape(f_h, f_w)
        )

        col_i = np.tile(stride * np.arange(new_width), new_height)
        col_j = np.tile(np.arange(f_w), f_h)
        col = np.add.outer(
            col_i.reshape(new_height, new_width),
            col_j.reshape(f_h, f_w)
        )
        return row, col

    @jit(forceobj=True)
    def forward(self, img):
        padding = self.padding
        stride = self.stride
        flt_h = self.filter_shape[0]
        flt_w = self.filter_shape[1]

        self.pad_img = np.lib.pad(img, ((0, 0),
                                        (padding, padding),
                                        (padding, padding),
                                        (0, 0)),
                                  'constant', constant_values=0)
        self.im2col_X = self.im2col_as_strided(self.pad_img, flt_h, flt_w, stride)
        feature_map = np.tensordot(self.im2col_X, self.filters, axes=[(3, 4, 5), (0, 1, 2)]) + self.bias
        # ReLu Layer
        self.relu_map = (1 * (feature_map > 0))
        return self.relu_map * feature_map

    @jit(forceobj=True)
    def backward(self, delta, rate):
        delta = self.relu_map * delta

        im2col_X = self.im2col_X
        batch = self.pad_img.shape[0]
        height = self.pad_img.shape[1]
        width = self.pad_img.shape[2]
        pad = self.padding
        zeros_delta = np.zeros(self.pad_img.shape)

        delta_b = np.sum(delta, axis=(0, 1, 2)).reshape(self.bias.shape)
        delta_b /= batch
        self.bias += rate * delta_b

        delta_filters = np.tensordot(im2col_X, delta, axes=[(0, 1, 2), (0, 1, 2)])
        delta_filters /= batch
        self.filters += rate * delta_filters

        back_delta = np.tensordot(delta, self.filters, axes=[3, 3])
        row_idx, col_idx = self.get_im2col_indcies(self.pad_img.shape,
                                                   self.filter_shape[0],
                                                   self.filter_shape[1],
                                                   self.stride)
        np.add.at(zeros_delta, (slice(None), row_idx, col_idx, slice(None)), back_delta)

        res_delta = zeros_delta[:, pad:height-pad, pad:width-pad, :]
        return res_delta

if __name__ == "__main__":
    # a = np.array([
    #     [1,1],
    #     [2,2]
    # ])
    # b = np.array([
    #     [0, 0],
    #     [1, 1]
    # ])
    # c = np.add.outer(a, b)
    # b = np.array([
    #     [
    #         [1, 2, 3],
    #         [4, 5, 6],
    #         [7, 8, 9]
    #     ],
    #     [
    #         [1, 1, 1],
    #         [1, 1, 1],
    #         [1, 1, 1]
    #     ]
    # ])
    # print(b[:,0,:])
    #
    # row = np.array([1, 1, 1, 1], dtype="int32")
    # i = [0, 0, 1, 1]
    # j = [0, 0, 1, 1]
    # np.add.at(b, (slice(None), [[0,1]], [[0,1]]), [ [[[1,1,1]],[[1,2,3]]], [[[1,1,1]],[[1,1,1]]] ])
    #
    # arry = np.array([
    #     [[
    #     [1,2,3],
    #     [4,5,6],
    #     [7,8,9]
    #         ]]
    # ])
    # k, i, j = get_pool2row_indices(arry.shape, 2, 2)

    img = np.random.randn(1, 4, 4, 1).astype(np.float64)
    padding = 1
    stride = 2
    shape = (3,3,1,1)

    conv = Convolution(shape, stride, padding)

    std_y = np.ones((1,2,2,1))

    for i in range(40):
        now_y = conv.forward(img)
        loss_mat = 2*(std_y - now_y)
        conv.backward(loss_mat, 0.08)
        loss_value = np.sum(loss_mat**2)
        print(loss_value)
