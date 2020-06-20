import numpy as np
# from numba import jit
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

        l = -1
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
        row_idx = row_i.reshape(-1, 1) + row_j.reshape(1, -1)

        col_i = stride * np.tile(np.arange(new_width), new_height)
        col_j = np.tile(np.arange(f_w), f_h)
        col_idx = col_i.reshape(-1, 1) + col_j.reshape(1, -1)
        C_idx = np.tile(np.arange(C), new_height*new_width*f_h*f_w)
        return row_idx.reshape(-1), col_idx.reshape(-1), C_idx, new_height*new_width, f_h*f_w, C

    # @jit(forceobj=True)
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
        return feature_map

    # @jit(forceobj=True)
    def backward(self, delta, rate):
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
        row_idx, col_idx, C_idx, row, col, C = self.get_im2col_indcies(self.pad_img.shape,
                                                                       self.filter_shape[0],
                                                                       self.filter_shape[1],
                                                                       self.stride)
        back_delta = back_delta.reshape(-1)
        np.add.at(zeros_delta, (slice(None), row_idx, col_idx, C_idx), back_delta)

        res_delta = zeros_delta[:, pad:height-pad, pad:width-pad, :]
        return res_delta

def get_pool2row_indices(x_shape, field_height, field_width, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H - field_height) % stride == 0
    assert (W - field_height) % stride == 0
    out_height = int((H - field_height) / stride + 1)
    out_width = int((W - field_width) / stride + 1)

    # 行坐标
    i0 = stride * np.repeat(np.arange(out_height), out_width)
    i0 = np.tile(i0, C)
    i1 = np.repeat(np.arange(field_height), field_width)

    # 列坐标
    j0 = stride * np.tile(np.arange(out_width), out_height * C)
    j1 = np.tile(np.arange(field_width), field_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), out_height * out_width).reshape(-1, 1)

    return (k, i, j)

if __name__ == "__main__":

    # a = np.zeros((3,3))
    b = np.array([
        [
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ],
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
    ])
    row = np.array([1,1,1,1], dtype="int32")
    i=[0,0,1,1]
    j=[0,0,1,1]
    np.add.at(b, (slice(None), [[0,1]], [[0,1]]), [ [[[1,1,1]],[[1,2,3]]], [[[1,1,1]],[[1,1,1]]] ])
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
        conv.backward(loss_mat, 0.03)
        loss_value = np.sum(loss_mat**2)
        print(loss_value)
