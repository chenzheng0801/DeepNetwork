import numpy as np
from numba import jit
from numpy.lib.stride_tricks import as_strided


class MaxPooling:
    """
    pooling_shape = (height * width * channel)
    strides = height_stride = width_stride
    padding = height_padding = width_padding)

    applying ReLu at the output
    """
    def __init__(self, pooling, stride=1, padding=0):
        self.pooling = pooling
        self.stride = stride
        self.padding = padding

    @jit(forceobj=True)
    def im2col_as_strided(self, X, flt_h, flt_w, stride):
        batch, height, width, channel = X.shape
        new_h = (height - flt_h) // stride + 1
        new_w = (width - flt_w) // stride + 1
        shape = (batch, new_h, new_w, flt_h, flt_w, channel)
        strides = (X.strides[0],) + (stride*X.strides[1],) + (stride*X.strides[2],) + X.strides[1:]
        A = as_strided(X, shape=shape, strides=strides)
        return A

    @jit(forceobj=True)
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
        return row, col, new_height, new_width

    @jit(forceobj=True)
    def forward(self, img):
        padding = self.padding
        stride = self.stride
        flt_h = self.pooling[0]
        flt_w = self.pooling[1]
        shape = img.shape

        batch = shape[0]
        ch = self.pooling[2]

        self.pad_img = np.lib.pad(img, ((0, 0),
                                        (padding, padding),
                                        (padding, padding),
                                        (0, 0)),
                                  'constant', constant_values=0)
        im2col_row, im2col_col, oh, ow = self.get_im2col_indcies(self.pad_img.shape, flt_h, flt_w, stride)
        im2col_X = self.im2col_as_strided(self.pad_img, flt_h, flt_w, stride)
        im2col_X = im2col_X.reshape((batch, oh, ow, flt_h*flt_w, ch))
        feature_map = np.max(im2col_X, axis=3)
        feature_idx = np.argmax(im2col_X, axis=3)

        ow_idx = np.tile(np.repeat(np.arange(ow), ch), batch*oh).reshape((batch, oh, ow, ch))
        oh_idx = np.tile(np.repeat(np.arange(oh), ch*ow), batch).reshape((batch, oh, ow, ch))

        im2col_row = im2col_row.reshape((oh, ow, flt_h*flt_w))
        im2col_col = im2col_col.reshape((oh, ow, flt_h*flt_w))

        self.im2col_row = im2col_row[oh_idx, ow_idx, feature_idx]
        self.im2col_col = im2col_col[oh_idx, ow_idx, feature_idx]

        return feature_map

    @jit(forceobj=True)
    def backward(self, delta, rate=0.0):
        zeros_delta = np.zeros(self.pad_img.shape)
        pad = self.padding

        height = self.pad_img.shape[1]
        width = self.pad_img.shape[2]
        batch = self.pad_img.shape[0]
        ch = self.pooling[2]
        oh = delta.shape[1]
        ow = delta.shape[2]

        batch_idx = np.repeat(np.arange(batch), ch*oh*ow).reshape((batch, oh, ow, ch))
        ch_idx = np.tile(np.arange(ch), batch*oh*ow).reshape((batch, oh, ow, ch))
        np.add.at(zeros_delta, (batch_idx, self.im2col_row, self.im2col_col, ch_idx), delta)

        res_delta = zeros_delta[:, pad:height-pad, pad:width-pad, :]
        return res_delta


if __name__ == "__main__":
    a = np.array([
        [
            [1, 2, 3],
            [2, 4, 6]
        ],
        [
            [9, 13, 23],
            [4, 5, 7]
        ]
    ])
    idx = np.array(
        [
            [0, 2],
            [1, 0]
        ]
    )
    b = a[:,:,[[0, 2],[1,1]]]
    c = np.argmax(a, axis=2)
    img = np.random.randn(3, 4, 4, 2).astype(np.float64)
    padding = 0
    stride = 2
    shape = (2, 2, 2)
    max_pool = MaxPooling(shape, stride=2, padding=0)
    res = max_pool.forward(img)
    delta = np.random.randn(3, 2, 2, 2).astype(np.float64)
    hehe = max_pool.backward(res, img)