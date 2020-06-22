import numpy as np
# from numba import jit
from numpy.lib.stride_tricks import as_strided


class MaxPooling:
    """
    pooling_shape = (height * width * channel)
    strides = (height_stride * width_stride)
    padding = (height_padding * width_padding)

    applying ReLu at the output
    """
    def __init__(self, pooling, stride=1, padding=0):
        self.pooling = pooling
        self.stride = stride
        self.padding = padding

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
        return row, col, new_height, new_width

    # @jit(forceobj=True)
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

        ow_idx = (np.tile(np.arange(ow, flt_h*flt_w), oh)).reshape((oh, ow, flt_h*flt_w))
        oh_idx = (np.arange(oh, flt_h*flt_w*ow)).reshape((oh, ow, flt_h*flt_w))

        im2col_row = im2col_row.reshape((oh, ow, flt_h*flt_w))
        im2col_col = im2col_col.reshape((oh, ow, flt_h*flt_w))

        self.im2col_row = im2col_row[oh_idx, ow_idx, feature_idx]
        self.im2col_col = im2col_col[oh_idx, ow_idx, feature_idx]

        # ReLu Layer
        return (1 * (feature_map > 0)) * feature_map

    # @jit(forceobj=True)
    def backward(self, delta, rate=0.0):
        relu_delta = (1 * (delta > 0)) * delta
        tmp_delta = np.zeros(self.pad_img.shape)
        h_stride = self.strides[0]
        w_stride = self.strides[1]
        h_pad = self.padding[0]
        w_pad = self.padding[1]
        h_pol = self.pooling[0]
        w_pol = self.pooling[1]

        height = self.pad_img.shape[1]
        width = self.pad_img.shape[2]
        batch = self.pad_img.shape[0]
        channel = self.pooling[2]

        for b in range(0, batch):
            for k in range(0, channel):
                for i in np.arange(0, height - h_pol + 1, h_stride):
                    for j in np.arange(0, width - w_pol + 1, w_stride):
                        img_i = i // h_stride
                        img_j = j // w_stride
                        flat_idx = np.argmax(self.pad_img[b, i:i + h_pol, j:j + w_pol, k])
                        h_idx = i + flat_idx // w_pol
                        w_idx = j + flat_idx % w_pol
                        tmp_delta[b, h_idx, w_idx, k] += relu_delta[b, img_i, img_j, k]
        ori_height = self.img_shape[1]
        ori_width = self.img_shape[2]
        res_delta = tmp_delta[:, h_pad:h_pad+ori_height, w_pad:w_pad+ori_width, :]
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
    padding = (1, 1)
    stride = (2, 2)
    shape = (3, 3, 2)
    max_pool = MaxPooling(shape, stride=2, padding=0)
    res = max_pool.forward(img)
    delta = np.random.randn(3, 2, 2, 2).astype(np.float64)
    hehe = max_pool.backward(delta)