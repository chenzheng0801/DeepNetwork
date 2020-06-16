import numpy as np
from numba import jit


class MaxPooling:
    """
    pooling_shape = (height * width * channel)
    strides = (height_stride * width_stride)
    padding = (height_padding * width_padding)

    applying ReLu at the output
    """
    def __init__(self, pooling, strides=(2, 2), padding=(0, 0)):
        self.pooling = pooling
        self.strides = strides
        self.padding = padding

    @jit(forceobj=True)
    def forward(self, img):
        h_stride = self.strides[0]
        w_stride = self.strides[1]
        h_pad = self.padding[0]
        w_pad = self.padding[1]
        h_pol = self.pooling[0]
        w_pol = self.pooling[1]
        self.pad_img = np.lib.pad(img, ((0, 0),
                                        (h_pad, h_pad),
                                        (w_pad, w_pad),
                                        (0, 0)),
                                  'constant', constant_values=0)
        self.img_shape = shape = img.shape
        height = shape[1] + h_pad * 2
        width = shape[2] + w_pad * 2
        batch = shape[0]
        channel = self.pooling[2]

        new_height = (height - h_pol) // h_stride + 1
        new_width = (width - w_pol) // w_stride + 1
        output_img = np.zeros((batch, new_height, new_width, channel))

        for i in np.arange(0, height - h_pol + 1, h_stride):
            for j in np.arange(0, width - w_pol + 1, w_stride):
                img_i = i // h_stride
                img_j = j // w_stride
                output_img[:, img_i, img_j, :] =\
                    np.max(self.pad_img[:, i:i + h_pol, j:j + w_pol, :], axis=(1, 2))
        # ReLu Layer
        return (1 * (output_img > 0)) * output_img

    @jit(forceobj=True)
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
    img = np.random.randn(3, 4, 4, 2).astype(np.float64)
    padding = (1, 1)
    stride = (2, 2)
    shape = (3, 3, 2)
    max_pool = MaxPooling(shape, stride, padding)
    res = max_pool.forward(img)
    delta = np.random.randn(3, 2, 2, 2).astype(np.float64)
    hehe = max_pool.backward(delta)