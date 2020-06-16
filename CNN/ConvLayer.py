import numpy as np
from numba import jit


class Convolution:
    """
    filter_shape = (height * width * channel * next_channel)
    strides = (height_stride * width_stride)
    padding = (height_padding * width_padding)
    """
    def __init__(self, filter_shape, strides, padding):
        self.flt_h = h = filter_shape[0]
        self.flt_w = w = filter_shape[1]
        self.flt_ch = ch = filter_shape[2]
        self.nxt_ch = nxt_ch = filter_shape[3]

        self.height_stride = strides[0]
        self.width_stride = strides[1]

        self.height_padding = padding[0]
        self.width_padding = padding[1]

        l = -np.sqrt(6.0 / (1.0 + self.flt_h * self.flt_w * self.flt_ch))
        r = -l
        self.filter = np.random.uniform(l, r, (h, w, ch, nxt_ch))
        self.b = np.random.uniform(l, r, (1, nxt_ch))

    # @jit(forceobj=True)
    def forward(self, img):
        h_stride = self.height_stride
        w_stride = self.width_stride
        h_pad = self.height_padding
        w_pad = self.width_padding
        self.pad_img = np.lib.pad(img, ((0, 0),
                                        (h_pad, h_pad),
                                        (w_pad, w_pad),
                                        (0, 0)),
                                  'constant', constant_values=0)
        self.img_shape = shape = img.shape
        height = shape[1] + h_pad*2
        width = shape[2] + w_pad*2
        batch = shape[0]
        channel = self.nxt_ch

        new_height = (height - self.flt_h) // h_stride + 1
        new_width = (width - self.flt_w) // w_stride + 1
        output_img = np.zeros((batch, new_height, new_width, channel))

        for i in np.arange(0, height - self.flt_h + 1, h_stride):
            for j in np.arange(0, width - self.flt_w + 1, w_stride):
                img_i = i // h_stride
                img_j = j // w_stride

                output_img[:, img_i, img_j, :] = np.einsum("Bijk,ijkl->Bl",
                                                               self.pad_img[:, i:i+self.flt_h, j:j+self.flt_w, :],
                                                               self.filter) + self.b
        return output_img

    @jit(forceobj=True)
    def backward(self, delta, rate):
        tmp_delta = np.zeros(self.pad_img.shape)
        h_stride = self.height_stride
        w_stride = self.width_stride
        h_pad = self.height_padding
        w_pad = self.width_padding

        height = self.pad_img.shape[1]
        width = self.pad_img.shape[2]
        batch = self.pad_img.shape[0]
        channel = self.nxt_ch

        delta_filter = np.zeros(self.filter.shape)
        delta_b = np.sum(delta, axis=(0, 1, 2)).reshape(self.b.shape)
        for b in np.arange(0, batch):
            for i in np.arange(0, height - self.flt_h + 1, h_stride):
                for j in np.arange(0, width - self.flt_w + 1, w_stride):
                    index_i = i // h_stride
                    index_j = j // w_stride
                    tmp_delta[b, i:i+self.flt_h, j:j+self.flt_w, :] +=\
                        np.sum(self.filter * delta[b:b+1, index_i:index_i+1, index_j:index_j+1, :], axis=3)
                    delta_filter += np.einsum("ijk,l->ijkl",
                                              self.pad_img[b, i:i+self.flt_h, j:j+self.flt_w, :],
                                              delta[b, index_i, index_j, :])
        delta_filter /= batch
        delta_b /= batch
        self.filter += rate * delta_filter
        self.b += rate * delta_b

        ori_height = self.img_shape[1]
        ori_width = self.img_shape[2]
        res_delta = tmp_delta[:, h_pad:h_pad+ori_height, w_pad:w_pad+ori_width, :]
        return res_delta


if __name__ == "__main__":
    img = np.random.randn(1, 4, 4, 1).astype(np.float64)
    padding = (1, 1)
    stride = (2, 2)
    shape = (3,3,1,1)

    conv = Convolution(shape, stride, padding)

    std_y = np.ones((1,2,2,1))

    for i in range(40):
        now_y = conv.foward(img)
        loss_mat = 2*(std_y - now_y)
        conv.backward(loss_mat, 0.03)
        loss_value = np.sum(loss_mat**2)
        print(loss_value)
