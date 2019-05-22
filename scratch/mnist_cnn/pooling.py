import numpy as np

from col2im import col2im
from im2col import im2col


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, padding=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = padding

        self.x = None
        self.arg_max = None

    def forward(self, x):
        data_num, channels, height, width = x.shape

        out_height = int(
            1 + (height - self.pool_h) / self.stride
        )
        out_width = int(
            1 + (width - self.pool_w) / self.stride
        )

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(
            data_num, out_height, out_width, channels
        ).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size),
             self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h,
                    self.pool_w, self.stride, self.pad)

        return dx
