import numpy as np

from col2im import col2im
from im2col import im2col


class Convolution:
    def __init__(self, W, b, stride=1, padding=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = padding

        # 中間データ（backward時に使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        filter_num, channels, filter_height, filter_width = self.W.shape
        data_num, channels, height, width = x.shape
        # FN, C, FH, FW = self.W.shape
        # N, C, H, W = x.shape

        # 畳み込み演算
        out_height = int(
            1 + (height + 2 * self.pad - filter_height) / self.stride
        )
        out_width = int(
            1 + (width + 2 * self.pad - filter_width) / self.stride
        )

        # 計算のしやすいように，行列の変換・整形を行う
        col = im2col(x, filter_height, filter_width, self.stride, self.pad)
        col_W = self.W.reshape(filter_num, -1).T

        out = np.dot(col, col_W) + self.b

        # 2次元配列を4次元配列に変換
        # transpose: 多次元配列の順番を入れ替える関数
        out = out.reshape(
            data_num, out_height, out_width, -1
        ).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        filter_num, channels, filter_height, filter_width = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, filter_num)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(
            filter_num, channels, filter_height, filter_width)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, filter_height,
                    filter_width, self.stride, self.pad)

        return dx
