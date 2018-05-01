# -* encoding: utf-8 *-


import chainer.functions as F
import chainer.links as L
import chainer
from chainer import training
import numpy as np
from PIL import Image

xp = np

class SuperResolutionModel(chainer.Chain):
    def __init__(self):

        w1 = chainer.initializers.Normal(scale=0.0378, dtype=None)
        w2 = chainer.initializers.Normal(scale=0.03536, dtype=None)
        w3 = chainer.initializers.Normal(scale=0.1179, dtype=None)
        w4 = chainer.initializers.Normal(scale=0.189, dtype=None)
        w5 = chainer.initializers.Normal(scale=0.0001, dtype=None)
        super(SuperResolutionModel, self).__init__()

        with self.init_scope():
            self.c1 = L.Convolution2D(1, 56, ksize=5, stride=1, pad=0, initialW=w1)
            self.l1 = L.PReLU()

            self.c2 = L.Convolution2D(56, 12, ksize=1, stride=1, pad=0, initialW=w2)
            self.l2 = L.PReLU()

            self.c3 = L.Convolution2D(12, 12, ksize=3, stride=1, pad=0, initialW=w3)
            self.l3 = L.PReLU()

            self.c4 = L.Convolution2D(12, 12, ksize=3, stride=1, pad=1, initialW=w3)
            self.l4 = L.PReLU()

            self.c5 = L.Convolution2D(12, 12, ksize=3, stride=1, pad=1, initialW=w3)
            self.l5 = L.PReLU()

            self.c6 = L.Convolution2D(12, 12, ksize=3, stride=1, pad=1, initialW=w3)
            self.l6 = L.PReLU()

            self.c7 = L.Convolution2D(12, 56, ksize=1, stride=1, pad=1, initialW=w4)
            self.l7 = L.PReLU()

            self.c8 = L.Convolution2D(56, 1, ksize=9, stride=1, pad=1, initialW=w5)

    def __call__(self, x, t=None, train=True):
        h1 = self.l1(self.c1(x))
        h2 = self.l2(self.c2(h1))
        h3 = self.l3(self.c3(h2))
        h4 = self.l4(self.c4(h3))
        h5 = self.l5(self.c5(h4))
        h6 = self.l6(self.c6(h5))
        h7 = self.l7(self.c7(h6))
        h8 = self.c8(h7)

        # 損失か結果を返す
        return F.mean_squared_error(h8, t) if train is True else h8


# 学習を管理する独自クラスの定義を行う
class SRUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, device):
        super(SRUpdater, self).__init__(
            train_iter,
            optimizer,
            devvice=device
        )


    def update_core(self):
        # ここに学習用のコードを書く
        batch = self.get_iterator('main').next()

        optimizer = self.get_optimizer('main')

        x_batch = [] # 入力データ
        y_batch = [] # 出力データ

        for img in batch:
            # 高解像度のデータ

            # YUV空間データのYを用いる
            hpix = xp.array(img, dtype=xp.float32) / 255.0
            y_batch.append(hpix[:, :, 0]) # Yのみの1chのデータにする
            low = img.resize(16, 16, Image.NEAREST)
            lpix = xp.array(low, dtype=xp.float32) / 255.0
            x_batch.append(lpix[:,:,0]) # Yのみの1chのデータにする




