# -* encoding: utf-8 *-
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
from chainer.training.extensions import LogReport
import numpy as np

class MNIST_Conv_MN(chainer.Chain):

    def __init__(self):
        super(MNIST_Conv_MN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 8, ksize=3) # フィルタサイズ=3
            self.linear1 = L.Linear(13*13*8, 10) # 出力数10

    def __call__(self, x, t=None, train=True):

        h1 = self.conv1(x) # 3*3のフィルタで処理->26*26*8
        h2 = F.relu(h1) # 活性化関数．出力のデータの数は変わらない
        h3 = F.average_pooling_2d(h2, 2) # pooling層(2*2ピクセルで平均値を出力) -> 13 * 13 * 8 のデータになる
        h4 = self.linear1(h3) # 全結合

        return F.softmax_cross_entropy(h4, t) if train else F.softmax(h4)

