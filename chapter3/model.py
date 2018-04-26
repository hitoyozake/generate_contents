# -* encoding: utf-8 *-


import chainer.functions as F
import chainer.links as L
import chainer

class Model(chainer.Chain):
    def __init__(self):
        super(Model, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 8, ksize=3)  # フィルタサイズ=3

