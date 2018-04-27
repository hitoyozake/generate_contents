# -* encoding: utf-8 *-


import chainer.functions as F
import chainer.links as L
import chainer

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

