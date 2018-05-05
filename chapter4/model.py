# -* encoding: utf-8 *-

import chainer
import chainer.links as L, chainer.functions as F
import numpy

batch_size = 10
use_device = -1
image_size = 128 # 生成画像のサイズ
neuron_size = 64 # 中間層のサイズ


# 鑑定側のNN
class DCGAN_Discreminator_NN(chainer.Chain):

    def __init__(self):
        weight_initializer = chainer.initializers.Normal(scale=0.02, dtype=None)

        super(DCGAN_Discreminator_NN, self).__init__()



        # NNの定義
        with self.init_scope(): # init_scopeの中でsuperクラスのメンバ変数の初期化を行う
            # // ・・・Pythonの除算演算子の１つ．結果を整数で得る(x/yだとfloatになるが，x//yだとintの結果を得る)
            self.c0_0 = L.Convolution2D(3, neuron_size // 8, 3, 1, 1, initialW=weight_initializer)
            self.c0_1 = L.Convolution2D(neuron_size // 8, neuron_size // 4, 3, 1, 1, initialW=weight_initializer)
            self.c1_0 = L.Convolution2D(neuron_size // 4, neuron_size // 2, 4, 2, 1, initialW=weight_initializer)
            self.c1_1 = L.Convolution2D(neuron_size // 2, neuron_size // 2, 3, 1, 1, initialW=weight_initializer)
            self.c2_0 = L.Convolution2D(neuron_size // 2, neuron_size // 2, 3, 1, 1, initialW=weight_initializer)
            self.c3_0 = L.Linear(neuron_size * image_size * image_size // 8 //8, 1, initialW=weight_initializer)


            self.bn0_1 = L.BatchNormalization(neuron_size //4, use_gamma=False)
            self.bn1_0 = L.BatchNormalization(neuron_size // 4, use_gamma=False)
            self.bn1_1 = L.BatchNormalization(neuron_size // 2, use_gamma=False)
            self.bn1_1 = L.BatchNormalization(neuron_size // 2, use_gamma=False)
            self.bn2_0 = L.BatchNormalization(neuron_size // 2, use_gamma=False)
            self.bn2_1 = L.BatchNormalization(neuron_size, use_gamma=False)
            self.bn3_0 = L.BatchNormalization(neuron_size, use_gamma=False)

