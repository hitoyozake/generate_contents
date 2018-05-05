# -* encoding: utf-8 *-

import chainer
import chainer.links as L, chainer.functions as F
import numpy

batch_size = 10
use_device = -1
image_size = 128 # 生成画像のサイズ
neuron_size = 64 # 中間層のサイズ

# 贋作側のNN
class DCGAN_Generator_NN(chainer.Chain):

    def __init__(self):
        weight_initializer = chainer.initializers.Normal(scale=0.02, dtype=None)

        super(DCGAN_Generator_NN, self).__init__()
        # class chainer.links.Deconvolution2D(self,
        # in_channels, out_channels, ksize=None, stride=1, pad=0,
        #         nobias=False, outsize=None, initialW=None,
        #         initial_bias=None, *, groups=1)
        with self.init_scope():
            self.l0 = L.Linear(100, (neuron_size **3) //8 //8, initialW=weight_initializer)
            self.dc1 = L.Deconvolution2D(in_channels=neuron_size, out_channels=neuron_size // 2, ksize=4,
                                         stride=2, pad=1, nobias=False, outsize=None, initialW=weight_initializer)
            self.dc2 = L.Deconvolution2D(neuron_size//2, neuron_size //4, 4, 2, 1, initialW=weight_initializer)
            self.bn1 = L.BatchNormalization(neuron_size//2)
            self.dc3 = L.Deconvolution2D(neuron_size//4, neuron_size //8, 4, 2, 1, nitialW=weight_initializer)
            self.bn2 = L.BatchNormalization(neuron_size//4)
            self.dc4 = L.Deconvolution2D(neuron_size//8,               3, 3, 1, 1, initialW=weight_initializer)
            self.bn3 = L.BatchNormalization(neuron_size // 8)


    def __call__(self, z):
        shape = (len(z), neuron_size, image_size // 8, image_size //8)
        h = F.reshape(F.relu(self.bn0()))


# 鑑定側のNN
class DCGAN_Discreminator_NN(chainer.Chain):

    def __init__(self):
        weight_initializer = chainer.initializers.Normal(scale=0.02, dtype=None)

        super(DCGAN_Discreminator_NN, self).__init__()



        # NNの定義
        with self.init_scope(): # init_scopeの中でsuperクラスのメンバ変数の初期化を行う
            # // ・・・Pythonの除算演算子の１つ．結果を整数で得る(x/yだとfloatになるが，x//yだとintの結果を得る)
            self.c0_0 = L.Convolution2D(3, neuron_size // 8, 3, 1, 1, initialW=weight_initializer)
            # Convolution2D(self, in_channels, out_channels, ksize=None,
            # stride=1, pad=0, nobias=False, initialW=None, initial_bias=None, *, dilate=1, groups=1)
            self.c0_1 = L.Convolution2D(neuron_size // 8, neuron_size // 4, ksize=4, stride=1, pad=1, nobias=False, initialW=weight_initializer)
            self.c1_0 = L.Convolution2D(neuron_size // 4, neuron_size // 4, 3, 2, 1, initialW=weight_initializer)
            self.c1_1 = L.Convolution2D(neuron_size // 2, neuron_size // 2, 4, 2, 1, initialW=weight_initializer)
            self.c2_0 = L.Convolution2D(neuron_size // 2, neuron_size // 2, 3, 1, 1, initialW=weight_initializer)
            self.c2_1 = L.Convolution2D(neuron_size // 2, neuron_size, 4, 2, 1, initialW=weight_initializer)
            self.c3_0 = L.Linear(neuron_size * image_size * image_size // 8 //8, 1, initialW=weight_initializer)

            self.l4 = L.Linear(neuron_size * image_size * image_size //8 // 6 ) # 全結合

            self.bn0_1 = L.BatchNormalization(neuron_size //4, use_gamma=False)
            self.bn1_0 = L.BatchNormalization(neuron_size // 4, use_gamma=False)
            self.bn1_1 = L.BatchNormalization(neuron_size // 2, use_gamma=False)
            self.bn2_0 = L.BatchNormalization(neuron_size // 2, use_gamma=False)
            self.bn2_1 = L.BatchNormalization(neuron_size, use_gamma=False)
            self.bn3_0 = L.BatchNormalization(neuron_size, use_gamma=False)



    def __call__(self, x):
        h = F.leaky_relu(self.c0_0(x))

        h = F.dropout(F.leaky_relu(self.bn0_1(self.c0_1(h))), ratio=0.2)
        h = F.dropout(F.leaky_relu(self.bn1_0(self.c1_0(h))), ratio=0.2)
        h = F.dropout(F.leaky_relu(self.bn1_1(self.c1_1(h))), ratio=0.2)
        h = F.dropout(F.leaky_relu(self.bn2_0(self.c2_0(h))), ratio=0.2)
        h = F.dropout(F.leaky_relu(self.bn2_1(self.c2_1(h))), ratio=0.2)
        h = F.dropout(F.leaky_relu(self.bn3_0(self.c3_0(h))), ratio=0.2)

        return self.l4(h)