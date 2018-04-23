# -* encoding: utf-8 *-
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
from chainer.training.extensions import LogReport
import numpy as np

batch_size = 10

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



def main(use_device):
    model = MNIST_Conv_MN()

    if use_device >= 1:
        print("use gpu")

    train, test = chainer.datasets.get_mnist(ndim=3)

    # 繰り返し条件の作成
    train_iter = iterators.SerialIterator(train, batch_size, shuffle=True)
    test_iter = iterators.SerialIterator(test, 1, repeat=False, shuffle=False)

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    updator=training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updator, (5, 'epoch'), out="result")

    # テストをTrainerに設定
    trainer.extend(extensions.Evaluator(test_iter, model))

    # 学習の進展を表示するようにする
    trainer.extend(extensions.ProgressBar())

    # 教師データとテストデータの正解率の表示
    #trainer.extend(extensions.LogReport())

    # trainer.extend(extensions.PrintReport(['main/accuracy', 'validation/main/accuracy']))

    trainer.run()

    # 学習データの保存
    chainer.serializers.save_hdf5('chapt02.hdf5', model)



import sys

if __name__ == '__main__':
    argv = sys.argv
    use_device = 0
    if len(argv) >= 2:
        if argv[1] == 1:
            print(argv[1])
            use_device = 0
    main(use_device)
