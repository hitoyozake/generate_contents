# -* encoding: utf-8 *-
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
from chainer.training.extensions import LogReport
import numpy as np
from chainer.backends import cuda

from second.model import MNIST_Conv_MN

batch_size = 10


def main(use_device):
    model = MNIST_Conv_MN()


    if use_device >= 1:
        print("use gpu")
        cuda.get_device_from_id(use_device).use()
        model.to_gpu(use_device)



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
