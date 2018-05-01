# -* encoding: utf-8 *-

import chapter3.model

import os
from PIL import Image
from chainer import iterators
from chainer import optimizers


def make_train_data():
    files = os.listdir('train')

    train_imgs = []

    for fn in files:
        img = Image.open('train/' + fn).resize((320, 320)).covert('YCbCr')

        cur_x = 0

        while cur_x <= 320 - 40:
            cur_y  = 0

            while cur_y <= 320 - 40:
                # 画像から切り出し

                rect = (cur_x, cur_y, cure_x + 40, cur_y + 40)

                coping = img.crop(rect).copy()

                # 配列に追加
                train_imgs.append(coping)

                cur_y += 20
            cur_x += 20

    return train_imgs

def main(use_device=0):
    print("main chapter3")

    model = chapter3.model.SuperResolutionModel()

    if use_device == 1:
        import cuda
        cuda.get_device_from_id(0).use()
        model.to_gpu(0)

    images = make_train_data()

    train_iter = iterators.SerialIterator(dataset=images, batch_size=20, shuffle=True)

    optimizer = optimizers.Adam()

    optimizer.setup(model)

    updater = chapter3.model.SRUpdater(train_iter=train_iter, optimizer=optimizer, device=use_device)

    from chainer import training
    trainer = training.Trainer(updater, (10000, 'epoch'), out='result')
    from chainer.training import extensions
    trainer.extend(extensions.ProgressBar())








if __name__ == '__main__':
    import sys

    use_device = 0

    if len(sys.argv) >= 2:
        use_device = sys.argv[1]

    main(use_device)
