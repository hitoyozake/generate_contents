# -* encoding: utf-8 *-
import numpy
import chapter4.model
import chainer
from PIL import Image
import os
import numpy as np
from chainer import optimizers
batch_size = 10

def make_train_data():
    files = os.listdir('train')

    train_imgs = []

    for fn in files:
        img = Image.open('train/' + fn).resize((128, 128)).convert('RGB')
        img = np.array(img, dtype=np.float32) /255.0
        img = img.transpose(2,0,1)
        train_imgs.append(img)

    return train_imgs

n_save = 0

def main(devices = -1):

    model_dis = chapter4.model.DCGAN_Discreminator_NN()
    model_gen = chapter4.model.DCGAN_Generator_NN()

    if devices == 1:
        import cuda
        cuda.get_device_from_id(0).use()

        model_dis.to_gpu(0)
        model_gen.to_gpu(0)
        try:
            import cupy
            xp = cupy
            cp = cupy
        except:
            xp = numpy
            cp = numpy
    imgs = make_train_data()

    train_iter = chainer.iterators.SerialIterator(imgs, batch_size, shuffle=True)

    optimizer_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
    optimizer_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)

    optimizer_gen.setup(model_gen)
    optimizer_dis.setup(model_dis)

    updater = chapter4.model.DCGANUpdater(train_iter, {'opt_gen':optimizer_gen, 'opt_dis':optimizer_dis}, device=int(devices))

    # 機械学習の実行
    trainer = chainer.training.Trainer(updater, (1500, 'epoch'), out='result')

    trainer.extend(chainer.training.extensions.ProgressBar())

    # 中韓結果の保存
    
    @chainer.training.make_extension(trigger=(1000, 'epoch'))
    def save_model(trainer):
        global n_save
        n_save = n_save + 1
        chainer.serializers.save_hdf5('chapt04-gen-{0:d}.hdf'.format(n_save), model_gen)
        chainer.serializers.save_hdf5('chapt04-dis-{0:d}.hdf'.format(n_save), model_dis)

    trainer.extend(save_model)

    trainer.run()

    # 結果の保存
    chainer.serializers.save_hdf5('chapt04gen_result.hdf', model_gen)


if __name__ == '__main__':
    import sys

    use_device = -1
    if len(sys.argv) >= 2:
        use_device = sys.argv[1]
    main(use_device)
