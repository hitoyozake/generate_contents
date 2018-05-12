# -* encoding: utf-8 *-
import numpy
import chapter4.model
import chainer
from PIL import Image
import os
import numpy as np


def make_train_data():
    files = os.listdir('train')

    train_imgs = []

    for fn in files:
        img = Image.open('train/' + fn).resize((128, 128)).convert('RGB')
        img = np.array(img, dtype=np.float32) /255.0
        train_imgs.append(img)

    return train_imgs


def main(devices = -1):

    model_dis = chapter4.model.DCGAN_Discreminator_NN()
    model_gen = chapter4.model.DCGAN_Generator_NN()

    if devices == 1:
        import cuda
        cuda.get_device_from_id(0).use()

        model_dis.to_gpu(0)
        model_gen.to_gpu(0)



    pass

if __name__ == '__main__':
    import sys

    use_device = -1
    if len(sys.argv) >= 2:
        use_device = sys.argv[1]
    main(use_device)