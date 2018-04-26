# -* encoding: utf-8 *-

import numpy as np
from second.model import MNIST_Conv_MN
from PIL import Image

uses_device = 0

if uses_device >= 0:
    import cupy as cp
else:
    cp = np

import chainer

model = MNIST_Conv_MN()
chainer.serializers.load_hdf5("chapter2.hdf5", model)

if uses_device >= 0:
    """
    """

image = Image.open('test/mnist-0.png').convert('L') #Lは8bitグレースケール変換
pixels = cp.asarray(image).astype(cp.float32).reshape(1, 1, 28, 28) # 28 * 28
pixels = pixels/255 #正規化

result = model(pixels, train=False)


