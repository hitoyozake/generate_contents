# -* # -* encoding: utf-8 *-
import numpy
import chapter4.model
import chainer
from PIL import Image
import os
import numpy as np
from chainer import optimizers
batch_size = 10
import chainer.cuda
try:
    import cupy as cp
except:
    pass

n_save = 0

import chainer.computational_graph as c

def main(devices = -1):

    model_gen = chapter4.model.DCGAN_Generator_NN()
    xp = np

    if devices == 1:
        import cuda
        cuda.get_device_from_id(0).use()

        model_gen.to_gpu(0)
        xp = cp

    chainer.serializers.load_hdf5('chapt04gen_result.hdf', model_gen)

    num_generate = 100
    optimizer_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)

    from numpy import random
    rnd = random.uniform(-1, 1, (num_generate, 100, 1, 1))
    rnd = xp.array(rnd, dtype=xp.float32)

    with chainer.using_config('train', False):
        result = model_gen(rnd)
    import codecs

    f = codecs.open('vectors.txt', 'w', 'utf-8')

    for i in range(num_generate):
        data = np.zeros((128, 128, 3), dtype=np.uint8)

        dst = result.data[i]*255

        if int(use_device) >= 0:
            dst = chainer.cuda.to_cpu(dst)

        data[:,:,0] = dst[0]
        data[:,:,1] = dst[1]
        data[:,:,2] = dst[2]

        himg = Image.fromarray(data, 'RGB')

        himg.save('gen-' + str(i) + '.png')

        f.write(','.join([str(j) for j in rnd[i][:, 0][:, 0]]))

        f.write('\n')

    f.close()



if __name__ == '__main__':
    import sys

    use_device = -1
    if len(sys.argv) >= 2:
        use_device = sys.argv[1]
    main(use_device)
