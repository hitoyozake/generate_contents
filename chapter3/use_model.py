# -* encoding: utf-8 *-
import chainer
import chapter3.model
from PIL import Image

def main(use_device = -1):
    print("use model")

    model = chapter3.model.SuperResolutionModel()
    chainer.serializers.load_hdf5("chapt03.hdf", model)

    # 画像の読み込み


import sys

if __name__ == '__main__':
    use_device = -1
    if len(sys.argv) >= 2:
        use_device = sys.argv[1]
    main(use_device)

