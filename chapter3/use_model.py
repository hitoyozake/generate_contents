# -* encoding: utf-8 *-
import chainer
import chapter3.model
from PIL import Image
import math
import numpy as np

xp = np

def main(use_device = -1):
    print("use model")

    model = chapter3.model.SuperResolutionModel()
    chainer.serializers.load_hdf5("chapt03.hdf", model)

    # 画像の読み込み
    input = Image.open('input.png').convert('YCbCr')

    org_w, org_h = w, h = input.size[0], input.size[1]

    if w % 16 != 0:
        w = (math.floor(w/16)+1)*16
    if h % 16 != 0:
        h = (math.floor(h/16)+1)*16

    if w != org_w or h != org_h:
        input = input.resize((w, h))

    output_img = Image.new('YCbCr', (10*w/4, 10*h/4), 'white')

    # 入力画像を分割
    cur_x = 0

    while cur_x <= input.size[0] - 16:
        cur_y = 0
        while cur_y <= input.size[1] - 16:

            # 画像の切り出し
            rect = (cur_x, cur_y, cur_x + 16, cur_y + 16)
            cropimg = input.crop(rect)

            hpix = xp.array(cropimg, dtype=xp.float32)
            hpix = hpix[:,:,0]/255
            x = xp.array([[hpix]], dtype=xp.float32)

            t=model(x, train=False)

            result_img = cropimg.resize((40, 40), Image.BICUBIC)

            hpix = np.array(result_img, dtype=np.float32)

            hpix.flags.writable=True

            bytes = np.array(hpix.clip(0, 255), dtype=np.uint8)
            himg = Image.fromarray(bytes, 'YCbCr')
            output_img.paster(himg, (10*cur_x/4, 10*cur_y/4, 10*cur_x/4 + 40, 10*cur_y/4 + 40))

            cur_y += 16

        cur_x += 16

    output_img = output_img.convert('RGB')
    output_img.save('output.png')
import sys

if __name__ == '__main__':
    use_device = -1
    if len(sys.argv) >= 2:
        use_device = sys.argv[1]
    main(use_device)

