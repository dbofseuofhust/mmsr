import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import glob
import numpy as np
import PIL.Image as pil_image
import tensorflow as tf
from matplotlib import pyplot as plt

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.enable_eager_execution(config=config)

'''
# 这个DataLoader是根据高清图片和scale自动生成低分辨率的图片的

class Dataset(object):
    def __init__(self, images_dir, patch_size, scale, use_fast_loader=False):
        self.image_files = sorted(glob.glob(images_dir + '/*'))
        self.patch_size = patch_size
        self.scale = scale
        self.use_fast_loader = use_fast_loader

    def __getitem__(self, idx):
        if self.use_fast_loader:
            hr = tf.read_file(self.image_files[idx])
            hr = tf.image.decode_jpeg(hr, channels=3)
            hr = pil_image.fromarray(hr.numpy())
        else:
            hr = pil_image.open(self.image_files[idx]).convert('RGB')

        # randomly crop patch from training set
        crop_x = random.randint(0, hr.width - self.patch_size * self.scale)
        crop_y = random.randint(0, hr.height - self.patch_size * self.scale)
        hr = hr.crop((crop_x, crop_y, crop_x + self.patch_size * self.scale, crop_y + self.patch_size * self.scale))

        # degrade lr with Bicubic
        lr = hr.resize((self.patch_size, self.patch_size), resample=pil_image.BICUBIC)

        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)

        hr = np.transpose(hr, axes=[2, 0, 1])
        lr = np.transpose(lr, axes=[2, 0, 1])

        # normalization
        hr /= 255.0
        lr /= 255.0

        return lr, hr

    def __len__(self):
        return len(self.image_files)
        
'''

# 对原有的DataLoader进行改造，使得自动读取Youku自带的高清和低清图片
'''
数据集的格式为:
    datasets
    |----Youku
    |--------Youku
    |------------train
    |----------------HR
    |----------------LR    
'''

class Dataset(object):
    def __init__(self, images_dir, crop_size, scale):
        self.image_files = sorted(glob.glob(images_dir + '/*'))
        self.crop_size = crop_size
        self.scale = scale

    def __getitem__(self, idx):

        hr = pil_image.open(self.image_files[idx]).convert('RGB')

        # 修改此地方，加载自带低分辨率图片
        lr = pil_image.open(self.image_files[idx].replace('HR','LR')).convert('RGB')

        lr,hr = self.crop(lr,hr)

        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)

        hr = np.transpose(hr, axes=[2, 0, 1])
        lr = np.transpose(lr, axes=[2, 0, 1])

        # normalization
        hr /= 255.0
        lr /= 255.0

        return lr, hr

    def __len__(self):
        return len(self.image_files)

    def crop(self, lr, hr):
        hr_crop_size = self.crop_size
        lr_crop_size = hr_crop_size // self.scale
        lr_w = np.random.randint(lr.size[0] - lr_crop_size + 1)
        lr_h = np.random.randint(lr.size[1] - lr_crop_size + 1)
        hr_w = lr_w * self.scale
        hr_h = lr_h * self.scale
        lr = lr.crop([lr_w, lr_h, lr_w + lr_crop_size, lr_h + lr_crop_size])
        hr = hr.crop([hr_w, hr_h, hr_w + hr_crop_size, hr_h + hr_crop_size])
        return lr, hr

if __name__ == '__main__':

    # 检查裁剪的是否正确

    hr_path = r'G:\youku\HR\train\Images\00000_h_GT_003.bmp'
    lr_path = r'G:\youku\LR\train\Images\00000_l_003.bmp'

    hr = pil_image.open(hr_path).convert('RGB')
    lr = pil_image.open(hr_path).convert('RGB')

    crop_size = 512
    scale = 4

    hr_crop_size = crop_size
    lr_crop_size = hr_crop_size // scale
    lr_w = np.random.randint(lr.size[0] - lr_crop_size + 1)
    lr_h = np.random.randint(lr.size[1] - lr_crop_size + 1)
    hr_w = lr_w * scale
    hr_h = lr_h * scale
    lr = lr.crop([lr_w, lr_h, lr_w + lr_crop_size, lr_h + lr_crop_size])
    hr = hr.crop([hr_w, hr_h, hr_w + hr_crop_size, hr_h + hr_crop_size])

    plt.figure("lr")
    plt.imshow(lr)
    plt.show()

    plt.figure("hr")
    plt.imshow(hr)
    plt.show()