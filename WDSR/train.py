# -*- coding: utf-8 -*-
"""
https://github.com/wmylxmj/Anime-Super-Resolution
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from keras import backend as K
from keras.losses import mean_absolute_error, mean_squared_error
from keras.models import load_model
from keras.optimizers import Adam
import random

import sys
sys.path.append('/home/dongbin/DL/WDSR/')

from model import wdsr_a, wdsr_b
from optimizer import AdamWithWeightsNormalization
from utils import DataLoader
from ssim import SSIM

class SuperResolution(object):
    
    def __init__(self, scale=4, num_res_blocks=32, pretrained_weights=None, name=None):
    # def __init__(self, scale=4, num_res_blocks=48, pretrained_weights=None, name=None):
        self.scale = scale
        self.num_res_blocks = num_res_blocks
        self.model = wdsr_b(scale=scale, num_res_blocks=num_res_blocks)
        self.model.compile(optimizer=AdamWithWeightsNormalization(lr=0.0001), loss=self.mae, metrics=[self.psnr])
        # self.model.compile(optimizer=AdamWithWeightsNormalization(lr=0.0001), loss=self.ssim_loss, metrics=[self.psnr])
        # self.model.compile(optimizer=AdamWithWeightsNormalization(lr=0.001), loss=self.psnr_loss, metrics=[self.psnr])
        if pretrained_weights != None:
            self.model.load_weights(pretrained_weights)
            print("[OK] weights loaded.")
            pass
        self.data_loader = DataLoader(scale=scale, crop_size=1024)
        self.pretrained_weights = pretrained_weights
        self.default_weights_save_path = 'experiments/ckpt/wdsr-b-' + \
        str(self.num_res_blocks) + '-x' + str(self.scale) + '.h5'
        self.name = name
        pass

    def ssim_loss(self,hr,sr):
        margin = (tf.shape(hr)[1] - tf.shape(sr)[1]) // 2
        hr_crop = tf.cond(tf.equal(margin, 0), lambda: hr, lambda: hr[:, margin:-margin, margin:-margin, :])
        hr = K.in_train_phase(hr_crop, hr)
        hr.uses_learning_phase = True
        return -SSIM(hr,sr)

    def mae(self, hr, sr):
        margin = (tf.shape(hr)[1] - tf.shape(sr)[1]) // 2
        hr_crop = tf.cond(tf.equal(margin, 0), lambda: hr, lambda: hr[:, margin:-margin, margin:-margin, :])
        hr = K.in_train_phase(hr_crop, hr)
        hr.uses_learning_phase = True
        return mean_absolute_error(hr, sr)

    def psnr(self, hr, sr):
        margin = (tf.shape(hr)[1] - tf.shape(sr)[1]) // 2
        hr_crop = tf.cond(tf.equal(margin, 0), lambda: hr, lambda: hr[:, margin:-margin, margin:-margin, :])
        hr = K.in_train_phase(hr_crop, hr)
        hr.uses_learning_phase = True
        return tf.image.psnr(hr, sr, max_val=255)

    def psnr_loss(self, hr, sr):
        margin = (tf.shape(hr)[1] - tf.shape(sr)[1]) // 2
        hr_crop = tf.cond(tf.equal(margin, 0), lambda: hr, lambda: hr[:, margin:-margin, margin:-margin, :])
        hr = K.in_train_phase(hr_crop, hr)
        hr.uses_learning_phase = True
        return 1-tf.image.psnr(hr, sr, max_val=255)/100
    
    def train(self, epoches=100, batch_size=2, weights_save_path=None):
        if weights_save_path == None:
            weights_save_path = self.default_weights_save_path
            pass
        for epoch in range(epoches):
            for batch_i, (lrs, hrs) in enumerate(self.data_loader.batches(batch_size=batch_size)):
                temp_loss, temp_psnr = self.model.train_on_batch(lrs, hrs)
                print("[epoch: {}/{}][batch: {}/{}][loss: {}][psnr: {}]".format(epoch+1, epoches, \
                      batch_i+1, self.data_loader.n_batches, temp_loss, temp_psnr))
                if (batch_i+1) % 500 == 0:
                    self.sample(epoch=epoch+1, batch=batch_i+1)
                    pass
                pass
            self.model.save_weights(weights_save_path)
            print("[OK] weights saved.")
            pass
        pass
    
    def sample(self, setpath='datasets/Youku/Youku/train/HR', save_folder='experiments/samples', epoch=1, batch=1):
        images = self.data_loader.search(setpath)
        image = random.choice(images)
        hr = self.data_loader.imread(image)
        lr = self.data_loader.downsampling(hr)
        lr_resize = lr.resize(hr.size)
        lr =  np.asarray(lr)
        sr = self.model.predict(np.array([lr]))[0]
        sr = np.clip(sr, 0, 255)
        sr = sr.astype('uint8')
        lr = Image.fromarray(lr)
        sr = Image.fromarray(sr)
        lr_resize.save(save_folder + "/" + "epoch_" + str(epoch) + "_batch_" + str(batch) + "_lr.jpg")
        sr.save(save_folder + "/" + "epoch_" + str(epoch) + "_batch_" + str(batch) + "_sr.jpg")
        hr.save(save_folder + "/" + "epoch_" + str(epoch) + "_batch_" + str(batch) + "_hr.jpg")
        pass
    
    pass


sr = SuperResolution(pretrained_weights='./weights/wdsr-b-32-x4.h5')
# sr = SuperResolution(pretrained_weights=None)
sr.train()
