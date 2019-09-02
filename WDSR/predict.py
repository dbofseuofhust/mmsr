# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from keras import backend as K
from keras.losses import mean_absolute_error, mean_squared_error
from keras.models import load_model
from keras.optimizers import Adam
import random
import os

import sys
sys.path.append('/home/dongbin/DL/WDSR/')
from model import wdsr_a, wdsr_b
from utils import DataLoader

# from deeplearning.WDSR.model import wdsr_a, wdsr_b
# from deeplearning.WDSR.utils import DataLoader

model = wdsr_b(scale=4, num_res_blocks=32)
model.load_weights('./experiments/ckpt/wdsr-b-32-x4.h5')

data_loader = DataLoader(scale=4)

def predict(model, fp, sp):
    lr = Image.open(fp)
    lr = np.asarray(lr)
    x = np.array([lr])
    y = model.predict(x)
    y = np.clip(y, 0, 255)
    y = y.astype('uint8')
    sr = Image.fromarray(y[0])
    sr.save(sp)
    pass

def resize(fp, sp, scale=4):
    lr = Image.open(fp)
    lr = lr.resize((scale*lr.size[0], scale*lr.size[1]))
    lr.save(sp)
    pass

def downsampling(fp, sp):
    hr = data_loader.imread(fp)
    lr = data_loader.downsampling(hr)
    lr.save(sp)
    pass

def copy(fp, sp):
    lr = Image.open(fp)
    lr.save(sp)
    pass

# 原来的
# def predict_testset(setpath='datasets/traffic'):
#     files = data_loader.search(setpath)
#     for index, file in enumerate(files):
#         copy(fp=file, sp='outputs/lr_' + str(index+1) + '.jpg')
#         predict(model, fp=file, sp='outputs/sr_' + str(index+1) + '.jpg')
#         pass
#     pass

def predict_testset(setpath='datasets/Youku/Youku/test'):

    sub_dirs = os.listdir(setpath)
    sub_dirs = [os.path.join(setpath,val) for val in sub_dirs]

    for line in sub_dirs:
        search_dir = os.path.join(line,'LR')
        output_dir = os.path.join(line,'HR')
        files = data_loader.search(search_dir)
        for index, file in enumerate(files):
            predict(model, fp=file, sp=os.path.join(output_dir,str(index) + '.bmp'))
            pass
        pass





predict_testset()

