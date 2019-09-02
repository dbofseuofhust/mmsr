import cv2
import matplotlib.pyplot as plt
import os

def rename(path):

    imgs = os.listdir(path)

    for idx,val in enumerate(imgs):
        ori_path = os.path.join(path,val)
        new_path = os.path.join(path,str(idx)+'_.'+val.split('.')[1])
        os.rename(ori_path,new_path)

if __name__ == '__main__':

    path = r'/deeplearning/Youku/Youku/train/HR'
    rename(path)

    path = r'/deeplearning/Youku/Youku/train/LR'
    rename(path)

    path = r'/deeplearning/Youku/Youku/valid/HR'
    rename(path)

    path = r'/deeplearning/Youku/Youku/valid/LR'
    rename(path)