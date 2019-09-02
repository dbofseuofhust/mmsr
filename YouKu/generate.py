import numpy as np
import os
import cv2

# 参考文献，注意，脚本中的变量比命令行的变量要多一个%
# https://www.cnblogs.com/riddick/p/7744531.html

# def generate_scripts(path):
#
#     with open(path,'a+') as f:
#         for i in range(150):
#             length = len(str(i))
#             num = '0'*(5-length)+str(i)
#             line = 'ffmpeg -i Youku_{}_h_GT.y4m -vsync 0 Images\{}_h_GT_%%3d.bmp -y'.format(num,num)
#             f.writelines(line+'\n')

# def generate_scripts(path):
#
#     with open(path,'a+') as f:
#         for i in range(150):
#             length = len(str(i))
#             num = '0'*(5-length)+str(i)
#             line = 'ffmpeg -i Youku_{}_l.y4m -vsync 0 Images\{}_l_%%3d.bmp -y'.format(num,num)
#             f.writelines(line+'\n')

# def generate_scripts(path):
#
#     # for i in range(200,250):
#     #     length = len(str(i))
#     #     num = '0' * (5 - length) + str(i)
#     #     sub_dir = r'G:\youku\test\Test\Youku_{}'.format(num)
#     #     os.mkdir(sub_dir)
#     #     os.mkdir(os.path.join(sub_dir,'LR'))
#     #     os.mkdir(os.path.join(sub_dir,'HR'))
#
#     with open(path,'a+') as f:
#         for i in range(200,250):
#             length = len(str(i))
#             num = '0'*(5-length)+str(i)
#             line = 'ffmpeg -i Youku_{}_l.y4m -vsync 0 Test\Youku_{}\LR\%%3d.bmp -y'.format(num,num)
#             f.writelines(line+'\n')

# 将需要抽帧的视频分出来
def split(path):

    dirs = []
    for i in range(205,250):
        length = len(str(i))
        num = '0'*(5-length)+str(i)
        dirs.append(os.path.join(path,'Youku_{}'.format(num)))

    for val in dirs:
        HR = os.path.join(val,'HR')
        Sub = os.path.join(val,'Sub')

        # os.mkdir(Sub)
        # imgs = os.listdir(Sub)

        imgs = os.listdir(HR)
        for idx,line in enumerate(imgs):

            # ori_path = os.path.join(HR, line)
            # new_path = ori_path.replace('HR','Sub')
            # os.remove(new_path)

            yushu = (int(line.split('.')[0])) % 25
            zhengshu = int((int(line.split('.')[0])) // 25) +1
            if yushu == 0:
                ori_path = os.path.join(HR,line)
                new_path = os.path.join(Sub,str(zhengshu)+'.bmp')
                src = cv2.imread(ori_path)
                cv2.imwrite(new_path,src)

# def generate_scripts(path):
#
#     with open(path,'a+') as f:
#         for i in range(200,205):
#             length = len(str(i))
#             num = '0'*(5-length)+str(i)
#             line = 'ffmpeg -i Test\Youku_{}\HR\%%1d.bmp  -pix_fmt yuv420p  -vsync 0 Submit\Youku_{}_h_Res.y4m -y'.format(num,num)
#             f.writelines(line+'\n')
#
#         for i in range(205,250):
#             length = len(str(i))
#             num = '0' * (5 - length) + str(i)
#             line = 'ffmpeg -i Test\Youku_{}\Sub\%%1d.bmp  -pix_fmt yuv420p  -vsync 0 Submit\Youku_{}_h_Sub25_Res.y4m -y'.format(num,num)
#             f.writelines(line+'\n')

if __name__ == '__main__':

    # path = r'C:\Users\admin\PycharmProjects\Crack\deeplearning\YouKu\valid.txt'
    # generate_scripts(path)

    path = r'G:\youku\test\Test'
    split(path)

