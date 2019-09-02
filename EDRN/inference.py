import os
import math
from decimal import Decimal
from torch import nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

import sys
sys.path.append('/home/dongbin/DL/EDRN/')

from dataset import Dataset
from utils import AverageMeter
# from model.frn_updown import make_model # 暂时不可用
from model.edsr import make_model # 可用
# from model.ddbpn import make_model # 可用
# from model.ercan import make_model # 可用
from model.awsrns import make_model # 可用
from tqdm import tqdm
import argparse
from PIL import Image
from torchvision import transforms
import imageio
import cv2

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Predictor():

    def __init__(self, opt):
        self.opt = opt

    def quantize(self,img):
        pixel_range = 255 / self.opt.rgb_range
        out = img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
        out = out.squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
        return out

    def predict(self,input_path,output_path):

        model = make_model(self.opt)

        state_dict = model.state_dict()
        for n, p in torch.load(opt.weights_path, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

        model = model.to(device)
        model.eval()

        input = Image.open(input_path).convert('RGB')
        input = transforms.ToTensor()(input).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input)
        output = pred.mul_(255).clamp_(0.0, 255).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
        print(output)
        output = Image.fromarray(output, mode='RGB')
        output.save(output_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='AWSRN',help='RCAN,DDBPN,ERCAN,AWSRN')
    parser.add_argument('--images_dir', type=str, default='')
    parser.add_argument('--outputs_dir', type=str, default='')
    parser.add_argument('--weights_path', type=str, default='/home/dongbin/DL/EDRN/weights/AWSRN_epoch_22.pth')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--num_rg', type=int, default=10)
    parser.add_argument('--num_rcab', type=int, default=20)
    parser.add_argument('--reduction', type=int, default=16)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--use_fast_loader', action='store_false')
    parser.add_argument('--n_resgroups', type=int, default=5,
                        help='number of residual groups in fractal residual groups')
    parser.add_argument('--n_resblocks', type=int, default=8,
                        help='number of innermost blocks of FRN (i.e. RCAN-PS)')
    parser.add_argument('--n_feats', type=int, default=64,
                        help='number of feature maps')
    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')
    parser.add_argument('--res_scale', type=float, default=1,
                        help='residual scaling')
    parser.add_argument('--px', type=int, default=1,
                        help='pixshuff downup scale (in RCAN-PS)')
    parser.add_argument('--n_awru', type=int, default=4,
                    help='number of n_awru in one LFB')
    parser.add_argument('--block_feats', type=int, default=128,
                    help='number of feature maps')

    opt = parser.parse_args()

    predictor = Predictor(opt)

    input_path = r'/home/dongbin/DL/EDRN/0802.png'
    output_path = r'/home/dongbin/DL/EDRN/0802_x4.png'
    predictor.predict(input_path,output_path)

