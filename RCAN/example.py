import argparse
import os
import PIL.Image as pil_image
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

import sys
sys.path.append('/home/dongbin/DL/RCAN/')

from model import RCAN
import ercan
import rdn
import idn
import drrn

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    # RCAN特有的
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--arch', type=str, default='RCAN')
    # parser.add_argument('--weights_path', type=str, required=True)
    # parser.add_argument('--image_path', type=str, required=True)
    # parser.add_argument('--outputs_dir', type=str, required=True)
    # parser.add_argument('--scale', type=int, required=True)
    # parser.add_argument('--num_features', type=int, default=64)
    # parser.add_argument('--num_rg', type=int, default=10)
    # parser.add_argument('--num_rcab', type=int, default=20)
    # parser.add_argument('--reduction', type=int, default=16)

    # RDN特有的
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='RDN')
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--scale', type=int, required=True)
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--growth_rate', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=16)

    # IDN特有的
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--arch', type=str, default='IDN')
    # parser.add_argument('--weights_path', type=str, required=True)
    # parser.add_argument('--image_path', type=str, required=True)
    # parser.add_argument('--outputs_dir', type=str, required=True)
    # parser.add_argument('--scale', type=int, required=True)
    # parser.add_argument('--num_features', type=int, default=64)
    # parser.add_argument('--d', type=int, default=16)
    # parser.add_argument('--s', type=int, default=4)

    # DRRN特有的
    # parser.add_argument('--arch', type=str, default='DRRN')
    # parser.add_argument('--weights_path', type=str, required=True)
    # parser.add_argument('--image_path', type=str, required=True)
    # parser.add_argument('--outputs_dir', type=str, required=True)
    # parser.add_argument('--scale', type=int, required=True)
    # parser.add_argument('--num_features', type=int, default=64)
    # parser.add_argument('--d', type=int, default=16)
    # parser.add_argument('--s', type=int, default=4)


    # ERCAN特有的
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--arch', type=str, default='ERCAN')
    # parser.add_argument('--weights_path', type=str, required=True)
    # parser.add_argument('--image_path', type=str, required=True)
    # parser.add_argument('--outputs_dir', type=str, required=True)
    # parser.add_argument('--scale', type=int, required=True)
    # parser.add_argument('--num_features', type=int, default=64)
    # parser.add_argument('--num_rg', type=int, default=10)
    # parser.add_argument('--num_rcab', type=int, default=20)
    # parser.add_argument('--reduction', type=int, default=16)
    # parser.add_argument('--n_resgroups', type=list, nargs='+', default=[2, 4, 8],
    #                     help='number of residual groups in fractal residual groups')
    # parser.add_argument('--n_resblocks', type=int, default=4,
    #                     help='number of innermost blocks of FRN (i.e. RCAN-PS)')
    # parser.add_argument('--use_fast_loader', action='store_false')
    # parser.add_argument('--n_feats', type=int, default=32,
    #                     help='number of feature maps')
    # parser.add_argument('--rgb_range', type=int, default=255,
    #                     help='maximum value of RGB')
    # parser.add_argument('--n_colors', type=int, default=3,
    #                     help='number of color channels to use')
    # parser.add_argument('--res_scale', type=float, default=1,
    #                     help='residual scaling')
    # parser.add_argument('--px', type=int, default=1,
    #                     help='pixshuff downup scale (in RCAN-PS)')
    # parser.add_argument('--n_awru', type=int, default=4,
    #                 help='number of n_awru in one LFB')
    # parser.add_argument('--block_feats', type=int, default=128,
    #                 help='number of feature maps')
    # parser.add_argument('--dilation', action='store_true',
    #                     help='use dilated convolution')
    # parser.add_argument('--shift_mean', default=True,
    #                     help='subtract pixel mean from the input')

    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    # model = RCAN(opt)
    # model = ercan.ERCAN(opt).to(device)
    model = rdn.RDN(scale_factor=opt.scale,
                num_channels=3,
                num_features=opt.num_features,
                growth_rate=opt.growth_rate,
                num_blocks=opt.num_blocks,
                num_layers=opt.num_layers).to(device)
    # model = idn.IDN(opt).to(device)
    # model = drrn.DRRN(B=opt.B, U=opt.U, num_features=opt.num_features).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(opt.weights_path, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model = model.to(device)
    model.eval()

    # 单张图片预测
    # filename = os.path.basename(opt.image_path).split('.')[0]
    # input = pil_image.open(opt.image_path).convert('RGB')
    # input = transforms.ToTensor()(input).unsqueeze(0).to(device)
    # with torch.no_grad():
    #     pred = model(input)
    # output = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
    # output = pil_image.fromarray(output, mode='RGB')
    # output.save(os.path.join(opt.outputs_dir, '{}_x{}_{}.png'.format(filename, opt.scale, opt.arch)))

    # 对文件夹下的文件进行预测
    sub_dirs = os.listdir(opt.image_path)
    sub_dirs = [os.path.join(opt.image_path, val) for val in sub_dirs]

    for line in sub_dirs:
        search_dir = os.path.join(line, 'LR')
        output_dir = os.path.join(line, 'HR')
        files = os.listdir(search_dir)
        for index, file in enumerate(files):
            filename = os.path.join(search_dir,file)
            input = pil_image.open(filename).convert('RGB')
            input = transforms.ToTensor()(input).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(input)
            output = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
            output = pil_image.fromarray(output, mode='RGB')
            output.save(os.path.join(output_dir, str(index)+'.bmp'.format(filename, opt.scale, opt.arch)))

