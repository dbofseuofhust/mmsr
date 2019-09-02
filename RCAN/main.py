import argparse
import os
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import sys
sys.path.append('/home/dongbin/DL/RCAN/')

# Github地址
# https://github.com/yjn870/RCAN-pytorch
# 论文解读
# https://blog.csdn.net/aaa958099161/article/details/82836846

from model import RCAN
import ercan
import rdn
import idn
import drrn
from dataset import Dataset
from utils import AverageMeter
from ssim import MS_SSIM

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # RCAN特有的
    # parser.add_argument('--arch', type=str, default='RCAN')
    # parser.add_argument('--images_dir', type=str, required=True)
    # parser.add_argument('--outputs_dir', type=str, required=True)
    # parser.add_argument('--scale', type=int, required=True)
    # parser.add_argument('--num_features', type=int, default=64)
    # parser.add_argument('--num_rg', type=int, default=10)
    # parser.add_argument('--num_rcab', type=int, default=20)
    # parser.add_argument('--reduction', type=int, default=16)
    # parser.add_argument('--crop_size', type=int, default=224)
    # parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--num_epochs', type=int, default=60)
    # parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--threads', type=int, default=8)
    # parser.add_argument('--seed', type=int, default=123)

    # RDN特有的
    # https://github.com/yjn870/RDN-pytorch
    parser.add_argument('--arch', type=str, default='RDN')
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--growth_rate', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=16)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=800)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--crop_size', type=int, default=128)

    # IDN特有的
    # https://github.com/yjn870/IDN-pytorch
    # parser.add_argument('--arch', type=str, default='IDN')
    # parser.add_argument('--images_dir', type=str, required=True)
    # parser.add_argument('--outputs_dir', type=str, required=True)
    # parser.add_argument('--scale', type=int, default=4)
    # parser.add_argument('--num_features', type=int, default=64)
    # parser.add_argument('--d', type=int, default=16)
    # parser.add_argument('--s', type=int, default=4)
    # parser.add_argument('--loss', type=str, default='l1')
    # parser.add_argument('--patch_size', type=int, default=40)
    # parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--num_epochs', type=int, default=20)
    # parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--threads', type=int, default=8)
    # parser.add_argument('--seed', type=int, default=123)
    # parser.add_argument('--crop_size', type=int, default=224)

    # DRRN特有的
    # https://github.com/yjn870/DRRN-pytorch
    # parser.add_argument('--arch', type=str, default='DRRN')
    # parser.add_argument('--scale', type=int, required=True)
    # parser.add_argument('--images_dir', type=str, required=True)
    # parser.add_argument('--outputs_dir', type=str, required=True)
    # parser.add_argument('--crop_size', type=int, default=224)
    # parser.add_argument('--B', type=int, default=1)
    # parser.add_argument('--U', type=int, default=9)
    # parser.add_argument('--num_features', type=int, default=128)
    # parser.add_argument('--lr', type=float, default=0.1)
    # parser.add_argument('--clip_grad', type=float, default=0.01)
    # parser.add_argument('--batch_size', type=int, default=128)
    # parser.add_argument('--num_epochs', type=int, default=50)
    # parser.add_argument('--seed', type=int, default=123)
    # parser.add_argument('--threads', type=int, default=8)


    # ERCAN特有的
    # parser.add_argument('--arch', type=str, default='ERCAN')
    # parser.add_argument('--images_dir', type=str, required=True)
    # parser.add_argument('--outputs_dir', type=str, required=True)
    # parser.add_argument('--scale', type=int, required=True)
    # parser.add_argument('--num_features', type=int, default=64)
    # parser.add_argument('--num_rg', type=int, default=10)
    # parser.add_argument('--num_rcab', type=int, default=20)
    # parser.add_argument('--reduction', type=int, default=16)
    # parser.add_argument('--crop_size', type=int, default=224)
    # parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--num_epochs', type=int, default=60)
    # parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--threads', type=int, default=8)
    # parser.add_argument('--seed', type=int, default=123)
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

    torch.manual_seed(opt.seed)

    # model = RCAN(opt).to(device)
    # model = ercan.ERCAN(opt).to(device)
    model = rdn.RDN(scale_factor=opt.scale,
                num_channels=3,
                num_features=opt.num_features,
                growth_rate=opt.growth_rate,
                num_blocks=opt.num_blocks,
                num_layers=opt.num_layers).to(device)
    # model = idn.IDN(opt).to(device)
    # model = drrn.DRRN(B=opt.B, U=opt.U, num_features=opt.num_features).to(device)

    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    dataset = Dataset(opt.images_dir, opt.crop_size, opt.scale)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.threads,
                            pin_memory=True,
                            drop_last=True)

    for epoch in range(opt.num_epochs):
        epoch_losses = AverageMeter()

        with tqdm(total=(len(dataset) - len(dataset) % opt.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, opt.num_epochs))
            for data in dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                _tqdm.update(len(inputs))

        torch.save(model.state_dict(), os.path.join(opt.outputs_dir, '{}_epoch_{}.pth'.format(opt.arch, epoch)))
