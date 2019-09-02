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
# from model.awsrns import make_model # 可用
from model.drca import make_model
from tqdm import tqdm
import argparse

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer():

    def __init__(self, opt):
        self.opt = opt

    def train(self):

        model = make_model(self.opt)
        model = model.to(device)
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=self.opt.lr)

        dataset = Dataset(self.opt.images_dir, self.opt.crop_size, self.opt.scale)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.opt.batch_size,
                                shuffle=True,
                                num_workers=self.opt.threads,
                                pin_memory=True,
                                drop_last=True)

        for epoch in range(self.opt.num_epochs):
            epoch_losses = AverageMeter()
            with tqdm(total=(len(dataset) - len(dataset) % self.opt.batch_size)) as _tqdm:
                _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, self.opt.num_epochs))
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

            torch.save(model.state_dict(), os.path.join(self.opt.outputs_dir, '{}_epoch_{}.pth'.format(self.opt.arch, epoch)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='DRCA',help='EDSR,RCAN,DDBPN,ERCAN,AWSRN,DRCA')
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--scale', type=int, required=True)
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--num_rg', type=int, default=10)
    parser.add_argument('--num_rcab', type=int, default=20)
    parser.add_argument('--reduction', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
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

    trainer = Trainer(opt)
    trainer.train()

