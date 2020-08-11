# Copyright (c) Lixin YANG. All Rights Reserved.
r"""
Networks for heatmap estimation from RGB images using Hourglass Network
"Stacked Hourglass Networks for Human Pose Estimation", Alejandro Newell, Kaiyu Yang, Jia Deng, ECCV 2016
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored, cprint

from bihand.models.bases.bottleneck import BottleneckBlock
from bihand.models.bases.hourglass import HourglassBisected


class SeedNet(nn.Module):
    def __init__(
        self,
        nstacks=2,
        nblocks=1,
        njoints=21,
        block=BottleneckBlock,
    ):
        super(SeedNet, self).__init__()
        self.njoints  = njoints
        self.nstacks  = nstacks
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.layer1 = self._make_residual(block, nblocks, self.in_planes, 2*self.in_planes)
        # current self.in_planes is 64 * 2 = 128
        self.layer2 = self._make_residual(block, nblocks, self.in_planes, 2*self.in_planes)
        # current self.in_planes is 128 * 2 = 256
        self.layer3 = self._make_residual(block, nblocks, self.in_planes, self.in_planes)

        ch = self.in_planes # 256

        hg2b, res1, res2, fc1, _fc1, fc2, _fc2= [],[],[],[],[],[],[]
        hm, _hm, mask, _mask = [], [], [], []
        for i in range(nstacks): # 2
            hg2b.append(HourglassBisected(block, nblocks, ch, depth=4))
            res1.append(self._make_residual(block, nblocks, ch, ch))
            res2.append(self._make_residual(block, nblocks, ch, ch))
            fc1.append(self._make_fc(ch, ch))
            fc2.append(self._make_fc(ch, ch))
            hm.append(nn.Conv2d(ch, njoints, kernel_size=1, bias=True))
            mask.append(nn.Conv2d(ch, 1, kernel_size=1, bias=True))

            if i < nstacks-1:
                _fc1.append(nn.Conv2d(ch, ch, kernel_size=1, bias=False))
                _fc2.append(nn.Conv2d(ch, ch, kernel_size=1, bias=False))
                _hm.append(nn.Conv2d(njoints, ch, kernel_size=1, bias=False))
                _mask.append(nn.Conv2d(1, ch, kernel_size=1, bias=False))

        self.hg2b  = nn.ModuleList(hg2b) # hgs: hourglass stack
        self.res1  = nn.ModuleList(res1)
        self.fc1   = nn.ModuleList(fc1)
        self._fc1  = nn.ModuleList(_fc1)
        self.res2  = nn.ModuleList(res2)
        self.fc2   = nn.ModuleList(fc2)
        self._fc2  = nn.ModuleList(_fc2)
        self.hm   = nn.ModuleList(hm)
        self._hm  = nn.ModuleList(_hm)
        self.mask  = nn.ModuleList(mask)
        self._mask = nn.ModuleList(_mask)


    def _make_fc(self, in_planes, out_planes):
        bn = nn.BatchNorm2d(in_planes)
        conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        return nn.Sequential(conv, bn, self.relu)


    def _make_residual(self, block, nblocks, in_planes, out_planes):
        layers = []
        layers.append( block( in_planes, out_planes) )
        self.in_planes = out_planes
        for i in range(1, nblocks):
            layers.append(block( self.in_planes, out_planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        l_hm, l_mask, l_enc = [], [], []
        x = self.conv1(x) # x: (N,64,128,128)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x) # x: (N,128,64,64)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.nstacks): #2
            y_1, y_2, _ = self.hg2b[i](x)

            y_1 = self.res1[i](y_1)
            y_1 = self.fc1[i](y_1)
            est_hm = self.hm[i](y_1)
            l_hm.append(est_hm)

            y_2 = self.res2[i](y_2)
            y_2 = self.fc2[i](y_2)
            est_mask = self.mask[i](y_2)
            l_mask.append(est_mask)

            if i < self.nstacks-1:
                _fc1 = self._fc1[i](y_1)
                _hm = self._hm[i](est_hm)
                _fc2 = self._fc2[i](y_2)
                _mask = self._mask[i](est_mask)
                x = x + _fc1 + _fc2 + _hm + _mask
                l_enc.append(x)
            else:
                l_enc.append(x + y_1 + y_2)
        assert len(l_hm) == self.nstacks
        return l_hm, l_mask, l_enc