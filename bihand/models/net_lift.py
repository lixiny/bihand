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


class LiftNet(nn.Module):
    def __init__(
            self,
            nstacks=2,
            nblocks=1,
            njoints=21,
            block=BottleneckBlock,
    ):
        super(LiftNet, self).__init__()
        self.njoints = njoints
        self.nstacks = nstacks
        self.relu = nn.ReLU(inplace=True)
        self.in_planes = 256
        ch = self.in_planes
        z_res = [32, 64]

        # encoding previous hm and mask
        self._hm_prev = nn.Conv2d(njoints, self.in_planes, kernel_size=1, bias=True)
        self._mask_prev = nn.Conv2d(1, self.in_planes, kernel_size=1, bias=True)

        hg3d2b, res1, res2, fc1, _fc1, fc2, _fc2 = [], [], [], [], [], [], []
        hm3d, _hm3d, dep, _dep = [], [], [], []
        for i in range(nstacks):
            hg3d2b.append(HourglassBisected(block, nblocks, ch, depth=4))
            res1.append(self._make_residual(block, nblocks, ch, ch))
            res2.append(self._make_residual(block, nblocks, ch, ch))
            fc1.append(self._make_fc(ch, ch))
            fc2.append(self._make_fc(ch, ch))

            hm3d.append(
                nn.Sequential(
                    nn.Conv2d(ch, z_res[i] * njoints, kernel_size=1, bias=True),
                    self.relu,
                )
            )
            dep.append(
                nn.Sequential(
                    nn.Conv2d(ch, 1, kernel_size=1, bias=True),
                    self.relu,
                )
            )

            if i < nstacks - 1:
                _fc1.append(nn.Conv2d(ch, ch, kernel_size=1, bias=False))
                _fc2.append(nn.Conv2d(ch, ch, kernel_size=1, bias=False))
                _hm3d.append(nn.Conv2d(z_res[i] * njoints, ch, kernel_size=1, bias=False))
                _dep.append(nn.Conv2d(1, ch, kernel_size=1, bias=False))

        self.z_res = z_res
        self.hg3d2b = nn.ModuleList(hg3d2b)  # hgs: hourglass stack
        self.res1 = nn.ModuleList(res1)
        self.fc1 = nn.ModuleList(fc1)
        self._fc1 = nn.ModuleList(_fc1)
        self.res2 = nn.ModuleList(res2)
        self.fc2 = nn.ModuleList(fc2)
        self._fc2 = nn.ModuleList(_fc2)
        self.hm3d = nn.ModuleList(hm3d)
        self._hm3d = nn.ModuleList(_hm3d)
        self.dep = nn.ModuleList(dep)
        self._dep = nn.ModuleList(_dep)

    def _make_fc(self, in_planes, out_planes):
        bn = nn.BatchNorm2d(in_planes)
        conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        return nn.Sequential(conv, bn, self.relu)

    def _make_residual(self, block, nblocks, in_planes, out_planes):
        layers = []
        layers.append(block(in_planes, out_planes))
        self.in_planes = out_planes
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, out_planes))
        return nn.Sequential(*layers)

    def forward(self, est_hm, est_mask, enc):
        x = self._hm_prev(est_hm) + self._mask_prev(est_mask) + enc

        l_hm3d, l_dep, l_latent = [], [], []

        for i in range(self.nstacks):
            y_1, y_2, latent = self.hg3d2b[i](x)

            l_latent.append(latent)

            y_1 = self.res1[i](y_1)
            y_1 = self.fc1[i](y_1)
            hm3d = self.hm3d[i](y_1)
            hm3d_out = hm3d.view(
                hm3d.shape[0],  # B
                self.njoints,  # 21
                self.z_res[i],  # z_res (32, 64)
                hm3d.shape[-2],  # H=64
                hm3d.shape[-1]  # W=64
            )
            hm3d_out = hm3d_out / (
                    torch.sum(hm3d_out, dim=[2, 3, 4], keepdim=True) + 1e-6
            )
            l_hm3d.append(hm3d_out)

            y_2 = self.res2[i](y_2)
            y_2 = self.fc2[i](y_2)
            est_dep = self.dep[i](y_2)
            l_dep.append(est_dep)

            if i < self.nstacks - 1:
                _fc1 = self._fc1[i](y_1)
                _hm3d = self._hm3d[i](hm3d)

                _fc2 = self._fc2[i](y_2)
                _dep = self._dep[i](est_dep)
                x = x + _fc1 + _fc2 + _hm3d + _dep

        return l_hm3d, l_dep, l_latent[-1]


def main():
    """
    net = SeedNet()
    x = torch.rand(4, 3, 256, 256).float()
    l_hmpred, l_maskpred, enc = net(x)
    print(l_hmpred[0].shape, l_hmpred[1].shape)
    print(l_maskpred[0].shape, l_maskpred[1].shape)
    """
    net = LiftNet()
    batch_size = 1
    from bihand.utils.misc import param_count
    cprint('params Net3dStage: {:.3f}M'.format(
        param_count(net)), 'yellow')
    est_hm = torch.rand(batch_size, 21, 64, 64).float()
    est_mask = torch.rand(batch_size, 1, 64, 64).float()
    enc = torch.rand(batch_size, 256, 64, 64).float()

    l_est_hm3d, l_est_dep, latent = net(est_hm, est_mask, enc)
    for i in range(len(l_est_hm3d)):
        cprint("stack{}:  {}".format(i, l_est_hm3d[i].shape), 'blue', end='  ')
        cprint("{}".format(l_est_dep[i].shape), 'green')
    latent = latent.view(batch_size, -1)
    print(latent.shape)


if __name__ == "__main__":
    main()
