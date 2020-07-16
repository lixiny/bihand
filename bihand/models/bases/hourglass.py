import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored, cprint

from bihand.models.bases.bottleneck import BottleneckBlock


class Hourglass(nn.Module):
    def __init__(self, block, nblocks, in_planes, depth=4):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.hg = self._make_hourglass(block, nblocks, in_planes, depth)

    def _make_hourglass(self, block, nblocks, in_planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, nblocks, in_planes))
            if i == 0:
                res.append(self._make_residual(block, nblocks, in_planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _make_residual(self, block, nblocks, in_planes):
        layers = []
        for i in range(0, nblocks):
            layers.append(block(in_planes, in_planes))
        return nn.Sequential(*layers)

    def _hourglass_foward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hourglass_foward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hourglass_foward(self.depth, x)


class HourglassBisected(nn.Module):
    def __init__(
            self,
            block,
            nblocks,
            in_planes,
            depth=4
    ):
        super(HourglassBisected, self).__init__()
        self.depth = depth
        self.hg = self._make_hourglass(block, nblocks, in_planes, depth)

    def _make_hourglass(self, block, nblocks, in_planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                _res = []
                if j == 1:
                    _res.append(self._make_residual(block, nblocks, in_planes))
                else:
                    _res.append(self._make_residual(block, nblocks, in_planes))
                    _res.append(self._make_residual(block, nblocks, in_planes))

                res.append(nn.ModuleList(_res))

            if i == 0:
                _res = []
                _res.append(self._make_residual(block, nblocks, in_planes))
                _res.append(self._make_residual(block, nblocks, in_planes))
                res.append(nn.ModuleList(_res))

            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _make_residual(self, block, nblocks, in_planes):
        layers = []
        for i in range(0, nblocks):
            layers.append(block(in_planes, in_planes))
        return nn.Sequential(*layers)

    def _hourglass_foward(self, n, x):
        up1_1 = self.hg[n - 1][0][0](x)
        up1_2 = self.hg[n - 1][0][1](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1][0](low1)

        if n > 1:
            low2_1, low2_2, latent = self._hourglass_foward(n - 1, low1)
        else:
            latent = low1
            low2_1 = self.hg[n - 1][3][0](low1)
            low2_2 = self.hg[n - 1][3][1](low1)

        low3_1 = self.hg[n - 1][2][0](low2_1)
        low3_2 = self.hg[n - 1][2][1](low2_2)

        up2_1 = F.interpolate(low3_1, scale_factor=2)
        up2_2 = F.interpolate(low3_2, scale_factor=2)
        out_1 = up1_1 + up2_1
        out_2 = up1_2 + up2_2

        return out_1, out_2, latent

    def forward(self, x):
        return self._hourglass_foward(self.depth, x)


class NetStackedHourglass3d(nn.Module):
    def __init__(
            self,
            nstacks=2,
            nblocks=1,
            nclasses=21,
            block=BottleneckBlock,
    ):
        super(NetStackedHourglass3d, self).__init__()
        self.nclasses = nclasses
        self.nstacks = nstacks
        self.relu = nn.ReLU(inplace=True)
        self.in_planes = 256
        ch = 256
        z_res = [32, 64]

        # encode previous heatmaps
        self._score_prev = nn.Conv2d(nclasses, self.in_planes, kernel_size=1, bias=True)
        self._fc_prev = nn.Conv2d(self.in_planes, self.in_planes, kernel_size=1, bias=True)

        hg3ds, res, fc, _fc, score, _score = [], [], [], [], [], []
        for i in range(nstacks):
            hg3ds.append(Hourglass(block, nblocks, ch, depth=4))
            res.append(self._make_residual(block, nblocks, ch, ch))
            fc.append(self._make_residual(block, nblocks, ch, ch))
            score.append(
                nn.Sequential(
                    nn.Conv2d(ch, z_res[i] * nclasses, kernel_size=1, bias=True),
                    self.relu,
                )
            )

            if i < nstacks - 1:
                _fc.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                _score.append(nn.Conv2d(z_res[i] * nclasses, ch, kernel_size=1, bias=True))

        self.z_res = z_res
        self.hg3ds = nn.ModuleList(hg3ds)  # hgs: hourglass stack
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self._fc = nn.ModuleList(_fc)
        self.score = nn.ModuleList(score)
        self._score = nn.ModuleList(_score)

    def _make_residual(self, block, nblocks, in_planes, out_planes):
        layers = []
        layers.append(block(in_planes, out_planes))
        self.in_planes = out_planes
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, out_planes))
        return nn.Sequential(*layers)

    def forward(self, l_est_hm, l_hm_enc):
        """stackhourglass to predict 3d heatmap

        Arguments:
            x {torch} -- (B, 256, 64, 64)

        Returns:
            out3d --
                    torch.Size([B, 21, 32, 64, 64])
                    torch.Size([B, 21, 64, 64, 64])]
            hm3d_enc --
                    torch.Size([16, 256, 64, 64])
                    torch.Size([16, 256, 64, 64])]
        """

        # same as: x = _fc + _score
        x = self._fc_prev(l_hm_enc[-1]) + \
            self._score_prev(l_est_hm[-1])

        out3d = []
        hm3d_enc = []

        for i in range(self.nstacks):  # 2
            y = self.hg3ds[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            score_out = score.view(
                score.shape[0],  # B
                self.nclasses,  # 21
                self.z_res[i],  # z_res (32, 64)
                score.shape[-2],  # H=64
                score.shape[-1]  # W=64
            )
            score_out = score_out / (
                    torch.sum(score_out, dim=[2, 3, 4], keepdim=True) + 1e-6
            )
            out3d.append(score_out)
            if i < self.nstacks - 1:
                _fc = self._fc[i](y)
                _socre = self._score[i](score)
                x = x + _fc + _socre
                hm3d_enc.append(x)
            else:
                hm3d_enc.append(y)
        return out3d, hm3d_enc


### classical
class NetStackedHourglass(nn.Module):
    def __init__(
            self,
            nstacks=2,
            nblocks=1,
            nclasses=21,
            block=BottleneckBlock
    ):
        super(NetStackedHourglass, self).__init__()
        self.nclasses = nclasses
        self.nstacks = nstacks
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.layer1 = self._make_residual(block, nblocks, self.in_planes, 2 * self.in_planes)
        # current self.in_planes is 64 * 2 = 128
        self.layer2 = self._make_residual(block, nblocks, self.in_planes, 2 * self.in_planes)
        # current self.in_planes is 128 * 2 = 256
        self.layer3 = self._make_residual(block, nblocks, self.in_planes, self.in_planes)

        ch = self.in_planes  # 256
        self.nfeats = ch

        hgs, res, fc, _fc, score, _score = [], [], [], [], [], []
        for i in range(nstacks):  # stacking the hourglass
            hgs.append(Hourglass(block, nblocks, ch, depth=4))
            res.append(self._make_residual(block, nblocks, ch, ch))
            fc.append(self._make_residual(block, nblocks, ch, ch))
            score.append(nn.Conv2d(ch, nclasses, kernel_size=1, bias=True))

            if i < nstacks - 1:
                _fc.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                _score.append(nn.Conv2d(nclasses, ch, kernel_size=1, bias=True))

        self.hgs = nn.ModuleList(hgs)  # hgs: hourglass stack
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)  ### change back to use the pre-trainded
        self._fc = nn.ModuleList(_fc)
        self.score = nn.ModuleList(score)
        self._score = nn.ModuleList(_score)

    def _make_residual(self, block, nblocks, in_planes, out_planes):
        layers = []
        layers.append(block(in_planes, out_planes))
        self.in_planes = out_planes
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, out_planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = []
        hm_enc = []  # heatmaps encoding
        # x: (N,3,256,256)
        x = self.conv1(x)  # x: (N,64,128,128)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # x: (N,128,128,128)
        x = self.maxpool(x)  # x: (N,128,64,64)
        x = self.layer2(x)  # x: (N,256,64,64)
        x = self.layer3(x)  # x: (N,256,64,64)
        hm_enc.append(x)

        for i in range(self.nstacks):
            y = self.hgs[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.nstacks - 1:
                _fc = self._fc[i](y)
                _score = self._score[i](score)
                x = x + _fc + _score
                hm_enc.append(x)
            else:
                hm_enc.append(y)
        return out, hm_enc
