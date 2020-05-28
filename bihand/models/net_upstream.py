from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn

from .net_seed import SeedNet
from .net_lift import LiftNet
from bihand.utils.handutils import uvd2xyz
from bihand.utils.misc import param_count
from termcolor import cprint



class IntegralPose(nn.Module):
    def __init__(self,):
        super(IntegralPose, self).__init__()

    def forward(self, hm3d):
        """integral heatmap3d to uvd bihand

        Arguments:
            hm3d {tensor (B, 21, D, H, W)}

        Returns:
            uvd {tensor (B, 21, 3)}
        """

        d_accu = torch.sum(hm3d, dim=[3,4])
        v_accu = torch.sum(hm3d, dim=[2,4])
        u_accu = torch.sum(hm3d, dim=[2,3])

        weightd = torch.arange(d_accu.shape[-1], dtype=d_accu.dtype, device=d_accu.device) / d_accu.shape[-1]
        weightv = torch.arange(v_accu.shape[-1], dtype=v_accu.dtype, device=v_accu.device) / v_accu.shape[-1]
        weightu = torch.arange(u_accu.shape[-1], dtype=u_accu.dtype, device=u_accu.device) / u_accu.shape[-1]

        d_ = d_accu.mul(weightd)
        v_ = v_accu.mul(weightv)
        u_ = u_accu.mul(weightu)

        d_ = torch.sum(d_, dim=-1, keepdim=True)
        v_ = torch.sum(v_, dim=-1, keepdim=True)
        u_ = torch.sum(u_, dim=-1, keepdim=True)

        uvd = torch.cat([u_,v_,d_], dim=-1)
        return uvd


class NetUpstream(nn.Module):
    def __init__(
        self,
        net_modules,
        hg_stacks=2,
        hg_blocks=1,
        njoints=21,
        inp_res=256,
        out_hm_res=64,
        out_dep_res=64,
    ):
        super(NetUpstream, self).__init__()
        self.inp_res       = inp_res # 256
        self.out_hm_res    = out_hm_res # 64
        self.out_dep_res   = out_dep_res # 64
        self.njoints       = njoints
        self.integral_pose = IntegralPose()
        self.net_modules = ['seed'] + [
            'lift' if ('lift' in net_modules) else ''
        ]

        self.seednet = SeedNet(
            nstacks=hg_stacks,
            nblocks=hg_blocks,
            njoints=njoints
        )
        cprint('params seednet: {:.3f}M'.format(
            param_count(self.seednet)), 'green')

        if 'lift' in self.net_modules:
            self.liftnet = LiftNet(
                nstacks=hg_stacks,
                nblocks=hg_blocks,
                njoints=njoints,
            )
            cprint('params liftnet: {:.3f}M'.format(
                param_count(self.liftnet)), 'green')

    def forward(self, img, infos):
        """[summary]
            x (B, 3, H=256, W=256)
        Returns:
            [type] -- [description]
        """

        joint_root = infos['joint_root']
        joint_bone = infos['joint_bone']

        l_hm, l_mask, l_enc = self.seednet(img)

        l_uvd = []
        l_dep = []
        l_joint = []
        latent = 0

        if 'lift' in self.net_modules:
            (
                l_hm3d,
                l_dep,
                latent
            ) = self.liftnet(
                l_hm[-1],
                l_mask[-1],
                l_enc[-1]
            )

            for i in range(len(l_hm3d)):
                hm3d = l_hm3d[i]
                uvd = self.integral_pose(hm3d)
                l_uvd.append(uvd)
            for i in range(len(l_uvd)):
                joint = uvd2xyz(
                    l_uvd[i], joint_root, joint_bone,
                    intr=infos['intr'], mode='persp'
                )
                l_joint.append(joint)

        ups_result = {
            "l_hm"   : l_hm,
            "l_mask" : l_mask,
            "l_uvd"  : l_uvd,
            "l_joint": l_joint,
            "l_dep"  : l_dep,
        }
        ups_enc = {
            "enc": l_enc[-1],
            "latent": latent
        }

        return ups_result, ups_enc
