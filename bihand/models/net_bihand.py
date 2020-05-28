from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import torch
import torch.nn as nn
import bihand.utils.misc as misc
import bihand.utils.handutils as handutils

from termcolor import colored, cprint
from bihand.models.net_upstream import NetUpstream
from bihand.models.net_sik import SIKNet
from bihand.utils.misc import param_count


class NetBiHand(nn.Module):
    def __init__(
        self,
        net_modules=['seed', 'lift', 'sik'],#['seed','lift'],['seed','lift','sik']
        njoints=21,
        inp_res=256,
        out_hm_res=64,
        out_dep_res=64,
        upstream_hg_stacks=2,
        upstream_hg_blocks=1,
    ):

        super(NetBiHand, self).__init__()
        self.inp_res = inp_res
        self.net_modules = net_modules

        self.upstream = NetUpstream (
            net_modules = net_modules,
            hg_stacks=upstream_hg_stacks,
            hg_blocks=upstream_hg_blocks,
            njoints=njoints,
            inp_res=inp_res,
            out_hm_res=out_hm_res,
            out_dep_res=out_dep_res,
        )

        cprint('params upstream: {:.3f}M'.format(
            param_count(self.upstream)), 'green', attrs=['bold'])
        if 'sik' in net_modules:
            self.siknet = SIKNet()
            cprint('params siknet: {:.3f}M'.format(
                param_count(self.siknet)), 'blue', attrs=['bold'])


    def load_checkpoints(
        self,
        ckp_seednet=None,
        ckp_liftnet=None,
        ckp_siknet=None,
    ):
        if (
            ckp_seednet
            and os.path.isfile(ckp_seednet)
            and 'seed' in self.net_modules
        ):
            misc.load_checkpoint(
                self.upstream.seednet, ckp_seednet
            )

        if (
            ckp_liftnet
            and os.path.isfile(ckp_liftnet)
            and 'lift' in self.net_modules
        ):
            misc.load_checkpoint(
                self.upstream.liftnet, ckp_liftnet
            )

        if (
            ckp_siknet
            and os.path.isfile(ckp_siknet)
            and 'sik' in self.net_modules
        ):
            misc.load_checkpoint(
                self.siknet, ckp_siknet
            )

    def forward(self, img, infos):

        ''' ***** net upstream ***** '''
        ups_result, ups_enc = self.upstream(img, infos)

        ''' ***** net mano ***** '''
        sik_reslut = {}
        if 'sik' in self.net_modules:
            joint = ups_result['l_joint'][-1] # B, 21, 3
            jointRS, kin_chain = self.siknet.parse_input(joint, infos)
            sik_reslut = self.siknet(jointRS, kin_chain)

        result = {**ups_result, **sik_reslut}
        return result







