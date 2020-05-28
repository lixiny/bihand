import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_f
import bihand.utils.quatutils as quatutils
import bihand.utils.handutils as handutils
from manopth.manolayer import ManoLayer
import bihand.config as cfg


class SIKNet(nn.Module):
    def __init__(
        self,
        njoints=21,
        dropout=0,
    ):
        super(SIKNet, self).__init__()

        ''' quat '''
        hidden_neurons = [256, 512, 1024, 1024, 512, 256]
        in_neurons = 3 * njoints + 3 * (njoints - 1)
        out_neurons = 16 * 4 # 16 quats
        neurons = [in_neurons] + hidden_neurons

        invk_layers = []
        for layer_idx, (inps, outs) in enumerate(
            zip(neurons[:-1], neurons[1:])
        ):
            if dropout:
                invk_layers.append(nn.Dropout(p=dropout))
            invk_layers.append(nn.Linear(inps, outs))
            invk_layers.append(nn.ReLU())

        invk_layers.append(nn.Linear(neurons[-1], out_neurons))

        self.invk_layers = nn.Sequential(*invk_layers)

        ''' shape '''
        hidden_neurons = [128, 256, 512, 256, 128]
        in_neurons = njoints * 3
        out_neurons = 10
        neurons = [in_neurons] + hidden_neurons

        shapereg_layers = []
        for layer_idx, (inps, outs) in enumerate(
            zip(neurons[:-1], neurons[1:])
        ):
            if dropout:
                shapereg_layers.append(nn.Dropout(p=dropout))
            shapereg_layers.append(nn.Linear(inps, outs))
            shapereg_layers.append(nn.ReLU())

        shapereg_layers.append(nn.Linear(neurons[-1], out_neurons))
        self.shapereg_layers = nn.Sequential(*shapereg_layers)

        self.mano_layer = ManoLayer(
            center_idx=9,
            side="right",
            mano_root="manopth/mano/models",
            use_pca=False,
            flat_hand_mean=True,
        )

        self.ref_bone_link = (0, 9)  # mid mcp
        self.joint_root_idx = 9  # root

    def parse_input(self, joint, infos):
        root = infos['joint_root'].unsqueeze(1)  # (B, 1, 3)
        bone = handutils.get_joint_bone(joint, self.ref_bone_link) # (B, 1)
        bone = bone.unsqueeze(1) #(B,1,1)

        jointR = joint - root  # (B,1,3)
        jointRS = jointR / bone
        kin_chain = [
            jointRS[:, i, :] - jointRS[:, cfg.SNAP_PARENT[i], :]
            for i in range(21)
        ]
        kin_chain = kin_chain[1:] # id 0's parent is itself
        kin_chain = torch.stack(kin_chain, dim=1) #(B, 20, 3)
        len = torch.norm(kin_chain, dim=-1, keepdim=True) #(B, 20, 1)
        kin_chain = kin_chain / (len + 1e-5)
        return jointRS, kin_chain


    def forward(self, pred_jointRS, kin_chain):
        batch_size = pred_jointRS.shape[0]
        x = torch.cat((pred_jointRS, kin_chain), dim=1)
        x = x.reshape(batch_size, -1)
        quat = self.invk_layers(x)
        quat = quat.reshape(batch_size, 16, 4)

        y = pred_jointRS.reshape(batch_size, -1)
        beta = self.shapereg_layers(y)
        quatN = quatutils.normalize_quaternion(quat)
        so3 = quatutils.quaternion_to_angle_axis(quatN)
        so3 = so3.reshape(batch_size, -1)

        vertsR, jointR, _ = self.mano_layer(
            th_pose_coeffs = so3,
            th_betas = beta
        )

        bone_pred = handutils.get_joint_bone(jointR, self.ref_bone_link) # (B, 1)
        bone_pred = bone_pred.unsqueeze(1) # (B,1,1)
        jointRS = jointR / bone_pred
        vertsRS = vertsR / bone_pred

        results = {
            'vertsRS':vertsRS,
            'jointRS':jointRS,
            'quat':quat,
            'beta':beta,
            'so3':so3
        }
        return results







