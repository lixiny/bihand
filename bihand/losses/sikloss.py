import numpy as np
import torch
from torch import nn
import torch.nn.functional as torch_f
import bihand.utils.quatutils as quatutils
import bihand.config as cfg


class SIKLoss:
    def __init__(
            self,
            lambda_quat=1.0,
            lambda_joint=1.0,
            lambda_shape=1.0
    ):
        self.lambda_quat = lambda_quat
        self.lambda_joint = lambda_joint
        self.lambda_shape = lambda_shape

    def compute_loss(self, preds, targs):
        batch_size = targs['batch_size']
        final_loss = torch.Tensor([0]).cuda()
        invk_losses = {}

        quat_total_loss = torch.Tensor([0]).cuda()

        predquat = preds['quat']  # (B, 16, 4)
        quat_norm = torch.norm(predquat, dim=-1, keepdim=True)  # (B, 16, 1)
        norm_loss = 100.0 * torch_f.mse_loss(
            quat_norm,
            torch.ones_like(quat_norm)
        )

        if self.lambda_quat:
            targquat = targs['quat']
            predquat = quatutils.normalize_quaternion(preds['quat'])
            l2_loss = torch_f.mse_loss(predquat, targquat)

            inv_predquat = quatutils.quaternion_inv(predquat)
            real_part = quatutils.quaternion_mul(
                targquat, inv_predquat
            )[..., 0]  # (B, 16)
            cos_loss = torch_f.l1_loss(
                real_part,
                torch.ones_like(real_part)
            )
            quat_total_loss = norm_loss + l2_loss + cos_loss
            final_loss += self.lambda_quat * quat_total_loss
        else:
            quat_total_loss = norm_loss
            final_loss += quat_total_loss
            l2_loss, cos_loss = None, None

        invk_losses['quat_norm'] = norm_loss
        invk_losses['quat_l2'] = l2_loss
        invk_losses['quat_cos'] = cos_loss

        if self.lambda_joint:
            joint_loss = torch_f.mse_loss(
                1000 * preds['jointRS'] * targs['joint_bone'].unsqueeze(1),
                1000 * targs['jointRS'] * targs['joint_bone'].unsqueeze(1)
            )
            final_loss += self.lambda_joint * joint_loss
        else:
            joint_loss = None
        invk_losses["joint"] = joint_loss

        if self.lambda_shape:
            shape_reg_loss = 10.0 * torch_f.mse_loss(
                preds["beta"],
                torch.zeros_like(preds["beta"])
            )
            # calculate kinematic chain len on _predjoint_:
            predjointRS = preds['jointRS']
            predkin_chain = [
                (
                        predjointRS[:, i, :] -
                        predjointRS[:, cfg.SNAP_PARENT[i], :]
                ) for i in range(21)
            ]
            predkin_chain = torch.stack(predkin_chain[1:], dim=1)  # (B,20,3)
            predkin_len = torch.norm(predkin_chain, dim=-1, keepdim=True)  # (B,20,1)
            kin_len_loss = torch_f.mse_loss(
                predkin_len.reshape(batch_size, -1),
                targs['kin_len'].reshape(batch_size, -1)
            )
            shape_total_loss = kin_len_loss + shape_reg_loss
            final_loss += self.lambda_shape * shape_total_loss
        else:
            shape_reg_loss, kin_len_loss = None, None
        invk_losses['shape_reg'] = shape_reg_loss
        invk_losses['kin_len'] = kin_len_loss

        return final_loss, invk_losses
