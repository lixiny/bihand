import numpy as np
import torch
from torch import nn
import torch.nn.functional as torch_f

class UpstreamLoss:
    def __init__(
        self,
        lambda_hm=1.0,
        lambda_mask=1.0,
        lambda_joint=1.0,
        lambda_dep=1.0,
    ):
        self.lambda_dep   = lambda_dep
        self.lambda_hm    = lambda_hm
        self.lambda_joint = lambda_joint
        self.lambda_mask  = lambda_mask

    def compute_loss(self, preds, targs, infos):
        hm_veil     = infos['hm_veil']
        dep_veil    = infos['dep_veil']
        ndep_valid  = infos['ndep_valid']
        batch_size  = infos['batch_size']
        final_loss  = torch.Tensor([0]).cuda()
        upstream_losses = {}

        # compute hmloss anyway
        hm_loss = torch.Tensor([0]).cuda()
        if self.lambda_hm:
            hm_veil = hm_veil.unsqueeze(-1)
            for pred_hm in preds['l_hm']:
                njoints = pred_hm.size(1)
                pred_hm = pred_hm.reshape((batch_size, njoints, -1)).split(1,1)
                targ_hm = targs['hm'].reshape((batch_size, njoints, -1)).split(1,1)
                for idx in range(njoints):
                    pred_hmi = pred_hm[idx].squeeze()  # (B, 1, 4096)->(B, 4096)
                    targ_hmi = targ_hm[idx].squeeze()
                    hm_loss += 0.5 * torch_f.mse_loss(
                        pred_hmi.mul(hm_veil[:, idx]), # (B, 4096) mul (B, 1)
                        targ_hmi.mul(hm_veil[:,idx])
                    )
            final_loss += self.lambda_hm * hm_loss
        upstream_losses["ups_hm"] = hm_loss

        # compute mask loss anyway
        mask_loss = torch.Tensor([0]).cuda()
        if self.lambda_mask:
            for pred_mask in preds['l_mask']:
                pred_mask = pred_mask.view(batch_size, -1) #(B, 64x64)
                targ_mask = targs["mask"].view(batch_size, -1)

                mloss = torch_f.binary_cross_entropy_with_logits(
                    pred_mask, targ_mask, reduction="none"
                ) * dep_veil
                mloss = torch.sum(mloss, dim=1) / pred_mask.shape[1]
                mloss = torch.sum(mloss)
                if ndep_valid != 0:
                    mloss = torch.sum(mloss) / ndep_valid
                mask_loss += mloss
            final_loss += self.lambda_mask * mask_loss
        upstream_losses["ups_mask"] = mask_loss

        joint_loss = torch.Tensor([0]).cuda()
        if self.lambda_joint:
            for pred_joint in preds['l_joint']:
                joint_loss += torch_f.mse_loss(
                    pred_joint * 1000.0,
                    targs["joint"] * 1000.0
                )
            final_loss += self.lambda_joint * joint_loss
        upstream_losses["ups_joint"] = joint_loss

        dep_loss = torch.Tensor([0]).cuda()
        if self.lambda_dep:
            for pred_dep in preds["l_dep"]:
                pred_dep = pred_dep.view(batch_size, -1) #(B, 64x64)
                targ_dep = targs["dep"].view(batch_size, -1)
                dloss = torch_f.smooth_l1_loss(
                    pred_dep.mul(dep_veil),
                    targ_dep.mul(dep_veil)
                )
                dep_loss += dloss
            final_loss += self.lambda_dep * dep_loss
        upstream_losses["ups_dep"] = dep_loss

        upstream_losses["ups_total_loss"] = final_loss
        return final_loss, upstream_losses







