import os
import shutil
import torch
import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from termcolor import colored, cprint
import bihand.utils.func as func
from collections import OrderedDict


def print_args(args):
    opts = vars(args)
    cprint("{:>30}  Options  {}".format("=" * 15, "=" * 15), 'yellow')
    for k, v in sorted(opts.items()):
        print("{:>30}  :  {}".format(k, v))
    cprint("{:>30}  Options  {}".format("=" * 15, "=" * 15), 'yellow')


def param_count(net):
    return sum(p.numel() for p in net.parameters()) / 1e6


def save_checkpoint(
        state,
        checkpoint='checkpoint',
        filename='checkpoint.pth.tar',
        snapshot=None,
        is_best=False
):
    # preds = to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    fileprefix = filename.split('.')[0]
    torch.save(state, filepath)

    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(
            filepath,
            os.path.join(
                checkpoint,
                '{}_{}.pth.tar'.format(fileprefix, state['epoch'])
            )
        )

    if is_best:
        shutil.copyfile(
            filepath,
            os.path.join(
                checkpoint,
                '{}_best.pth.tar'.format(fileprefix)
            )
        )


def load_checkpoint(model:torch.nn.Module, checkpoint_pth):
    checkpoint = torch.load(checkpoint_pth)
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict_old = checkpoint["state_dict"]
        state_dict = OrderedDict()
        # delete 'module.' because it is saved from DataParallel module
        for key in state_dict_old.keys():
            if key.startswith("module."):
                state_dict[key[7:]] = state_dict_old[key]  # delete "module." (in nn.parallel)
            else:
                state_dict[key] = state_dict_old[key]
    else:
         raise RuntimeError(f"=> No state_dict found in checkpoint file {checkpoint_pth}")

    model.load_state_dict(state_dict, strict=True)
    print(colored('loaded {}'.format(checkpoint_pth), 'cyan'))


def save_pred(preds, checkpoint='checkpoint', filename='preds_valid.mat'):
    preds = func.to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    scipy.io.savemat(filepath, mdict={'preds': preds})


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        print("adjust learning rate to: %.3e" % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def adjust_learning_rate_in_group(optimizer, group_id, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        print("adjust learning rate of group %d to: %.3e" % (group_id, lr))
        optimizer.param_groups[group_id]['lr'] = lr
    return lr


def resume_learning_rate(optimizer, epoch, lr, schedule, gamma):
    for decay_id in schedule:
        if epoch > decay_id:
            lr *= gamma
    print("adjust learning rate to: %.3e" % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def resume_learning_rate_in_group(optimizer, group_id, epoch, lr, schedule, gamma):
    for decay_id in schedule:
        if epoch > decay_id:
            lr *= gamma
    print("adjust learning rate of group %d to: %.3e" % (group_id, lr))
    optimizer.param_groups[group_id]['lr'] = lr
    return lr
