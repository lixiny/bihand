from __future__ import print_function, absolute_import

import os, sys
import argparse
import torch
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from termcolor import colored, cprint
import signal
import matplotlib.pyplot as plt

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
# There is BN issue for early version of PyTorch
# see https://github.com/bearpaw/pytorch-pose/issues/33

import bihand.models as models
import bihand.utils.misc as misc
import bihand.utils.handutils as handutils
from progress.progress.bar import Bar
from bihand.utils.eval.zimeval import EvalUtil
from bihand.datasets.handataset import HandDataset
from bihand.vis.drawer import HandDrawer

def main(args):
    if (
        not args.fine_tune
        or not args.fine_tune in ['rhd', 'stb']
    ):
        raise Exception('expect --fine_tune in [rhd|stb], got {}'
                        .format(args.fine_tune))
    args.datasets = [args.fine_tune, ]
    misc.print_args(args)
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)
    print("\nCREATE NETWORK")
    model = models.NetBiHand(
        net_modules=['seed','lift','sik'],
        njoints=args.njoints,
        inp_res=256,
        out_hm_res=64,
        out_dep_res=64,
        upstream_hg_stacks=args.hg_stacks,
        upstream_hg_blocks=args.hg_blocks,
    )
    model = model.to(device)

    # define loss function (criterion) and optimizer

    print("\nCREATE TESTSET")
    val_dataset = HandDataset(
        data_split='test',
        train=False,
        subset_name=args.datasets,
        data_root=args.data_root,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    print("Total test dataset size: {}".format(len(val_dataset)))
    print("\nLOAD CHECKPOINT")
    model.load_checkpoints(
        ckp_seednet=os.path.join(args.checkpoint, 'ckp_seednet_all.pth.tar'),
        ckp_liftnet=os.path.join(args.checkpoint, args.fine_tune,
                                 'ckp_liftnet_{}.pth.tar'.format(args.fine_tune)),
        ckp_siknet=os.path.join(args.checkpoint, args.fine_tune,
                                'ckp_siknet_{}.pth.tar'.format(args.fine_tune))
    )

    validate(val_loader, model, vis=args.vis)
    return 0


def one_forward_pass(metas, model):
    """ prepare infos """
    joint_root = metas['joint_root'].to(device, non_blocking=True)  # (B, 3)
    joint_bone = metas['joint_bone'].to(device, non_blocking=True)  # (B, 1)
    intr = metas['intr'].to(device, non_blocking=True)
    hm_veil = metas['hm_veil'].to(device, non_blocking=True)
    dep_veil = metas['dep_veil'].to(device, non_blocking=True)  # (B, 1)
    ndep_valid = torch.sum(dep_veil).item()
    infos = {
        'joint_root': joint_root,
        'intr': intr,
        'joint_bone': joint_bone,
        'hm_veil': hm_veil,
        'dep_veil': dep_veil,
        'batch_size': joint_root.shape[0],
        'ndep_valid': ndep_valid,
    }
    ''' prepare targets '''
    clr = metas['clr'].to(device, non_blocking=True)
    hm = metas['hm'].to(device, non_blocking=True)
    dep = metas['dep'].to(device, non_blocking=True)  # (B, 64, 64)
    kp2d = metas['kp2d'].to(device, non_blocking=True)
    joint = metas['joint'].to(device, non_blocking=True)  # (B, 21, 3)
    jointR = metas['jointR'].to(device, non_blocking=True)
    mask = metas['mask'].to(device, non_blocking=True)  # (B, 64, 64)

    targets = {
        'clr': clr,
        'hm': hm,
        'joint': joint,
        'kp2d': kp2d,
        'jointR': jointR,
        'dep': dep,
        'mask': mask,
    }
    ''' ----------------  Forward Pass  ---------------- '''
    results = model(clr, infos)
    ''' ----------------  Forward End   ---------------- '''

    return results, {**targets, **infos}


def validate(val_loader, model, vis=False):
    # switch to evaluate mode
    evaluator = EvalUtil()
    drawer = HandDrawer(reslu=256)
    model.eval()
    if vis:
        drawer.daemon = True
        drawer.start()
    bar = Bar(colored("EVAL", color='yellow'), max=len(val_loader))
    with torch.no_grad():
        for i, metas in enumerate(val_loader):
            results, targets = one_forward_pass(metas, model)
            pred_jointRS = results['jointRS'] # B, 21, 3
            targ_joint = targets['joint']  # B, 21, 3
            joint_bone = targets['joint_bone'].unsqueeze(1)  # B, 21, 1
            joint_root = targets['joint_root'].unsqueeze(1)  # B, 21, 3
            pred_joint = pred_jointRS * joint_bone + joint_root # B, 21, 3

            # quantitative
            for targj, predj in zip(targ_joint, pred_joint):
                evaluator.feed(targj * 1000.0, predj * 1000.0)

            pck20 = evaluator.get_pck_all(20)
            pck30 = evaluator.get_pck_all(30)
            pck40 = evaluator.get_pck_all(40)
            bar.suffix = (
                '({batch}/{size}) '
                'pck20avg: {pck20:.3f} | '
                'pck30avg: {pck30:.3f} | '
                'pck40avg: {pck40:.3f} | '
            ).format(
                batch=i + 1,
                size=len(val_loader),
                pck20=pck20,
                pck30=pck30,
                pck40=pck40,
            )
            bar.next()

            ## visualize
            if vis: # little bit time comsuming
                clr = targets['clr'].detach().cpu()
                uvd = handutils.xyz2uvd(
                    pred_joint, targets['joint_root'], targets['joint_bone'],
                    intr=targets['intr'], mode='persp'
                ).detach().cpu()
                uv = uvd[:, :, :2] * clr.shape[-1]

                vertsRS = results['vertsRS'].detach().cpu()
                mean_bone_len = torch.Tensor([0.1]) # 0.1 m
                fixed_root = torch.Tensor([0.0, 0.0, 0.5]) # 0.5 m
                verts = vertsRS * mean_bone_len + fixed_root
                drawer.feed(clr, verts, uv)

        bar.finish()
        drawer.set_stop()
        (
            _1, _2, _3,
            auc_all,
            pck_curve_all,
            thresholds
        ) = evaluator.get_measures(
            20, 50, 20
        )
        print("AUC all: {}".format(auc_all))

    return auc_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Train BiHand')

    ### Adjustable ###
    parser.add_argument(
        '-dr',
        '--data_root',
        type=str,
        default='data',
        help='dataset root directory'
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=['stb', 'rhd'],
        type=str,
        help="sub modules contained in model"
    )
    parser.add_argument(
        '--fine_tune',
        type=str,
        default='',
        help='fine tune dataset. should in: [rhd|stb|freihand]'
    )
    parser.add_argument(
        '-ckp',
        '--checkpoint',
        default='checkpoints',
        type=str,
        metavar='PATH',
        help='path to save checkpoint (default: checkpoint)'
    )
    parser.add_argument(
        '-j', '--workers',
        default=8,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 8)'
    )
    parser.add_argument(
        '-b', '--batch_size',
        default=16,
        type=int,
        metavar='N',
        help='batch size'
    )
    parser.add_argument(
        '--vis',
        dest='vis',
        action='store_true',
        help='visualization'
    )

    # Model Structure, YOU SHOULDN'T CHANGE BELOW
    ## hourglass:
    parser.add_argument(
        '-hgs',
        '--hg-stacks',
        default=2,
        type=int,
        metavar='N',
        help='Number of hourglasses to stack'
    )
    parser.add_argument(
        '-hgb',
        '--hg-blocks',
        default=1,
        type=int,
        metavar='N',
        help='Number of residual modules at each location in the hourglass'
    )
    parser.add_argument(
        '-nj',
        '--njoints',
        default=21,
        type=int,
        metavar='N',
        help='Number of heatmaps calsses (hand joints) to predict in the hourglass'
    )
    parser.add_argument(
        "--net_modules",
        nargs="+",
        default=['seed', 'lift', 'sik'],
        type=str,
        help="sub modules contained in model"
    )
    main(parser.parse_args())
