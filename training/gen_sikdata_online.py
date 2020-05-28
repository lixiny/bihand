from __future__ import print_function, absolute_import

import os
import argparse

import numpy as np
import torch
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import pickle
from termcolor import colored, cprint

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
# There is BN issue for early version of PyTorch
# see https://github.com/bearpaw/pytorch-pose/issues/33

import bihand.models as models
import bihand.utils.misc as misc
from progress.progress.bar import Bar
from bihand.utils.eval.zimeval import EvalUtil
from bihand.datasets.handataset import HandDataset



def main(args):
    if (
        not args.fine_tune
        or not args.fine_tune in ['rhd', 'stb']
    ):
        raise Exception('expect --fine_tune in [rhd|stb], got {}'
                        .format(args.fine_tune))
    if (
        not args.data_split
        or not args.data_split in ['train', 'test']
    ):
        raise Exception('expect --data_split in [train|test], got {}'
                        .format(args.data_split))

    args.datasets = [args.fine_tune, ]
    is_train = (args.data_split == 'train')
    if not is_train: args.epochs = 1
    if not os.path.isdir(args.sik_genpath):
        os.makedirs(args.sik_genpath)
    misc.print_args(args)

    print("\nCREATE NETWORK")
    model = models.NetBiHand(
        net_modules=args.net_modules,
        njoints=21,
        inp_res=256,
        out_hm_res=64,
        out_dep_res=64,
        upstream_hg_stacks=2,
        upstream_hg_blocks=1,
    )
    model = model.to(device)

    criterion = {}

    print("\nCREATE DATASET")
    print(colored(args.datasets, 'yellow',  attrs=['bold']),
          colored(args.data_split, 'blue', attrs=['bold']),
          colored('is_train:{}'.format(is_train), 'red', attrs=['bold']))

    gen_set = HandDataset(
        data_split=args.data_split,
        train=is_train,
        subset_name=args.datasets,
        scale_jittering=0.2,
        center_jettering=0.2,
        max_rot=0.5 * np.pi,
        data_root=args.data_root,
    )

    gen_loader = torch.utils.data.DataLoader(
        gen_set,
        batch_size=args.train_batch,
        shuffle=is_train,
        num_workers=args.workers,
        pin_memory=True
    )

    print("\nLOAD CHECKPOINT")
    model.load_checkpoints(
        ckp_seednet=os.path.join(args.checkpoint, 'ckp_seednet_all.pth.tar'),
        ckp_liftnet=os.path.join(args.checkpoint, args.fine_tune,
                                 'ckp_liftnet_{}.pth.tar'.format(args.fine_tune))
    )
    model = torch.nn.DataParallel(model)
    print("\nUSING {} GPUs".format(torch.cuda.device_count()))
    all_save_at = []
    for i in range(args.epochs):
        saving = validate(gen_loader, model, criterion, args=args)
        save_at = os.path.join(args.sik_genpath, "sik_{}_{}_{}.pkl"
                               .format(args.data_split, args.fine_tune, i))
        with open(save_at, 'wb') as fid:
            pickle.dump(saving, fid)
        fid.close()
        cprint("saving {} epoch data at {}".format(i, save_at), 'yellow')
        all_save_at.append(save_at)

    # merge all temp files
    allJointGt_, allJointImpl_= [], []
    for save_at in all_save_at:
        with open(save_at, 'rb') as fid:
            raw = dict(pickle.load(fid))
            fid.close()
        allJointGt_.append(raw['jointGt_'])
        allJointImpl_.append(raw['jointImpl_'])

    allJointGt_ = np.concatenate(allJointGt_, axis=0)
    allJointImpl_ = np.concatenate(allJointImpl_,axis=0)
    sikdata = {
        'jointGt_': allJointGt_,
        'jointImpl_':allJointImpl_
    }
    sikdata_at = os.path.join(
        args.sik_genpath,
        'sik_{}_{}{}.pkl'.format(
            args.data_split,
            args.fine_tune,
            '_{}epochs'.format(args.epochs) if is_train else ''
        )
    )
    with open(sikdata_at, 'wb') as fid:
        pickle.dump(sikdata, fid)
        fid.close()
    cprint('Saved {} samples at {}'
           .format(allJointGt_.shape[0], sikdata_at), 'yellow', attrs=['bold'])

    # delete intermediate outputs
    for save_at in all_save_at:
        os.remove(save_at)
    cprint('All Done', 'yellow',attrs=['bold'])
    return 0 # end of main



def one_forward_pass(metas, model, criterion, args, train=True):
    ''' prepare infos '''
    joint_root   = metas['joint_root'].to(device, non_blocking=True) # (B, 3)
    joint_bone   = metas['joint_bone'].to(device, non_blocking=True) # (B, 1)
    intr         = metas['intr'].to(device, non_blocking=True)
    hm_veil      = metas['hm_veil'].to(device, non_blocking=True)
    dep_veil     = metas['dep_veil'].to(device, non_blocking=True) # (B, 1)
    ndep_valid   = torch.sum(dep_veil).item()
    infos = {
        'joint_root':joint_root,
        'intr':intr,
        'joint_bone':joint_bone,
        'hm_veil':hm_veil,
        'dep_veil':dep_veil,
        'batch_size':joint_root.shape[0],
        'ndep_valid':ndep_valid,
    }
    ''' prepare targets '''
    clr     = metas['clr'].to(device, non_blocking=True)
    hm      = metas['hm'].to(device, non_blocking=True)
    dep     = metas['dep'].to(device, non_blocking=True) # (B, 64, 64)
    kp2d    = metas['kp2d'].to(device, non_blocking=True)
    joint   = metas['joint'].to(device, non_blocking=True)  # (B, 21, 3)
    jointR  = metas['jointR'].to(device, non_blocking=True)
    mask    = metas['mask'].to(device, non_blocking=True)  # (B, 64, 64)
    targets = {
        'clr':clr,
        'hm':hm,
        'joint':joint,
        'kp2d':kp2d,
        'jointR':jointR,
        'dep':dep,
        'mask':mask,
    }
    ''' ----------------  Forward Pass  ---------------- '''
    results = model(clr, infos)
    ''' ----------------  Forward End   ---------------- '''

    total_loss = torch.Tensor([0]).cuda()
    losses = {}

    return results, {**targets, **infos}, total_loss, losses


def validate(val_loader, model, criterion, args, stop=-1):
    # switch to evaluate mode
    evaluator = EvalUtil()
    model.eval()
    bar = Bar(colored('Eval', 'yellow'), max=len(val_loader))
    jointImpl_ = []
    jointGt_ = []
    with torch.no_grad():
        for i, metas in enumerate(val_loader):
            results, targets, _1, _2 = one_forward_pass(
                metas, model, criterion, args=None, train=False
            )
            joint_root = targets['joint_root'].unsqueeze(1)  # (B, 1, 3)
            predjoint = results['l_joint'][-1]  #(B 21 3)
            predjointR = predjoint - joint_root
            predjointR = np.array(predjointR.detach().cpu())

            jointImpl_.append(predjointR)

            targjointR = np.array(targets['jointR'].detach().cpu())
            jointGt_.append(targjointR)

            for targj, predj in zip(targjointR, predjointR):
                evaluator.feed(targj * 1000.0, predj * 1000.0)

            pck20 = evaluator.get_pck_all(20)
            pck30 = evaluator.get_pck_all(30)
            pck40 = evaluator.get_pck_all(40)
            bar.suffix  = (
                '({batch}/{size}) '
                't: {total:}s | '
                'eta:{eta:}s | '
                'pck20avg: {pck20:.3f} | '
                'pck30avg: {pck30:.3f} | '
                'pck40avg: {pck40:.3f} | '
            ).format(
                batch = i + 1,
                size = len(val_loader),
                total = bar.elapsed_td,
                eta   = bar.eta_td,
                pck20 = pck20,
                pck30 = pck30,
                pck40 = pck40,
            )
            bar.next()
    bar.finish()
    (
        _1, _2, _3,
        auc_all,
        pck_curve_all,
        thresholds
    ) = evaluator.get_measures(
        20, 50, 20
    )
    print("AUC all: {}".format(auc_all))
    jointGt_ = np.concatenate(jointGt_, axis=0)
    jointImpl_ = np.concatenate(jointImpl_, axis=0)
    saving = {
        'jointGt_':jointGt_,
        'jointImpl_':jointImpl_
    }
    return saving


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Train 3d-Hand-Circle')
    # Dataset setting
    parser.add_argument(
        '-dr',
        '--data_root',
        type=str,
        default='/media/sirius/Lixin213G/Dataset',
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
        '--data_split',
        type=str,
        default='',
        help='dataset split. should in: [train|test]'
    )
    parser.add_argument(
        '--sik_genpath',
        type=str,
        default='data/SIK-online',
        help='path to save generated sik-online data'
    )


    # Model Structure
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

    # Miscs
    parser.add_argument(
        '-ckp',
        '--checkpoint',
        default='/home/sirius/Documents/BiHand/checkpoints',
        type=str,
        metavar='PATH',
        help='path to save checkpoint (default: checkpoint)'
    )
    # Training Parameters
    parser.add_argument(
        '-j', '--workers',
        default=8,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 8)'
    )
    parser.add_argument(
        '--epochs',
        default=100,
        type=int,
        metavar='N',
        help='number of total epochs to run'
    )
    parser.add_argument(
        '-b','--train_batch',
        default=32,
        type=int,
        metavar='N',
        help='train batchsize'
    )

    parser.add_argument(
        "--net_modules",
        nargs="+",
        default=['seed', 'lift'],
        type=str,
        help="sub modules contained in model"
    )

    main(parser.parse_args())
