from bihand.datasets.handataset import HandDataset
from bihand.utils.eval.zimeval import EvalUtil
from bihand.utils.eval.evalutils import AverageMeter
from progress.progress.bar import Bar
from termcolor import colored, cprint
import bihand.utils.misc as misc
import bihand.losses as losses
import bihand.models as models

import os
import argparse
import time
import cv2
import numpy as np
import torch
import torch.nn.parallel
import torch.nn.utils
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import matplotlib.pyplot as plt
import pickle

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
# There is BN issue for early version of PyTorch
# see https://github.com/bearpaw/pytorch-pose/issues/33


def main(args):
    if (
            not args.fine_tune
            or not args.fine_tune in ['rhd', 'stb']
    ):
        raise Exception('expect --fine_tune in [rhd|stb], got {}'
                        .format(args.fine_tune))
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    args.datasets = [args.fine_tune, ]
    misc.print_args(args)
    auc_best = 0
    print("\nCREATE NETWORK")

    model = models.NetBiHand(
        net_modules=['seed', 'lift', 'sik'],
        njoints=21,
        inp_res=256,
        out_hm_res=64,
        out_dep_res=64,
        upstream_hg_stacks=2,
        upstream_hg_blocks=1,
    )
    model = model.to(device)

    criterion = losses.SIKLoss(
        lambda_quat=0.0,
        lambda_joint=1.0,
        lambda_shape=1.0
    )

    optimizer = torch.optim.Adam(
        [
            {
                'params': model.siknet.parameters(),
                'initial_lr': args.learning_rate
            },

        ],
        lr=args.learning_rate,
    )

    train_dataset = HandDataset(
        data_split='train',
        train=True,
        scale_jittering=0.2,
        center_jettering=0.2,
        max_rot=0.5 * np.pi,
        subset_name=args.datasets,
        data_root=args.data_root,
    )

    val_dataset = HandDataset(
        data_split='test',
        train=False,
        subset_name=args.datasets,
        data_root=args.data_root,
    )

    print("Total train dataset size: {}".format(len(train_dataset)))
    print("Total val dataset size: {}".format(len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    model.load_checkpoints(
        ckp_seednet=os.path.join(args.checkpoint, 'ckp_seednet_all.pth.tar'),
        ckp_liftnet=os.path.join(args.checkpoint, args.fine_tune,
                                 'ckp_liftnet_{}.pth.tar'.format(args.fine_tune)),
        ckp_siknet=os.path.join(args.checkpoint, 'ckp_siknet_synth.pth.tar')
    )
    for params in model.upstream.parameters():
        params.requires_grad = False

    if args.evaluate or args.resume:
        model.load_checkpoints(
            ckp_siknet=os.path.join(
                args.checkpoint, args.fine_tune,
                'ckp_siknet_{}.pth.tar'.format(args.fine_tune)
            )
        )

    if args.evaluate:
        validate(val_loader, model, criterion, args=args)
        cprint('Eval All Done', 'yellow', attrs=['bold'])
        return 0

    model = torch.nn.DataParallel(model)
    print("\nUSING {} GPUs".format(torch.cuda.device_count()))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_decay_step, gamma=args.gamma,
        last_epoch=args.start_epoch
    )

    for epoch in range(args.start_epoch, args.epochs + 1):
        print('\nEpoch: %d' % (epoch))
        for i in range(len(optimizer.param_groups)):
            print('group %d lr:' % i, optimizer.param_groups[i]['lr'])
        #############  trian for on epoch  ###############
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            args=args,
        )
        ##################################################
        auc_all = validate(val_loader, model, criterion, args)
        misc.save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
            },
            checkpoint=args.checkpoint,
            filename='{}_{}.pth.tar'.format(args.saved_prefix, args.fine_tune),
            snapshot=args.snapshot,
            is_best=auc_all > auc_best
        )
        if auc_all > auc_best:
            auc_best = auc_all

        scheduler.step()
    cprint('All Done', 'yellow', attrs=['bold'])
    return 0  # end of main


def validate(val_loader, model, criterion, args, stop=-1):
    am_quat_norm = AverageMeter()
    evaluator = EvalUtil()
    model.eval()
    bar = Bar(colored('Eval', 'yellow'), max=len(val_loader))

    with torch.no_grad():
        for i, metas in enumerate(val_loader):
            results, targets, total_loss, losses = one_fowrard_pass(
                metas, model, criterion, args, train=True
            )
            am_quat_norm.update(
                losses['quat_norm'].item(), targets['batch_size']
            )

            joint_bone = targets['joint_bone'].unsqueeze(1)

            predjointR = results['jointRS'] * joint_bone
            targjointR = targets['jointRS'] * joint_bone

            for targj, predj in zip(targjointR, predjointR):
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
        bar.finish()
        (
            _1, _2, _3,
            auc_all,
            pck_curve_all,
            thresholds
        ) = evaluator.get_measures(
            20, 50, 20
        )
        print(pck_curve_all)
        print("AUC all: {}".format(auc_all))

    return auc_all


def train(train_loader, model, criterion, optimizer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    am_quat_norm = AverageMeter()
    am_joint = AverageMeter()
    am_kin_len = AverageMeter()

    last = time.time()
    # switch to trian
    model.train()
    bar = Bar('\033[31m Train \033[0m', max=len(train_loader))
    for i, metas in enumerate(train_loader):
        data_time.update(time.time() - last)
        results, targets, total_loss, losses = one_fowrard_pass(
            metas, model, criterion, args, train=True
        )
        am_quat_norm.update(
            losses['quat_norm'].item(), targets['batch_size']
        )
        am_joint.update(
            losses['joint'].item(), targets['batch_size']
        )
        am_kin_len.update(
            losses['kin_len'].item(), targets['batch_size']
        )
        ''' backward and step '''
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        ''' progress '''
        batch_time.update(time.time() - last)
        last = time.time()
        bar.suffix = (
            '({batch}/{size}) '
            'd: {data:.2f}s | '
            'b: {bt:.2f}s | '
            't: {total:}s | '
            'eta:{eta:}s | '
            'lJ: {lossJ:.5f} | '
            'lK: {lossK:.5f} | '
            'lN: {lossN:.5f} | '
        ).format(
            batch=i + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            lossJ=am_joint.avg,
            lossK=am_kin_len.avg,
            lossN=am_quat_norm.avg,
        )
        bar.next()
    bar.finish()


def one_fowrard_pass(metas, model, criterion, args, train=True):
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
    jointRS = metas['jointRS'].to(device, non_blocking=True)
    kin_len = metas['kin_len'].to(device, non_blocking=True)
    mask = metas['mask'].to(device, non_blocking=True)  # (B, 64, 64)

    targets = {
        'clr': clr,
        'hm': hm,
        'joint': joint,
        'kp2d': kp2d,
        'jointRS': jointRS,
        'kin_len': kin_len,
        'dep': dep,
        'mask': mask,
    }
    ''' ----------------  Forward Pass  ---------------- '''
    results = model(clr, infos)
    ''' ----------------  Forward End   ---------------- '''

    total_loss = torch.Tensor([0]).cuda()
    losses = {}
    targets = {**targets, **infos}
    if not train:
        return results, targets, total_loss, losses

    ''' conpute losses '''
    total_loss, losses = criterion.compute_loss(results, targets)
    return results, targets, total_loss, losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch Train BiHand Stage 3: SIKNet (with SeedNet, LiftNet Freeze)')
    # Miscs
    parser.add_argument(
        '-dr',
        '--data_root',
        type=str,
        default='data',
        help='dataset root directory'
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
        '-sp',
        '--saved_prefix',
        default='ckp_siknet',
        type=str,
        metavar='PATH',
        help='path to save checkpoint (default: checkpoint)'
    )
    parser.add_argument(
        '--fine_tune',
        type=str,
        default='',
        help='fine tune dataset. should in: [rhd|stb]'
    )
    parser.add_argument(
        '--snapshot',
        default=1, type=int,
        help='save models for every #snapshot epochs (default: 1)'
    )

    parser.add_argument(
        '-e', '--evaluate',
        dest='evaluate',
        action='store_true',
        help='evaluate model on validation set'
    )

    parser.add_argument(
        '-r', '--resume',
        dest='resume',
        action='store_true',
        help='resume model on validation set'
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
        '-se', '--start_epoch',
        default=0,
        type=int,
        metavar='N',
        help='manual epoch number (useful on restarts)'
    )
    parser.add_argument(
        '-b', '--train_batch',
        default=16,
        type=int,
        metavar='N',
        help='train batchsize'
    )
    parser.add_argument(
        '-tb', '--test_batch',
        default=32,
        type=int,
        metavar='N',
        help='test batchsize'
    )

    parser.add_argument(
        '-lr', '--learning-rate',
        default=1.0e-4,
        type=float,
        metavar='LR',
        help='initial learning rate'
    )
    parser.add_argument(
        "--lr_decay_step",
        default=40,
        type=int,
        help="Epochs after which to decay learning rate",
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.1,
        help='LR is multiplied by gamma on schedule.'
    )
    main(parser.parse_args())
