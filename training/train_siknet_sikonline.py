from __future__ import print_function, absolute_import

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

import bihand.datasets as datasets
import bihand.models as models
import bihand.losses as losses
import bihand.utils.misc as misc

from termcolor import colored, cprint
from progress.progress.bar import Bar
from bihand.utils.eval.evalutils import AverageMeter
from bihand.utils.eval.zimeval import EvalUtil

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
    model = models.SIKNet()
    model = model.to(device)
    criterion = losses.SIKLoss(
        lambda_quat = 0.0,
        lambda_joint = 1.0,
        lambda_shape = 1.0
    )

    optimizer = torch.optim.Adam(
        [
            {
                'params': model.invk_layers.parameters(),
                'initial_lr':args.learning_rate
            },
            {
                'params': model.shapereg_layers.parameters(),
                'initial_lr':args.learning_rate
            },

        ],
        lr=args.learning_rate,
    )

    train_dataset = datasets.SIKONLINE(
        data_root=args.data_root,
        data_split="train",
        data_source=args.datasets,
    )

    val_dataset = datasets.SIKONLINE(
        data_root=args.data_root,
        data_split="test",
        data_source=args.datasets,
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

    if args.evaluate or args.resume:
        misc.load_checkpoint(
            model,
            os.path.join(
                args.checkpoint, args.fine_tune,
                '{}_{}.pth.tar'.format(args.saved_prefix, args.fine_tune)
            )
        )

    if args.evaluate:
        validate(val_loader, model, criterion, args=args)
        cprint('Eval All Done', 'yellow',attrs=['bold'])
        return 0

    model = torch.nn.DataParallel(model)
    print("\nUSING {} GPUs".format(torch.cuda.device_count()))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_decay_step, gamma=args.gamma,
        last_epoch=args.start_epoch
    )

    for epoch in range(args.start_epoch, args.epochs+1):
        print('\nEpoch: %d' % (epoch))
        for i in range(len(optimizer.param_groups)):
            print('group %d lr:'%i, optimizer.param_groups[i]['lr'])
        #############  trian for on epoch  ###############
        train (
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
            is_best= auc_all > auc_best
        )
        if auc_all > auc_best:
            auc_best = auc_all

        scheduler.step()
    cprint('All Done', 'yellow',attrs=['bold'])
    return 0 # end of main

def validate(val_loader,  model, criterion, args, stop=-1):

    evaluator = EvalUtil()
    model.eval()
    bar = Bar(colored('Eval', 'yellow'), max=len(val_loader))

    with torch.no_grad():
        for i, (impls, targs) in enumerate(val_loader):
            results, targets, total_loss, losses = one_fowrard_pass(
                impls, targs, model, criterion, args, train=True
            )

            joint_bone = targets['joint_bone'].unsqueeze(1)
            targjointR = targets['jointRS'] * joint_bone
            predjointR = results['jointRS'] * joint_bone

            for targj, predj in zip(targjointR, predjointR):
                evaluator.feed(targj * 1000.0, predj * 1000.0)

            pck20 = evaluator.get_pck_all(20)
            pck30 = evaluator.get_pck_all(30)
            pck40 = evaluator.get_pck_all(40)

            bar.suffix  = (
                '({batch}/{size}) '
                'pck20avg: {pck20:.3f} | '
                'pck30avg: {pck30:.3f} | '
                'pck40avg: {pck40:.3f} | '
            ).format(
                batch = i + 1,
                size = len(val_loader),
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
        print(pck_curve_all)
        print("AUC all: {}".format(auc_all))

    return auc_all


def train(train_loader, model, criterion, optimizer, args):
    batch_time   = AverageMeter()
    data_time    = AverageMeter()
    am_quat_norm   = AverageMeter()
    am_joint = AverageMeter()
    am_kin_len = AverageMeter()

    last = time.time()
    # switch to trian
    model.train()
    bar = Bar(colored('Train', 'red'), max=len(train_loader))
    for i, (impls, targs) in enumerate(train_loader):
        data_time.update(time.time() - last)
        results, targets, total_loss, losses = one_fowrard_pass(
            impls, targs, model, criterion, args, train=True
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
        bar.suffix  = (
            '({batch}/{size}) '
            'd: {data:.2f}s | '
            'b: {bt:.2f}s | '
            't: {total:}s | '
            'eta:{eta:}s | '
            'lJ: {lossJ:.5f} | '
            'lK: {lossK:.5f} | '
            'lN: {lossN:.5f} | '
        ).format(
            batch = i+1,
            size  = len(train_loader),
            data  = data_time.avg,
            bt    = batch_time.avg,
            total = bar.elapsed_td,
            eta   = bar.eta_td,
            lossJ = am_joint.avg,
            lossK = am_kin_len.avg,
            lossN = am_quat_norm.avg,
        )
        bar.next()
    bar.finish()

def one_fowrard_pass(impls, targs, model, criterion, args, train=True):
    ''' prepare targets '''
    impljointRS    = impls['jointRS'].to(device, non_blocking=True)
    implkin_chain  = impls['kin_chain'].to(device, non_blocking=True)

    ''' ----------------  Forward Pass  ---------------- '''
    results = model( impljointRS, implkin_chain)
    ''' ----------------  Forward End   ---------------- '''

    targkin_len     = targs['kin_len'].to(device, non_blocking=True)
    targjoint_bone  = targs['joint_bone'].to(device, non_blocking=True)
    targjointRS     = targs['jointRS'].to(device, non_blocking=True)
    targets = {
        'batch_size':targjointRS.shape[0],
        'jointRS':targjointRS,
        'kin_len':targkin_len,
        'joint_bone':targjoint_bone
    }

    total_loss = torch.Tensor([0]).cuda()
    losses = {}
    if not train:
        return results, targets, total_loss, losses

    ''' conpute losses '''
    total_loss, losses = criterion.compute_loss(results, targets)
    return results, targets, total_loss, losses

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Train 3d-Hand-Circle')
    # Miscs
    parser.add_argument(
        '-dr',
        '--data_root',
        type=str,
        default='/media/sirius/Lixin213G/Dataset',
        help='dataset root directory'
    )
    parser.add_argument(
        '-ckp',
        '--checkpoint',
        default='/home/sirius/Documents/BiHand/checkpoints',
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
        help='fine tune dataset. should in: [rhd|stb|freihand]'
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
        default=6,
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
        '-b','--train_batch',
        default=5,
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
        default=3,
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
