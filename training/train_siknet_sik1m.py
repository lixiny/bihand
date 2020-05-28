from __future__ import print_function, absolute_import

import os
import argparse
import time

import torch
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

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

def print_args(args):
    opts = vars(args)
    cprint("{:>30}  Options  {}".format("="*15, "="*15), 'yellow')
    for k, v in sorted(opts.items()):
        print("{:>30}  :  {}".format(k, v))
    cprint("{:>30}  Options  {}".format("="*15, "="*15), 'yellow')


def main(args):
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)
    print_args(args)
    print("\nCREATE NETWORK")
    model = models.SIKNet()
    model = model.to(device)
    criterion = losses.SIKLoss(
        lambda_quat = 1.0,  # only perform quaternion loss
        lambda_joint = 0.0,
        lambda_shape = 0.0
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

    train_dataset = datasets.SIK1M(
        data_root=args.data_root,
        data_split="train"
    )

    val_dataset = datasets.SIK1M(
         data_root=args.data_root,
        data_split="test"
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
            model, os.path.join(args.checkpoint, 'ckp_siknet_synth.pth.tar')
        )
        if args.evaluate:
            for params in model.invk_layers.parameters():
                params.requires_grad = False

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
        print('\nEpoch: %d' % (epoch + 1))
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
        misc.save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
            },
            checkpoint=args.checkpoint,
            filename='{}.pth.tar'.format(args.saved_prefix),
            snapshot=args.snapshot,
            is_best=False
        )
        validate(val_loader, model, criterion, args)
        scheduler.step()
    cprint('All Done', 'yellow',attrs=['bold'])
    return 0 # end of main

def validate(val_loader, model, criterion, args, stop=-1):
    am_quat_norm   = AverageMeter()
    am_quat_l2 = AverageMeter()
    am_quat_cos  = AverageMeter()
    evaluator = EvalUtil()
    model.eval()
    total_quat = []
    total_beta = []
    bar = Bar('\033[33m Eval  \033[0m', max=len(val_loader))
    with torch.no_grad():
        for i, metas in enumerate(val_loader):
            results, targets, total_loss, losses = one_forward_pass(
                metas, model, criterion, args, train=True
            )
            am_quat_norm.update(
                losses['quat_norm'].item(), targets['batch_size']
            )
            am_quat_l2.update(
                losses['quat_l2'].item(), targets['batch_size']
            )
            am_quat_cos.update(
                losses['quat_cos'].item(), targets['batch_size']
            )
            predjointRS = results['jointRS']
            joint_bone = targets['joint_bone'].unsqueeze(1)
            predjointR = predjointRS * joint_bone

            targjointRS = targets['jointRS']
            targjointR = targjointRS * joint_bone

            predjointR = predjointR.detach().cpu()
            targjointR = targjointR.detach().cpu()
            for targj, predj in zip(targjointR, predjointR):
                evaluator.feed(targj * 1000.0, predj * 1000.0)

            pck20 = evaluator.get_pck_all(20)
            pck30 = evaluator.get_pck_all(30)
            pck40 = evaluator.get_pck_all(40)

            bar.suffix  = (
                '({batch}/{size}) '
                'lN: {lossN:.5f} | '
                'lL2: {lossL2:.5f} | '
                'lC: {lossC:.3f} |'
                'pck20avg: {pck20:.3f} | '
                'pck30avg: {pck30:.3f} | '
                'pck40avg: {pck40:.3f} | '
            ).format(
                batch = i + 1,
                size = len(val_loader),
                pck20 = pck20,
                pck30 = pck30,
                pck40 = pck40,
                lossN = am_quat_norm.avg,
                lossL2 = am_quat_l2.avg,
                lossC = am_quat_cos.avg,
            )

            bar.next()
        bar.finish()

    return 0


def train(train_loader, model, criterion, optimizer, args):
    batch_time   = AverageMeter()
    data_time    = AverageMeter()
    am_quat_norm   = AverageMeter()
    am_quat_l2 = AverageMeter()
    am_quat_cos  = AverageMeter()
    am_joint = AverageMeter()
    am_kin_len = AverageMeter()

    last = time.time()
    # switch to trian
    model.train()
    bar = Bar('\033[31m Train \033[0m', max=len(train_loader))
    for i, metas in enumerate(train_loader):
        data_time.update(time.time() - last)
        results, targets, total_loss, losses = one_forward_pass(
            metas, model, criterion, args, train=True
        )
        am_quat_norm.update(
            losses['quat_norm'].item(), targets['batch_size']
        )
        am_quat_l2.update(
            losses['quat_l2'].item(), targets['batch_size']
        )
        am_quat_cos.update(
            losses['quat_cos'].item(), targets['batch_size']
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
            # 'lJ: {lossJ:.5f} | '
            # 'lK: {lossK:.5f} | '
            'lN: {lossN:.5f} | '
            'lL2: {lossL2:.5f} | '
            'lC: {lossC:.3f} |'
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
            lossL2 = am_quat_l2.avg,
            lossC = am_quat_cos.avg,
        )
        bar.next()
    bar.finish()

def one_forward_pass(metas, model, criterion, args, train=True):
    ''' prepare targets '''
    jointRS     = metas['jointRS'].to(device, non_blocking=True)
    kin_chain   = metas['kin_chain'].to(device, non_blocking=True)
    kin_len     = metas['kin_len'].to(device, non_blocking=True)
    joint_bone  = metas['joint_bone'].to(device, non_blocking=True)
    quat        = metas['quat'].to(device, non_blocking=True)
    targets = {
        'batch_size':jointRS.shape[0],
        'quat':quat,
        'kin_chain':kin_chain,
        'jointRS':jointRS,
        'kin_len':kin_len,
        'joint_bone':joint_bone
    }
    ''' ----------------  Forward Pass  ---------------- '''
    results = model(jointRS, kin_chain)
    ''' ----------------  Forward End   ---------------- '''

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
        '-ckp',
        '--checkpoint',
        default='/home/sirius/Documents/BiHand/checkpoints',
        type=str,
        metavar='PATH',
        help='path to save checkpoint (default: checkpoint)'
    )

    parser.add_argument(
        '-dr',
        '--data_root',
        type=str,
        default='/media/sirius/Lixin213G/Dataset',
        help='dataset root directory'
    )

    parser.add_argument(
        '-sp',
        '--saved_prefix',
        default='ckp_siknet_synth',
        type=str,
        metavar='PATH',
        help='path to save checkpoint (default: checkpoint)'
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
        default=150,
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
        default=64,
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
        default=1.0e-3,
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
