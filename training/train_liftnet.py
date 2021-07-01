from genericpath import isfile
from bihand.datasets.handataset import HandDataset
from bihand.utils.eval.zimeval import EvalUtil
from bihand.utils.eval.evalutils import AverageMeter
from progress.progress.bar import Bar
from termcolor import colored, cprint

import bihand.utils.func as func
import bihand.utils.misc as misc
import bihand.utils.imgutils as imutils
import bihand.losses as losses
import bihand.models as models
import os
import argparse
import time
import numpy as np
import torch
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import matplotlib.pyplot as plt

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
# There is BN issue for early version of PyTorch
# see https://github.com/bearpaw/pytorch-pose/issues/33


def main(args):
    best_acc = 0
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)
    if args.fine_tune:
        args.datasets = [args.fine_tune, ]
    else:
        args.fine_tune = 'all'
    misc.print_args(args)
    print("\nCREATE NETWORK")
    model = models.NetBiHand(
        net_modules=args.net_modules,  # only train hm
        njoints=21,
        inp_res=256,
        out_hm_res=64,
        out_dep_res=64,
        upstream_hg_stacks=2,
        upstream_hg_blocks=1,
    )
    model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion_ups = losses.UpstreamLoss(
        lambda_hm=0.0,
        lambda_mask=0.0,
        lambda_joint=1.0,
        lambda_dep=1.0
    )

    criterion = {
        'ups': criterion_ups,
    }
    optimizer = torch.optim.Adam(
        [
            {
                'params': model.upstream.liftnet.parameters(),
                'initial_lr': args.learning_rate
            },
        ],
        lr=args.learning_rate,
    )

    print("\nCREATE DATASET")
    train_dataset = HandDataset(
        data_split='train',
        train=True,
        scale_jittering=0.1,
        center_jettering=0.1,
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
    print("Total test dataset size: {}".format(len(val_dataset)))
    print("\nLOAD CHECKPOINT")

    if not args.frozen_seednet_pth or not isfile(args.frozen_seednet_pth):
        raise ValueError("No frozen_seednet_pth is provided")

    model.load_checkpoints(ckp_seednet=args.frozen_seednet_pth)
    for params in model.upstream.seednet.parameters(): # frozen
        params.requires_grad = False

    if args.resume_liftnet_pth:
        model.load_checkpoints(ckp_liftnet=args.resume_liftnet_pth,)
    else:
        for m in model.upstream.liftnet.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

    if args.evaluate_liftnet_pth:
        model.load_checkpoints(ckp_liftnet=args.evaluate_liftnet_pth)
        validate(val_loader, model, criterion, args=args)
        return 0

    model = torch.nn.DataParallel(model)
    print("\nUSING {} GPUs".format(torch.cuda.device_count()))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_decay_step, gamma=args.gamma,
        last_epoch=args.start_epoch
    )

    for epoch in range(args.start_epoch, args.epochs + 1):
        print('\nEpoch: %d' % (epoch + 1))
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
        auc = best_acc
        if epoch >= 20 and epoch % 2 == 0:
            auc = validate(val_loader, model, criterion, args=args)
        misc.save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.module.upstream.liftnet.state_dict(),
            },
            checkpoint=args.checkpoint,
            filename='{}_{}.pth.tar'.format(args.saved_prefix, args.fine_tune),
            snapshot=args.snapshot,
            is_best=auc > best_acc
        )
        if auc > best_acc:
            best_acc = auc
        scheduler.step()
    cprint('All Done', 'yellow', attrs=['bold'])
    return 0  # end of main


def one_forward_pass(metas, model, criterion, args=None, train=True):
    ''' prepare infos '''
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
    mask = metas['mask'].to(device, non_blocking=True)  # (B, 64, 64)
    targets = {
        'clr': clr,
        'hm': hm,
        'joint': joint,
        'kp2d': kp2d,
        'jointRS': jointRS,
        'dep': dep,
        'mask': mask,
    }
    ''' ----------------  Forward Pass  ---------------- '''
    results = model(clr, infos)
    ''' ----------------  Forward End   ---------------- '''

    total_loss = torch.Tensor([0]).cuda()
    losses = {}
    if not train:
        return results, {**targets, **infos}, total_loss, losses

    ''' compute losses '''
    if args.ups_loss:
        ups_total_loss, ups_losses = criterion['ups'].compute_loss(
            results, targets, infos
        )
        total_loss += ups_total_loss
        losses.update(ups_losses)

    return results, {**targets, **infos}, total_loss, losses


def train(train_loader, model, criterion, optimizer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    am_loss_joint = AverageMeter()
    am_loss_dep = AverageMeter()
    am_loss_all = AverageMeter()

    last = time.time()
    # switch to trian
    model.train()
    bar = Bar(colored('Train', 'red'), max=len(train_loader))
    for i, metas in enumerate(train_loader):
        data_time.update(time.time() - last)
        results, targets, total_loss, losses = one_forward_pass(
            metas, model, criterion, args, train=True
        )
        am_loss_joint.update(
            losses['ups_joint'].item(), targets['batch_size']
        )
        am_loss_dep.update(
            losses['ups_dep'].item(), targets['batch_size']
        )
        am_loss_all.update(
            total_loss.item(), targets['batch_size']
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
            'lD: {lossD:.5f} | '
            'lA: {lossA:.3f} |'
        ).format(
            batch=i + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            lossJ=am_loss_joint.avg,
            lossD=am_loss_dep.avg,
            lossA=am_loss_all.avg,
        )
        bar.next()
    bar.finish()


def validate(val_loader, model, criterion, args, stop=-1):
    # switch to evaluate mode
    evaluator = EvalUtil()
    model.eval()
    bar = Bar(colored('Eval', 'yellow'), max=len(val_loader))
    with torch.no_grad():
        for i, metas in enumerate(val_loader):
            results, targets, _1, _2 = one_forward_pass(
                metas, model, criterion, args=None, train=False
            )
            pred_joint = results['l_joint'][-1].detach().cpu()
            targ_joint = targets['joint'].detach().cpu()

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
            if stop != -1 and i >= stop:
                break
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

    return auc_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch Train BiHand Stage 2: LiftNet')
    # Dataset setting
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
        help="sub datasets, should be listed in: [rhd|stb]"
    )
    parser.add_argument(
        '--fine_tune',
        type=str,
        default='',
        help='fine tune dataset. should in: [rhd|stb]'
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
        default='checkpoints',
        type=str,
        metavar='PATH',
        help='path to save checkpoint (default: checkpoint)'
    )
    parser.add_argument(
        '--frozen_seednet_pth',
        default='',
        type=str,
        metavar='PATH',
        help='You must provide the destination to load the frozen SeedNet checkpoints (default: none)'
    )
    parser.add_argument(
        '--resume_liftnet_pth',
        default='',
        type=str,
        metavar='PATH',
        help='whether to load LiftNet resume checkpoints pth (default: none)'
    )
    parser.add_argument(
        '--evaluate_liftnet_pth',
        default='',
        type=str,
        metavar='PATH',
        help='whether to load LiftNet checkpoints pth for evaluation ONLY (default: none)'
    )
    parser.add_argument(
        '-sp',
        '--saved_prefix',
        default='ckp_liftnet',
        type=str,
        metavar='PATH',
        help='path to save checkpoint (default: checkpoint)'
    )
    parser.add_argument(
        '--snapshot',
        default=1, type=int,
        help='save models for every #snapshot epochs (default: 0)'
    )
    parser.add_argument(
        '-d', '--debug',
        dest='debug',
        action='store_true',
        default=False,
        help='show intermediate results'
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
        default=1.0e-4,
        type=float,
        metavar='LR',
        help='initial learning rate'
    )
    parser.add_argument(
        "--lr_decay_step",
        default=50,
        type=int,
        help="Epochs after which to decay learning rate",
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.1,
        help='LR is multiplied by gamma on schedule.'
    )
    parser.add_argument(
        "--net_modules",
        nargs="+",
        default=['seed', 'lift'],
        type=str,
        help="sub modules contained in model"
    )
    parser.add_argument(
        '--ups_loss',
        dest='ups_loss',
        action='store_true',
        help='Calculate upstream loss',
        default=True
    )

    main(parser.parse_args())
