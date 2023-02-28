from __future__ import print_function
import argparse
import os.path
import os
import logging
import time
import datetime

import torch
import torchvision
import PIL
import torch.optim as optim
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from core.datasets.image_list import ImageList
from core.models.network import ResNetFc
from core.active.active import DUC_active
from core.utils.utils import set_random_seed, mkdir, LrScheduler
from core.datasets.transforms import build_transform
from core.active.loss import EDL_Loss
from core.utils.metric_logger import MetricLogger
from core.utils.logger import setup_logger
from core.config import cfg


def test(model, test_loader):
    start_test = True
    model.eval()
    with torch.no_grad():
        for batch_idx, test_data in enumerate(test_loader):
            img, labels = test_data['img0'], test_data['label']
            img = img.cuda()
            logits = model(img, return_feat=False)
            alpha = torch.exp(logits)
            total_alpha = torch.sum(alpha, dim=1, keepdim=True)  # total_alpha.shape: [B, 1]
            outputs = alpha / total_alpha
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, dim=1)
    acc = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0]) * 100

    return acc


def train(cfg, task):
    logger = logging.getLogger("main.trainer")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU_ID)

    kwargs = {'num_workers': cfg.DATALOADER.NUM_WORKERS, 'pin_memory': True}

    # prepare data
    source_transform = build_transform(cfg, is_train=True, choices=cfg.INPUT.SOURCE_TRANSFORMS)
    target_transform = build_transform(cfg, is_train=True, choices=cfg.INPUT.TARGET_TRANSFORMS)
    test_transform = build_transform(cfg, is_train=False, choices=cfg.INPUT.TEST_TRANSFORMS)

    src_train_ds = ImageList(os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME, cfg.DATASET.SOURCE_TRAIN_DOMAIN),
                             transform=source_transform)
    src_train_loader = DataLoader(src_train_ds, batch_size=cfg.DATALOADER.SOURCE.BATCH_SIZE, shuffle=True,
                                  drop_last=True, **kwargs)

    tgt_unlabeled_ds = ImageList(os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME, cfg.DATASET.TARGET_TRAIN_DOMAIN),
                                 transform=target_transform)
    tgt_unlabeled_loader = DataLoader(tgt_unlabeled_ds, batch_size=cfg.DATALOADER.TARGET.BATCH_SIZE, shuffle=True,
                                      drop_last=True, **kwargs)
    tgt_unlabeled_loader_full = DataLoader(tgt_unlabeled_ds, batch_size=cfg.DATALOADER.TARGET.BATCH_SIZE,
                                              shuffle=True, drop_last=False, **kwargs)

    tgt_test_ds = ImageList(os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME, cfg.DATASET.TARGET_VAL_DOMAIN),
                            transform=test_transform)
    tgt_test_loader = DataLoader(tgt_test_ds, batch_size=cfg.DATALOADER.TEST.BATCH_SIZE, shuffle=False, **kwargs)

    # active target dataset & loader
    tgt_selected_ds = ImageList(empty=True, transform=source_transform)
    tgt_selected_loader = DataLoader(tgt_selected_ds, batch_size=cfg.DATALOADER.SOURCE.BATCH_SIZE,
                                     shuffle=True, drop_last=False, **kwargs)

    iter_per_epoch = max(len(src_train_loader), len(tgt_unlabeled_loader))
    max_iters = cfg.TRAINER.MAX_EPOCHS * iter_per_epoch

    # model
    model = ResNetFc(class_num=cfg.DATASET.NUM_CLASS, cfg=cfg).cuda()

    # optimizer
    optimizer = optim.SGD(model.parameters_list(), lr=cfg.OPTIM.LR, momentum=0.9, weight_decay=1e-3, nesterov=True)
    lr_sheduler = LrScheduler(optimizer, max_iters, init_lr=cfg.OPTIM.LR, gamma=cfg.OPTIM.GAMMA, decay_rate=cfg.OPTIM.DECAY_RATE)

    # evidence deep learning loss function
    edl_criterion = EDL_Loss(cfg)

    # total number of target samples
    totality = tgt_unlabeled_ds.__len__()
    print("totality={}".format(totality))

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    start_training_time = time.time()
    end = time.time()

    final_acc = 0.
    final_model = None
    best_acc = 0.
    best_model = None
    all_epoch_result = []
    all_selected_images = None
    active_round = 1
    ckt_path = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.NAME, task)
    mkdir(ckt_path)
    for epoch in range(1, cfg.TRAINER.MAX_EPOCHS + 1):
        model.train()
        for batch_idx in range(iter_per_epoch):
            data_time = time.time() - end

            if batch_idx % len(src_train_loader) == 0:
                src_iter = iter(src_train_loader)
            if batch_idx % len(tgt_unlabeled_loader) == 0:
                tgt_unlabeled_iter = iter(tgt_unlabeled_loader)
            if not tgt_selected_ds.empty:
                if batch_idx % len(tgt_selected_loader) == 0:
                    tgt_selected_iter = iter(tgt_selected_loader)

            src_data = src_iter.next()
            tgt_unlabeled_data = tgt_unlabeled_iter.next()

            src_img, src_lbl = src_data['img0'], src_data['label']
            src_img, src_lbl = src_img.cuda(), src_lbl.cuda()

            tgt_unlabeled_img = tgt_unlabeled_data['img']
            tgt_unlabeled_img = tgt_unlabeled_img.cuda()

            lr_sheduler.step()
            optimizer.zero_grad()

            total_loss = 0

            # evidence deep learning loss on labeled source data
            src_out = model(src_img, return_feat=False)
            Loss_nll_s, Loss_KL_s = edl_criterion(src_out, src_lbl)
            Loss_KL_s = Loss_KL_s / cfg.DATASET.NUM_CLASS

            if cfg.DATASET.NAME == 'visda2017' and epoch > 1:
                Loss_nll_s = 0.5 * Loss_nll_s
                Loss_KL_s = 0.5 * Loss_KL_s

            total_loss += Loss_nll_s
            meters.update(Loss_nll_s=Loss_nll_s.item())

            total_loss += Loss_KL_s
            meters.update(Loss_KL_s=Loss_KL_s.item())


            if cfg.TRAINER.BETA > 0:
                # uncertainty reduction loss on unlabeled target data
                tgt_unlabeled_out = model(tgt_unlabeled_img, return_feat=False)
                alpha_t = torch.exp(tgt_unlabeled_out)
                total_alpha_t = torch.sum(alpha_t, dim=1, keepdim=True)  # total_alpha.shape: [B, 1]
                expected_p_t = alpha_t / total_alpha_t
                eps = 1e-7
                point_entropy_t = - torch.sum(expected_p_t * torch.log(expected_p_t + eps), dim=1)
                data_uncertainty_t = torch.sum((alpha_t / total_alpha_t) * (torch.digamma(total_alpha_t + 1) - torch.digamma(alpha_t + 1)), dim=1)
                loss_Udis = torch.sum(point_entropy_t - data_uncertainty_t) / tgt_unlabeled_out.shape[0]
                loss_Udata = torch.sum(data_uncertainty_t) / tgt_unlabeled_out.shape[0]

                total_loss += cfg.TRAINER.BETA * loss_Udis
                meters.update(loss_Udis=(loss_Udis).item())
                total_loss += cfg.TRAINER.LAMBDA * loss_Udata
                meters.update(loss_Udata=(loss_Udata).item())

            # evidence deep learning loss on selected target data
            if not tgt_selected_ds.empty:
                tgt_selected_data = tgt_selected_iter.next()
                tgt_selected_img, tgt_selected_lbl = tgt_selected_data['img0'], tgt_selected_data['label']
                tgt_selected_img, tgt_selected_lbl = tgt_selected_img.cuda(), tgt_selected_lbl.cuda()

                if tgt_selected_img.size(0) == 1:
                    # avoid bs=1, can't pass through BN layer
                    tgt_selected_img = torch.cat((tgt_selected_img, tgt_selected_img), dim=0)
                    tgt_selected_lbl = torch.cat((tgt_selected_lbl, tgt_selected_lbl), dim=0)

                tgt_selected_out = model(tgt_selected_img, return_feat=False)
                selected_Loss_nll_t, selected_Loss_KL_t = edl_criterion(tgt_selected_out, tgt_selected_lbl)
                selected_Loss_KL_t = selected_Loss_KL_t / cfg.DATASET.NUM_CLASS
                total_loss += selected_Loss_nll_t
                meters.update(selected_Loss_nll_t=selected_Loss_nll_t.item())
                total_loss += selected_Loss_KL_t
                meters.update(selected_Loss_KL_t=selected_Loss_KL_t.item())

            total_loss.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)
            eta_seconds = meters.time.global_avg * (iter_per_epoch * cfg.TRAINER.MAX_EPOCHS - batch_idx * epoch)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if batch_idx % cfg.TRAIN.PRINT_FREQ == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "task: {task}",
                            "epoch: {epoch}",
                            f"[iter: {batch_idx}/{iter_per_epoch}]",
                            "{meters}",
                            "max mem: {memory:.2f} GB",
                        ]
                    ).format(
                        task=task,
                        eta=eta_string,
                        epoch=epoch,
                        meters=str(meters),
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
                    )
                )

        # test every 5 epoch
        if cfg.DATASET.NAME == 'visda2017':
            if epoch % 2 == 0:
                testacc = test(model, tgt_test_loader)
                logger.info('Task: {} Test Epoch: {} testacc: {:.2f}'.format(task, epoch, testacc))
                all_epoch_result.append({'epoch': epoch, 'acc': testacc})
                if epoch == cfg.TRAINER.MAX_EPOCHS:
                    final_model = model.state_dict()
                    final_acc = testacc
                if testacc > best_acc:
                    best_acc = testacc
                    if cfg.SAVE:
                        torch.save(model.state_dict(), os.path.join(ckt_path, "best_model_{}.pth".format(task)))
        else:
            if epoch % 5 == 0:
                testacc = test(model, tgt_test_loader)
                logger.info('Task: {} Test Epoch: {} testacc: {:.2f}'.format(task, epoch, testacc))
                all_epoch_result.append({'epoch': epoch, 'acc': testacc})
                if epoch == cfg.TRAINER.MAX_EPOCHS:
                    final_model = model.state_dict()
                    final_acc = testacc
                if testacc > best_acc:
                    best_acc = testacc
                    if cfg.SAVE:
                        torch.save(model.state_dict(), os.path.join(ckt_path, "best_model_{}.pth".format(task)))

        # active selection rounds
        if epoch in cfg.TRAINER.ACTIVE_ROUND:
            logger.info('Task: {} Active Epoch: {}'.format(task, epoch))
            active_samples = DUC_active(tgt_unlabeled_loader_full=tgt_unlabeled_loader_full,
                                             tgt_unlabeled_ds=tgt_unlabeled_ds,
                                             tgt_selected_ds=tgt_selected_ds,
                                             active_ratio=0.01,
                                             totality=totality,
                                             model=model,
                                             cfg=cfg,
                                             logger=logger)
            active_round += 1

            # record all selected target images
            if all_selected_images is None:
                all_selected_images = active_samples
            else:
                all_selected_images = np.concatenate((all_selected_images, active_samples), axis=0)

    if all_selected_images is not None:
        logger.info("totality*0.05={}  all_selected_images.shape={}".format(totality * 0.05, all_selected_images.shape))
        logger.info(all_selected_images)
        # record all selected images

    if cfg.SAVE:
        # if all_selected_images is not None:
        #     torch.save(all_selected_images, os.path.join(ckt_path, "all_selected_images.pth"))
        torch.save(final_model, os.path.join(ckt_path, "final_model_{}.pth".format(task)))

    # record results for test epochs
    best_acc = 0.0
    best_epoch = 0
    with open(os.path.join(ckt_path, 'all_epoch_result.csv'), 'w') as handle:
        for i, rec in enumerate(all_epoch_result):
            keys_list = list(rec.keys())
            if rec[keys_list[1]] > best_acc:
                best_acc = rec[keys_list[1]]
                best_epoch = rec[keys_list[0]]
            if i == 0:
                handle.write(','.join(list(rec.keys())) + '\n')
            line = [str(rec[key]) for key in rec.keys()]
            handle.write(','.join(line) + '\n')
        handle.write(','.join(['best epoch', 'best acc']) + '\n')
        line = [str(best_epoch), str(best_acc)]
        handle.write(','.join(line) + '\n')

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / ep)".format(
            total_time_str, total_training_time / cfg.TRAINER.MAX_EPOCHS
        )
    )

    return task, final_acc, best_acc


def main():
    parser = argparse.ArgumentParser(description='PyTorch Activate Domain Adaptation')
    parser.add_argument('--cfg',
                        default='',
                        metavar='FILE',
                        help='path to config file',
                        type=str)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.NAME)
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("main", output_dir, 0, filename=cfg.LOG_NAME)
    logger.info("PTL.version = {}".format(PIL.__version__))
    logger.info("torch.version = {}".format(torch.__version__))
    logger.info("torchvision.version = {}".format(torchvision.__version__))
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.SEED >= 0:
        print('Setting fixed seed: {}'.format(cfg.SEED))
        set_random_seed(cfg.SEED)

    torch.multiprocessing.set_sharing_strategy('file_system')
    cudnn.deterministic = True
    cudnn.benchmark = True

    all_task_result = []
    for source in cfg.DATASET.SOURCE_DOMAINS:
        for target in cfg.DATASET.TARGET_DOMAINS:
            if source != target:
                cfg.DATASET.SOURCE_TRAIN_DOMAIN = os.path.join(source + '_train.txt')
                cfg.DATASET.TARGET_TRAIN_DOMAIN = os.path.join(target + '_train.txt')
                cfg.DATASET.TARGET_VAL_DOMAIN = os.path.join(target + '_test.txt')

                print("{}2{}: cfg.OPTIM.LR={}".format(source, target, cfg.OPTIM.LR))
                logger.info("{}2{}: cfg.OPTIM.LR={}".format(source, target, cfg.OPTIM.LR))

                cfg.freeze()
                task, final_acc, best_acc = train(cfg, task=source + '2' + target)
                all_task_result.append({'task': task, 'final_acc': final_acc, 'best_acc': best_acc})
                cfg.defrost()

    # record all results for all tasks
    with open(os.path.join(output_dir, 'all_task_result.csv'), 'w') as handle:
        for i, rec in enumerate(all_task_result):
            if i == 0:
                handle.write(','.join(list(rec.keys())) + '\n')
            line = [str(rec[key]) for key in rec.keys()]
            handle.write(','.join(line) + '\n')


if __name__ == '__main__':
    main()
