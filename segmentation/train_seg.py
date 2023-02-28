import argparse
import os
import datetime
import logging
import time

import torch
import torch.nn as nn
import torch.utils
import torch.distributed
from torch.utils.data import DataLoader


from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_feature_extractor, build_classifier
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.utils.utils import set_random_seed
from core.active.build import *
from core.datasets.dataset_path_catalog import DatasetCatalog
from core.active.loss import EDL_Loss, get_valid_pixels

import warnings
warnings.filterwarnings('ignore')


def train(cfg):
    logger = logging.getLogger("main_DUC.trainer")

    device = torch.device(cfg.MODEL.DEVICE)

    # create network
    feature_extractor = build_feature_extractor(cfg)
    feature_extractor.to(device)

    classifier = build_classifier(cfg)
    classifier.to(device)

    print(classifier)

    # init optimizer
    optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()

    optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=cfg.SOLVER.BASE_LR * 10, momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_cls.zero_grad()

    # load checkpoint
    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        classifier.load_state_dict(checkpoint['classifier'])

    # init mask for cityscape
    if "cityscapes" in cfg.DATASETS.TARGET_TRAIN:
        DatasetCatalog.initMask(cfg)
    else:
        print("Unknown target dataset!")

    # init data loader
    src_train_data = build_dataset(cfg, mode='train', is_source=True)
    tgt_train_data = build_dataset(cfg, mode='train', is_source=False)
    tgt_epoch_data = build_dataset(cfg, mode='active', is_source=False, epochwise=True)

    src_train_loader = DataLoader(src_train_data, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=4,
                                  pin_memory=True, drop_last=True)
    tgt_train_loader = DataLoader(tgt_train_data, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=4,
                                  pin_memory=True, drop_last=True)
    tgt_epoch_loader = DataLoader(tgt_epoch_data, batch_size=1, shuffle=False, num_workers=4,
                                  pin_memory=True, drop_last=False)

    totality = int(cfg.INPUT.TARGET_INPUT_SIZE_TRAIN[0] * cfg.INPUT.TARGET_INPUT_SIZE_TRAIN[1])

    # evidence deep learning loss function
    edl_criterion = EDL_Loss(cfg)

    iteration = 0
    start_training_time = time.time()
    end = time.time()
    max_iters = cfg.SOLVER.MAX_ITER
    meters = MetricLogger(delimiter="  ")

    logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
    feature_extractor.train()
    classifier.train()
    active_round = 1
    for batch_index, (src_data, tgt_data) in enumerate(zip(src_train_loader, tgt_train_loader)):
        data_time = time.time() - end
        progress = (iteration + 1) / max_iters
        current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, max_iters,
                                          power=cfg.SOLVER.LR_POWER)
        for index in range(len(optimizer_fea.param_groups)):
            optimizer_fea.param_groups[index]['lr'] = current_lr
        for index in range(len(optimizer_cls.param_groups)):
            optimizer_cls.param_groups[index]['lr'] = current_lr * 10

        optimizer_fea.zero_grad()
        optimizer_cls.zero_grad()

        src_input, src_label = src_data['img'], src_data['label']
        src_input = src_input.cuda(non_blocking=True)
        src_label = src_label.cuda(non_blocking=True)

        # target data
        # tgt_mask is active label
        tgt_input, tgt_mask, tgt_label = tgt_data['img'], tgt_data['label_mask'], tgt_data['label']
        tgt_input = tgt_input.cuda(non_blocking=True)
        tgt_mask = tgt_mask.cuda(non_blocking=True)
        tgt_label = tgt_label.cuda(non_blocking=True)

        src_size = src_input.shape[-2:]
        src_out = classifier(feature_extractor(src_input), size=src_size)
        tgt_size = tgt_input.shape[-2:]
        tgt_out = classifier(feature_extractor(tgt_input), size=tgt_size)   # tgt_out.shape:[B, C, H, W]

        # convert logits to the parameters of Dirichlet distribution
        src_alpha = torch.exp(src_out)
        tgt_alpha = torch.exp(tgt_out)

        valid_src_alpha, valid_src_label, _ = get_valid_pixels(src_alpha.cpu(), src_label.cpu(), src_label.cpu())  # valid_src_alpha.shape[BHW, C], valid_src_label.shape[BHW, 1]
        valid_tgt_alpha, valid_tgt_label, invalid_tgt_alpha = get_valid_pixels(tgt_alpha.cpu(), tgt_label.cpu(), tgt_mask.cpu())

        # evidence deep learning loss on labeled source data
        total_loss = torch.Tensor([0]).to(device)
        Loss_nll_s, Loss_KL_s = edl_criterion(valid_src_alpha.cpu(), valid_src_label.cpu())
        Loss_KL_s = Loss_KL_s / cfg.MODEL.NUM_CLASSES
        total_loss += Loss_nll_s
        meters.update(Loss_nll_s=Loss_nll_s.item())
        total_loss += Loss_KL_s
        meters.update(Loss_KL_s=Loss_KL_s.item())

        # uncertainty reduction loss on unlabeled target data
        if cfg.SOLVER.BETA > 0 and (invalid_tgt_alpha is not None):
            total_alpha_t = torch.sum(invalid_tgt_alpha, dim=1, keepdim=True)  # total_alpha.shape: [M, 1]
            expected_p_t = invalid_tgt_alpha / total_alpha_t
            eps = 1e-7
            point_entropy_t = - torch.sum(expected_p_t * torch.log(expected_p_t + eps), dim=1)
            data_uncertainty_t = torch.sum((invalid_tgt_alpha / total_alpha_t) * (torch.digamma(total_alpha_t + 1) - torch.digamma(invalid_tgt_alpha + 1)), dim=1)
            loss_Udis = torch.sum(point_entropy_t - data_uncertainty_t) / invalid_tgt_alpha.shape[0]
            loss_Udata = torch.sum(data_uncertainty_t) / invalid_tgt_alpha.shape[0]

            total_loss += cfg.SOLVER.BETA * loss_Udis
            meters.update(loss_Udis=(loss_Udis).item())
            total_loss += cfg.SOLVER.LAMBDA * loss_Udata
            meters.update(loss_Udata=(loss_Udata).item())

        # evidence deep learning loss on selected target data
        if torch.sum((tgt_mask != 255)) != 0:  # target has labeled pixels
            selected_Loss_nll_t, selected_Loss_KL_t = edl_criterion(valid_tgt_alpha.cpu(), valid_tgt_label.cpu())
            selected_Loss_KL_t = selected_Loss_KL_t / cfg.MODEL.NUM_CLASSES
            total_loss += selected_Loss_nll_t
            meters.update(selected_Loss_nll_t=selected_Loss_nll_t.item())
            total_loss += selected_Loss_KL_t
            meters.update(selected_Loss_KL_t=selected_Loss_KL_t.item())

        total_loss.backward()
        optimizer_fea.step()
        optimizer_cls.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (cfg.SOLVER.STOP_ITER - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        iteration += 1
        if iteration % 500 == 0 or iteration == max_iters:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.02f} GB"
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer_fea.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0
                )
            )

        if iteration == cfg.SOLVER.MAX_ITER or iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            filename = os.path.join(cfg.OUTPUT_DIR, "model_iter{:06d}.pth".format(iteration))
            torch.save({'iteration': iteration,
                        'feature_extractor': feature_extractor.state_dict(),
                        'classifier': classifier.state_dict(),
                        'optimizer_fea': optimizer_fea.state_dict(),
                        'optimizer_cls': optimizer_cls.state_dict(),
                        }, filename)

        # active learning
        if iteration in cfg.ACTIVE.SELECT_ITER:
            # if iteration in cfg.ACTIVE.SELECT_ITER or cfg.DEBUG:
            print("iter={}".format(iteration))
            DUC_active(cfg=cfg,
                       feature_extractor=feature_extractor.cpu(),
                       classifier=classifier.cpu(),
                       tgt_epoch_loader=tgt_epoch_loader,
                       active_round=active_round,
                       totality=totality
                       )
            active_round += 1

        if iteration == cfg.SOLVER.MAX_ITER:
            break
        if iteration == cfg.SOLVER.STOP_ITER:
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / cfg.SOLVER.STOP_ITER
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Pytorch Active Domain Adaptation")
    parser.add_argument("--cfg",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("main_DUC", output_dir, 0, filename=cfg.LOG_NAME)
    logger.info(args)

    import PIL
    import torchvision
    logger.info("PTL.version = {}".format(PIL.__version__))
    logger.info("torch.version = {}".format(torch.__version__))
    logger.info("torchvision.version = {}".format(torchvision.__version__))

    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    logger.info("Initializing %s label mask..." % (cfg.DATASETS.TARGET_TRAIN.split('_')[0]))

    if cfg.SEED >= 0:
        print('Setting fixed seed: {}'.format(cfg.SEED))
        set_random_seed(cfg.SEED)

    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    train(cfg)

if __name__ == '__main__':
    main()
