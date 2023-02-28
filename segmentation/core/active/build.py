import random
import math
import torch
import os


import numpy as np
import torch.nn as nn
import torch.nn.functional as F


from PIL import Image
from math import ceil
from tqdm import tqdm

# from .compared_score import ComparedScore
# from .floating_region import FloatingRegionScore
# from .semantic_boundary import DetectSPBoundary
# from .spatial_purity import SpatialPurity


def transform_color(pred):
    synthia_to_city = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 10,
        10: 11,
        11: 12,
        12: 13,
        13: 15,
        14: 17,
        15: 18,
    }
    label_copy = 255 * np.ones(pred.shape, dtype=np.float32)
    for k, v in synthia_to_city.items():
        label_copy[pred == k] = v
    return label_copy.copy()


def get_color_pallete(npimg, dataset='voc'):
    out_img = Image.fromarray(npimg.astype('uint8')).convert('P')
    if dataset == 'city':
        cityspallete = [
            128, 64, 128,
            244, 35, 232,
            70, 70, 70,
            102, 102, 156,
            190, 153, 153,
            153, 153, 153,
            250, 170, 30,
            220, 220, 0,
            107, 142, 35,
            152, 251, 152,
            0, 130, 180,
            220, 20, 60,
            255, 0, 0,
            0, 0, 142,
            0, 0, 70,
            0, 60, 100,
            0, 80, 100,
            0, 0, 230,
            119, 11, 32,
        ]
        out_img.putpalette(cityspallete)
    else:
        vocpallete = _getvocpallete(256)
        out_img.putpalette(vocpallete)
    return out_img


def _getvocpallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while (lab > 0):
            pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return pallete


def save_for_visualization(cfg, round_path, active_mask):
    if cfg.MODEL.NUM_CLASSES == 16:
        active_mask = transform_color(active_mask)
    active_mask = get_color_pallete(active_mask, "city")
    mask_filename = round_path.replace('.png', '_color.png')
    if active_mask.mode == 'P':
        active_mask = active_mask.convert('RGB')
    active_mask.save(mask_filename)




def DUC_active(cfg, feature_extractor, classifier, tgt_epoch_loader, active_round, totality):
    feature_extractor.cuda()
    classifier.cuda()
    feature_extractor.eval()
    classifier.eval()

    all_pixels = 0
    select_pixels = 0
    with torch.no_grad():
        for tgt_data in tqdm(tgt_epoch_loader):
            tgt_input, path2mask = tgt_data['img'], tgt_data['path_to_mask']
            origin_mask, origin_label = tgt_data['origin_mask'], tgt_data['origin_label']
            origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            tgt_input = tgt_input.cuda(non_blocking=True)
            tgt_size = tgt_input.shape[-2:]
            tgt_feat = feature_extractor(tgt_input)
            tgt_out = classifier(tgt_feat, size=tgt_size)    # tgt_out.shape:[B, C, H, W]

            origin_size_output = F.interpolate(tgt_out, size=origin_size, mode='bilinear', align_corners=True)  # interpolate to the original size

            # convert logits to the parameters of Dirichlet distribution
            alpha = torch.exp(origin_size_output)

            total_alpha = torch.sum(alpha, dim=1, keepdim=True)  # total_alpha.shape: [B, 1, H, W]
            expected_p = alpha / total_alpha       # expected_p.shape: [B, C, H, W]
            pseudo_label = torch.argmax(expected_p, dim=1)  # shape of pseudo_label_i: [B, H, W]
            eps = 1e-7

            for i in range(len(origin_mask)):
                # for the i-th example
                active_mask = origin_mask[i].cuda(non_blocking=True)
                ground_truth = origin_label[i].cuda(non_blocking=True)
                all_pixels += tgt_size[0] * tgt_size[1]
                active = active_indicator[i]
                selected = selected_indicator[i]

                expected_p_i = expected_p[i, :, :, :]   # expected_p_i.shape: [C, H, W]
                alpha_i = alpha[i, :, :, :]
                total_alpha_i = total_alpha[i, :, :, :]

                # distributional uncertainty
                point_entropy_i = - torch.sum(expected_p_i * torch.log(expected_p_i + eps), dim=0)
                data_uncertainty_i = torch.sum((alpha_i / total_alpha_i) * (torch.digamma(total_alpha_i + 1) - torch.digamma(alpha_i + 1)), dim=0)
                distributional_uncertainty_i = point_entropy_i - data_uncertainty_i

                # Previously selected pixels no longer participate in the selection process
                distributional_uncertainty_i[active] = -float('inf')
                data_uncertainty_i[active] = -float('inf')

                # first sample according to the distributional uncertainty
                fisrt_round_num = math.ceil(totality * cfg.ACTIVE.RATIO * cfg.ACTIVE.KAPPA)
                arr_score_distri = distributional_uncertainty_i.reshape([-1])  # arr_score_distri.shape: [HW]
                tops, indices = torch.topk(arr_score_distri, k=fisrt_round_num)
                threshold_distri = tops.min().item()

                # second sample according to the expected data uncertainty
                arr_score_data = data_uncertainty_i.reshape([-1])[indices]  # arr_score_data.shape: [HW]
                tops, indices = torch.topk(arr_score_data, k=math.ceil(totality * cfg.ACTIVE.RATIO))
                threshold_data = tops.min().item()
                cur_round = (data_uncertainty_i > threshold_data) * (distributional_uncertainty_i > threshold_distri)

                active[cur_round] = True
                selected[cur_round] = True
                active_mask[cur_round] = ground_truth[cur_round]
                select_pixels += math.ceil(totality * cfg.ACTIVE.RATIO)

                # save for visualization
                round_path = path2mask[i].replace('gtMask/train/', 'gtMask/active_round_{}/'.format(active_round))
                save_for_visualization(cfg, round_path, active_mask.cpu().numpy())

                pseudo_label_i = pseudo_label[i, :, :]  # pseudo_label_i.shape: [H, W]
                pred_path = round_path.replace('active_round', 'pred_round')
                save_for_visualization(cfg, pred_path, pseudo_label_i.cpu().numpy())
                #######################################################################################################

                active_mask = Image.fromarray(np.array(active_mask.cpu().numpy(), dtype=np.uint8))
                active_mask.save(path2mask[i])
                indicator = {
                    'active': active,
                    'selected': selected
                }
                torch.save(indicator, path2indicator[i])

    print("{}/{}({:.8f}%) pixels have been selected!".format(select_pixels, all_pixels, select_pixels / all_pixels * 100))

    feature_extractor.train()
    classifier.train()


