import random
import math
import numpy as np
import torch




def DUC_active(tgt_unlabeled_loader_full, tgt_unlabeled_ds, tgt_selected_ds, active_ratio, totality, model, cfg, logger):
    model.eval()
    first_stat = list()
    with torch.no_grad():
        for _, data in enumerate(tgt_unlabeled_loader_full):
            tgt_img, tgt_lbl = data['img'], data['label']
            tgt_path, tgt_index = data['path'], data['index']
            tgt_img, tgt_lbl = tgt_img.cuda(), tgt_lbl.cuda()
            tgt_out = model(tgt_img, return_feat=False)

            alpha = torch.exp(tgt_out)
            total_alpha = torch.sum(alpha, dim=1, keepdim=True)  # total_alpha.shape: [B, 1]
            expected_p = alpha / total_alpha
            eps = 1e-7

            # distributional uncertainty of each sample
            point_entropy = - torch.sum(expected_p * torch.log(expected_p + eps), dim=1)
            data_uncertainty = torch.sum((alpha / total_alpha) * (torch.digamma(total_alpha + 1) - torch.digamma(alpha + 1)), dim=1)
            distributional_uncertainty = point_entropy - data_uncertainty

            for i in range(len(distributional_uncertainty)):
                first_stat.append([tgt_path[i], tgt_lbl[i].item(), tgt_index[i].item(),
                                   distributional_uncertainty[i].item(),
                                   data_uncertainty[i].item()])

    sample_num = math.ceil(totality * active_ratio)
    fisrt_round_num = math.ceil(totality * active_ratio * cfg.TRAINER.KAPPA)

    # distributional uncertainty: higher value, higher consideration
    first_stat = sorted(first_stat, key=lambda x: x[3], reverse=True)  # reverse=True: descending order
    second_stat = first_stat[:fisrt_round_num]

    # data uncertainty: higher value, higher consideration
    second_stat = sorted(second_stat, key=lambda x: x[4], reverse=True)   # reverse=True: descending order
    second_stat = np.array(second_stat)

    selected_active_samples = second_stat[:sample_num, 0:2, ...]
    selected_candidate_ds_index = second_stat[:sample_num, 2, ...]
    selected_candidate_ds_index = np.array(selected_candidate_ds_index, dtype=np.int)

    tgt_selected_ds.add_item(selected_active_samples)
    tgt_unlabeled_ds.remove_item(selected_candidate_ds_index)

    return selected_active_samples



