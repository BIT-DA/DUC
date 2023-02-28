import torch
import torch.nn as nn


class EDL_Loss(nn.Module):
    """
    evidence deep learning loss
    """
    def __init__(self, cfg):
        super(EDL_Loss, self).__init__()
        self.cfg = cfg

    def forward(self, logits, labels=None):
        alpha = torch.exp(logits)
        total_alpha = torch.sum(alpha, dim=1, keepdim=True)  # total_alpha.shape: [B, 1]
        if labels is None:
            labels = torch.max(alpha, dim=1)[1]

        one_hot_y = torch.eye(logits.shape[1]).cuda()
        one_hot_y = one_hot_y[labels]
        one_hot_y.requires_grad = False

        loss_nll = torch.sum(one_hot_y * (total_alpha.log() - alpha.log())) / logits.shape[0]

        uniform_bata = torch.ones((1, logits.shape[1])).cuda()
        uniform_bata.requires_grad = False
        total_uniform_beta = torch.sum(uniform_bata, dim=1)  # new_total_alpha.shape: [1]
        new_alpha = one_hot_y + (1.0 - one_hot_y) * alpha
        new_total_alpha = torch.sum(new_alpha, dim=1)  # new_total_alpha.shape: [B]
        loss_KL = torch.sum(
            torch.lgamma(new_total_alpha) - torch.lgamma(total_uniform_beta) - torch.sum(torch.lgamma(new_alpha), dim=1) \
            + torch.sum((new_alpha - 1) * (torch.digamma(new_alpha) - torch.digamma(new_total_alpha.unsqueeze(1))), dim=1)
        ) / logits.shape[0]

        return loss_nll, loss_KL



