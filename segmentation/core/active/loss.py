import torch
import torch.nn as nn



def get_valid_pixels(out, label, mask):
    out, label, mask = out.cuda(), label.cuda(), mask.cuda()

    assert not label.requires_grad
    assert not mask.requires_grad
    assert out.dim() == 4
    assert label.dim() == 3
    assert label.dim() == mask.dim()
    assert out.size(0) == label.size(0), "{0} vs {1} ".format(out.size(0), label.size(0))
    assert out.size(2) == label.size(1), "{0} vs {1} ".format(out.size(2), label.size(1))
    assert out.size(3) == label.size(2), "{0} vs {1} ".format(out.size(3), label.size(3))

    # convert to pred (BHW, C) and label(BHW, 1)
    n, c, h, w, = out.size()
    out = out.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    label = label.view(-1, 1)
    mask = mask.view(-1, 1)

    target_mask = (label != 255) * (mask != 255)
    target_mask = target_mask.view(-1)

    # select pixels that are not 255 and their mask is True
    valid_out = out[target_mask]
    valid_label = label[target_mask]

    # unselect pixels that are 255 and their mask is False
    target_invalid_mask = (mask == 255)
    if torch.sum(target_invalid_mask) > 0:
        target_invalid_mask = target_invalid_mask.view(-1)
        invalid_out = out[target_invalid_mask]
    else:
        invalid_out = None

    return valid_out, valid_label, invalid_out



class EDL_Loss(nn.Module):
    """
    evidence deep learning loss
    """
    def __init__(self, cfg):
        super(EDL_Loss, self).__init__()
        self.cfg = cfg

    def forward(self, alpha, label):
        # alpha.shape: [BHW, C], label.shape:[BHW, 1]
        alpha, label = alpha.cuda(), label.cuda()
        label = label.squeeze(1)      # convert label.shape to [BHW]
        total_alpha = torch.sum(alpha, dim=1, keepdim=True)  # total_alpha.shape[BHW, 1]

        one_hot_y = torch.eye(alpha.shape[1]).cuda()
        one_hot_y = one_hot_y[label]
        one_hot_y.requires_grad = False

        temp_loss = total_alpha.log() - alpha.log()
        temp_loss = torch.gather(temp_loss, dim=1, index=label.unsqueeze(1))
        loss_nll = torch.sum(temp_loss) / label.shape[0]

        uniform_bata = torch.ones((1, alpha.shape[1])).cuda()
        uniform_bata.requires_grad = False
        total_uniform_beta = torch.sum(uniform_bata, dim=1)  # new_total_alpha.shape: [1]

        new_alpha = one_hot_y + (1.0 - one_hot_y) * alpha
        new_total_alpha = torch.sum(new_alpha, dim=1)  # new_total_alpha.shape: [B]

        eps=1e-7
        loss_KL = torch.sum(
            torch.lgamma(new_total_alpha) - torch.lgamma(total_uniform_beta) - torch.sum(torch.lgamma(new_alpha + eps), dim=1) \
            + torch.sum((new_alpha - 1) * (torch.digamma(new_alpha + eps) - torch.digamma(new_total_alpha.unsqueeze(1))), dim=1)
        ) / label.shape[0]

        return loss_nll, loss_KL

