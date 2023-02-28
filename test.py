from __future__ import print_function
import argparse
import os.path
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from core.datasets.image_list import ImageList
from core.models.network import ResNetFc
from core.datasets.transforms import build_transform
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


def evaluate(cfg, task, weight_path, args):
    torch.multiprocessing.set_sharing_strategy('file_system')
    cudnn.deterministic = True
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU_ID)
    kwargs = {'num_workers': cfg.DATALOADER.NUM_WORKERS, 'pin_memory': True}

    # prepare data
    test_transform = build_transform(cfg, is_train=False, choices=cfg.INPUT.TEST_TRANSFORMS)
    tgt_test_ds = ImageList(os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME, os.path.join(args.target + '_test.txt')), transform=test_transform)
    tgt_test_loader = DataLoader(tgt_test_ds, batch_size=cfg.DATALOADER.TEST.BATCH_SIZE, shuffle=False, **kwargs)

    # load model
    model = ResNetFc(class_num=cfg.DATASET.NUM_CLASS, cfg=cfg).cuda()
    weight = torch.load(weight_path)
    print(weight_path)
    model.load_state_dict(weight)

    print("Start testing")
    testacc = test(model, tgt_test_loader)
    print('Task: {}  testacc: {:.2f}'.format(task, testacc))




def main():
    parser = argparse.ArgumentParser(description='PyTorch Activate Domain Adaptation Testing')
    parser.add_argument('--cfg',
                        default='',
                        metavar='FILE',
                        help='path to config file',
                        type=str)
    parser.add_argument('--source', type=str, default='Art', help="The source domain")
    parser.add_argument('--target', type=str, default='Clipart', help="The target domain")
    parser.add_argument('--weight_path', type=str, default=None, help="the path of model weights")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    task = args.source + '2' + args.target
    ckt_path = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.NAME, task)
    if args.weight_path == 'None':
        weight_path = os.path.join(ckt_path, "final_model_{}.pth".format(task))
    else:
        weight_path = args.weight_path

    evaluate(cfg, task, weight_path, args)



if __name__ == '__main__':
    main()
