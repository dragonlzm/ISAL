import argparse
import os
from datetime import timedelta
import mmcv
import torch
import torch.distributed as dist
from torch.autograd import grad
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.apis import multi_gpu_test, single_gpu_test, set_random_seed
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mining.utils import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--in-path',
        type=str,
        help='the path to read the gradient on valset')    
    parser.add_argument(
        '--out-path',
        type=str,
        help='the path to save the s_test')
    parser.add_argument(
        '--img-num',
        type=int,
        default=1000,
        help='the max iteration for s_test calculation')  
    parser.add_argument(
        '--data-path',
        type=str, default=None,
        help='the path for raw data')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    all_pt_file = [(int(dir.split('_')[1].split('.')[0]), dir) for dir in os.listdir(args.in_path) if dir.startswith('tensor_')]
    all_pt_file = sorted(all_pt_file)
    val_sum_grad = [torch.load(os.path.join(args.in_path,file[1])) for file in all_pt_file]

    cfg = Config.fromfile(args.config)
    cfg.data_path = args.data_path

    if args.options is not None:
        cfg.merge_from_dict(args.options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None
    
    # set random seeds
    set_random_seed(42, deterministic=True)
    cfg.seed = 42

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, timeout=timedelta(0, 7200), **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    train_dataset = build_dataset(cfg.data.train_influ)
    train_data_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    test_cfg = cfg.test_cfg if cfg.get('test_cfg', False) else None
    model = build_detector(cfg.model, train_cfg=None, test_cfg=test_cfg)

    # select the parameter which need to calculate the gradient in FCOS
    for param in model.parameters():
        param.requires_grad = False
    for param in model.bbox_head.conv_centerness.parameters():
        param.requires_grad = True
    for param in model.bbox_head.conv_cls.parameters():
        param.requires_grad = True
    for param in model.bbox_head.conv_reg.parameters():
        param.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad and len(p.size()) != 1]
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = train_data_loader.CLASSES

    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)

    loss_val_mul_hessian_reverse = multi_gpu_s_test(val_sum_grad, model, params, train_data_loader, img_num=args.img_num)

    rank, _ = get_dist_info()
    if rank == 0: 
        for i in range(len(loss_val_mul_hessian_reverse[0])):
            per_param_grad = [torch.unsqueeze(torch.from_numpy(per_node_grad[i]), dim=0) for per_node_grad in loss_val_mul_hessian_reverse]
            num_of_node = len(per_param_grad)
            per_param_grad = torch.cat(per_param_grad, dim=0)
            per_param_grad = torch.sum(per_param_grad, dim=0)
            per_param_grad /= num_of_node
            torch.save(per_param_grad, os.path.join(args.out_path,"s_test_" + str(i) + '.pt'))    
    dist.barrier()


if __name__ == '__main__':
    main()


