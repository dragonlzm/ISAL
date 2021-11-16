import argparse
import os
import os.path as osp
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.autograd import grad
import mmcv
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.apis import multi_gpu_test, single_gpu_test, set_random_seed
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mining.utils import *
from mining import mining, convert_inference_to_training
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
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--mining-method',
        type=str, default='random',
        help='method to mine unlabeled data')
    parser.add_argument(
        '--collect-loss',
        action='store_true',
        help='return loss per img instead of bbox')
    parser.add_argument(
        '--sorted-reverse',
        action='store_true',
        help='whether to reversely sort the confidence scores')
    parser.add_argument(
        '--in-path',
        type=str,
        help='the path to read the s_test')
    parser.add_argument(
        '--model-result',
        help='result file/dir by previous trained model')
    parser.add_argument('--deploy', action='store_true', help='whether for deployment')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--gt', action='store_true', help='whether to use the gt')
    parser.add_argument(
        '--noised-score-thresh',
        type=float, default=0.5,
        help='threshold of score from model result')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--score-thresh',
        type=float, default=0.0,
        help='threshold of score from model result')
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
    
    all_pt_file = [(int(dir.split('_')[-1].split('.')[0]), dir) for dir in os.listdir(args.in_path) if dir.startswith('s_test_')]
    all_pt_file = sorted(all_pt_file)
    s_test = [torch.load(os.path.join(args.in_path,file[1])) for file in all_pt_file]

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg.data_path = args.data_path

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
    if args.gt:
        train_dataset = build_dataset(cfg.data.train_influ)
        train_data_loader = build_dataloader(
            train_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
    else:
        cfg.deploy = args.deploy
        cfg.mining_method = args.mining_method
        cfg.ratio = 0
        cfg.noised_student = True
        cfg.noised_score_thresh = args.noised_score_thresh
        cfg.model_result = args.model_result
        cfg.work_dir = args.work_dir

        timestamp = time.strftime('%Y%m%d_%H%M%S'+ '_mining', time.localtime())
        log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

        target_file = cfg.data.train_influ
        train_dataset_cfg = convert_inference_to_training(logger, cfg, target_file)
        train_dataset = build_dataset(train_dataset_cfg)

        #train_dataset = build_dataset(cfg.data.test)
        train_data_loader = build_dataloader(
            train_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)


    # build the model and load checkpoint
    train_cfg = cfg.train_cfg if cfg.get('train_cfg', False) else None
    test_cfg = cfg.test_cfg if cfg.get('test_cfg', False) else None
    model = build_detector(
        cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

    # select the parameter which need to calculate the gradient in FCOS
    for param in model.parameters():
        param.requires_grad = False
    #for param in model.rpn_head.rpn_conv.parameters():
    #    param.requires_grad = True
    for param in model.rpn_head.rpn_cls.parameters():
        param.requires_grad = True
    for param in model.rpn_head.rpn_reg.parameters():
        param.requires_grad = True                
    for param in model.roi_head.bbox_head.fc_cls.parameters():
        param.requires_grad = True
    for param in model.roi_head.bbox_head.fc_reg.parameters():
        param.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad and len(p.size()) != 1]
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = train_data_loader.CLASSES

    if not distributed:
        print("while calculating the loss for all validation img, the undistributed version have not been implemented")
        return
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        influences = multi_gpu_cal_influence(model, train_data_loader, tmpdir=args.tmpdir, gpu_collect=args.gpu_collect, params=params, s_test=s_test, model_name=cfg.model.type)
    
    rank, _ = get_dist_info()
    if rank == 0:
        kwargs = {} if args.options is None else args.options
        if args.format_only:
            kwargs.update({"collect_influence":True})
            train_dataset.format_results(influences, **kwargs)
    dist.barrier()

if __name__ == '__main__':
    main()