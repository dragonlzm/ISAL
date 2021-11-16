import argparse
import copy
import os
import os.path as osp
import time
import mmcv
import torch
import glob
from datetime import timedelta
from mmcv import Config, DictAction
from mmcv.runner import init_dist, get_dist_info
from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from utils import *

from mining import mining


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--deploy', action='store_true', help='whether for deployment')
    parser.add_argument(
        '--ratio',
        type=str, default='1/10',
        help='ratio of training data to use')
    parser.add_argument(
        '--initial-ratio',
        type=str, default='1/10',
        help='initial ratio of training data to random sample')
    parser.add_argument(
        '--model-result',
        help='result file/dir by previous trained model')
    parser.add_argument(
        '--score-thresh',
        type=float, default=0.0,
        help='threshold of score from model result')
    parser.add_argument(
        '--mining-method',
        type=str, default='random',
        help='method to mine unlabeled data')
    parser.add_argument(
        '--sorted-reverse',
        action='store_true',
        help='whether to reversely sort the confidence scores')
    parser.add_argument(
        '--unlabeled-data-as-val',
        action='store_true',
        help='use the unlabeled dataset as validation')
    parser.add_argument(
        '--noised-student',
        action='store_true',
        help='whether to use noised student')
    parser.add_argument(
        '--ceal',
        action='store_true',
        help='whether to use ceal')
    parser.add_argument(
        '--noised-score-thresh',
        type=float, default=0.5,
        help='threshold of pseudo label score')
    parser.add_argument(
        '--finetune-model',
        action='store_true',
        help='whether to finetune model trained from previous step')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--divided-dataset',
        action='store_true',
        help='adding noise on the pseudo-label only')
    parser.add_argument(
        '--stratified-sample',
        type=int,
        default=1,
        help='level number to use the stratified sampling')
    parser.add_argument(
        '--data-path',
        type=str, default=None,
        help='the path for raw data')     

    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    args.ratio = eval(args.ratio)
    args.initial_ratio = eval(args.initial_ratio)

    cfg = Config.fromfile(args.config)
    cfg.deploy = args.deploy
    cfg.ratio = args.ratio
    cfg.initial_ratio = args.initial_ratio
    cfg.score_thresh = args.score_thresh
    cfg.mining_method = args.mining_method
    cfg.sorted_reverse = args.sorted_reverse
    cfg.noised_student = args.noised_student
    cfg.ceal = args.ceal
    cfg.noised_score_thresh = args.noised_score_thresh
    cfg.finetune_model = args.finetune_model 
    cfg.divided_dataset = args.divided_dataset
    cfg.stratified_sample = args.stratified_sample
    cfg.data_path = args.data_path
    if args.model_result != None:
        cfg.model_result = sorted(glob.glob(args.model_result))

    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, timeout=timedelta(0, 7200), **cfg.dist_params)

    # use the unlabeled data as validation set
    if args.unlabeled_data_as_val:
        cfg.data.val.ann_file = cfg.work_dir + "/unlabeled_datasets.json"
        cfg.data.val.img_prefix = cfg.data.train.full.img_prefix

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    if cfg.divided_dataset and cfg.noised_student:
        labeled_datasets, pseudo_datasets = mining(logger, cfg)
        datasets = [build_dataset(labeled_datasets), build_dataset(pseudo_datasets)]
    else:
        labeled_datasets = mining(logger, cfg)
        datasets = [build_dataset(labeled_datasets)]

    train_cfg = cfg.train_cfg if cfg.get('train_cfg', False) else None
    test_cfg = cfg.test_cfg if cfg.get('test_cfg', False) else None
    model = build_detector(
        cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)
    # For the latest mmdetection version
    # model.init_weights()
    if len(cfg.workflow) == 2 and not (cfg.divided_dataset and cfg.noised_student):
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES)
        # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta) 


if __name__ == '__main__':
    main()
