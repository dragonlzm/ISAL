import os
import json
import torch
import random
import argparse
import numpy as np
import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.trainer as trainer
from pycls.core.config import cfg, dump_cfg
import datetime

from mining.datasets import Cifar10, Cifar100, SVHN
from mining.mining import *
from tools.save_results import save_results_to_json


_DATASETS = {
    "cifar10": Cifar10,
    "cifar100": Cifar100,
    "svhn": SVHN,
}
_MINING_METHOD = {
    "random": random_sampling,
    "influence": influence_sampling,
    "grad_sampling": grad_sampling,
}

def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier in one active learning step')
    parser.add_argument('--config-path', help='train config file path')
    parser.add_argument('--data-dir', help='dataset path')
    parser.add_argument('--work-dir', help='the dir to save logs and models and temp data')
    parser.add_argument(
        '--dataset',
        type=str, default='cifar10',
        help='training dataset name')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpu-ids',
        type=int, default=0,
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--random-seed', type=int, default=42, help='random seed')
    # active learning config
    parser.add_argument(
        '--mining-method',
        type=str, default='random',
        help='method to mine unlabeled data')
    parser.add_argument(
        '--budget-ratio',
        type=str, default='1/10',
        help='budget ratio of training data')
    parser.add_argument(
        '--prev-labeled-file',
        type=str, default=None,
        help='labeled data file in previous step')   
    parser.add_argument(
        '--prev-unlabeled-file',
        type=str, default=None,
        help='unlabeled data file in previous step')   
    parser.add_argument(
        '--prev-model-result',
        type=str, default=None,
        help='model resutls on unlabeled data in previous step')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # calculate ratio
    args.budget_ratio = eval(args.budget_ratio)
    # load config file
    # split config_path
    (config_dir, config_name) = os.path.split(args.config_path)
    config.load_cfg(config_dir,config_name)
    cfg.OUT_DIR = args.work_dir
    # set gpu device
    if cfg.NUM_GPUS == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    # not auto resume from check point
    cfg.TRAIN.AUTO_RESUME = False
    # not output additional info to log
    cfg.VERBOSE = False
    # set cfg random
    cfg.RNG_SEED = args.random_seed
    cfg.CUDNN.BENCHMARK = False
    # according to mmdetection/mmdet/apis/train.py set_random_seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    
    if not args.prev_labeled_file and not args.prev_unlabeled_file:
        """on first active leraning step:
        -init dataset
        -train with random initial labeled dataset
        """   
        # generate initial dataset
        dataset = _DATASETS[args.dataset](args.data_dir, args.work_dir, initial_ratio=args.budget_ratio)
    else:
        """on other active leraning step:
        -mine data to get query index and create pickle file in work dir
        -train with prev + query labeled dataset
        """
        # query sample index
        query_indices = []
        query_indices = _MINING_METHOD[args.mining_method](args.prev_unlabeled_file, args.budget_ratio, args.prev_model_result)
        #generate new data file(labeled_data.pickle/unlabeled_data.pickle)
        dataset = _DATASETS[args.dataset](args.data_dir, args.work_dir,
                                          query_indices=query_indices, prev_labeled_file=args.prev_labeled_file, 
                                          prev_unlabeled_file=args.prev_unlabeled_file)
    
    # set training data path
    cfg.TRAIN.DATAPATH = os.path.join(args.work_dir, "labeled_data.pickle")
    cfg.TEST.DATAPATH = dataset._test_data_path
    
    # validate config
    config.assert_and_infer_cfg()
    # save config to OUT_DIR+CFG_DEST
    cfg.CFG_DEST = 'train.config'
    dump_cfg()

    # training classification model with labeled data and test on test set
    print("Training: started")
    inner_loop_start = datetime.datetime.now()
    dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.train_model)
    print("Training: finished")
    training_time = datetime.datetime.now() - inner_loop_start
    print("Training took {}".format(training_time))
    # save time in json file
    time_file_path = os.path.join(os.path.abspath(os.path.join(args.work_dir, "..")), "training_time.json")
    if os.path.exists(time_file_path):
        with open(time_file_path, 'r') as load_f:
            load_dict = json.load(load_f)
        load_dict["time"].append(training_time.seconds)
    else:
        load_dict = {}
        load_dict["time"] = [training_time.seconds]
    with open(time_file_path, 'w') as load_f:
            load_f.write(json.dumps(load_dict))
    print("Time file saved in ", time_file_path)

    # save results in timestamp/
    json_path = os.path.join(args.work_dir, "results.json")
    # get best err among all epoch
    al_results_accuarcy = (100.0 - cfg.TEST.BEST_TOP1_ERR)
    cur_data_num = len(dataset.labeled_indices)
    # set method_name
    mining_method_name = "_".join([args.mining_method, str(args.random_seed)])
    # combine current step results to the json file
    save_results_to_json(mining_method_name, cur_data_num, al_results_accuarcy, json_path)
    

if __name__ == '__main__':
    main()
