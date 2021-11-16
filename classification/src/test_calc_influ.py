"""Test a trained classification model."""

import argparse
import os

import pycls.core.builders as builders
import pycls.core.checkpoint as cp
import pycls.core.config as config
import pycls.core.logging as logging
import pycls.core.trainer as trainer
import pycls.datasets.loader as data_loader
import pycls.core.net as net
import json
import torch
from torch.autograd import grad
from pycls.core.config import cfg
from tools.save_results import save_predictions_to_pkl
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='test a trained classifier')
    parser.add_argument('--config-path', help='train config file path')
    parser.add_argument('--data-dir', help='dataset path')
    parser.add_argument('--work-dir', help='the dir to save results')
    parser.add_argument('--model-path', help='the path of the trained model(check point file)')
    parser.add_argument('--test-data-file', help='the test data path to evaluate')
    parser.add_argument('--in-path', type=str, help='the path to load the s_test')  
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    all_pt_file = [(int(dir.split('_')[-1].split('.')[0]), dir) for dir in os.listdir(args.in_path) if dir.startswith('s_test_')]
    all_pt_file = sorted(all_pt_file)
    s_test = [torch.load(os.path.join(args.in_path,file[1])) for file in all_pt_file]
    # load config file
    # split config_path
    (config_dir, config_name) = os.path.split(args.config_path)
    config.load_cfg(config_dir,config_name)
    cfg.OUT_DIR = args.work_dir
    # init the logger before other steps
    cfg.LOG_DEST = 'file'
    log_file = 'test.log'
    logging.setup_logging(log_file)
    logger = logging.get_logger(__name__)
    # init and get test set path
    cfg.TEST.DATAPATH = args.test_data_file
    cfg.TEST.BATCH_SIZE = 1
    # check point file which contained model weight
    cfg.TEST.WEIGHTS = args.model_path

    # validate config
    config.assert_and_infer_cfg()

    """Evaluates a trained model."""
    # Setup training/testing environment
    trainer.setup_env()
    # Construct the model
    model = trainer.setup_model()

    # select the parameter which need to calculate the gradient in FCOS
    #for param in model.parameters():
    #    param.requires_grad = False
    #for param in model.linear.parameters():
    #    param.requires_grad = True
    #for param in model.layer4.parameters():
    #    param.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad and len(p.size()) != 1]

    # Load model weights
    cp.load_checkpoint(cfg.TEST.WEIGHTS, model)
    # Create data loaders and meters
    test_loader = data_loader.construct_test_loader(ouput_id=True)
    # Evaluate the model
    model.eval()
    influence_list, idxes_list = calc_influence_funtion(model, test_loader, params, s_test)
    pkl_file_path = os.path.join(args.work_dir, "result_on_unlabeled.pickle")
    save_predictions_to_pkl(influence_list, idxes_list, pkl_file_path)

if __name__ == "__main__":
    main()
