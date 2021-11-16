"""Test a trained classification model."""

import argparse
import os
import pycls.core.checkpoint as cp
import pycls.core.config as config
import pycls.core.logging as logging
import pycls.core.trainer as trainer
import pycls.datasets.loader as data_loader
import torch
from pycls.core.config import cfg
import random
import numpy as np

from tools.save_results import save_predictions_to_pkl


def parse_args():
    parser = argparse.ArgumentParser(description='test a trained classifier')
    parser.add_argument('--config-path', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save results')
    parser.add_argument('--model-path', help='the path of the trained model(check point file)')
    parser.add_argument('--test-data-file', help='the test data path to evaluate')
    parser.add_argument('--random-seed', type=int, default=42, help='random seed')
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
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
    # check point file which contained model weight
    cfg.TEST.WEIGHTS = args.model_path
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

    # validate config
    config.assert_and_infer_cfg()

    """Evaluates a trained model."""
    # Setup training/testing environment
    trainer.setup_env()
    # Construct the model
    model = trainer.setup_model()
    # Load model weights
    cp.load_checkpoint(cfg.TEST.WEIGHTS, model)
    # Create data loaders and meters
    test_loader = data_loader.construct_test_loader(ouput_id=True)
    # Evaluate the model
    model.eval()
    preds_list = [] #data_num *class_num
    id_list = []
    for cur_iter, (inputs, labels, ids) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Compute the predictions
        preds = model(inputs) #batch_size * class_num
        # foward softmax
        preds = torch.nn.functional.softmax(preds, dim=1)
        # save predictions and image_id
        preds_list.extend(preds.tolist())
        id_list.extend(ids)
    
    assert len(preds_list) == len(id_list), "pred not match id"
    # save results
    pkl_file_path = os.path.join(args.work_dir, "result_on_unlabeled.pickle")
    if not os.path.exists(pkl_file_path):
        save_predictions_to_pkl(preds_list, id_list, pkl_file_path)
    else:
        #save resutls with pseudo
        pseudo_pkl_file_path = os.path.join(args.work_dir, "pseudo_model_result_on_unlabeled.pickle")
        save_predictions_to_pkl(preds_list, id_list, pseudo_pkl_file_path)
        

if __name__ == "__main__":
    main()
