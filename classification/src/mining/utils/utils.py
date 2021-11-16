import os
import pickle
import random
import math

import numpy as np
from pycls.core.config import cfg
import pycls.core.checkpoint as cp
import pycls.core.trainer as trainer
import pycls.datasets.loader as data_loader
import torch
import torch.nn.functional as F
from tools.save_results import read_predictions_from_pkl

def load_indices(data_file_path):
    """load data from data_file_path and return the data indices
    param:
        data_file_path: the pickle data file path
    return：
        the indices of the data in pickle file 
    """
    # read data
    with open(data_file_path, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    
    return data[b"img_id"]


def load_indices_labels(data_file_path):
    """load data from data_file_path and return the data indices and labels
    param:
        data_file_path: the pickle data file path
    return:
        the indices and labels of the data in pickle file 
    """
    # read data
    with open(data_file_path, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    
    return data[b"img_id"], data[b"labels"]    


def save_query_info(score, score_sample_indices, query_indices, save_file_name = "mining_info.pickle"):
    """save query score and results to pkl file for analysis
    """
    # combine to one dict
    data = {}
    data[b"score"] = score
    data[b"score_sample_indices"] = score_sample_indices
    data[b"query_indices"] = query_indices
  
    # save in data.pickle
    with open(os.path.join(cfg.OUT_DIR, save_file_name), 'wb') as f:
        pickle.dump(data, f)


@torch.no_grad()
def get_sample_feature(test_data_file, model_path):
    """test data from test_data_file and return the data feature forward model"""
    # init and get test set path
    cfg.TEST.DATAPATH = test_data_file
    # model output mid-feature
    cfg.MODEL.OUTPUT_MID_LAYER = True

    """Evaluates a trained model."""
    # Setup training/testing environment
    trainer.setup_env()
    # Construct the model
    model = trainer.setup_model()
    # Load model weights
    cp.load_checkpoint(model_path, model)
    # Create data loaders and meters
    test_loader = data_loader.construct_test_loader(ouput_id=True)
    # Evaluate the model
    model.eval()
    feature_list = [] #data_num * feature_dim
    id_list = []
    for cur_iter, (inputs, labels, ids) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Compute the predictions
        _, features = model(inputs) #model output pred and features
        feature = features[3]
        feature = F.avg_pool2d(feature, 4)
        feature = feature.view(feature.size(0), -1)
        feature_list.extend(feature)
        id_list.extend(ids)
    
    assert len(feature_list) == len(id_list), "pred not match id"
    # reset for next training
    cfg.MODEL.OUTPUT_MID_LAYER = False
    return feature_list, id_list        