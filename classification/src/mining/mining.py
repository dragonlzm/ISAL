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
from .utils import load_indices, load_indices_labels, save_query_info, get_sample_feature


def random_sampling(prev_unlabeled_file, budget_ratio, prev_model_result=None):
    """sampling query samples randomly
    param:
        prev_unlabeled_file: unlabelede data file in previous step
        budget_ratio: the num of query sample
    return:
        the indices of query data sample
    """
    unlabeled_indices = load_indices(prev_unlabeled_file)
    # sample randomly
    query_indices = random.sample(unlabeled_indices, math.ceil(budget_ratio * len(unlabeled_indices)))

    return query_indices


def influence_sampling(prev_unlabeled_file, budget_ratio, prev_model_result):
    """sampling the query sample which has lowest influence value"""
    unlabeled_indices = load_indices(prev_unlabeled_file)
    pred_array, id_array = read_predictions_from_pkl(prev_model_result)

    # ascending index
    id_array = list(id_array)
    ranked_preds_index = np.argsort(pred_array)[:math.ceil(budget_ratio * len(unlabeled_indices))]
    query_indices = [id_array[i] for i in ranked_preds_index]

    return query_indices


def greedy_k_center(labeled_representation, unlabeled_representation, amount):
    greedy_indices = []
    # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
    min_dist = torch.min(torch.cdist(labeled_representation[0, :].view(1, labeled_representation.size()[1]), unlabeled_representation), dim=0)[0]
    min_dist = min_dist.view(1, min_dist.size()[0])
    for j in range(1, labeled_representation.size()[0], 100):
        if j + 100 < labeled_representation.size()[0]:
            dist = torch.cdist(labeled_representation[j:j+100, :], unlabeled_representation)
        else:
            dist = torch.cdist(labeled_representation[j:, :], unlabeled_representation)
        min_dist = torch.stack((min_dist, torch.min(dist, dim=0)[0].view(1, min_dist.size()[1])), dim=0)
        min_dist = torch.min(min_dist, dim=0)[0]

    # iteratively insert the farthest index and recalculate the minimum distances:
    farthest = torch.argmax(min_dist)
    greedy_indices.append(farthest)
    for i in range(amount-1):
        dist = torch.cdist(unlabeled_representation[greedy_indices[-1], :].view(1,unlabeled_representation.size()[1]), unlabeled_representation)
        min_dist = torch.stack((min_dist, dist.view(1, min_dist.size()[1])), dim=0)
        min_dist = torch.min(min_dist, dim=0)[0]
        farthest = torch.argmax(min_dist)
        greedy_indices.append(farthest)

    return greedy_indices


def grad_sampling(prev_unlabeled_file, budget_ratio, prev_model_result):
    """sampling the query sample which has highest gradient similarity value"""
    unlabeled_indices = load_indices(prev_unlabeled_file)
    pred_array, id_array = read_predictions_from_pkl(prev_model_result)

    # ascending index
    id_array = list(id_array)
    ranked_preds_index = np.argsort(pred_array)[:math.ceil(budget_ratio * len(unlabeled_indices))]
    query_indices = [id_array[i] for i in ranked_preds_index]

    return query_indices