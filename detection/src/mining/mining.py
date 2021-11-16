import time
import argparse
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info

from .utils import *
from .methods import * 
from .datasets import *
from .registry import *


def mining(logger, cfg):
    full_datasets = cfg.data.train.full
    labeled_datasets = cfg.data.train.initial.copy()
    unlabeled_datasets = cfg.data.train.unlabeled.copy()
    save_dir = cfg.work_dir        

    if cfg.divided_dataset:
        pseudo_datasets = cfg.data.train.pseudo.copy()
        pseudo_datasets.ann_file = save_dir + "/pseudo_datasets.json"

    ann_file = full_datasets.ann_file
    labeled_ann_file = labeled_datasets.ann_file
    unlabeled_ann_file = unlabeled_datasets.ann_file
    labeled_datasets.ann_file = save_dir + "/labeled_datasets.json"
    unlabeled_datasets.ann_file = save_dir + "/unlabeled_datasets.json"

    rank, _ = get_dist_info()
    if rank == 0:
        #if labeled_datasets.type == 'ISALCocoDataset':
        load_func = coco_load
        combine_func = coco_combine
        save_func = coco_save        
        if cfg.noised_student:
            logger.info('Use teacher model to pseudo label and retrain student model')
            labeled_image_set, meta = load_func(labeled_ann_file, config=cfg)
            unlabeled_image_set, _  = load_func(unlabeled_ann_file, not cfg.deploy, config=cfg)
            miner = MINERS.get(cfg.mining_method)(logger, cfg)
            selected_valued_image_set = miner.pseudo_label(labeled_image_set, unlabeled_image_set)
            remained_valued_image_set = unlabeled_image_set
            if not cfg.divided_dataset:
                selected_valued_image_set = combine_func([selected_valued_image_set, labeled_image_set])
            else:
                save_func(selected_valued_image_set, meta, pseudo_datasets.ann_file, config=cfg)
                selected_valued_image_set = labeled_image_set
        elif cfg.ratio == 0:
            logger.info(f'Randomly mine {cfg.initial_ratio} unlabeled data')
            labeled_image_set = []
            unlabeled_image_set, meta = load_func(ann_file, config=cfg)
            cfg.sorted_reverse = False
            miner = MINERS.get('random')(logger, cfg)
            selected_valued_image_set, remained_valued_image_set = miner.run(unlabeled_image_set, cfg.initial_ratio)
        elif cfg.ceal:
            logger.info(f'Mine {cfg.ratio} unlabeled data by {cfg.mining_method}')
            # load the annotations used in last step and filter the pseudo-label in the last step
            labeled_image_set, meta = load_func(labeled_ann_file, config=cfg)
            unlabeled_image_set, _  = load_func(unlabeled_ann_file, config=cfg)
            miner = MINERS.get(cfg.mining_method)(logger, cfg)
            
            # mine the new gt for this round 
            selected_valued_image_set, remained_valued_image_set = miner.run(unlabeled_image_set, cfg.ratio)
            save_func(selected_valued_image_set, meta, save_dir + "/new_added_data.json", config=cfg)
            if not cfg.finetune_model:
                selected_valued_image_set = combine_func([selected_valued_image_set, labeled_image_set])

            # add the pseudo-label for this round
            pseudo_label_image_set = miner.pseudo_label(selected_valued_image_set, remained_valued_image_set)
            selected_valued_image_set = combine_func([selected_valued_image_set, pseudo_label_image_set])
        else:
            logger.info(f'Mine {cfg.ratio} unlabeled data by {cfg.mining_method}')
            
            labeled_image_set, meta = load_func(labeled_ann_file, config=cfg)
            unlabeled_image_set, _  = load_func(unlabeled_ann_file, not cfg.deploy, config=cfg)
            miner = MINERS.get(cfg.mining_method)(logger, cfg)
            selected_valued_image_set, remained_valued_image_set = miner.run(unlabeled_image_set, cfg.ratio)
            save_func(selected_valued_image_set, meta, save_dir + "/new_added_data.json", config=cfg)
            if not cfg.finetune_model:
                selected_valued_image_set = combine_func([selected_valued_image_set, labeled_image_set])
        
        logger.info(f'{len(selected_valued_image_set)} images are selected as labeled.')
        logger.info(f'{len(remained_valued_image_set)} images are selected as unlabeled.') 
        logger.info(f'{len(labeled_image_set)} images are used as labeled in previous step.')
        logger.info(f'{len(unlabeled_image_set)} images are used as unlabeled in previous step.')
        save_func(selected_valued_image_set, meta, labeled_datasets.ann_file, config=cfg)
        save_func(remained_valued_image_set, meta, unlabeled_datasets.ann_file, config=cfg)
    dist.barrier()
    if cfg.divided_dataset and cfg.noised_student:
        return labeled_datasets, pseudo_datasets
    else:
        return labeled_datasets


def convert_inference_to_training(logger, cfg, target_file):
    load_func = coco_load
    save_func = coco_save   

    labeled_datasets = cfg.data.train.initial.copy()
    unlabeled_datasets = cfg.data.train.unlabeled.copy()
    labeled_ann_file = labeled_datasets.ann_file
    unlabeled_ann_file = unlabeled_datasets.ann_file
    target_file.ann_file = cfg.work_dir + "/pseudo_datasets.json"

    rank, _ = get_dist_info()
    if rank == 0:
        #logger.info('Use teacher model to pseudo label and retrain student model')
        labeled_image_set, meta = load_func(labeled_ann_file, config=cfg)
        unlabeled_image_set, _  = load_func(unlabeled_ann_file, not cfg.deploy, config=cfg)
        miner = MINERS.get(cfg.mining_method)(logger, cfg)
        selected_valued_image_set = miner.pseudo_label(labeled_image_set, unlabeled_image_set)
        save_func(selected_valued_image_set, meta, target_file.ann_file, config=cfg)
    dist.barrier()

    return target_file