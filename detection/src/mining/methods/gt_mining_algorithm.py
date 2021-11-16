import json
import torch
import random
import os
import io

from ..utils import *
from ..registry import MINERS
from .mining_algorithm import MiningAlgorithm


@MINERS.register_module(name='gt_loss')
class GTLossAlgorithm(MiningAlgorithm):
    def load(self, score_thresh):
        image_dict = {}
        if len(self.config.model_result) == 1:
            self.config.model_result = self.config.model_result[0]     
        data = json.load(open(self.config.model_result, 'r'))

        for annotation in data:
            image_name = annotation['image_id']
            if image_name not in image_dict:
                image_dict[image_name] = {key:annotation[key] for key in annotation.keys() if key != 'image_id'}
        return image_dict

    def mining(self, unlabled_data, score_dict, ratio):
        scored_data, unscored_data = [], []
        for data in unlabled_data:
            image_id = int(data['info']['id'])
            if image_id not in score_dict:
                unscored_data.append(data)
                continue
            loss_dict = score_dict[image_id]

            data['value'] = self.value(loss_dict)
            scored_data.append(data)
        
        self.logger.info(f'{len(scored_data)} images\' score is higher than score_thresh')
        self.logger.info(f'{len(unscored_data)} images\' score is lower than score_thresh')

        if ratio <= 1:
            to_select_num = int(ratio * len(unlabled_data))
        else:
            to_select_num = int(ratio)

        if len(scored_data) >= to_select_num:
            selected_data, remained_data = self.select_func(scored_data, to_select_num, self.config.sorted_reverse)
            remained_data.extend(unscored_data)
        else:
            random.shuffle(unscored_data)
            to_add_num = to_select_num - len(scored_data)
            selected_data = scored_data + unscored_data[:to_add_num]
            remained_data = unscored_data[to_add_num:]
        return selected_data, remained_data 
    
    def value(self, loss_dict):
        loss = 0
        for key in loss_dict.keys():
            loss += loss_dict[key]
        return loss


@MINERS.register_module(name='influence')
class InfluenceAlgorithm(GTLossAlgorithm):    
    def value(self, loss_dict):
        return loss_dict['influence']


@MINERS.register_module(name='coreset')
class CoresetAlgorithm(GTLossAlgorithm):    
    def value(self, loss_dict):
        return loss_dict['coreset']