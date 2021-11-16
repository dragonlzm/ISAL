import abc
import json
import numpy as np
import io
from ..utils import *
from ..registry import MINERS


class MiningAlgorithm(object):
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        if self.config.ratio != 0:
            self.select_func = stratified_sampling(self.config.stratified_sample)
        else:
            self.select_func = select


    def load_pseudo_label_bbox(self, score_thresh, bbox_start_id):
        image_dict = {}
        if len(self.config.model_result) == 1:
            self.config.model_result = self.config.model_result[0] 
        data = json.load(open(self.config.model_result, 'r'))
        for annotation in data:
            if annotation['score'] < score_thresh:
                continue
            image_id = annotation['image_id']
            if image_id not in image_dict:
                image_dict[image_id] = []
            tmp = {
                'iscrowd': 0,
                'segmentation': [],
                'id': bbox_start_id,
                'is_pseudo_label': True,
                'area': annotation['bbox'][2] * annotation['bbox'][3],
                'score': annotation['score']
            }
            tmp.update(annotation)
            image_dict[image_id].append(tmp)
            bbox_start_id += 1
        return image_dict

    def pseudo_label(self, labeled_data, unlabeled_data):
        assert self.config.noised_student or self.config.ceal
        all_bbox_data = labeled_data + unlabeled_data
        max_bbox_id = 0
        for data in all_bbox_data:
            bbox_id = max([inst['id'] for inst in data['instances']])
            max_bbox_id = max(max_bbox_id, bbox_id)
        noised_score_thresh = self.config.get("noised_score_thresh", 0.5)
        bbox_dict = self.load_pseudo_label_bbox(noised_score_thresh, max_bbox_id+1)
        # pseudo label is from remained data,
        # so we only consider image in remained data
        to_add_data = []
        for data in unlabeled_data:
            image_id = int(data['info']['id'])
            if image_id not in bbox_dict:
                continue
            tmp = {
                'info': data['info'],
                'instances': bbox_dict[image_id]
            }
            to_add_data.append(tmp) 
        self.logger.info(f'{len(to_add_data)} pseudo images are added as labeled')
        return to_add_data

    def run(self, unlabeled_data, ratio):
        score_thresh = self.config.get("score_thresh", 0)
        score_dict = self.load(score_thresh)
        selected_data, remained_data = self.mining(unlabeled_data, score_dict, ratio)
        return selected_data, remained_data

    @abc.abstractmethod
    def load(self, score_thresh=0):
        pass

    @abc.abstractmethod
    def mining(self, unlabeled_data, score_dict, ratio):
        pass


@MINERS.register_module(name='random')
class RandomMiningAlgorithm(MiningAlgorithm):
    def load(self, score_thresh=0):
        return None

    def mining(self, unlabeled_data, score_dict, ratio):
        for data in unlabeled_data:
            data['value'] = np.random.rand()
        
        # if the ratio is large than 1 then use the ratio as select_num
        select_num = ratio
        if ratio <= 1:
            select_num = int(ratio * 1.0 * len(unlabeled_data))

        return self.select_func(unlabeled_data, select_num, reverse=self.config.sorted_reverse)
