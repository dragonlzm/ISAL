import json
import torch
import random
import os

from ..utils import *
from ..registry import MINERS
from .mining_algorithm import MiningAlgorithm


@MINERS.register_module(name='localization_stability')
class LocalizationStabilityAlgorithm(MiningAlgorithm):
    def load(self, score_thresh=0):
        if len(self.config.model_result) <= 1:
            self.logger.info(f'Error! In {self.__class__.__name__}.load() the len of model_result is smaller or equal to 1\n')
            os._exit(0)
        all_dict = []
        for i in range(len(self.config.model_result)):
            image_dict = {}
            file = self.config.model_result[i]
            data = json.load(open(file, 'r'))
            for annotation in data:
                image_name = annotation['image_id']
                if image_name not in image_dict:
                    image_dict[image_name] = {'confidences': [], 'bboxes': []}
                instance_dict = image_dict[image_name]
                if annotation['score'] < score_thresh:
                    continue
                instance_dict['confidences'].append(annotation['score'])
                instance_dict['bboxes'].append(annotation['bbox'])
            all_dict.append(image_dict)
        return all_dict   

    def mining(self, unlabled_data, score_dict, ratio):
        scored_data, unscored_data = [], []
        for data in unlabled_data:
            image_id = int(data['info']['id'])
            if image_id not in score_dict[0]:
                unscored_data.append(data)
                continue
            all_ref_bboxes = score_dict[0][image_id]['bboxes']
            all_ref_scores = score_dict[0][image_id]['confidences']
            if len(all_ref_bboxes) == 0:
                unscored_data.append(data)
            else:
                data['value'] = self.value(image_id, all_ref_bboxes, all_ref_scores, score_dict)
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
    
    def value(self, image_id, all_ref_bboxes, all_ref_scores, score_dict):
        image_score = 0
        bbox_score_sum = 0
        # For each reference box, finding out the max iou between the reference box and all the predicted boxes 
        # under each noise level and calculating the average over the noise levels.
        # Then calculating the weight average of all ref_bbox to represent the score of an img,
        # the weight is the score of the refence bbox
        for i, ref_bbox in enumerate(all_ref_bboxes):
            average_iou = 0
            for level in range(1, len(score_dict)):
                if image_id not in score_dict[level]:
                    continue
                all_pred_bboxes = score_dict[level][image_id]['bboxes']
                if len(all_pred_bboxes) == 0:
                    continue
                average_iou += max([calculate_iou(_, ref_bbox) for _ in all_pred_bboxes])
            average_iou /= (len(score_dict) - 1)
            image_score += (average_iou * all_ref_scores[i])
            bbox_score_sum += (all_ref_scores[i])
        image_score /= bbox_score_sum
        return image_score


@MINERS.register_module(name='localization_stability_min_confidence')
class LocalizationStabilityAndMinConfidenceAlgorithm(LocalizationStabilityAlgorithm): 
    def mining(self, unlabled_data, score_dict, ratio):
        scored_data, unscored_data = [], []
        for data in unlabled_data:
            image_id = int(data['info']['id'])
            if image_id not in score_dict[0]:
                unscored_data.append(data)
                continue
            all_ref_bboxes = score_dict[0][image_id]['bboxes']
            all_ref_scores = score_dict[0][image_id]['confidences']
            if len(all_ref_bboxes) == 0:
                unscored_data.append(data)
            else:
                # Obtain the localization stability score
                ls_score = self.value(image_id, all_ref_bboxes, all_ref_scores, score_dict)
                confidences = torch.tensor(all_ref_scores)
                # Use the min socre of all reference bboxes to represent the confidence
                # of img and calculate the uncertainty
                uncertainty = 1 + self.value_confidence(confidences)
                data['value'] = uncertainty - ls_score
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
    
    def value_confidence(self, confidences):
        if len(confidences.size()) == 2:
            confidences = torch.max(confidences, dim=1)[0]
        confidences = -confidences
        return torch.max(confidences).item()