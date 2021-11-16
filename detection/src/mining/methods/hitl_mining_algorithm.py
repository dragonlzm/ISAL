import json
import torch
import random
import os.path as osp

from ..utils import *
from ..registry import MINERS
from .mining_algorithm import MiningAlgorithm


class GeneralMiningAlgorithm(MiningAlgorithm):
    def load(self, score_thresh):
        image_dict = {}
        if len(self.config.model_result) == 1:
            self.config.model_result = self.config.model_result[0]
        data = json.load(open(self.config.model_result, 'r'))
        for annotation in data:
            if annotation['score'] < score_thresh:
                continue
            image_id = annotation['image_id']
            if image_id not in image_dict:
                image_dict[image_id] = {'confidences': []}
            instance_dict = image_dict[image_id]
            instance_dict['confidences'].append(annotation['score'])
        return image_dict

    def mining(self, unlabled_data, score_dict, ratio):
        scored_data, unscored_data = [], []
        for data in unlabled_data:
            image_id = int(data['info']['id'])
            if image_id not in score_dict:
                unscored_data.append(data)
                continue
            confidences = torch.tensor(score_dict[image_id]['confidences'])
            if len(confidences) == 0:
                unscored_data.append(data)
            else:
                data['value'] = self.value(confidences)
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

    def value(self, confidences):
        raise NotImplementedError


@MINERS.register_module(name='max_confidence')
class MaxConfidenceMiningAlgorithm(GeneralMiningAlgorithm):
    def value(self, confidences):
        if len(confidences.size()) == 2:
            confidences = torch.max(confidences, dim=1)[0]
        return torch.max(confidences).item()


@MINERS.register_module(name='least_confidence')
class LeastConfidenceMiningAlgorithm(GeneralMiningAlgorithm):
    def value(self, confidences):
        if len(confidences.size()) == 2:
            confidences = torch.max(confidences, dim=1)[0]
        confidences = -confidences
        return torch.max(confidences).item()

@MINERS.register_module(name='mean_confidence')
class MeanConfidenceMiningAlgorithm(GeneralMiningAlgorithm):
    def value(self, confidences):
        if len(confidences.size()) == 2:
            confidences = torch.max(confidences, dim=1)[0]
        return torch.mean(confidences).item()

@MINERS.register_module(name='max_entropy')
class MaxEntropyMiningAlgorithm(GeneralMiningAlgorithm):
    def value(self, confidences, eps=1e-10):
        if len(confidences.size()) == 1:
            confidences = confidences.view(-1, 1)
            confidences = torch.cat([1.0 - confidences, confidences], dim=1)
        elif torch.min(torch.sum(confidences, dim=1)).item() < 1.0 - eps:
            confidences = torch.cat([1.0 - torch.sum(confidences, dim=1, keepdim=True), confidences], dim=1)
        ent = confidences * torch.log(confidences + eps)
        ent = -1.0 * torch.sum(ent, dim=1)
        return torch.max(ent).item()


@MINERS.register_module(name='marginal_sampling')
class MarginalSamplingMiningAlgorithm(GeneralMiningAlgorithm):
    def value(self, confidences, eps=1e-10):
        # check if background class is missing
        if len(confidences.size()) == 1:
            confidences = confidences.view(-1, 1)
            confidences = torch.cat([1.0 - confidences, confidences], dim=1)
        elif torch.min(torch.sum(confidences, dim=1)).item() < 1.0 - eps:
            confidences = torch.cat([1.0 - torch.sum(confidences, dim=1, keepdim=True), confidences], dim=1)
        confidences, _ = torch.sort(confidences, dim=1, descending=True)
        diff = confidences[:, 1] - confidences[:, 0]
        return torch.max(diff).item()


@MINERS.register_module(name='random_plus_least_confidence')
class RandomPlusLeastConfidence(LeastConfidenceMiningAlgorithm):
    def mining(self, unlabled_data, score_dict, ratio):
        scored_data, unscored_data = [], []
        for data in unlabled_data:
            image_id = int(data['info']['id'])
            if image_id not in score_dict:
                unscored_data.append(data)
                continue
            confidences = torch.tensor(score_dict[image_id]['confidences'])
            if len(confidences) == 0:
                unscored_data.append(data)
            else:
                data['value'] = self.value(confidences)
                scored_data.append(data)
        self.logger.info(f'{len(scored_data)} images\' score is higher than score_thresh')
        self.logger.info(f'{len(unscored_data)} images\' score is lower than score_thresh')
        if ratio <= 1:
            least_confidence_select_num = int(ratio * len(unlabled_data) / 2) 
            random_select_num = int(ratio * len(unlabled_data)) - least_confidence_select_num
        else:
            least_confidence_select_num = int(ratio / 2)
            random_select_num = int(ratio) - least_confidence_select_num
        if len(scored_data) >= least_confidence_select_num:
            selected_data, remained_data = self.select_func(scored_data, least_confidence_select_num, self.config.sorted_reverse)
            remained_data.extend(unscored_data)
            to_add_num = random_select_num
        else:
            selected_data = scored_data
            remained_data = unscored_data
            to_add_num = random_select_num + (least_confidence_select_num - len(scored_data))

        random.shuffle(remained_data)
        selected_data = selected_data + remained_data[:to_add_num]
        remained_data = remained_data[to_add_num:]
        return selected_data, remained_data