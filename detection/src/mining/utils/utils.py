import numpy as np
import torch
import time
import pickle
import mmcv
import torch
import torch.distributed as dist
from torch.autograd import grad


def sort_dict(valued_image_set, sort_key='value', reverse=False):
    sorted_valued_image_set = sorted(valued_image_set, key=lambda k: k[sort_key], reverse=reverse)
    return sorted_valued_image_set


def select(valued_image_set, select_num, reverse):    
    select_num = int(select_num)

    sorted_valued_image_set = sort_dict(valued_image_set, sort_key='value', reverse=reverse)
    selected_valued_image_set = sorted_valued_image_set[:select_num]
    remained_valued_image_set = sorted_valued_image_set[select_num:]
    return selected_valued_image_set, remained_valued_image_set


def stratified_sampling(preset_layers=10):
    def _stratified_sampling(valued_image_set, select_num, reverse):
        layers = preset_layers
        if len(valued_image_set) <= select_num:
            return valued_image_set, []
        
        select_num = int(select_num)
        select_num_per_layer = int(select_num / layers)
        img_per_layer = int(len(valued_image_set) / layers)
        selected_valued_image_set, remained_valued_image_set = [], []
        sorted_valued_image_set = sort_dict(valued_image_set, sort_key='value', reverse=reverse)

        for i in range(layers):
            if i == layers - 1:
                img_need_to_add = select_num - len(selected_valued_image_set)
                selected_valued_image_set += sorted_valued_image_set[i * img_per_layer : i * img_per_layer + img_need_to_add]
                remained_valued_image_set += sorted_valued_image_set[i * img_per_layer + img_need_to_add : ]
            else:
                selected_valued_image_set += sorted_valued_image_set[i * img_per_layer : i * img_per_layer + select_num_per_layer]
                remained_valued_image_set += sorted_valued_image_set[i * img_per_layer + select_num_per_layer : (i + 1)* img_per_layer]

        return selected_valued_image_set, remained_valued_image_set 
    
    return _stratified_sampling


def calculate_iou(ref_box, pred_box):
    # convert the representation of box (left_up_x, left_up_y, width, hight) 
    # to (left_bottom_x, left_bottom_y, right_top_x, right_top_y)
    pred_box = [pred_box[0], pred_box[1] - pred_box[3], pred_box[0] + pred_box[2], pred_box[1]]
    ref_box = [ref_box[0], ref_box[1] - ref_box[3], ref_box[0] + ref_box[2], ref_box[1]]

    # get the coordinate of inters
    ixmin = max(pred_box[0], ref_box[0])
    ixmax = min(pred_box[2], ref_box[2])
    iymin = max(pred_box[1], ref_box[1])
    iymax = min(pred_box[3], ref_box[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)

    # calculate the area of inters
    inters = iw * ih

    # calculate the area of union
    uni = ((pred_box[2] - pred_box[0] + 1.) * (pred_box[3] - pred_box[1] + 1.) +
           (ref_box[2] - ref_box[0] + 1.) * (ref_box[3] - ref_box[1] + 1.) - inters)

    # calculate the overlaps between pred_box and gt_box
    iou = inters / uni
    return iou