import numpy as np
import math


def cal_iou(pred_box, ref_box):
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


def l1_dist(p1_x, p1_y, p2_x, p2_y):
    return max(abs(p1_x - p2_x), abs(p1_y - p2_y))


def l2_dist(p1_x, p1_y, p2_x, p2_y):
    return math.sqrt((p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2)


def xywh2xyxy(box):
    # convert the format of the box from topleft_x, topleft_y, width, hight
    # to bottomleft_x, bottomleft_y, topright_x, toprigth_y
    return [box[0], box[1] - box[3], box[0] + box[2], box[1]]