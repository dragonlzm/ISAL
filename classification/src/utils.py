import numpy as np
import torch
import time
import pickle
import torch
import torch.distributed as dist
from torch.autograd import grad
import pycls.core.net as net
import pycls.core.builders as builders

import numpy as np
from pycls.core.config import cfg
import pycls.core.checkpoint as cp
import pycls.core.trainer as trainer
import pycls.datasets.loader as data_loader
import torch
import torch.nn.functional as F
from tools.save_results import read_predictions_from_pkl


def s_test(val_grad, model, params, train_data_loader, damp=0.01, scale=10000.0, img_num=250):
    """ non-distributed version of calculation of s_test.
    s_test is the Inverse Hessian Vector Product.

    Args:
        val_grad: the sum of the gradient of whole validation set.
        model (nn.Module): Model to calculate the loss.
        params: the parameters of model to calculate the gradients.
        data_loader (nn.Dataloader): Pytorch data loader with the train data.
        damp: float, dampening factor.
        scale: float, scaling factor, a list, you can set up a specific scale for each parameter.
        img_num: use how many img to calculate s_test per node.

    Returns:
        list [tensor]: The s_test results. The list contain the gradient calculated for per parameter.
    """
    loss_fun = builders.build_loss_fun().cuda()
    h_estimate = val_grad.copy()
    h_estimate_list = []

    for cur_iter, (inputs, labels) in enumerate(train_data_loader):   
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)    
        labels_one_hot = net.smooth_one_hot_labels(labels)
        #inputs, labels_one_hot, labels = net.mixup(inputs, labels_one_hot)
        preds = model(inputs)
        loss = loss_fun(preds, labels_one_hot)

        hv = hessian_vec_prod(loss, params, h_estimate)

        h_estimate_updated = []
        for _v, _h_e, _hv in zip(val_grad, h_estimate, hv):
            if True in torch.isnan(_hv):
                h_estimate_updated.append(_v)
            else:
                h_estimate_updated.append(_v + (1 - damp) * _h_e - _hv / scale)
        h_estimate = h_estimate_updated

        if cur_iter % 100 == 0:
            print(cur_iter)
        if cur_iter >= img_num*4:
            break
        if cur_iter % img_num == img_num - 1:
            h_estimate_list.append(h_estimate)
            h_estimate = val_grad.copy()
    return h_estimate_list


def hessian_vec_prod(loss, params, h_estimate):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.

    Arguments:
        loss: The loss from each training data
        params: The parameters of model which need to calculate the gradients
        h_estimate: the hessian_vec_prod in the last iteration

    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.

    Raises:
        ValueError: `y` and `w` have a different length."""
    if len(params) != len(h_estimate):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(loss, params, retain_graph=True, create_graph=True, allow_unused=True)
    
    elemwise_products = [torch.unsqueeze(torch.sum(grad_elem.cuda() * v_elem.cuda()), dim=0) for grad_elem, v_elem in zip(first_grads, h_estimate)]
    elemwise_products = torch.cat(elemwise_products, dim=0)
    per_param_grad = torch.sum(elemwise_products, dim=0)

    # Second backprop
    return_grads = grad(per_param_grad, params, create_graph=True, allow_unused=True)
    return_grads = [ele.detach() for ele in return_grads]

    return return_grads


def calc_influence_funtion(model, test_loader, params, s_test):
    """ non-distributed version of calculation of influence value.

    Args:
        model (nn.Module): Model to calculate the loss.
        test_loader (nn.Dataloader): Pytorch data loader with the unlabeled data.
        params: the parameters of model to calculate the gradients.
        s_test: the s_test value.
    Returns:
        list [tensor]: The s_test results. The list contain the gradient calculated for per parameter.
    """

    loss_fun = builders.build_loss_fun().cuda()
    influence_list = []
    idxes_list = []
    s_test = torch.cat([ele.view(-1).cuda() for ele in s_test])

    for cur_iter, (inputs, labels, idx) in enumerate(test_loader):    
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        preds = model(inputs)
        pseudo_label = torch.nn.functional.softmax(preds).argmax().unsqueeze(dim=0)
        labels_one_hot = net.smooth_one_hot_labels(pseudo_label)
        loss = loss_fun(preds, labels_one_hot)
        grad_result = grad(loss, params)

        grad_result = [ele.detach().view(-1) for ele in grad_result]
        grad_concat = torch.cat(grad_result)
        result = - torch.sum(grad_concat * s_test).cpu().numpy() / len(test_loader.dataset)

        influence_list.append(result)
        idxes_list.append(idx)

    return influence_list, idxes_list


def calc_influence_funtion_expectation(model, test_loader, params, s_test, topk_idx=10):
    loss_fun = builders.build_loss_fun().cuda()
    influence_list = []
    idxes_list = []

    s_test = torch.cat([ele.view(-1).cuda() for ele in s_test])  

    for cur_iter, (inputs, labels, idx) in enumerate(test_loader):    
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        preds = model(inputs)

        loss_weight = torch.nn.functional.softmax(preds).detach()
        loss_list = []
        need_calculated_idx = torch.topk(loss_weight, topk_idx)[1]

        for label in need_calculated_idx[0]:
            labels_one_hot = net.smooth_one_hot_labels(label.unsqueeze(dim=0)).cuda()
            loss_list.append((loss_fun(preds, labels_one_hot) * loss_weight[0][label]).unsqueeze(dim=0))

        loss_list = torch.cat(loss_list)
        loss = torch.sum(loss_list)       
        grad_result = grad(loss, params, retain_graph=True)
        grad_result = [ele.detach().view(-1) for ele in grad_result]
        grad_concat = torch.cat(grad_result)
        result = - torch.sum(grad_concat * s_test).cpu().numpy() / len(test_loader.dataset)

        influence_list.append(result)
        idxes_list.append(idx)

    return influence_list, idxes_list
   

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