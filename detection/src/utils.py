import os.path as osp
import pickle
import shutil
import tempfile
import time
import mmcv
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd import grad
from mmcv.runner import get_dist_info
import numpy as np
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset


def multi_gpu_calculate_grad(model, data_loader, tmpdir=None, gpu_collect=False, params=None, model_name='FCOS'):
    """Calculate the sum of the samples' gradients with multiple gpus.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        params (tensor): the parameters that are used to calculate the model gradients.
        model_name (str): the name of the model

    Returns:
        list: It only has one ele in the list. It's the sum of the gradients of all samples.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    for i, data in enumerate(data_loader):
        loss_per_img = model(return_loss=True, **data)
        if model_name == 'FCOS':
            all_loss = torch.cat([torch.unsqueeze(loss_per_img[loss_key], dim=0) for loss_key in loss_per_img.keys()], dim=0)
            all_loss = torch.sum(all_loss)
            grad_result = grad(all_loss, params)

        elif model_name == 'FasterRCNN':
            loss_list = []
            for loss_key in loss_per_img.keys():
                if loss_key == 'acc':
                    loss_list.append(loss_per_img[loss_key])
                    continue
                if isinstance(loss_per_img[loss_key], list):
                    for ele in loss_per_img[loss_key]:
                        loss_list.append(torch.unsqueeze(ele, dim=0))
                else:       
                    loss_list.append(torch.unsqueeze(loss_per_img[loss_key], dim=0))
    
            all_loss = torch.cat(loss_list, dim=0)
            all_loss = torch.sum(all_loss)
            grad_result = grad(all_loss, params)

        result = []
        sum_res = None
        if len(results) != 0:
            sum_res = results.pop()
        for i, grad_ele in enumerate(grad_result):
            if sum_res != None:
                result.append(grad_ele.detach().cpu().numpy() + sum_res[i])
            else:    
                result.append(grad_ele.detach().cpu().numpy())
                
        del loss_per_img
        del all_loss
        del grad_result
        results.append(result)

        if rank == 0:
            for _ in range(1 * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def multi_gpu_s_test(val_grad, model, params, train_data_loader, damp=0.01, scale=None, img_num=1000, model_name='FCOS'):
    """ distributed version of calculation of s_test.
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
        list [[tensor]]: The s_test results. The inner list contain the gradient calculated for per parameter on one gpus.
                         The outer list contain the result from different gpu.
    """
    if model_name == 'FCOS':
        scale = [100.0, 8000.0, 5000.0]
    elif model_name == 'FasterRCNN':
        #scale = [1000.0, 1000.0, 1000.0, 5000.0, 1000.0]
        scale = [1000.0, 1000.0, 5000.0, 1000.0]
        #scale = [5000.0, 1000.0]
    model.eval()
    dataset = train_data_loader.dataset
    rank, world_size = get_dist_info()

    if rank == 0:
        total_data = world_size * img_num
        if total_data > len(dataset):
            total_data = len(dataset)
        prog_bar = mmcv.ProgressBar(total_data)
    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    h_estimate = val_grad.copy()

    for i, data in enumerate(train_data_loader):
        loss_per_img = model(return_loss=True, **data)
        if model_name == 'FCOS':
            all_loss = torch.cat([torch.unsqueeze(loss_per_img[loss_key], dim=0) for loss_key in loss_per_img.keys()], dim=0)
            all_loss = torch.sum(all_loss)
        elif model_name == 'FasterRCNN':
            loss_list = []
            for loss_key in loss_per_img.keys():
                if loss_key == 'acc':
                    loss_list.append(loss_per_img[loss_key])
                    continue
                if isinstance(loss_per_img[loss_key], list):
                    for ele in loss_per_img[loss_key]:
                        loss_list.append(torch.unsqueeze(ele, dim=0))
                else:       
                    loss_list.append(torch.unsqueeze(loss_per_img[loss_key], dim=0))    
            all_loss = torch.cat(loss_list, dim=0)
            all_loss = torch.sum(all_loss)        

        hv = hessian_vec_prod(all_loss, params, h_estimate)
        # Recursively caclulate h_estimate
        h_estimate_updated = []
        for _v, _h_e, _hv, _scale in zip(val_grad, h_estimate, hv, scale):
            if True in torch.isnan(_hv):
                # to avoid the NAN gradient deteriorate the calculation process
                h_estimate_updated.append(_v.cuda())
            else:
                h_estimate_updated.append(_v.cuda() + (1 - damp) * _h_e.cuda() - _hv.cuda() / _scale)

        h_estimate = h_estimate_updated
        if i > img_num:
            break

        if rank == 0:
            for _ in range(1 * world_size):
                prog_bar.update()

    result = [[]]
    for ele in h_estimate:
        result[0].append(ele.detach().cpu().numpy())
    del h_estimate

    results = collect_results_gpu(result, world_size)
    return results


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
    
    # Elementwise products
    elemwise_products = [torch.unsqueeze(torch.sum(grad_elem.cuda() * v_elem.cuda()), dim=0) for grad_elem, v_elem in zip(first_grads, h_estimate)]
    elemwise_products = torch.cat(elemwise_products, dim=0)
    per_param_grad = torch.sum(elemwise_products, dim=0)

    # Second backprop
    return_grads = grad(per_param_grad, params, create_graph=True, allow_unused=True)
    return_grads = [ele.detach() for ele in return_grads]

    return return_grads


def multi_gpu_cal_influence(model, data_loader, tmpdir=None, gpu_collect=False, params=None, s_test=None, model_name='FCOS'):
    """Calculate the influence value with multiple GPUs
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        params (tensor): the parameters that are needed to be used to calculate influence value.
        s_test (tensor): the s_test value for calculate the influence.
        model_name (str): the name of the model

    Returns:
        list: The influence value.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    for i, data in enumerate(data_loader):
        loss_per_img = model(return_loss=True, **data)
        if model_name == 'FCOS':
            all_loss = torch.cat([torch.unsqueeze(loss_per_img[loss_key], dim=0) for loss_key in loss_per_img.keys()], dim=0)
            all_loss = torch.sum(all_loss)
            grad_result = grad(all_loss, params)
            result = []
            for grad_ele in grad_result:
                if grad_ele != None:
                    result.append(grad_ele.detach().cpu().numpy())
        elif model_name == 'FasterRCNN':
            loss_list = []
            for loss_key in loss_per_img.keys():
                if loss_key == 'acc':
                    loss_list.append(loss_per_img[loss_key])
                    continue
                if isinstance(loss_per_img[loss_key], list):
                    for ele in loss_per_img[loss_key]:
                        loss_list.append(torch.unsqueeze(ele, dim=0))
                else:       
                    loss_list.append(torch.unsqueeze(loss_per_img[loss_key], dim=0))     
            all_loss = torch.cat(loss_list, dim=0)
            all_loss = torch.sum(all_loss)
            grad_result = grad(all_loss, params)

        grad_result = [ele.detach() for ele in grad_result]
        result = -sum(
            [
                torch.sum(k * j.cuda()).data.cpu().numpy()
                for k, j in zip(grad_result, s_test) if not (True in torch.isnan(k) or True in torch.isnan(j))
            ]) / len(dataset)
        
        results.append(result)

        if rank == 0:
            for _ in range(1 * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


@DATASETS.register_module()
class ISALCocoDataset(CocoDataset):

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_scores = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])
                if 'score' not in ann.keys():
                    gt_scores.append(1)
                else: 
                    gt_scores.append(float(ann['score']))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            scores=gt_scores)

        return ann

    def _influence2json(self, results):
        """Convert detection influence value to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            data = dict()
            data['image_id'] = img_id
            data['influence'] = result
            json_results.append(data)
        return json_results

    def _coreset2json(self, results):
        """Convert detection coreset value to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            data = dict()
            data['image_id'] = img_id
            data['coreset'] = result
            json_results.append(data)
        return json_results        

    def results2json(self, results, outfile_prefix, **kwargs):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()

        if "collect_influence" in kwargs.keys():
            json_results = self._influence2json(results)
            result_files['influence'] = f'{outfile_prefix}.influence.json'
            mmcv.dump(json_results, result_files['influence']) 
        elif "coreset" in kwargs.keys():
            json_results = self._coreset2json(results)
            result_files['coreset'] = f'{outfile_prefix}.coreset.json'
            mmcv.dump(json_results, result_files['coreset'])
        else:
            if isinstance(results[0], list):
                json_results = self._det2json(results)
                result_files['bbox'] = f'{outfile_prefix}.bbox.json'
                result_files['proposal'] = f'{outfile_prefix}.bbox.json'
                mmcv.dump(json_results, result_files['bbox']) 
            elif isinstance(results[0], tuple):
                json_results = self._segm2json(results)
                result_files['bbox'] = f'{outfile_prefix}.bbox.json'
                result_files['proposal'] = f'{outfile_prefix}.bbox.json'
                result_files['segm'] = f'{outfile_prefix}.segm.json'
                mmcv.dump(json_results[0], result_files['bbox'])
                mmcv.dump(json_results[1], result_files['segm'])
            elif isinstance(results[0], np.ndarray):
                json_results = self._proposal2json(results)
                result_files['proposal'] = f'{outfile_prefix}.proposal.json'
                mmcv.dump(json_results, result_files['proposal'])
            else:
                raise TypeError('invalid type of results')
        return result_files

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix, **kwargs)
        return result_files, tmp_dir