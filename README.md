# Influence Selection for Active Learning (ISAL)

This project hosts the code for implementing the ISAL algorithm for object detection and image classification, as presented in our paper:

    Influence Selection for Active Learning;
    Zhuoming Liu, Hao Ding, Huaping Zhong, Weijia Li, Jifeng Dai, Conghui He;
    In: Proc. Int. Conf. Computer Vision (ICCV), 2021.
    arXiv preprint arXiv:2108.09331

The full paper is available at: [https://arxiv.org/abs/2108.09331](https://arxiv.org/abs/2108.09331). 

Implementation based on MMDetection is included in [MMDetection](https://github.com/open-mmlab/mmdetection).

## Highlights
- **Task agnostic:** We evaluate ISAL in both object detection and image classification. Compared with previous methods, ISAL decreases the annotation cost at least by 12%, 12%, 3%, 13% and 16% on CIFAR10, SVHN, CIFAR100, VOC2012 and COCO, respectively.


- **Model agnostic:** We evaluate ISAL with different model in object detection. On COCO dataset, with one-stage anchor-free detector FCOS, ISAL decreases the annotation cost at least by 16%. With two-stage anchor-based detector Faster R-CNN, ISAL decreases the annotation cost at least by 10%.

ISAL just needs to use the model gradients, which can be easily obtained in a neural network no matter 
what task is and how complex the model structure is, our proposed ISAL is task-agnostic and model-agnostic.

## Required hardware
We use 4 NVIDIA V100 GPUs for object detection. We use 1 NVIDIA TITAN Xp GPUs for image classification.

## Installation
Our ISAL implementation for object detection is based on mmdetection v2.4.0 with mmcv v1.1.1.
Their need Pytorch version = 1.5, CUDA version = 10.1, CUDNN version = 7.
We provide a docker file (./detection/Dockerfile) to prepare the environment.
Once the environment is prepared, please copy all the files under the folder ./detection into the directory /mmdetection in the docker.

Our ISAL implementation for image classification is based on pycls v0.1.
It need Pytorch version = 1.6, CUDA version = 10.1, CUDNN version = 7.

## Training
The following command line will perform the ISAL algorithm with FCOS detector on COCO dataset, the active learning algorithm will iterate 20 steps with 4 GPUS:

    bash dist_run_isal.sh /workdir /datadir \
        /mmdetection/configs/mining_experiments/ \
        fcos/fcos_r50_caffe_fpn_1x_coco_influence_function.py \
        --mining-method=influence --seed=42 --deterministic \
        --noised-score-thresh=0.1

Note that:
1) If you want to use fewer GPUs, please change `GPUS` in shell script. In addition, you may need to change the `samples_per_gpu` in the config file to mantain the total batch size is equal to 8.
2) The models and all inference results will be saved into `/workdir`.
3) The data should be place in `/datadir`.
4) If you want to run our code on VOC or your own dataset, we suggest that you should change the data format into COCO format.
5) If you want to change the active learning iteration steps, please change the `TRAIN_STEP` in shell script. If you want to change the image selected by step_0 or the following steps, please change the `INIT_IMG_NUM` or `IMG_NUM` in shell script, respectively.
6) The shell script will delete all the trained models after all the active learning steps. If you want to maintain the models please change the `DELETE_MODEL` in shell script.

The following command line will perform the ISAL algorithm with ResNet-18 on CIFAR10 dataset, the active learning algorithm will iterate 10 steps with 1 GPU:

    bash run_isal.sh /workdir /datadir \
        pycls/configs/archive/cifar/resnet/R-18_nds_1gpu_cifar10.yaml \
        --mining-method=influence --random-seed=0

Note that:
1) The models and all inference results will be saved into `/workdir`.
2) The data should be place in `/datadir`.
3) If you want to train SHVN or your own dataset, we suggest that you should change the data format into CIFAR10 format.
4) The `STEP` in shell script indicates that in each active learning step the algorithm will add (1/STEP)% of the whole dataset into labeled dataset. The `TRAIN_STEP` indicates the total steps of active learning algorithm.

## Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```
@inproceedings{liu2021influence,
  title={Influence selection for active learning},
  author={Liu, Zhuoming and Ding, Hao and Zhong, Huaping and Li, Weijia and Dai, Jifeng and He, Conghui},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9274--9283},
  year={2021}
}

```

## Acknowledgments
We thank Zheng Zhu for implementing the classification pipeline.
We thank Bin Wang and Xizhou Zhu for discussion and helping with the experiments.
We thank Yuan Tian and Jiamin He for discussing the mathematic derivation.    

## License
For academic use only. For commercial use, please contact the authors. 

