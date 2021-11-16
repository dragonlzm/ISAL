_base_ = './fcos_r50_caffe_fpn_1x_coco.py'

data_root = "data/detection/voc_converted/"
classes = ('person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 
'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor')

data = dict(
    train=dict(
        initial=dict(
            classes=classes,
            ann_file=data_root + 'annotations/initial.json',
            img_prefix=data_root + 'all_imgs/'),
        unlabeled=dict(
            classes=classes,
            ann_file=data_root + 'annotations/unlabeled.json',
            img_prefix=data_root + 'all_imgs/'),
        full=dict(
            classes=classes,
            ann_file=data_root + 'annotations/voc2012train.json',
            img_prefix=data_root + 'all_imgs/')),
    val=dict(
        classes=classes,
        ann_file=data_root + 'annotations/voc2012val.json',
        img_prefix=data_root + 'all_imgs/'),
    test=dict(
        classes=classes,
        ann_file=data_root + 'annotations/voc2012train.json',
        img_prefix=data_root + 'all_imgs/'))

model = dict(bbox_head=dict(num_classes=20))

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=8, norm_type=2))