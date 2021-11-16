_base_ = './fcos_r50_caffe_fpn_1x_voc2012.py'

img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)

influence_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

dataset_type = 'ISALCocoDataset'
data_root = "data/detection/voc_converted/"
data = dict(
    val_influ=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/voc2012val.json',
        img_prefix=data_root + 'all_imgs/',
        pipeline=influence_pipeline),
    train_influ=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/voc2012train.json',
        img_prefix=data_root + 'all_imgs/',
        pipeline=influence_pipeline))

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=8, norm_type=2))
