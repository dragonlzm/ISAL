_base_ = '../../fcos/fcos_r50_caffe_fpn_4x4_1x_coco.py'

checkpoint_config = dict(interval=12)
#evaluation = dict(interval=12, metric='bbox')
evaluation = dict(interval=6, metric='bbox', classwise=True)
# optimizer
optimizer = dict(
    lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=10, norm_type=2))
# training and testing settings
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
dataset_type = 'ISALCocoDataset'
data_root = 'data/detection/coco/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        initial=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/initial.json',
            img_prefix=data_root + 'train2017/',
            pipeline=train_pipeline),
        unlabeled=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/unlabeled.json',
            img_prefix=data_root + 'train2017/',
            pipeline=train_pipeline),
        full=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=test_pipeline))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])