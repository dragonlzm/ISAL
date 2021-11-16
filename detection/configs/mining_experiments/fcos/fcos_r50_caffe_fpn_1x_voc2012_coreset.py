_base_ = './fcos_r50_caffe_fpn_1x_voc2012.py'

img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)

coreset_pipeline = [
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
data_root = "data/detection/voc_converted/"
data = dict(
    coreset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/voc2012train.json',
        img_prefix=data_root + 'all_imgs/',
        pipeline=coreset_pipeline))

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=8, norm_type=2))