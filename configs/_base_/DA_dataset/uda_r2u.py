# dataset settings
dataset_type = 'LoveDADataset'
data_root = '/root/autodl-tmp/dataset'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(512, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='UDA_dataset',
        source_dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
                img_path='loveda_uda/rural/train/img_dir', seg_map_path='loveda_uda/rural/train/ann_dir'),
            pipeline=train_pipeline),
        target_dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
                img_path='loveda_uda/urban/train/img_dir', seg_map_path='loveda_uda/urban/train/ann_dir'),
        pipeline=train_pipeline))
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='loveda_uda/urban/val/img_dir', seg_map_path='loveda_uda/urban/val/ann_dir'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
