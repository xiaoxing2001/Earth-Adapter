_base_ = [
    "./urbansyn_512x512.py",
    "./synthia_512x512.py",
    "./gta_512x512.py",
    "./cityscapes_512x512.py",
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            {{_base_.train_urbansyn}},
            {{_base_.train_gta}},
            {{_base_.train_syn}},
        ],
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset={{_base_.val_cityscapes}},
)
test_dataloader = val_dataloader
val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator
