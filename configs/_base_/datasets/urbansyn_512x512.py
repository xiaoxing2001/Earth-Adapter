urbansyn_type = "CityscapesDataset"
urbansyn_root = "data/UrbanSyn"
urbansyn_crop_size = (512, 512)
urbansyn_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(1024, 512)),
    dict(type="RandomCrop", crop_size=urbansyn_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
urbansyn_train_pipeline_mask2former = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomChoiceResize",
        scales=[int(512 * x * 0.1) for x in range(5, 21)],
        resize_type="ResizeShortestEdge",
        max_size=2048,
    ),
    dict(type="RandomCrop", crop_size=urbansyn_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
urbansyn_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1024, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_urbansyn = dict(
    type=urbansyn_type,
    data_root=urbansyn_root,
    data_prefix=dict(
        img_path="rgb",
        seg_map_path="ss",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=urbansyn_train_pipeline,
)
train_urbansyn_mask2former = dict(
    type=urbansyn_type,
    data_root=urbansyn_root,
    data_prefix=dict(
        img_path="rgb",
        seg_map_path="ss",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=urbansyn_train_pipeline_mask2former,
)
