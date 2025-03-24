# dataset config
_base_ = [
    "../../../_base_/DA_dataset/uda_pr2vi.py",
    "../../../_base_/default_runtime.py",
    "../../../_base_/models/dinov2_mask2former.py"
]


model = dict(
    type = 'DACS_encoder_decoder',
    backbone = dict(
        type = 'MOE_Adpter_DinoVisionTransformer',
        moe_adapter_type = 'earth_adapter'
    ),
    decode_head = dict(
    num_classes= 6,
    loss_cls = dict(
        class_weight = [1.0]*6+[0.1],
        )
    )
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    constructor="PEFTOptimWrapperConstructor",
    optimizer=dict(
        type="AdamW", lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        custom_keys={
            "norm": dict(decay_mult=0.0),
            "query_embed": embed_multi,
            "level_embed": embed_multi,
            "learnable_tokens": embed_multi,
            "reins.scale": embed_multi,
        },
        norm_decay_mult=0.0,
    ),
)
param_scheduler = [
    dict(type="PolyLR", eta_min=0, power=0.9, begin=0, end=20000, by_epoch=False)
]

# training schedule for 160k
train_cfg = dict(type="IterBasedTrainLoop", max_iters=20000, val_interval=2000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", by_epoch=False, interval=2000, max_keep_ckpts=1, save_best='mIoU'
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
exp_name = 'DA_spatial_64_token_64'
randomness = dict(seed = 0)
