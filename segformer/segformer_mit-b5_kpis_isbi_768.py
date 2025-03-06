albu_train_transforms = [
    dict(
        brightness_limit=(
            -0.25,
            0.25,
        ),
        contrast_limit=(
            -0.25,
            0.25,
        ),
        p=0.35,
        type='RandomBrightnessContrast'),
    dict(p=0.3, type='HorizontalFlip'),
    dict(p=0.3, type='VerticalFlip'),
    dict(blur_limit=(
        3,
        11,
    ), p=0.2, type='Blur'),
]
batch_size = 2
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'
crop_size = 768
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        768,
        768,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = ''
data_val_for_train = dict(
    data_prefix=dict(
        img_path='KPIs24_Validation_Data/Task1_patch_level/validation'),
    data_root='',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadKPIsAnnotations'),
        dict(
            bgr_to_rgb=False,
            keymap=dict(gt_seg_map='mask', img='image'),
            transforms=[
                dict(
                    brightness_limit=(
                        -0.25,
                        0.25,
                    ),
                    contrast_limit=(
                        -0.25,
                        0.25,
                    ),
                    p=0.35,
                    type='RandomBrightnessContrast'),
                dict(p=0.3, type='HorizontalFlip'),
                dict(p=0.3, type='VerticalFlip'),
                dict(blur_limit=(
                    3,
                    11,
                ), p=0.2, type='Blur'),
            ],
            type='Albu'),
        dict(cat_max_ratio=0.75, crop_size=(
            768,
            768,
        ), type='RandomCrop'),
        dict(type='PackSegInputs'),
    ],
    type='KPIsDataset')
dataset_type = 'KPIsDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=435,
        max_keep_ckpts=7,
        save_best='mDice',
        type='CheckpointHook'),
    logger=dict(interval=100, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw=True, interval=435, type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
epoch = 30
iter_scale = 1.618
iters_per_train = 871
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
max_iters = 42278
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dims=64,
        in_channels=3,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth',
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[
            1,
            2,
            5,
            8,
        ],
        num_layers=[
            3,
            6,
            40,
            3,
        ],
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_sizes=[
            7,
            3,
            3,
            3,
        ],
        qkv_bias=True,
        sr_ratios=[
            8,
            4,
            2,
            1,
        ],
        type='MixVisionTransformer'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            768,
            768,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=256,
        dropout_ratio=0.1,
        in_channels=[
            64,
            128,
            320,
            512,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=[
            dict(
                avg_non_ignore=True,
                loss_name='loss_ce',
                loss_weight=1.0,
                type='CrossEntropyLoss',
                use_sigmoid=False),
        ],
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        type='SegformerHead'),
    pretrained=None,
    test_cfg=dict(crop_size=(
        768,
        768,
    ), mode='slide', stride=(
        576,
        576,
    )),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
num_gpus = 4
num_imgs = 6974
num_workers = 4
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=6e-05, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0),
            pos_block=dict(decay_mult=0.0))),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
ori_train_data = dict(
    data_prefix=dict(img_path='KPIs24_Training_Data/Task1_patch_level/train'),
    data_root='',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadKPIsAnnotations'),
        dict(
            bgr_to_rgb=False,
            keymap=dict(gt_seg_map='mask', img='image'),
            transforms=[
                dict(
                    brightness_limit=(
                        -0.25,
                        0.25,
                    ),
                    contrast_limit=(
                        -0.25,
                        0.25,
                    ),
                    p=0.35,
                    type='RandomBrightnessContrast'),
                dict(p=0.3, type='HorizontalFlip'),
                dict(p=0.3, type='VerticalFlip'),
                dict(blur_limit=(
                    3,
                    11,
                ), p=0.2, type='Blur'),
            ],
            type='Albu'),
        dict(cat_max_ratio=0.75, crop_size=(
            768,
            768,
        ), type='RandomCrop'),
        dict(type='PackSegInputs'),
    ],
    type='KPIsDataset')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1500, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1500,
        by_epoch=False,
        end=42278,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
resume = False
stride_size = 576
test_cfg = None
test_data_as_val = dict(
    data_prefix=dict(img_path='KPIs24_Testing_Data/Task1_patch_level/test'),
    data_root='',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadKPIsAnnotations'),
        dict(type='PackSegInputs'),
    ],
    type='KPIsDataset')
train_cfg = dict(max_iters=42278, type='IterBasedTrainLoop', val_interval=435)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        datasets=[
            dict(
                data_prefix=dict(
                    img_path='KPIs24_Training_Data/Task1_patch_level/train'),
                data_root='',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadKPIsAnnotations'),
                    dict(
                        bgr_to_rgb=False,
                        keymap=dict(gt_seg_map='mask', img='image'),
                        transforms=[
                            dict(
                                brightness_limit=(
                                    -0.25,
                                    0.25,
                                ),
                                contrast_limit=(
                                    -0.25,
                                    0.25,
                                ),
                                p=0.35,
                                type='RandomBrightnessContrast'),
                            dict(p=0.3, type='HorizontalFlip'),
                            dict(p=0.3, type='VerticalFlip'),
                            dict(blur_limit=(
                                3,
                                11,
                            ), p=0.2, type='Blur'),
                        ],
                        type='Albu'),
                    dict(
                        cat_max_ratio=0.75,
                        crop_size=(
                            768,
                            768,
                        ),
                        type='RandomCrop'),
                    dict(type='PackSegInputs'),
                ],
                type='KPIsDataset'),
            dict(
                data_prefix=dict(
                    img_path=
                    'KPIs24_Validation_Data/Task1_patch_level/validation'),
                data_root='',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadKPIsAnnotations'),
                    dict(
                        bgr_to_rgb=False,
                        keymap=dict(gt_seg_map='mask', img='image'),
                        transforms=[
                            dict(
                                brightness_limit=(
                                    -0.25,
                                    0.25,
                                ),
                                contrast_limit=(
                                    -0.25,
                                    0.25,
                                ),
                                p=0.35,
                                type='RandomBrightnessContrast'),
                            dict(p=0.3, type='HorizontalFlip'),
                            dict(p=0.3, type='VerticalFlip'),
                            dict(blur_limit=(
                                3,
                                11,
                            ), p=0.2, type='Blur'),
                        ],
                        type='Albu'),
                    dict(
                        cat_max_ratio=0.75,
                        crop_size=(
                            768,
                            768,
                        ),
                        type='RandomCrop'),
                    dict(type='PackSegInputs'),
                ],
                type='KPIsDataset'),
        ],
        type='ConcatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadKPIsAnnotations'),
    dict(
        bgr_to_rgb=False,
        keymap=dict(gt_seg_map='mask', img='image'),
        transforms=[
            dict(
                brightness_limit=(
                    -0.25,
                    0.25,
                ),
                contrast_limit=(
                    -0.25,
                    0.25,
                ),
                p=0.35,
                type='RandomBrightnessContrast'),
            dict(p=0.3, type='HorizontalFlip'),
            dict(p=0.3, type='VerticalFlip'),
            dict(blur_limit=(
                3,
                11,
            ), p=0.2, type='Blur'),
        ],
        type='Albu'),
    dict(cat_max_ratio=0.75, crop_size=(
        768,
        768,
    ), type='RandomCrop'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(type='LoadKPIsAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_prefix=dict(
            img_path='KPIs24_Testing_Data/Task1_patch_level/test'),
        data_root='',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadKPIsAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='KPIsDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mDice',
        'mFscore',
    ], type='IoUMetric')
val_freq = 0.5
val_interval = 435
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadKPIsAnnotations'),
    dict(type='PackSegInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'segformer_mit-b5_kpis_768'
