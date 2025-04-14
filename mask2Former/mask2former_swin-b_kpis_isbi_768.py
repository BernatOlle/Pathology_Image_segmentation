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
auto_scale_lr = dict(base_batch_size=8, enable=False)
backbone_embed_multi = dict(decay_mult=0.0, lr_mult=0.1)
backbone_norm_multi = dict(decay_mult=0.0, lr_mult=0.1)
batch_size = 2
crop_size = 768
custom_keys = dict({
    'absolute_pos_embed':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.patch_embed.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.0.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.0.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.0.downsample.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.1.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.1.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.1.downsample.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.10.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.11.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.12.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.13.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.14.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.15.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.16.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.17.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.2.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.3.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.4.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.5.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.6.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.7.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.8.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.9.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.downsample.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.3.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.3.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'level_embed':
    dict(decay_mult=0.0, lr_mult=1.0),
    'query_embed':
    dict(decay_mult=0.0, lr_mult=1.0),
    'query_feat':
    dict(decay_mult=0.0, lr_mult=1.0),
    'relative_position_bias_table':
    dict(decay_mult=0.0, lr_mult=0.1)
})
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
    data_root='/mnt/work/datasets/BKidney/KPIS/KPIs24 Validation Data/Task1_patch_level',
    data_prefix=dict(
        img_path='validation',  # Las imágenes están directamente en los subdirectorios de data_root
        seg_map_path='validation'  # Las máscaras están en la misma estructura paralela
    ),
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
        max_keep_ckpts=3,
        save_best='mDice',
        type='CheckpointHook'),
    logger=dict(interval=100, log_metric_by_epoch=False, type='LoggerHook'),
    visualization=dict(draw=True, interval=435, type='SegVisualizationHook'))
default_scope = 'mmseg'
depths = [
    2,
    2,
    18,
    2,
]
embed_multi = dict(decay_mult=0.0, lr_mult=1.0)
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
        depths=[
            2,
            2,
            18,
            2,
        ],
        drop_path_rate=0.3,
        drop_rate=0.0,
        embed_dims=128,
        frozen_stages=-1,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth',
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[
            4,
            8,
            16,
            32,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_norm=True,
        pretrain_img_size=384,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        window_size=12,
        with_cp=False),
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
        enforce_decoder_input_project=False,
        feat_channels=256,
        in_channels=[
            128,
            256,
            512,
            1024,
        ],
        loss_cls=dict(
            class_weight=[
                1.0,
                1.0,
                0.1,
            ],
            loss_weight=2.0,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False),
        loss_dice=dict(
            activate=True,
            eps=1.0,
            loss_weight=5.0,
            naive_dice=True,
            reduction='mean',
            type='mmdet.DiceLoss',
            use_sigmoid=True),
        loss_mask=dict(
            loss_weight=5.0,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        num_classes=2,
        num_queries=100,
        num_transformer_feat_level=3,
        out_channels=256,
        pixel_decoder=dict(
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                init_cfg=None,
                layer_cfg=dict(
                    ffn_cfg=dict(
                        act_cfg=dict(inplace=True, type='ReLU'),
                        embed_dims=256,
                        feedforward_channels=1024,
                        ffn_drop=0.0,
                        num_fcs=2),
                    self_attn_cfg=dict(
                        batch_first=True,
                        dropout=0.0,
                        embed_dims=256,
                        im2col_step=64,
                        init_cfg=None,
                        norm_cfg=None,
                        num_heads=8,
                        num_levels=3,
                        num_points=4)),
                num_layers=6),
            init_cfg=None,
            norm_cfg=dict(num_groups=32, type='GN'),
            num_outs=3,
            positional_encoding=dict(normalize=True, num_feats=128),
            type='mmdet.MSDeformAttnPixelDecoder'),
        positional_encoding=dict(normalize=True, num_feats=128),
        strides=[
            4,
            8,
            16,
            32,
        ],
        train_cfg=dict(
            assigner=dict(
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(
                        type='mmdet.CrossEntropyLossCost',
                        use_sigmoid=True,
                        weight=5.0),
                    dict(
                        eps=1.0,
                        pred_act=True,
                        type='mmdet.DiceCost',
                        weight=5.0),
                ],
                type='mmdet.HungarianAssigner'),
            importance_sample_ratio=0.75,
            num_points=12544,
            oversample_ratio=3.0,
            sampler=dict(type='mmdet.MaskPseudoSampler')),
        transformer_decoder=dict(
            init_cfg=None,
            layer_cfg=dict(
                cross_attn_cfg=dict(
                    attn_drop=0.0,
                    batch_first=True,
                    dropout_layer=None,
                    embed_dims=256,
                    num_heads=8,
                    proj_drop=0.0),
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    add_identity=True,
                    dropout_layer=None,
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.0,
                    num_fcs=2),
                self_attn_cfg=dict(
                    attn_drop=0.0,
                    batch_first=True,
                    dropout_layer=None,
                    embed_dims=256,
                    num_heads=8,
                    proj_drop=0.0)),
            num_layers=9,
            return_intermediate=True),
        type='Mask2FormerHead'),
    test_cfg=dict(crop_size=(
        768,
        768,
    ), mode='slide', stride=(
        576,
        576,
    )),
    train_cfg=dict(),
    type='EncoderDecoder')
num_classes = 2
num_gpus = 4
num_imgs = 6974
num_workers = 4
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.0001,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict({
            'absolute_pos_embed':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone':
            dict(decay_mult=1.0, lr_mult=0.1),
            'backbone.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.patch_embed.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.0.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.0.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.0.downsample.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.1.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.1.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.1.downsample.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.10.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.11.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.12.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.13.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.14.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.15.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.16.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.17.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.2.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.3.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.4.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.5.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.6.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.7.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.8.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.9.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.downsample.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.3.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.3.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'level_embed':
            dict(decay_mult=0.0, lr_mult=1.0),
            'query_embed':
            dict(decay_mult=0.0, lr_mult=1.0),
            'query_feat':
            dict(decay_mult=0.0, lr_mult=1.0),
            'relative_position_bias_table':
            dict(decay_mult=0.0, lr_mult=0.1)
        }),
        norm_decay_mult=0.0),
    type='OptimWrapper')
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ),
    eps=1e-08,
    lr=0.0001,
    type='AdamW',
    weight_decay=0.05)
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
pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth'
resume = False
stride_size = 576
test_cfg = None
test_data_as_val = dict(
    data_root='/mnt/work/datasets/BKidney/KPIS/KPIs24 Testing Data/Task1_patch_level',
                data_prefix=dict(
                    img_path='test',
                    seg_map_path='test'
                ),
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
                data_root='/mnt/work/datasets/BKidney/KPIS/KPIs24 Training Data/Task1_patch_level',
                data_prefix=dict(
                    img_path='train',  # Images are in [case]/[subcase]/img/
                    seg_map_path='train'  # Masks are in [case]/[subcase]/mask/
                ),
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
                data_root='/mnt/work/datasets/BKidney/KPIS/KPIs24 Validation Data/Task1_patch_level',
                data_prefix=dict(
                    img_path='validation',
                    seg_map_path='validation'
                ),
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
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root='/mnt/work/datasets/BKidney/KPIS/KPIs24 Testing Data/Task1_patch_level',
        data_prefix=dict(
            img_path='test',  # Images are in [data_root]/[case]/[subcase]/img/
            seg_map_path='test'  # Masks are in [data_root]/[case]/[subcase]/mask/
        ),
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
work_dir = 'mask2former_swin-b_kpis_768'
