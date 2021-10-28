_base_ = [
    '../_base_/datasets/cityscapes_768x768.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    neck=dict(
        type='VitUpV2',
        img_size=96,
        patch_size=2,
        in_chans=2048,
        embed_dim=512,
        depth=12,
        num_heads=8,
        num_classes=19,
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='VisionTransformerUpHead',
        in_channels=512,
        channels=256,
        in_index=-1,
        img_size=768,
        embed_dim=512,
        num_classes=19,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='ASPPHead',
        in_channels=128,
        in_index=0,
        channels=64,
        dilations=(1, 6, 12, 18),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.6)))
# model training and testing settings
optimizer = dict(lr=0.01, weight_decay=0.0,
                 paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)})
                 )

crop_size = (768, 768)
train_cfg = dict()
test_cfg = dict(mode='slide', crop_size=crop_size, stride=(512, 512))
find_unused_parameters = True
data = dict(samples_per_gpu=2)


# model settings
# 1. resnet output [H/4, W/4, 256], [H/8, W/8, 512], [H/16, W/16, 1024], [H/32, W/32, 1024], [192, 96, 48, 24]
# 2. patch size [4, 4, 4, 4], L[48*48, 24*24, 12*12, 6*6] pyramid size
# 3. transformer [16, 32, 64, 128]
# 4. img_size 768*768, batch size 2 for each gpu, 8682M for each gpu, use 2 gpu
# 5. official cityscapes lable should be converted by scripts tools/convert_datasets/cityscapes.py else raise runtimeerror cuda error
# 6. reduce channel of embed_dim, [256, 512, 1024, 2048] reduce to [32, 64, 128, 256]
# 7. increase all four transformer block depth from 3 to 6
