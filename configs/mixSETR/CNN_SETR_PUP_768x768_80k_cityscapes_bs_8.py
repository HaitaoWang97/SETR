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
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    neck=[dict(
        type='VitFpn',
        img_size=[192, 96, 48, 24],
        patch_size=[4, 4, 4, 4],
        in_chans=[256, 512, 1024, 2048],
        embed_dim=[256, 512, 1024, 2048],
        depth=3,
        num_heads=8,
        num_classes=19,
        drop_rate=0.1,
        norm_cfg=norm_cfg,
        pos_embed_interp=True,
        align_corners=False),
        dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4,
        )],
    decode_head=dict(
        type='TransFpnHead',
        in_channels=256,
        channels=128,
        in_index=23,
        img_size=768,
        embed_dim=256,
        num_classes=19,
        norm_cfg=norm_cfg,
        num_conv=4,
        upsampling_method='bilinear',
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))
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
# 3. transformer [256, 512, 1024, 2048]
# 4. img_size 768*768, batch size 2 for each gpu, 8682M for each gpu
# 5. official cityscapes lable should be converted by scripts tools/convert_datasets/cityscapes.py else raise runtimeerror cuda error

# model result
# 2021-10-04 08:35:52,622 - mmseg - INFO - Iter(val) [80000]
# Class                  IoU        Acc
# road                 96.98      98.51
# sidewalk             77.47      86.84
# building             88.15      94.70
# wall                 45.00      53.99
# fence                43.74      58.45
# pole                 39.98      49.01
# traffic light        46.02      58.42
# traffic sign         56.44      66.34
# vegetation           88.85      94.80
# terrain              52.96      67.15
# sky                  90.97      95.54
# person               66.23      81.86
# rider                33.90      43.57
# car                  90.36      95.72
# truck                56.61      69.87
# bus                  58.94      79.92
# train                44.69      55.91
# motorcycle           20.97      25.95
# bicycle              62.36      79.14
# Summary:
# Scope                 mIoU       mAcc       aAcc
# global               61.09      71.35      93.44