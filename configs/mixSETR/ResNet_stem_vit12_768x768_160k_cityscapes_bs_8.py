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
        type='ResNetVit',
        in_channels=3,
        stem_channels=64,
        embed_channels=256,
        img_size=192,
        patch=2,
        trans_depth=12,
        num_heads=8,
        style='pytorch',
        conv_cfg=None,
        norm_cfg=norm_cfg,
        norm_eval=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=256,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)))
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
