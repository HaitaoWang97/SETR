_base_ = [
    '../_base_/models/fcn_r50-d8.py',
    '../_base_/datasets/cityscapes_769x769.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(align_corners=True),
    auxiliary_head=dict(align_corners=True))
test_cfg = dict(mode='slide', crop_size=(769, 769), stride=(513, 513))


# 1. norm_cfg = dict(type='BN', requires_grad=True)
# 2, without pretrained in imagenet 1k
# [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 500/500, 0.9 task/s, elapsed: 544s, ETA:     0s2021-10-05 07:56:31,768 - mmseg - INFO - per class results:
# Class                  IoU        Acc
# road                 96.05      98.38
# sidewalk             73.09      84.27
# building             86.98      95.90
# wall                 17.25      18.57
# fence                27.75      30.62
# pole                 55.57      65.70
# traffic light        52.72      56.41
# traffic sign         71.37      76.45
# vegetation           89.64      94.91
# terrain              37.07      39.72
# sky                  89.75      96.59
# person               72.63      87.06
# rider                31.91      36.36
# car                  85.80      97.69
# truck                19.09      23.88
# bus                   9.30       9.58
# train                 8.92       9.05
# motorcycle           23.39      30.22
# bicycle              63.74      70.73
# Summary:
# Scope                 mIoU       mAcc       aAcc
# global               53.26      59.06      92.82

# 2021-10-05 07:56:31,771 - mmseg - INFO - Exp name: fcn_r50-d8_769x769_80k_cityscapes.py
# 2021-10-05 07:56:31,771 - mmseg - INFO - Iter(val) [80000]      mIoU: 0.5326, mAcc: 0.5906, aAcc: 0.9282