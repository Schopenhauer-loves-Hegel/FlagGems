"""
Torch参考实现 - conv3d
算子类型: general
描述: The conv3d operator
"""

import torch

def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return torch.conv3d(input, weight, bias, stride, padding, dilation, groups)