"""
Torch参考实现 - conv2d
算子类型: conv
描述: The conv2d operator
"""

import torch

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return torch.conv2d(input, weight, bias, stride, padding, dilation, groups)