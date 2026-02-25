"""
Torch参考实现 - conv1d
算子类型: conv
描述: The conv1d operator
"""

import torch

def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return torch.conv1d(input, weight, bias, stride, padding, dilation, groups)