"""
Torch参考实现 - batch_norm
算子类型: normalization
描述: The batch_norm operator
"""

import torch

def batch_norm(*args, **kwargs):
    return torch.nn.functional.batch_norm(input, running_mean, running_var, weight=weight, bias=bias, training=training, momentum=momentum, eps=eps)