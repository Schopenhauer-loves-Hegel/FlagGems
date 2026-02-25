"""
Torch参考实现 - addcdiv
算子类型: general
描述: The addcdiv operator
"""

import torch

def addcdiv(inp, tensor1, tensor2, value=1.0, out=None):
    return torch.addcdiv(inp, tensor1, tensor2, value, out)