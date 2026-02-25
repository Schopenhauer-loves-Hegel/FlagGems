"""
Torch参考实现 - multinomial
算子类型: general
描述: The multinomial operator
"""

import torch

def multinomial(prob, n_samples, with_replacement=False, *, gen=None):
    return torch.multinomial(prob, n_samples, with_replacement, gen)