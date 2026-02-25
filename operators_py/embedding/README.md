# embedding

## 基本信息

- **算子名**: embedding
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The embedding operator

## 查询语句

操作符名字是 embedding，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| weight | torch.Tensor | Parameter weight |
| indices | Any | Parameter indices |
| padding_idx | torch.Tensor | Parameter padding_idx |
| scale_grad_by_freq | torch.Tensor | Parameter scale_grad_by_freq |
| sparse | bool | Parameter sparse |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `embedding_triton.py` - Triton kernel实现（FlagGems原始代码）
- `embedding_torch.py` - PyTorch参考实现（groundtruth）
- `embedding_test.py` - 测试代码（bench格式）
