# resolve_neg

## 基本信息

- **算子名**: resolve_neg
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The resolve_neg operator

## 查询语句

操作符名字是 resolve_neg，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| A: torch.Tensor | torch.Tensor | Parameter A: torch.Tensor |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `resolve_neg_triton.py` - Triton kernel实现（FlagGems原始代码）
- `resolve_neg_torch.py` - PyTorch参考实现（groundtruth）
- `resolve_neg_test.py` - 测试代码（bench格式）
