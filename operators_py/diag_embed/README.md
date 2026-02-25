# diag_embed

## 基本信息

- **算子名**: diag_embed
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The diag_embed operator

## 查询语句

操作符名字是 diag_embed，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| x | torch.Tensor | Parameter x |
| offset | int | Parameter offset |
| dim1 | int | Parameter dim1 |
| dim2 | int | Parameter dim2 |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `diag_embed_triton.py` - Triton kernel实现（FlagGems原始代码）
- `diag_embed_torch.py` - PyTorch参考实现（groundtruth）
- `diag_embed_test.py` - 测试代码（bench格式）
