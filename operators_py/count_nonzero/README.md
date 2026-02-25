# count_nonzero

## 基本信息

- **算子名**: count_nonzero
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The count_nonzero operator

## 查询语句

操作符名字是 count_nonzero，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| x | torch.Tensor | Parameter x |
| dim | int | Parameter dim |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `count_nonzero_triton.py` - Triton kernel实现（FlagGems原始代码）
- `count_nonzero_torch.py` - PyTorch参考实现（groundtruth）
- `count_nonzero_test.py` - 测试代码（bench格式）
