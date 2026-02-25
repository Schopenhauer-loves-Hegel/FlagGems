# softmax

## 基本信息

- **算子名**: softmax
- **算子类型**: normalization
- **目标硬件**: nvidia
- **描述**: The softmax operator

## 查询语句

操作符名字是 softmax，是一个 normalization 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| self | Any | Parameter self |
| dim | int | Parameter dim |
| half_to_float | bool | Parameter half_to_float |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `softmax_triton.py` - Triton kernel实现（FlagGems原始代码）
- `softmax_torch.py` - PyTorch参考实现（groundtruth）
- `softmax_test.py` - 测试代码（bench格式）
