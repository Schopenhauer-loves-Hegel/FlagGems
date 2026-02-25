# bitwise_or

## 基本信息

- **算子名**: bitwise_or
- **算子类型**: pointwise
- **目标硬件**: nvidia
- **描述**: The bitwise_or operator

## 查询语句

操作符名字是 bitwise_or，是一个 pointwise 算子，处理硬件是 Nvidia。

## 输入参数

无参数信息

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `bitwise_or_triton.py` - Triton kernel实现（FlagGems原始代码）
- `bitwise_or_torch.py` - PyTorch参考实现（groundtruth）
- `bitwise_or_test.py` - 测试代码（bench格式）
