# groupnorm

## 基本信息

- **算子名**: groupnorm
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The groupnorm operator

## 查询语句

操作符名字是 groupnorm，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

无参数信息

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `groupnorm_triton.py` - Triton kernel实现（FlagGems原始代码）
- `groupnorm_torch.py` - PyTorch参考实现（groundtruth）
- `groupnorm_test.py` - 测试代码（bench格式）
