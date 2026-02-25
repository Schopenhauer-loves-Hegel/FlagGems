# isfinite

## 基本信息

- **算子名**: isfinite
- **算子类型**: pointwise
- **目标硬件**: nvidia
- **描述**: The isfinite operator

## 查询语句

操作符名字是 isfinite，是一个 pointwise 算子，处理硬件是 Nvidia。

## 输入参数

无参数信息

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `isfinite_triton.py` - Triton kernel实现（FlagGems原始代码）
- `isfinite_torch.py` - PyTorch参考实现（groundtruth）
- `isfinite_test.py` - 测试代码（bench格式）
