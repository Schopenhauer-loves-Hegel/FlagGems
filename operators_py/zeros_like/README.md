# zeros_like

## 基本信息

- **算子名**: zeros_like
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The zeros_like operator

## 查询语句

操作符名字是 zeros_like，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

无参数信息

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `zeros_like_triton.py` - Triton kernel实现（FlagGems原始代码）
- `zeros_like_torch.py` - PyTorch参考实现（groundtruth）
- `zeros_like_test.py` - 测试代码（bench格式）
