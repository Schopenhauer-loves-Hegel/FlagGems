# flip

## 基本信息

- **算子名**: flip
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The flip operator

## 查询语句

操作符名字是 flip，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

无参数信息

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `flip_triton.py` - Triton kernel实现（FlagGems原始代码）
- `flip_torch.py` - PyTorch参考实现（groundtruth）
- `flip_test.py` - 测试代码（bench格式）
