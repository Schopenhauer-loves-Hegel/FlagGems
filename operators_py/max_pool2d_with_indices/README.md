# max_pool2d_with_indices

## 基本信息

- **算子名**: max_pool2d_with_indices
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The max_pool2d_with_indices operator

## 查询语句

操作符名字是 max_pool2d_with_indices，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

无参数信息

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `max_pool2d_with_indices_triton.py` - Triton kernel实现（FlagGems原始代码）
- `max_pool2d_with_indices_torch.py` - PyTorch参考实现（groundtruth）
- `max_pool2d_with_indices_test.py` - 测试代码（bench格式）
