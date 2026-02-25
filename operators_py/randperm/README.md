# randperm

## 基本信息

- **算子名**: randperm
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The randperm operator

## 查询语句

操作符名字是 randperm，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

无参数信息

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `randperm_triton.py` - Triton kernel实现（FlagGems原始代码）
- `randperm_torch.py` - PyTorch参考实现（groundtruth）
- `randperm_test.py` - 测试代码（bench格式）
