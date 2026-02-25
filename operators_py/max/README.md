# max

## 基本信息

- **算子名**: max
- **算子类型**: reduction
- **目标硬件**: nvidia
- **描述**: The max operator

## 查询语句

操作符名字是 max，是一个 reduction 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| inp | Any | Parameter inp |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `max_triton.py` - Triton kernel实现（FlagGems原始代码）
- `max_torch.py` - PyTorch参考实现（groundtruth）
- `max_test.py` - 测试代码（bench格式）
