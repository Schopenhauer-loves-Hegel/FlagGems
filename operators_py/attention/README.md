# attention

## 基本信息

- **算子名**: attention
- **算子类型**: attention
- **目标硬件**: nvidia
- **描述**: The attention operator

## 查询语句

操作符名字是 attention，是一个 attention 算子，处理硬件是 Nvidia。

## 输入参数

无参数信息

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `attention_triton.py` - Triton kernel实现（FlagGems原始代码）
- `attention_torch.py` - PyTorch参考实现（groundtruth）
- `attention_test.py` - 测试代码（bench格式）
