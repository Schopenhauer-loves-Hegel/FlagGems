# index

## 基本信息

- **算子名**: index
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The index operator

## 查询语句

操作符名字是 index，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| inp | Any | Parameter inp |
| indices | Any | Parameter indices |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `index_triton.py` - Triton kernel实现（FlagGems原始代码）
- `index_torch.py` - PyTorch参考实现（groundtruth）
- `index_test.py` - 测试代码（bench格式）
