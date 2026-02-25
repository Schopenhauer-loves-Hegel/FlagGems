# any

## 基本信息

- **算子名**: any
- **算子类型**: reduction
- **目标硬件**: nvidia
- **描述**: The any operator

## 查询语句

操作符名字是 any，是一个 reduction 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| inp | Any | Parameter inp |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `any_triton.py` - Triton kernel实现（FlagGems原始代码）
- `any_torch.py` - PyTorch参考实现（groundtruth）
- `any_test.py` - 测试代码（bench格式）
