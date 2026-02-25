# lt

## 基本信息

- **算子名**: lt
- **算子类型**: pointwise
- **目标硬件**: nvidia
- **描述**: The lt operator

## 查询语句

操作符名字是 lt，是一个 pointwise 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| A | Any | Parameter A |
| B | Any | Parameter B |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `lt_triton.py` - Triton kernel实现（FlagGems原始代码）
- `lt_torch.py` - PyTorch参考实现（groundtruth）
- `lt_test.py` - 测试代码（bench格式）
