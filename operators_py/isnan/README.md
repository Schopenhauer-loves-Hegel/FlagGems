# isnan

## 基本信息

- **算子名**: isnan
- **算子类型**: pointwise
- **目标硬件**: nvidia
- **描述**: The isnan operator

## 查询语句

操作符名字是 isnan，是一个 pointwise 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| A | Any | Parameter A |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `isnan_triton.py` - Triton kernel实现（FlagGems原始代码）
- `isnan_torch.py` - PyTorch参考实现（groundtruth）
- `isnan_test.py` - 测试代码（bench格式）
