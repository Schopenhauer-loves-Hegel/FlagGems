# dropout

## 基本信息

- **算子名**: dropout
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The dropout operator

## 查询语句

操作符名字是 dropout，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| input | torch.Tensor | Parameter input |
| p | Any | Parameter p |
| train | bool | Parameter train |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `dropout_triton.py` - Triton kernel实现（FlagGems原始代码）
- `dropout_torch.py` - PyTorch参考实现（groundtruth）
- `dropout_test.py` - 测试代码（bench格式）
