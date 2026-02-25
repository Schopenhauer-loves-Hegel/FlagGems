# dot

## 基本信息

- **算子名**: dot
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The dot operator

## 查询语句

操作符名字是 dot，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| x | torch.Tensor | Parameter x |
| y | torch.Tensor | Parameter y |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `dot_triton.py` - Triton kernel实现（FlagGems原始代码）
- `dot_torch.py` - PyTorch参考实现（groundtruth）
- `dot_test.py` - 测试代码（bench格式）
