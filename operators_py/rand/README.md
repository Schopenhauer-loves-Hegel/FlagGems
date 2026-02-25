# rand

## 基本信息

- **算子名**: rand
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The rand operator

## 查询语句

操作符名字是 rand，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| size | int | Parameter size |
| dtype | torch.Tensor | Parameter dtype |
| layout | torch.Tensor | Parameter layout |
| device | Any | Parameter device |
| pin_memory | torch.Tensor | Parameter pin_memory |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `rand_triton.py` - Triton kernel实现（FlagGems原始代码）
- `rand_torch.py` - PyTorch参考实现（groundtruth）
- `rand_test.py` - 测试代码（bench格式）
