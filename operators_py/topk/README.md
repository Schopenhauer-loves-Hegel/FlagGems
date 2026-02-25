# topk

## 基本信息

- **算子名**: topk
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The topk operator

## 查询语句

操作符名字是 topk，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| x | torch.Tensor | Parameter x |
| k | Any | Parameter k |
| dim | int | Parameter dim |
| largest | bool | Parameter largest |
| sorted | bool | Parameter sorted |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `topk_triton.py` - Triton kernel实现（FlagGems原始代码）
- `topk_torch.py` - PyTorch参考实现（groundtruth）
- `topk_test.py` - 测试代码（bench格式）
