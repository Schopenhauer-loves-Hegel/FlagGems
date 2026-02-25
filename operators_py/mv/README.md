# mv

## 基本信息

- **算子名**: mv
- **算子类型**: blas
- **目标硬件**: nvidia
- **描述**: The mv operator

## 查询语句

操作符名字是 mv，是一个 blas 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| inp | Any | Parameter inp |
| vec | Any | Parameter vec |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `mv_triton.py` - Triton kernel实现（FlagGems原始代码）
- `mv_torch.py` - PyTorch参考实现（groundtruth）
- `mv_test.py` - 测试代码（bench格式）
