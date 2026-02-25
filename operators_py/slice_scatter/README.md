# slice_scatter

## 基本信息

- **算子名**: slice_scatter
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The slice_scatter operator

## 查询语句

操作符名字是 slice_scatter，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| inp | Any | Parameter inp |
| src | Any | Parameter src |
| dim | int | Parameter dim |
| start | Any | Parameter start |
| end | Any | Parameter end |
| step | int | Parameter step |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `slice_scatter_triton.py` - Triton kernel实现（FlagGems原始代码）
- `slice_scatter_torch.py` - PyTorch参考实现（groundtruth）
- `slice_scatter_test.py` - 测试代码（bench格式）
