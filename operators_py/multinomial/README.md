# multinomial

## 基本信息

- **算子名**: multinomial
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The multinomial operator

## 查询语句

操作符名字是 multinomial，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| prob | Any | Parameter prob |
| n_samples | Any | Parameter n_samples |
| with_replacement | bool | Parameter with_replacement |
| gen | Any | Parameter gen |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `multinomial_triton.py` - Triton kernel实现（FlagGems原始代码）
- `multinomial_torch.py` - PyTorch参考实现（groundtruth）
- `multinomial_test.py` - 测试代码（bench格式）
