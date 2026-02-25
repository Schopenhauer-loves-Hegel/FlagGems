# Pointwise Dynamic 转 Static Kernel 实现方案

## 📊 现状分析

**使用 pointwise_dynamic 的算子统计**：~73个算子

**主要模式**：
1. 单输入算子：abs, neg, sqrt, sin, cos, exp, log等
2. 双输入算子：add, sub, mul, div, eq, ne等
3. 多输入/多变体：div(12个变体), clamp(5个变体)

## 🎯 转换策略

### 策略A：模板化生成（推荐 - 平衡自动化和质量）

为不同类型的算子创建模板，半自动化生成：

```python
# 1. 单输入 unary 算子模板
def generate_unary_kernel(op_name, scalar_expr):
    return f'''
@triton.jit
def {op_name}_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    output = {scalar_expr}  # 例如: tl.abs(x)
    tl.store(output_ptr + offsets, output, mask=mask)
'''

# 2. 双输入 binary 算子模板
def generate_binary_kernel(op_name, scalar_expr):
    return f'''
@triton.jit
def {op_name}_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = {scalar_expr}  # 例如: x + y
    tl.store(output_ptr + offsets, output, mask=mask)
'''
```

**优点**：
- 代码简洁、高质量
- 易于理解和维护
- 70%算子可自动化

**缺点**：
- 不支持自动广播（需要预处理）
- 复杂算子需要手动处理

### 策略B：利用生成的缓存代码

1. 运行原始算子，让 pointwise_dynamic 生成完整代码
2. 从 `~/.flaggems/` 缓存提取生成的代码
3. 清理依赖、简化代码
4. 替换原文件

**优点**：
- 保留完整功能（广播、stride、类型提升）
- 100%自动化
- 代码完全来自 pointwise_dynamic，质量有保证

**缺点**：
- 生成的代码复杂（1000+行/算子）
- 需要清理大量依赖代码
- 可读性较差

### 策略C：混合方案（最佳实践）

1. **简单算子（~50个）**：使用模板化生成
   - 单输入：abs, neg, sqrt, exp, log, sin, cos, etc.
   - 简单双输入：add, sub, mul, eq, ne, etc.

2. **复杂算子（~23个）**：手动改写或使用缓存
   - 多变体：div(12变体), clamp(5变体)
   - 特殊逻辑：where, masked_fill等

## 🛠️ 实现步骤

### Step 1: 分类算子

```bash
# 运行分类脚本
python classify_pointwise_ops.py

# 输出:
# - simple_unary.txt    # 40个简单单输入算子
# - simple_binary.txt   # 20个简单双输入算子
# - complex_ops.txt     # 13个复杂算子
```

### Step 2: 自动生成简单算子

```bash
# 批量生成简单算子
python generate_simple_kernels.py --input simple_unary.txt --type unary
python generate_simple_kernels.py --input simple_binary.txt --type binary
```

### Step 3: 手动处理复杂算子

对于复杂算子，提供转换指导和辅助工具。

### Step 4: 验证和测试

```bash
# 运行测试验证转换正确性
python verify_conversion.py
```

## 📝 具体实现

### 工具1：算子分类器

```python
# classify_pointwise_ops.py
def classify_operator(op_path):
    """分类算子为 simple_unary, simple_binary, complex"""
    funcs = analyze_pointwise_usage(op_path)

    if len(funcs) == 1:
        func = funcs[0]
        params = parse_params(func['params'])

        if len(params) == 1:
            return 'simple_unary'
        elif len(params) == 2 and 'is_tensor=[True, True' in func['decorator_args']:
            return 'simple_binary'

    return 'complex'
```

### 工具2：模板生成器

```python
# generate_simple_kernels.py
UNARY_TEMPLATE = '''
import torch
import triton
import triton.language as tl

@triton.jit
def {op_name}_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    output = {compute_expr}
    tl.store(output_ptr + offsets, output, mask=mask)

def {op_name}(input):
    output = torch.empty_like(input)
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    {op_name}_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
    return output
'''

# 映射 scalar function 到 compute expression
SCALAR_EXPR_MAP = {
    'abs': 'tl.abs(x)',
    'neg': '-x',
    'sqrt': 'tl.sqrt(x)',
    'rsqrt': 'tl.rsqrt(x)',
    'exp': 'tl.exp(x)',
    'log': 'tl.log(x)',
    'sin': 'tl.sin(x)',
    'cos': 'tl.cos(x)',
    'tanh': 'tl.tanh(x)',
    # ... 更多映射
}
```

### 工具3：转换验证器

```python
# verify_conversion.py
def verify_operator(op_name):
    """验证转换后的算子输出正确"""
    # 导入原始实现（使用 pointwise_dynamic）
    old_impl = import_old(op_name)
    # 导入新实现（static kernel）
    new_impl = import_new(op_name)

    # 测试多种输入
    test_inputs = generate_test_inputs(op_name)

    for inp in test_inputs:
        old_out = old_impl(inp)
        new_out = new_impl(inp)
        assert torch.allclose(old_out, new_out)
```

## 🚀 执行计划

### Phase 1: 准备工作（1天）
- [ ] 创建分类工具
- [ ] 分析所有73个算子
- [ ] 生成分类列表

### Phase 2: 简单算子转换（2-3天）
- [ ] 实现模板生成器
- [ ] 批量转换 simple_unary (~40个)
- [ ] 批量转换 simple_binary (~20个)
- [ ] 验证转换正确性

### Phase 3: 复杂算子处理（3-5天）
- [ ] 手动分析复杂算子
- [ ] 逐个改写或使用缓存提取
- [ ] 验证正确性

### Phase 4: 清理和优化（1-2天）
- [ ] 移除所有 pointwise_dynamic 导入
- [ ] 优化生成的代码
- [ ] 完整测试

## 📌 注意事项

1. **广播处理**：简单模板不支持自动广播，需要在 wrapper 中预处理
2. **类型提升**：可能需要手动处理类型提升逻辑
3. **性能**：简化的 kernel 可能比 pointwise_dynamic 生成的略慢（但更易懂）
4. **兼容性**：保持相同的函数签名和行为

## 🎓 示例对比

### 转换前（使用 pointwise_dynamic）

```python
from flag_gems.utils import pointwise_dynamic

@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def abs_func(x):
    return tl.abs(x)

def abs(A):
    return abs_func(A)
```

### 转换后（静态 kernel）

```python
import torch
import triton
import triton.language as tl

@triton.jit
def abs_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    output = tl.abs(x)
    tl.store(output_ptr + offsets, output, mask=mask)

def abs(A):
    output = torch.empty_like(A)
    n_elements = A.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    abs_kernel[grid](A, output, n_elements, BLOCK_SIZE=1024)
    return output
```

## ✅ 成功标准

- [ ] 所有73个算子不再依赖 pointwise_dynamic
- [ ] 功能测试100%通过
- [ ] 代码可读性提升
- [ ] 无外部依赖（除 torch, triton）
