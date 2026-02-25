#!/usr/bin/env python3
"""
Step 2: 为 Simple Unary 算子生成标准 Triton kernel
"""

import re
from pathlib import Path

# Scalar function 到 Triton 表达式的映射
SCALAR_EXPR_MAP = {
    'abs': 'tl.abs(x)',
    'neg': '-x',
    'sqrt': 'tl.sqrt(x)',
    'rsqrt': 'tl.rsqrt(x)',
    'exp': 'tl.exp(x)',
    'exp2': 'tl.exp2(x)',
    'log': 'tl.log(x)',
    'sin': 'tl.sin(x)',
    'cos': 'tl.cos(x)',
    'tan': 'tl.tan(x)',
    'tanh': 'tl.tanh(x)',
    'erf': 'tl.erf(x)',
    'reciprocal': '1.0 / x',
    'isnan': 'tl.isnan(x)',
    'isinf': 'tl.isinf(x)',
    'isfinite': 'tl.isfinite(x)',
    'bitwise_not': '~x',
    'logical_not': 'not x',
    'atan': 'tl.atan(x)',
    'log_sigmoid': 'tl.log(tl.sigmoid(x))',
}

UNARY_TEMPLATE = '''import torch
import triton
import triton.language as tl


@triton.jit
def {op_name}_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for {op_name} operation.
    Applies {op_name} element-wise to the input tensor.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute
    output = {compute_expr}
    
    # Store output
    tl.store(output_ptr + offsets, output, mask=mask)


def {op_name}(input):
    """
    Apply {op_name} operation element-wise.
    
    Args:
        input: Input tensor
        
    Returns:
        Output tensor with same shape as input
    """
    output = torch.empty_like(input)
    n_elements = input.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    {op_name}_kernel[grid](
        input,
        output,
        n_elements,
        BLOCK_SIZE=1024,
    )
    
    return output
'''

def extract_scalar_function_body(triton_file, func_name):
    """从 triton 文件提取 scalar function 的函数体"""
    with open(triton_file, 'r') as f:
        content = f.read()
    
    # 查找函数定义
    pattern = rf'@triton\.jit\s*def\s+{re.escape(func_name)}\([^)]*\):\s*\n((?:    .*\n)*)'
    match = re.search(pattern, content)
    
    if match:
        body = match.group(1)
        # 提取 return 语句
        return_match = re.search(r'return\s+(.+)', body)
        if return_match:
            return return_match.group(1).strip()
    
    return None

def infer_compute_expr(op_name, scalar_body):
    """推断计算表达式"""
    
    # 1. 先查找预定义映射
    if op_name in SCALAR_EXPR_MAP:
        return SCALAR_EXPR_MAP[op_name]
    
    # 2. 如果 scalar body 很简单，直接使用
    if scalar_body and len(scalar_body) < 50:
        # 简单替换参数名
        expr = scalar_body.replace('tl.', 'tl.')
        return expr
    
    # 3. 默认
    return f'tl.{op_name}(x)'

def generate_kernel_for_op(op_name, triton_file):
    """为单个算子生成 kernel"""
    
    # 提取 scalar function（通常是 {op_name}_func 或 {op_name}_kernel）
    possible_func_names = [
        f'{op_name}_func',
        f'{op_name}_kernel',
        f'_{op_name}_kernel',
    ]
    
    scalar_body = None
    for func_name in possible_func_names:
        scalar_body = extract_scalar_function_body(triton_file, func_name)
        if scalar_body:
            break
    
    # 推断计算表达式
    compute_expr = infer_compute_expr(op_name, scalar_body)
    
    # 生成代码
    code = UNARY_TEMPLATE.format(
        op_name=op_name,
        compute_expr=compute_expr
    )
    
    return code

def extract_wrapper_functions(triton_file):
    """提取原文件中的其他 wrapper 函数（如 in-place 版本）"""
    with open(triton_file, 'r') as f:
        content = f.read()
    
    # 查找所有不使用 pointwise_dynamic 的函数定义
    # 例如 abs_ 这种 in-place 版本
    pattern = r'\ndef\s+(\w+_)\([^)]*\):[^\n]*\n(?:(?:    [^\n]*\n)*)'
    matches = re.finditer(pattern, content)
    
    wrappers = []
    for match in matches:
        func_def = match.group(0)
        if '@' not in func_def:  # 不是装饰器函数
            wrappers.append(func_def)
    
    return wrappers

def process_unary_op(op_name, dry_run=True):
    """处理单个 unary 算子"""
    op_dir = Path(f"/share/project/tj/workspace/FlagGems/operators_py/{op_name}")
    triton_file = op_dir / f"{op_name}_triton.py"
    
    if not triton_file.exists():
        return False
    
    # 生成新的 kernel 代码
    new_code = generate_kernel_for_op(op_name, triton_file)
    
    # 提取其他 wrapper 函数
    wrappers = extract_wrapper_functions(triton_file)
    
    # 如果有其他 wrapper，附加到新代码后面
    if wrappers:
        # 需要修改这些 wrapper 来调用新的 kernel
        for wrapper in wrappers:
            # 简单处理：替换调用
            wrapper = wrapper.replace(f'{op_name}_func(', f'{op_name}(')
            new_code += '\n' + wrapper
    
    if dry_run:
        print(f"\n{'='*60}")
        print(f"Generated code for: {op_name}")
        print(f"{'='*60}")
        print(new_code[:500] + "...")
        return True
    else:
        # 备份原文件
        backup_file = triton_file.with_suffix('.py.bak')
        if not backup_file.exists():
            import shutil
            shutil.copy(triton_file, backup_file)
        
        # 写入新代码
        with open(triton_file, 'w') as f:
            f.write(new_code)
        
        return True

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='只显示生成的代码，不修改文件')
    parser.add_argument('--ops', nargs='+', help='指定要转换的算子')
    args = parser.parse_args()
    
    print("="*60)
    print("Step 2: 生成 Simple Unary Kernels")
    print("="*60)
    
    # 读取 simple_unary 列表
    if args.ops:
        ops_to_process = args.ops
    else:
        unary_file = Path("/share/project/tj/workspace/FlagGems/simple_unary_ops.txt")
        with open(unary_file, 'r') as f:
            ops_to_process = [line.strip() for line in f if line.strip()]
    
    print(f"\n将处理 {len(ops_to_process)} 个算子")
    if args.dry_run:
        print("(DRY RUN 模式 - 不会修改文件)")
        # 只显示前3个
        ops_to_process = ops_to_process[:3]
    
    success = 0
    for op_name in ops_to_process:
        try:
            if process_unary_op(op_name, dry_run=args.dry_run):
                success += 1
        except Exception as e:
            print(f"\n✗ {op_name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"完成: {success}/{len(ops_to_process)} 个算子")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
