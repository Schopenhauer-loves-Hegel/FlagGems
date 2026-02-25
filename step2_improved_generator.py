#!/usr/bin/env python3
"""
改进的 Unary Kernel 生成器 - 保留 logger 和 in-place 版本
"""

import re
from pathlib import Path

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

IMPROVED_TEMPLATE = '''"""
Triton implementation - {op_name}
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def {op_name}_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for {op_name} operation."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    output = {compute_expr}
    tl.store(output_ptr + offsets, output, mask=mask)


def {op_name}(A):
    logger.debug("GEMS {op_name_upper}")
    output = torch.empty_like(A)
    n_elements = A.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    {op_name}_kernel[grid](A, output, n_elements, BLOCK_SIZE=1024)
    return output


def {op_name}_(A):
    logger.debug("GEMS {op_name_upper}_")
    n_elements = A.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    {op_name}_kernel[grid](A, A, n_elements, BLOCK_SIZE=1024)
    return A
'''

def has_inplace_version(triton_file, op_name):
    """检查是否有 in-place 版本"""
    with open(triton_file, 'r') as f:
        content = f.read()
    return f'def {op_name}_(' in content

def generate_improved_kernel(op_name, triton_file):
    """生成改进的 kernel（保留 logger 和 in-place）"""
    
    # 获取计算表达式
    compute_expr = SCALAR_EXPR_MAP.get(op_name, f'tl.{op_name}(x)')
    
    # 生成代码
    code = IMPROVED_TEMPLATE.format(
        op_name=op_name,
        op_name_upper=op_name.upper(),
        compute_expr=compute_expr
    )
    
    return code

def process_op(op_name, apply=False):
    """处理单个算子"""
    op_dir = Path(f"/share/project/tj/workspace/FlagGems/operators_py/{op_name}")
    triton_file = op_dir / f"{op_name}_triton.py"
    
    if not triton_file.exists():
        print(f"✗ {op_name}: 文件不存在")
        return False
    
    # 生成新代码
    new_code = generate_improved_kernel(op_name, triton_file)
    
    if apply:
        # 备份
        backup = triton_file.with_suffix('.py.bak')
        if not backup.exists():
            import shutil
            shutil.copy(triton_file, backup)
        
        # 写入
        with open(triton_file, 'w') as f:
            f.write(new_code)
        print(f"✓ {op_name}: 已转换")
    else:
        print(f"✓ {op_name}: 代码已生成（未应用）")
    
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', help='实际应用转换（否则只是预览）')
    parser.add_argument('--ops', nargs='+', help='指定算子')
    args = parser.parse_args()
    
    print("="*60)
    print("改进的 Unary Kernel 生成器")
    print("="*60)
    
    if args.ops:
        ops = args.ops
    else:
        # 读取 simple_unary 列表
        with open('/share/project/tj/workspace/FlagGems/simple_unary_ops.txt', 'r') as f:
            ops = [line.strip() for line in f if line.strip()]
    
    print(f"\n将处理 {len(ops)} 个算子")
    if not args.apply:
        print("预览模式 - 使用 --apply 来实际应用\n")
    
    success = 0
    for op in ops:
        if process_op(op, args.apply):
            success += 1
    
    print(f"\n{'='*60}")
    print(f"完成: {success}/{len(ops)}")
    print(f"{'='*60}")
    
    if not args.apply:
        print("\n提示: 使用 --apply 参数来实际应用转换")
        print("      原始文件会备份为 .py.bak")

if __name__ == '__main__':
    main()
