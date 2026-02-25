#!/usr/bin/env python3
"""
将使用 pointwise_dynamic 的算子转换为静态 Triton kernel
"""

import os
import sys
import re
import torch
from pathlib import Path
import importlib.util

# 添加 FlagGems 到 path
sys.path.insert(0, '/share/project/tj/workspace/FlagGems/src')

def extract_scalar_function(triton_file_path):
    """从 triton 文件中提取 scalar function 和装饰器信息"""
    with open(triton_file_path, 'r') as f:
        content = f.read()
    
    # 查找所有使用 pointwise_dynamic 的函数
    pattern = r'@pointwise_dynamic\((.*?)\)\s*@triton\.jit\s*def\s+(\w+)\((.*?)\):\s*(.*?)(?=\n(?:def|@|\Z))'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    functions = []
    for match in matches:
        decorator_args = match.group(1)
        func_name = match.group(2)
        func_params = match.group(3)
        func_body = match.group(4)
        
        functions.append({
            'name': func_name,
            'params': func_params,
            'body': func_body.strip(),
            'decorator_args': decorator_args
        })
    
    return functions

def generate_simple_kernel(func_info, op_name):
    """生成简化的 Triton kernel（不依赖 pointwise_dynamic）"""
    
    # 提取参数名
    params = [p.strip().split(':')[0].strip() for p in func_info['params'].split(',') if p.strip()]
    
    # 生成 kernel
    kernel_code = f'''import torch
import triton
import triton.language as tl

@triton.jit
def {op_name}_kernel(
'''
    
    # 添加输入指针
    for i, param in enumerate(params):
        kernel_code += f'    {param}_ptr,\n'
    
    kernel_code += f'''    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
'''
    
    for param in params:
        kernel_code += f'    {param} = tl.load({param}_ptr + offsets, mask=mask)\n'
    
    # 添加计算逻辑（从 scalar function body 提取）
    kernel_code += f'''    
    # Compute
    {func_info['body']}
    output = result  # Assuming the function returns result
    
    # Store output
    tl.store(output_ptr + offsets, output, mask=mask)

def {op_name}('''
    
    # 生成 wrapper 参数
    for i, param in enumerate(params):
        if i > 0:
            kernel_code += ', '
        kernel_code += param
    
    kernel_code += f'''):
    """
    Triton implementation of {op_name}
    """
    # 处理输入
    input_tensor = {params[0]}
    output = torch.empty_like(input_tensor)
    
    n_elements = input_tensor.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    {op_name}_kernel[grid](
'''
    
    for param in params:
        kernel_code += f'        {param},\n'
    
    kernel_code += f'''        output,
        n_elements,
        BLOCK_SIZE=1024,
    )
    
    return output
'''
    
    return kernel_code

def process_operator(op_path):
    """处理单个算子"""
    triton_file = op_path / f"{op_path.name}_triton.py"
    
    if not triton_file.exists():
        return False
    
    # 检查是否使用 pointwise_dynamic
    with open(triton_file, 'r') as f:
        content = f.read()
    
    if 'pointwise_dynamic' not in content:
        return False
    
    print(f"Processing {op_path.name}...")
    
    # 提取 scalar functions
    functions = extract_scalar_function(triton_file)
    
    if not functions:
        print(f"  No pointwise_dynamic functions found")
        return False
    
    # 生成新的代码（先简单打印，不修改文件）
    for func_info in functions:
        print(f"  Found function: {func_info['name']}")
        # TODO: 生成实际的 kernel 代码
    
    return True

def main():
    operators_dir = Path("/share/project/tj/workspace/FlagGems/operators_py")
    
    # 扫描所有算子
    processed = 0
    for op_dir in sorted(operators_dir.iterdir()):
        if not op_dir.is_dir():
            continue
        if op_dir.name in ['common', '__pycache__']:
            continue
            
        if process_operator(op_dir):
            processed += 1
    
    print(f"\n{'='*60}")
    print(f"Processed {processed} operators")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
