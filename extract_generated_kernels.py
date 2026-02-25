#!/usr/bin/env python3
"""
方案C实现：从 pointwise_dynamic 生成的缓存中提取代码，转换为静态实现

核心思路：
1. 运行每个算子一次，让 pointwise_dynamic 生成代码到缓存
2. 从缓存读取生成的 kernel 和 wrapper
3. 清理依赖，简化代码
4. 替换原始文件
"""

import os
import sys
import re
import torch
from pathlib import Path
import tempfile

# 添加 FlagGems 到 path
sys.path.insert(0, '/share/project/tj/workspace/FlagGems/src')

def get_cache_dir():
    """获取 flaggems 缓存目录"""
    cache_dir = Path.home() / '.flaggems'
    return cache_dir

def trigger_code_generation(op_name, triton_file):
    """触发代码生成：导入并运行算子，让 pointwise_dynamic 生成代码"""
    try:
        # 临时导入模块
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"temp_{op_name}", triton_file)
        if spec is None or spec.loader is None:
            return False
        
        module = importlib.util.module_from_spec(spec)
        
        # 执行模块（可能会失败，因为依赖问题，但没关系）
        try:
            spec.loader.exec_module(module)
            
            # 尝试运行算子函数触发代码生成
            if hasattr(module, op_name):
                func = getattr(module, op_name)
                # 用简单的输入触发生成（可能失败，但会生成代码）
                try:
                    test_input = torch.randn(10, device='cuda' if torch.cuda.is_available() else 'cpu')
                    func(test_input)
                except:
                    pass  # 即使失败也可能已经生成了代码
        except Exception as e:
            print(f"    Warning: Module execution failed: {e}")
            return False
        
        return True
    except Exception as e:
        print(f"    Error triggering generation: {e}")
        return False

def find_generated_files(op_name, cache_dir):
    """在缓存目录中查找为该算子生成的文件"""
    pattern = f"pointwise_dynamic_*{op_name}*.py"
    files = list(cache_dir.glob(pattern))
    return files

def extract_kernel_and_wrapper(generated_file):
    """从生成的文件中提取 kernel 和 wrapper 函数"""
    with open(generated_file, 'r') as f:
        content = f.read()
    
    # 移除依赖导入，替换为基本导入
    simplified_imports = """import torch
import triton
import triton.language as tl
"""
    
    # 查找第一个 import 块的结束位置
    import_end = content.find('\n\n\n')
    if import_end != -1:
        # 替换 import 部分
        content = simplified_imports + content[import_end:]
    
    return content

def simplify_generated_code(content):
    """简化生成的代码，移除不必要的依赖"""
    
    # 移除 libentry 装饰器
    content = re.sub(r'@libentry\(\)\s*\n', '', content)
    
    # 移除 tle 相关的调用，替换为标准 triton 调用
    content = content.replace('tle.program_id', 'tl.program_id')
    content = content.replace('tle.num_programs', 'tl.num_programs')
    
    # 移除 torch_device_fn 相关
    content = re.sub(r'with torch_device_fn\.device\([^)]+\):\s*\n', '', content)
    # 调整缩进
    content = re.sub(r'\n        ([^\n])', r'\n    \1', content)
    
    # 移除 StridedBuffer 相关（简化为直接使用 tensor）
    content = content.replace('Union[torch.Tensor, StridedBuffer]', 'torch.Tensor')
    content = content.replace('StridedBuffer', 'torch.Tensor')
    
    # 移除其他工具函数调用
    content = re.sub(r'from flag_gems\..*\n', '', content)
    
    return content

def create_simple_template(op_name, scalar_func_body):
    """为简单算子创建模板代码（当无法从缓存提取时）"""
    template = f'''import torch
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
    Triton kernel for {op_name} operation
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute
{scalar_func_body}
    
    # Store
    tl.store(output_ptr + offsets, output, mask=mask)


def {op_name}(input_tensor):
    """
    Apply {op_name} operation element-wise
    
    Args:
        input_tensor: Input tensor
        
    Returns:
        Output tensor with same shape as input
    """
    output = torch.empty_like(input_tensor)
    n_elements = input_tensor.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    {op_name}_kernel[grid](
        input_tensor,
        output,
        n_elements,
        BLOCK_SIZE=1024,
    )
    
    return output
'''
    return template

def analyze_pointwise_usage(triton_file):
    """分析 pointwise_dynamic 的使用模式"""
    with open(triton_file, 'r') as f:
        content = f.read()
    
    # 查找 scalar functions
    pattern = r'@pointwise_dynamic\((.*?)\)\s*@triton\.jit\s*def\s+(\w+)\((.*?)\):\s*\n((?:    .*\n)*)'
    matches = re.finditer(pattern, content)
    
    funcs = []
    for match in matches:
        decorator_args = match.group(1)
        func_name = match.group(2)
        func_params = match.group(3)
        func_body = match.group(4)
        
        funcs.append({
            'name': func_name,
            'params': func_params,
            'body': func_body,
            'decorator_args': decorator_args
        })
    
    return funcs

def process_operator(op_path):
    """处理单个算子"""
    op_name = op_path.name
    triton_file = op_path / f"{op_name}_triton.py"
    
    if not triton_file.exists():
        return False
    
    with open(triton_file, 'r') as f:
        content = f.read()
    
    if 'pointwise_dynamic' not in content:
        return False
    
    print(f"\nProcessing: {op_name}")
    
    # 分析使用模式
    funcs = analyze_pointwise_usage(triton_file)
    if not funcs:
        print(f"  ⚠️  No pointwise_dynamic functions found")
        return False
    
    print(f"  Found {len(funcs)} pointwise function(s)")
    for func in funcs:
        print(f"    - {func['name']}")
    
    # 暂时只显示信息，不修改文件
    return True

def main():
    print("="*60)
    print("Pointwise Dynamic to Static Conversion Tool")
    print("="*60)
    
    operators_dir = Path("/share/project/tj/workspace/FlagGems/operators_py")
    
    # 统计
    total = 0
    processed = 0
    
    for op_dir in sorted(operators_dir.iterdir()):
        if not op_dir.is_dir():
            continue
        if op_dir.name in ['common', '__pycache__', 'INDEX.md']:
            continue
        
        total += 1
        if process_operator(op_dir):
            processed += 1
    
    print("\n" + "="*60)
    print(f"Summary: {processed}/{total} operators use pointwise_dynamic")
    print("="*60)

if __name__ == '__main__':
    main()
