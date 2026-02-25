#!/usr/bin/env python3
"""
Step 1: 分类 pointwise_dynamic 算子
将算子分为 simple_unary, simple_binary, complex 三类
"""

import re
from pathlib import Path
from collections import defaultdict

def parse_params(param_str):
    """解析函数参数"""
    if not param_str.strip():
        return []
    return [p.strip().split(':')[0].strip() for p in param_str.split(',') if p.strip()]

def analyze_decorator_args(decorator_str):
    """分析装饰器参数"""
    info = {
        'is_tensor': None,
        'promotion_methods': None,
        'num_inputs': None,
    }
    
    # 提取 is_tensor
    is_tensor_match = re.search(r'is_tensor\s*=\s*\[(.*?)\]', decorator_str)
    if is_tensor_match:
        info['is_tensor'] = is_tensor_match.group(1)
    
    # 提取 promotion_methods
    promo_match = re.search(r'promotion_methods\s*=\s*\[(.*?)\]', decorator_str)
    if promo_match:
        info['promotion_methods'] = promo_match.group(1)
    
    return info

def classify_function(func_info):
    """分类单个函数"""
    params = parse_params(func_info['params'])
    num_params = len(params)
    
    decorator_info = analyze_decorator_args(func_info['decorator_args'])
    
    # Simple unary: 单个tensor输入，无特殊参数
    if num_params == 1:
        if decorator_info['is_tensor'] is None or 'True' in decorator_info['is_tensor']:
            return 'simple_unary'
    
    # Simple binary: 两个tensor输入
    if num_params == 2:
        if decorator_info['is_tensor']:
            if decorator_info['is_tensor'].count('True') == 2:
                return 'simple_binary'
    
    # 其他情况为 complex
    return 'complex'

def analyze_operator(op_path):
    """分析单个算子"""
    op_name = op_path.name
    triton_file = op_path / f"{op_name}_triton.py"
    
    if not triton_file.exists():
        return None
    
    with open(triton_file, 'r') as f:
        content = f.read()
    
    if 'pointwise_dynamic' not in content:
        return None
    
    # 查找所有 pointwise_dynamic 函数
    pattern = r'@pointwise_dynamic\((.*?)\)\s*@triton\.jit\s*def\s+(\w+)\((.*?)\):'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    functions = []
    for match in matches:
        decorator_args = match.group(1)
        func_name = match.group(2)
        func_params = match.group(3)
        
        functions.append({
            'name': func_name,
            'params': func_params,
            'decorator_args': decorator_args
        })
    
    if not functions:
        return None
    
    # 如果只有一个函数，直接分类
    if len(functions) == 1:
        category = classify_function(functions[0])
        return {
            'op_name': op_name,
            'category': category,
            'num_functions': 1,
            'functions': functions
        }
    else:
        # 多个函数通常是 complex
        return {
            'op_name': op_name,
            'category': 'complex',
            'num_functions': len(functions),
            'functions': functions,
            'reason': f'Multiple variants ({len(functions)})'
        }

def main():
    print("="*60)
    print("Step 1: 分类 Pointwise Dynamic 算子")
    print("="*60)
    
    operators_dir = Path("/share/project/tj/workspace/FlagGems/operators_py")
    
    categories = defaultdict(list)
    
    for op_dir in sorted(operators_dir.iterdir()):
        if not op_dir.is_dir():
            continue
        if op_dir.name in ['common', '__pycache__']:
            continue
        
        result = analyze_operator(op_dir)
        if result:
            categories[result['category']].append(result)
    
    # 打印统计
    print(f"\n统计结果:")
    print(f"  Simple Unary:  {len(categories['simple_unary']):3d} 个")
    print(f"  Simple Binary: {len(categories['simple_binary']):3d} 个")
    print(f"  Complex:       {len(categories['complex']):3d} 个")
    print(f"  Total:         {sum(len(v) for v in categories.values()):3d} 个")
    
    # 保存到文件
    output_dir = Path("/share/project/tj/workspace/FlagGems")
    
    for category, ops in categories.items():
        output_file = output_dir / f"{category}_ops.txt"
        with open(output_file, 'w') as f:
            for op in sorted(ops, key=lambda x: x['op_name']):
                f.write(f"{op['op_name']}\n")
        print(f"\n✓ Saved {output_file}")
    
    # 打印详细列表
    print(f"\n" + "="*60)
    print("详细分类:")
    print("="*60)
    
    for category in ['simple_unary', 'simple_binary', 'complex']:
        print(f"\n{category.upper()}:")
        for op in sorted(categories[category], key=lambda x: x['op_name']):
            reason = f" ({op.get('reason', '')})" if 'reason' in op else ''
            print(f"  - {op['op_name']}{reason}")

if __name__ == '__main__':
    main()
