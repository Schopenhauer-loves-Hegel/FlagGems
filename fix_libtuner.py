#!/usr/bin/env python3
"""
修复残留的 @libtuner 装饰器
"""

import re
from pathlib import Path

def remove_libtuner_decorator(content):
    """移除 @libtuner 装饰器（支持多行）"""
    # 匹配 @libtuner( ... ) 可能跨越多行
    pattern = r'@libtuner\([^)]*(?:\n[^)]*)*\)\s*\n'
    content = re.sub(pattern, '', content, flags=re.MULTILINE)
    return content

def process_file(file_path):
    """处理单个文件"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    if '@libtuner' not in content:
        return False
    
    original = content
    content = remove_libtuner_decorator(content)
    
    if content != original:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    operators_dir = Path("/share/project/tj/workspace/FlagGems/operators_py")
    
    modified = []
    for op_dir in operators_dir.iterdir():
        if not op_dir.is_dir() or op_dir.name in ['common', '__pycache__']:
            continue
        
        triton_file = op_dir / f"{op_dir.name}_triton.py"
        if triton_file.exists() and process_file(triton_file):
            modified.append(op_dir.name)
            print(f"✓ {op_dir.name}")
    
    print(f"\n修改了 {len(modified)} 个文件")

if __name__ == '__main__':
    main()
