#!/usr/bin/env python3
"""
最终清理：移除所有 @libtuner 装饰器
"""

import re
from pathlib import Path

def remove_libtuner(file_path):
    """移除 @libtuner 装饰器"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    i = 0
    modified = False
    
    while i < len(lines):
        line = lines[i]
        
        # 检测到 @libtuner 行
        if '@libtuner' in line:
            # 跳过所有属于这个装饰器的行（直到遇到闭合括号）
            paren_count = line.count('(') - line.count(')')
            i += 1
            
            while i < len(lines) and paren_count > 0:
                line = lines[i]
                paren_count += line.count('(') - line.count(')')
                i += 1
            
            modified = True
        else:
            new_lines.append(line)
            i += 1
    
    if modified:
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
        return True
    
    return False

def main():
    files_to_fix = [
        "sum", "mm", "mean", "prod", "min", "bmm", 
        "max", "baddbmm", "attention", "any", "all", "addmm"
    ]
    
    operators_dir = Path("/share/project/tj/workspace/FlagGems/operators_py")
    
    for op_name in files_to_fix:
        file_path = operators_dir / op_name / f"{op_name}_triton.py"
        if file_path.exists() and remove_libtuner(file_path):
            print(f"✓ {op_name}")

if __name__ == '__main__':
    main()
