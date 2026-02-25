#!/usr/bin/env python3
"""
安全清理：只做简单替换，不动缩进
"""

import re
from pathlib import Path

def safe_cleanup(file_path):
    """安全清理单个文件"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original = content
    
    # 1. 替换 tle. → tl.
    content = content.replace('tle.program_id', 'tl.program_id')
    content = content.replace('tle.num_programs', 'tl.num_programs')
    
    # 2. 移除单行 runtime 导入
    content = re.sub(r'^from flag_gems import runtime\s*$', '', content, flags=re.MULTILINE)
    
    # 3. 移除 tle 导入
    content = re.sub(r'^from flag_gems\.utils import triton_lang_extension as tle\s*$', '', content, flags=re.MULTILINE)
    
    # 4. 从多项导入中移除 libtuner
    content = re.sub(r',\s*libtuner\b', '', content)
    content = re.sub(r'\blibtuner\s*,\s*', '', content)
    
    # 5. 移除 @libtuner 装饰器（多行）
    lines = content.split('\n')
    new_lines = []
    skip_until = -1
    
    for i, line in enumerate(lines):
        if i < skip_until:
            continue
            
        if '@libtuner' in line:
            # 找到装饰器结束位置
            paren_count = line.count('(') - line.count(')')
            j = i + 1
            while j < len(lines) and paren_count > 0:
                paren_count += lines[j].count('(') - lines[j].count(')')
                j += 1
            skip_until = j
        else:
            new_lines.append(line)
    
    content = '\n'.join(new_lines)
    
    # 6. 清理多余空行（但不超过2个）
    content = re.sub(r'\n\n\n+', '\n\n', content)
    
    if content != original:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

operators_dir = Path("/share/project/tj/workspace/FlagGems/operators_py")

modified = []
for op_dir in sorted(operators_dir.iterdir()):
    if not op_dir.is_dir() or op_dir.name in ['common', '__pycache__']:
        continue
    
    triton_file = op_dir / f"{op_dir.name}_triton.py"
    if triton_file.exists():
        try:
            if safe_cleanup(triton_file):
                modified.append(op_dir.name)
                print(f"✓ {op_dir.name}")
        except Exception as e:
            print(f"✗ {op_dir.name}: {e}")

print(f"\n修改了 {len(modified)} 个文件")
