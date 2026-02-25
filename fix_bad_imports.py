#!/usr/bin/env python3
"""
修复错误的 import 语句
"""

import re
from pathlib import Path

files = [
    "cumsum", "eye", "eye_m", "ones", "rand", 
    "randn", "randperm", "upsample_bicubic2d_aa", 
    "upsample_nearest2d", "zeros"
]

operators_dir = Path("/share/project/tj/workspace/FlagGems/operators_py")

for op_name in files:
    file_path = operators_dir / op_name / f"{op_name}_triton.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 移除错误的 import 行
    content = re.sub(r'^from flag_gems\.runtime import.*$', '', content, flags=re.MULTILINE)
    
    # 清理多余空行
    content = re.sub(r'\n\n\n+', '\n\n', content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✓ {op_name}")

print("\n验证语法...")
import py_compile
errors = 0
for op_name in files:
    file_path = operators_dir / op_name / f"{op_name}_triton.py"
    try:
        py_compile.compile(str(file_path), doraise=True)
    except:
        print(f"✗ {op_name} 仍有错误")
        errors += 1

if errors == 0:
    print("✅ 所有文件语法正确!")
