#!/usr/bin/env python3
"""
验证所有 triton 文件的 Python 语法
"""

import py_compile
from pathlib import Path

operators_dir = Path("/share/project/tj/workspace/FlagGems/operators_py")

errors = []
success = 0

for op_dir in sorted(operators_dir.iterdir()):
    if not op_dir.is_dir() or op_dir.name in ['common', '__pycache__']:
        continue
    
    triton_file = op_dir / f"{op_dir.name}_triton.py"
    if not triton_file.exists():
        continue
    
    try:
        py_compile.compile(str(triton_file), doraise=True)
        success += 1
    except py_compile.PyCompileError as e:
        errors.append((op_dir.name, str(e)))
        print(f"✗ {op_dir.name}")

print(f"\n{'='*60}")
print(f"验证完成: {success} 个文件通过, {len(errors)} 个文件有错误")
print(f"{'='*60}")

if errors:
    print("\n有错误的文件:")
    for name, error in errors:
        print(f"\n{name}:")
        print(f"  {error[:200]}...")
