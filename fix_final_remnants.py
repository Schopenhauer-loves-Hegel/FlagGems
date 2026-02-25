#!/usr/bin/env python3
"""
修复最后的残留依赖
"""

from pathlib import Path
import re

def fix_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    original = content
    
    # 替换 tle.
    content = content.replace('tle.', 'tl.')
    
    # 移除 runtime 导入
    content = re.sub(r'^from flag_gems import runtime\s*\n', '', content, flags=re.MULTILINE)
    
    if content != original:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

files = [
    "cummin/cummin_triton.py",
    "cummax/cummax_triton.py",
    "index_put/index_put_triton.py",
    "index/index_triton.py",
    "scatter/scatter_triton.py",
    "gather/gather_triton.py",
]

operators_dir = Path("/share/project/tj/workspace/FlagGems/operators_py")

for f in files:
    file_path = operators_dir / f
    if fix_file(file_path):
        print(f"✓ {f.split('/')[0]}")

print("\n最终验证...")
import subprocess
result = subprocess.run(
    "find /share/project/tj/workspace/FlagGems/operators_py -name '*_triton.py' -exec grep -l 'tle\\.' {} \; | wc -l",
    shell=True, capture_output=True, text=True
)
print(f"tle残留: {result.stdout.strip()} 个")

result = subprocess.run(
    "find /share/project/tj/workspace/FlagGems/operators_py -name '*_triton.py' -exec grep -l 'from flag_gems import runtime' {} \; | wc -l",
    shell=True, capture_output=True, text=True
)
print(f"runtime残留: {result.stdout.strip()} 个")
