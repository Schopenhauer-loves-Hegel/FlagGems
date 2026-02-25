#!/usr/bin/env python3
"""
修复代码生成文件中的 tle 引用
"""

import re
from pathlib import Path

def process_file(file_path):
    """处理单个文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    modified = False

    # 1. 替换 tle import 为 tle 函数定义
    tle_import_pattern = r'code\.writeline\("from flag_gems\.utils import triton_lang_extension as tle"\)'

    if re.search(tle_import_pattern, content):
        # 构建替换的代码块
        replacement = '''code.writeline("")
    code.writeline("# Triton helper functions")
    code.writeline("@triton.jit")
    code.writeline("def program_id(axis: int):")
    code.writeline("    return tl.program_id(axis).to(tl.int64)")
    code.writeline("")
    code.writeline("@triton.jit")
    code.writeline("def num_programs(axis: int):")
    code.writeline("    return tl.num_programs(axis).to(tl.int64)")'''

        content = re.sub(tle_import_pattern, replacement, content)
        modified = True

    # 2. 替换 tle.program_id 为 program_id
    if 'tle.program_id' in content:
        content = content.replace('tle.program_id', 'program_id')
        modified = True

    # 3. 替换 tle.num_programs 为 num_programs
    if 'tle.num_programs' in content:
        content = content.replace('tle.num_programs', 'num_programs')
        modified = True

    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True

    return False

def main():
    print("=" * 60)
    print("修复代码生成文件中的 tle 引用")
    print("=" * 60)

    # 需要处理的文件列表
    files_to_process = [
        "/share/project/tj/workspace/FlagGems/operators_py/gather/gather_triton.py",
        "/share/project/tj/workspace/FlagGems/operators_py/tile/tile_triton.py",
        "/share/project/tj/workspace/FlagGems/operators_py/repeat/repeat_triton.py",
        "/share/project/tj/workspace/FlagGems/operators_py/pad/pad_triton.py",
        "/share/project/tj/workspace/FlagGems/operators_py/index/index_triton.py",
        "/share/project/tj/workspace/FlagGems/operators_py/index_put/index_put_triton.py",
        "/share/project/tj/workspace/FlagGems/operators_py/scatter/scatter_triton.py",
    ]

    modified_count = 0

    for file_path in files_to_process:
        file_path_obj = Path(file_path)
        op_name = file_path_obj.parent.name

        try:
            if process_file(file_path):
                print(f"✓ {op_name:30s} - 已修正")
                modified_count += 1
            else:
                print(f"  {op_name:30s} - 无需修改")
        except Exception as e:
            print(f"✗ {op_name:30s} - 错误: {e}")

    print("\n" + "=" * 60)
    print(f"处理完成: {modified_count} 个文件被修改")
    print("=" * 60)

if __name__ == '__main__':
    main()
