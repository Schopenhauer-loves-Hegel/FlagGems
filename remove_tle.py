#!/usr/bin/env python3
"""
移除 tle 模块依赖，将 tle 函数内嵌到文件中
"""

import re
from pathlib import Path

# tle 模块中的所有函数定义
TLE_FUNCTIONS = {
    'program_id': """@triton.jit
def program_id(axis: int):
    return tl.program_id(axis).to(tl.int64)""",

    'num_programs': """@triton.jit
def num_programs(axis: int):
    return tl.num_programs(axis).to(tl.int64)""",

    'promote_to_tensor': """@triton.jit
def promote_to_tensor(x):
    # Addition promotes to tensor for us
    return x + tl.zeros((1,), tl.int1)""",

    'is_floating': """@triton.jit
def is_floating(x):
    return promote_to_tensor(x).dtype.is_floating()""",

    'minimum_with_index_tie_break_right': """@triton.jit
def minimum_with_index_tie_break_right(a_value, a_index, b_value, b_index):
    mask = a_value < b_value
    equal = a_value == b_value
    if is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan and not b_isnan
        # Consider NaNs as equal
        equal |= a_isnan and b_isnan

    # Prefer highest index if values are equal
    mask |= equal & (a_index > b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)"""
}

# 函数依赖关系
FUNCTION_DEPENDENCIES = {
    'is_floating': ['promote_to_tensor'],
    'minimum_with_index_tie_break_right': ['is_floating', 'promote_to_tensor']
}

def extract_used_tle_functions(content):
    """提取文件中使用的 tle 函数"""
    used_functions = set()

    # 查找所有 tle.function_name 调用
    pattern = r'tle\.([a-z_]+)'
    matches = re.finditer(pattern, content)

    for match in matches:
        func_name = match.group(1)
        if func_name in TLE_FUNCTIONS:
            used_functions.add(func_name)

    return used_functions

def get_functions_with_dependencies(used_functions):
    """获取函数及其依赖"""
    result = set(used_functions)

    for func in used_functions:
        if func in FUNCTION_DEPENDENCIES:
            result.update(FUNCTION_DEPENDENCIES[func])

    return result

def process_file(file_path):
    """处理单个文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否使用了 tle
    if 'from ..utils import triton_lang_extension as tle' not in content:
        return False

    # 提取使用的函数
    used_functions = extract_used_tle_functions(content)
    if not used_functions:
        # 如果没有使用任何 tle 函数，直接移除 import
        content = content.replace('\nfrom ..utils import triton_lang_extension as tle\n', '\n')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True

    # 获取包含依赖的完整函数列表
    all_functions = get_functions_with_dependencies(used_functions)

    # 构建需要添加的函数定义（按依赖顺序）
    function_order = ['promote_to_tensor', 'is_floating', 'program_id', 'num_programs', 'minimum_with_index_tie_break_right']
    functions_to_add = [func for func in function_order if func in all_functions]

    if functions_to_add:
        function_defs = [TLE_FUNCTIONS[func] for func in functions_to_add]
        functions_block = '\n# Triton helper functions\n' + '\n\n'.join(function_defs) + '\n\n'

        # 找到合适的位置插入函数（在第一个 @libentry 或 @triton.jit 之前）
        insert_patterns = [
            r'@libentry\(\)',
            r'@triton\.jit',
            r'@triton\.autotune',
            r'@triton\.heuristics'
        ]

        insert_pos = -1
        for pattern in insert_patterns:
            match = re.search(pattern, content)
            if match:
                insert_pos = match.start()
                break

        if insert_pos == -1:
            # 如果找不到，插入到 import 之后
            last_import = max(
                content.rfind('\nimport '),
                content.rfind('\nfrom ')
            )
            if last_import != -1:
                insert_pos = content.find('\n', last_import + 1) + 1

        if insert_pos != -1:
            # 找到这一行的开始
            line_start = content.rfind('\n', 0, insert_pos) + 1
            content = content[:line_start] + functions_block + content[line_start:]

    # 替换 tle.function_name 为 function_name
    for func_name in used_functions:
        content = re.sub(rf'\btle\.{func_name}\b', func_name, content)

    # 移除 tle import
    content = content.replace('\nfrom ..utils import triton_lang_extension as tle\n', '\n')

    # 清理多余的空行
    content = re.sub(r'\n\n\n+', '\n\n', content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return True

def main():
    print("=" * 60)
    print("移除 tle 模块依赖")
    print("=" * 60)

    operators_dir = Path("/share/project/tj/workspace/FlagGems/operators_py")
    triton_files = list(operators_dir.glob("*/*_triton.py"))

    modified_count = 0

    for triton_file in sorted(triton_files):
        op_name = triton_file.parent.name

        try:
            if process_file(triton_file):
                print(f"✓ {op_name:30s} - 已处理")
                modified_count += 1
        except Exception as e:
            print(f"✗ {op_name:30s} - 错误: {e}")

    print("\n" + "=" * 60)
    print(f"处理完成: {modified_count} 个文件被修改")
    print("=" * 60)

if __name__ == '__main__':
    main()
