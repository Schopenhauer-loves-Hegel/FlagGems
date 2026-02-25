#!/usr/bin/env python3
"""
修正 heuristics 函数引用问题
将引用的函数定义添加到文件中
"""

import re
from pathlib import Path

# NVIDIA heuristics 函数定义
HEURISTICS_FUNCTIONS = """
import torch
import triton

def argmax_heur_block_m(args):
    return 4 if args["M"] < 4096 else 8

def argmax_heur_block_n(args):
    return min(4096, triton.next_power_of_2(args["N"]))

def argmin_heur_block_m(args):
    return 4 if args["M"] < 4096 else 8

def argmin_heur_block_n(args):
    return min(4096, triton.next_power_of_2(args["N"]))

def bmm_heur_divisible_m(args):
    return args["M"] % args["TILE_M"] == 0

def bmm_heur_divisible_n(args):
    return args["N"] % args["TILE_N"] == 0

def bmm_heur_divisible_k(args):
    return args["K"] % args["TILE_K"] == 0

def dropout_heur_block(args):
    return 512 if args["N"] <= 512 else 1024

def dropout_heur_num_warps(args):
    if args["N"] <= 512:
        return 4
    elif args["N"] <= 1024:
        return 8
    else:
        return 16

def exponential_heur_block(args):
    return 512 if args["N"] <= 512 else 1024

def exponential_heur_num_warps(args):
    if args["N"] <= 512:
        return 4
    elif args["N"] <= 1024:
        return 8
    else:
        return 16

def gather_heur_block_m(args):
    return min(4, triton.next_power_of_2(triton.cdiv(args["N"], 2048)))

def gather_heur_block_n(args):
    return min(2048, triton.next_power_of_2(args["N"]))

def index_select_heur_block_m(args):
    return min(4, triton.next_power_of_2(triton.cdiv(256, args["N"])))

def index_select_heur_block_n(args):
    m = min(triton.next_power_of_2(triton.cdiv(args["N"], 16)), 512)
    return max(m, 16)

def mm_heur_even_k(args):
    return args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0

def rand_heur_block(args):
    return 512 if args["N"] <= 512 else 1024

def rand_heur_num_warps(args):
    if args["N"] <= 512:
        return 4
    elif args["N"] <= 1024:
        return 8
    else:
        return 16

def randn_heur_block(args):
    return 512 if args["N"] <= 512 else 1024

def randn_heur_num_warps(args):
    if args["N"] <= 512:
        return 4
    elif args["N"] <= 1024:
        return 8
    else:
        return 16

def softmax_heur_tile_k(args):
    MAX_TILE_K = 8192
    NUM_SMS = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    tile_k = 1
    upper_bound = min(args["K"], MAX_TILE_K)
    while tile_k <= upper_bound:
        num_blocks = args["M"] * triton.cdiv(args["K"], tile_k)
        num_waves = num_blocks / NUM_SMS
        if (num_waves > 1) and (tile_k * 2 <= upper_bound):
            tile_k *= 2
        else:
            break
    return tile_k

def softmax_heur_tile_n_non_inner(args):
    return triton.cdiv(8192, args["TILE_K"])

def softmax_heur_one_tile_per_cta(args):
    return args["TILE_N"] >= args["N"]

def softmax_heur_num_warps_non_inner(args):
    tile_size = args["TILE_N"] * args["TILE_K"]
    if tile_size < 2048:
        return 4
    elif tile_size < 4096:
        return 8
    else:
        return 16

def softmax_heur_tile_n_inner(args):
    if args["N"] <= (32 * 1024):
        return triton.next_power_of_2(args["N"])
    else:
        return 4096

def softmax_heur_num_warps_inner(args):
    tile_size = args["TILE_N"]
    if tile_size < 2048:
        return 4
    elif tile_size < 4096:
        return 8
    else:
        return 16

def softmax_heur_tile_n_bwd_non_inner(args):
    return max(1, 1024 // args["TILE_K"])

def softmax_heru_tile_m(args):
    return max(1, 1024 // args["TILE_N"])

def uniform_heur_block(args):
    return 512 if args["N"] <= 512 else 1024

def uniform_heur_num_warps(args):
    if args["N"] <= 512:
        return 4
    elif args["N"] <= 1024:
        return 8
    else:
        return 16

def var_mean_heur_block_n(args):
    return triton.next_power_of_2(args["BLOCK_NUM"])

def upsample_nearest2d_SAME_H(args):
    return args["OH"] == args["IH"]

def upsample_nearest2d_SAME_W(args):
    return args["OW"] == args["IW"]

def batch_norm_heur_block_m(args):
    return min(2048, triton.next_power_of_2(args["batch_dim"]))

def batch_norm_heur_block_n(args):
    BLOCK_M = batch_norm_heur_block_m(args)
    BLOCK_N = triton.next_power_of_2(args["spatial_dim"])
    return min(BLOCK_N, max(1, 2**14 // BLOCK_M))

def vdot_heur_block_size(args):
    n = args["n_elements"]
    if n < 1024:
        return 32
    elif n < 8192:
        return 256
    else:
        return 1024
"""

def extract_used_functions(content):
    """提取文件中使用的 heuristics 函数"""
    used_functions = set()

    # 查找所有 heuristics 装饰器
    pattern = r'@triton\.heuristics\(\{([^}]+)\}\)'
    matches = re.finditer(pattern, content)

    for match in matches:
        config_str = match.group(1)
        # 提取函数名
        func_names = re.findall(r':\s*(\w+)', config_str)
        used_functions.update(func_names)

    return used_functions

def get_function_definition(func_name):
    """获取函数定义"""
    pattern = rf'^def {re.escape(func_name)}\(.*?\):\n(?:.*\n)*?(?=\ndef |\Z)'
    match = re.search(pattern, HEURISTICS_FUNCTIONS, re.MULTILINE)
    if match:
        return match.group(0).rstrip()
    return None

def process_file(file_path):
    """处理单个文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否有 @triton.heuristics
    if '@triton.heuristics' not in content:
        return False

    # 提取使用的函数
    used_functions = extract_used_functions(content)
    if not used_functions:
        return False

    # 获取所有需要的函数定义
    function_defs = []
    for func_name in sorted(used_functions):
        func_def = get_function_definition(func_name)
        if func_def:
            function_defs.append(func_def)

    if not function_defs:
        return False

    # 找到合适的位置插入函数（在第一个 @triton.heuristics 之前）
    insert_pos = content.find('@triton.heuristics')
    if insert_pos == -1:
        return False

    # 找到这一行的开始
    line_start = content.rfind('\n', 0, insert_pos) + 1

    # 构建插入的内容
    functions_block = '\n# Heuristics functions for NVIDIA\n' + '\n\n'.join(function_defs) + '\n\n'

    # 插入函数定义
    new_content = content[:line_start] + functions_block + content[line_start:]

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    return True

def main():
    print("=" * 60)
    print("修正 heuristics 函数引用")
    print("=" * 60)

    operators_dir = Path("/share/project/tj/workspace/FlagGems/operators_py")
    triton_files = list(operators_dir.glob("*/*_triton.py"))

    modified_count = 0

    for triton_file in sorted(triton_files):
        op_name = triton_file.parent.name

        try:
            if process_file(triton_file):
                print(f"✓ {op_name:30s} - 已添加 heuristics 函数")
                modified_count += 1
        except Exception as e:
            print(f"✗ {op_name:30s} - 错误: {e}")

    print("\n" + "=" * 60)
    print(f"处理完成: {modified_count} 个文件被修改")
    print("=" * 60)

if __name__ == '__main__':
    main()
