#!/usr/bin/env python3
"""
移除 Triton 代码中的 runtime 模块，用 NVIDIA 对应的实际函数替换
"""

import re
import json
from pathlib import Path
import yaml

# 读取 NVIDIA 的配置
NVIDIA_CONFIG_PATH = "/share/project/tj/workspace/FlagGems/src/flag_gems/runtime/backend/_nvidia"

def load_nvidia_configs():
    """加载 NVIDIA 的配置"""
    # 读取 tune_configs.yaml
    tune_config_file = Path(NVIDIA_CONFIG_PATH) / "tune_configs.yaml"
    with open(tune_config_file, 'r') as f:
        tune_configs = yaml.safe_load(f)

    # 读取 heuristics
    heuristics_file = Path(NVIDIA_CONFIG_PATH) / "heuristics_config_utils.py"
    with open(heuristics_file, 'r') as f:
        heuristics_content = f.read()

    return tune_configs, heuristics_content

def get_heuristic_config_dict(op_name, heuristics_content):
    """从 heuristics 内容中提取特定算子的配置"""
    # 查找 HEURISTICS_CONFIGS 字典中的对应配置
    pattern = rf'"{re.escape(op_name)}":\s*\{{([^}}]+)\}}'
    match = re.search(pattern, heuristics_content)

    if match:
        config_str = match.group(1)
        # 解析配置项
        items = re.findall(r'"([^"]+)":\s*([^,\n]+)', config_str)
        config = {}
        for key, value in items:
            config[key] = value.strip()
        return config
    return None

def format_tuned_config(config_data):
    """格式化 tuned config 为 Python 代码"""
    if not config_data:
        return "[]"

    if isinstance(config_data, list) and len(config_data) > 0:
        first_item = config_data[0]

        # 如果是自动生成配置
        if isinstance(first_item, dict) and first_item.get('gen'):
            param_map = first_item['param_map']
            meta_map = param_map.get('META', {})

            # 生成配置列表
            configs = []

            # 获取所有参数的可能值
            block_m_values = first_item.get('block_m', [64, 128])
            block_n_values = first_item.get('block_n', [32, 64, 128])
            pre_load_v_values = first_item.get('pre_load_v', [True, False])
            warps_values = first_item.get('warps', [4, 8])
            stages_values = first_item.get('stages', [1, 2, 3])

            # 生成一些代表性配置（不生成全部组合以避免过长）
            sample_configs = [
                (64, 32, True, 4, 1),
                (64, 64, False, 4, 2),
                (128, 64, True, 8, 2),
                (128, 128, False, 8, 3),
            ]

            for bm, bn, plv, w, s in sample_configs:
                meta_keys = list(meta_map.keys())
                meta_dict = {}
                if 'BLOCK_M' in meta_keys:
                    meta_dict['BLOCK_M'] = bm
                if 'BLOCK_N' in meta_keys:
                    meta_dict['BLOCK_N'] = bn
                if 'PRE_LOAD_V' in meta_keys:
                    meta_dict['PRE_LOAD_V'] = plv

                config_str = f"triton.Config({meta_dict}, num_warps={w}, num_stages={s})"
                configs.append(config_str)

            return "[" + ", ".join(configs) + "]"

        # 如果是直接配置列表
        else:
            configs = []
            for item in config_data[:4]:  # 最多取4个配置
                meta = item.get('META', {})
                num_warps = item.get('num_warps', 4)
                num_stages = item.get('num_stages', 2)
                num_ctas = item.get('num_ctas', 1)

                config_str = f"triton.Config({meta}, num_warps={num_warps}, num_stages={num_stages}, num_ctas={num_ctas})"
                configs.append(config_str)

            return "[" + ", ".join(configs) + "]"

    return "[]"

def format_heuristic_config(heuristic_dict):
    """格式化 heuristic config 为 Python 代码"""
    if not heuristic_dict:
        return "{}"

    items = []
    for key, func_name in heuristic_dict.items():
        items.append(f'"{key}": {func_name}')

    return "{" + ", ".join(items) + "}"

def process_triton_file(file_path, tune_configs, heuristics_content):
    """处理单个 Triton 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    modified = False

    # 1. 移除 runtime 相关的 import
    # 移除: from flag_gems.runtime import torch_device_fn
    if 'from flag_gems.runtime import torch_device_fn' in content:
        content = content.replace('from flag_gems.runtime import torch_device_fn', '')
        modified = True

    # 移除: from .. import runtime
    if 'from .. import runtime' in content:
        content = content.replace('from .. import runtime', '')
        modified = True

    # 移除: from ..runtime import torch_device_fn
    if 'from ..runtime import torch_device_fn' in content:
        content = content.replace('from ..runtime import torch_device_fn', '')
        modified = True

    # 2. 替换 torch_device_fn 为 torch.cuda
    if 'torch_device_fn' in content:
        content = re.sub(r'\btorch_device_fn\b', 'torch.cuda', content)
        modified = True

    # 3. 替换 runtime.get_tuned_config("xxx")
    tuned_config_pattern = r'runtime\.get_tuned_config\("([^"]+)"\)'
    tuned_matches = re.finditer(tuned_config_pattern, content)

    for match in tuned_matches:
        op_config_name = match.group(1)
        full_match = match.group(0)

        # 查找对应的配置
        if op_config_name in tune_configs:
            config_data = tune_configs[op_config_name]
            replacement = format_tuned_config(config_data)
            content = content.replace(full_match, replacement)
            modified = True
        else:
            # 没有找到配置，使用空列表
            content = content.replace(full_match, "[]")
            modified = True

    # 4. 替换 runtime.get_heuristic_config("xxx")
    heuristic_pattern = r'runtime\.get_heuristic_config\("([^"]+)"\)'
    heuristic_matches = re.finditer(heuristic_pattern, content)

    for match in heuristic_matches:
        op_config_name = match.group(1)
        full_match = match.group(0)

        # 查找对应的 heuristic 配置
        heuristic_dict = get_heuristic_config_dict(op_config_name, heuristics_content)
        replacement = format_heuristic_config(heuristic_dict)
        content = content.replace(full_match, replacement)
        modified = True

    # 5. 清理多余的空行
    content = re.sub(r'\n\n\n+', '\n\n', content)

    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True

    return False

def main():
    print("=" * 60)
    print("开始移除 runtime 模块")
    print("=" * 60)

    # 加载 NVIDIA 配置
    print("\n加载 NVIDIA 配置...")
    tune_configs, heuristics_content = load_nvidia_configs()
    print(f"✓ 加载了 {len(tune_configs)} 个 tuned configs")

    # 扫描所有 triton 文件
    operators_dir = Path("/share/project/tj/workspace/FlagGems/operators_py")
    triton_files = list(operators_dir.glob("*/*_triton.py"))

    print(f"\n找到 {len(triton_files)} 个 Triton 文件")
    print("\n开始处理...\n")

    modified_count = 0

    for triton_file in sorted(triton_files):
        op_name = triton_file.parent.name

        try:
            if process_triton_file(triton_file, tune_configs, heuristics_content):
                print(f"✓ {op_name:30s} - 已修改")
                modified_count += 1
            else:
                print(f"  {op_name:30s} - 无需修改")
        except Exception as e:
            print(f"✗ {op_name:30s} - 错误: {e}")

    print("\n" + "=" * 60)
    print(f"处理完成: {modified_count}/{len(triton_files)} 个文件被修改")
    print("=" * 60)

if __name__ == '__main__':
    main()
