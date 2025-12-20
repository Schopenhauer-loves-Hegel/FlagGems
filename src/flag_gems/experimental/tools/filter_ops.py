#!/usr/bin/env python3
"""
Filter and select operators for experimental import based on performance criteria

Usage:
    # Filter batch 1: Existing FlagGems ops with configurable speedup threshold
    python filter_ops.py --batch 1 \
        --gpt-data-dir eval_perf_gpt5_pass_10_20251117-114806 \
        --flaggems-excel vendor-test-1106.xlsx \
        --threshold 1.2 \
        --output selected_batch1.json

    # Filter batch 2: New ops with 80% CUDA performance
    python filter_ops.py --batch 2 \
        --gpt-data-dir <gpt_data_dir> \
        --threshold 0.8 \
        --output selected_batch2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class OperatorFilter:
    """Filter operators based on performance criteria"""

    def __init__(
        self,
        batch: int,
        gpt_data_dir: Path,
        flaggems_excel: Optional[Path] = None,
        threshold: Optional[float] = None,
        verbose: bool = True
    ):
        """
        Initialize filter

        Args:
            batch: 1 for existing ops, 2 for new ops
            gpt_data_dir: Path to GPT data directory
            flaggems_excel: Path to FlagGems Excel file (required for batch 1)
            threshold: Custom threshold (if None, use defaults)
            verbose: Print progress information
        """
        self.batch = batch
        self.gpt_data_dir = Path(gpt_data_dir)
        self.flaggems_excel = Path(flaggems_excel) if flaggems_excel else None
        self.verbose = verbose

        # Set threshold and criterion
        if batch == 1:
            # Batch 1: GPT speedup / FlagGems speedup >= threshold
            self.threshold = threshold if threshold is not None else 1.30
            self.criterion = "speedup_vs_flaggems"
            if not self.flaggems_excel:
                raise ValueError("--flaggems-excel is required for batch 1")
        elif batch == 2:
            # Batch 2: GPT speedup / CUDA >= threshold
            self.threshold = threshold if threshold is not None else 0.80
            self.criterion = "speedup_vs_cuda"
        else:
            raise ValueError(f"Invalid batch number: {batch}")

    def log(self, message: str, level: str = "info"):
        """Log message"""
        if not self.verbose:
            return

        prefix = {
            "info": "ℹ️ ",
            "success": "✅",
            "warning": "⚠️ ",
            "error": "❌"
        }.get(level, "  ")

        print(f"{prefix} {message}")

    def load_gpt_data(self) -> Dict[str, Dict]:
        """
        从 GPT 数据目录加载算子信息

        返回:
            {
                "op_name": {
                    "speedup_vs_cuda": float,
                    "code": str,
                    "success": bool
                }
            }
        """
        self.log("Loading GPT data...", "info")

        # 1. 读取 speedup_summary.json
        summary_file = self.gpt_data_dir / 'speedup_summary.json'
        if not summary_file.exists():
            raise FileNotFoundError(f"speedup_summary.json not found in {self.gpt_data_dir}")

        with open(summary_file, 'r') as f:
            summary = json.load(f)

        # 2. 遍历 log_X/result.json 获取代码
        result = {}
        for log_dir in sorted(self.gpt_data_dir.glob('log_*')):
            result_file = log_dir / 'result.json'
            if not result_file.exists():
                continue

            with open(result_file, 'r') as f:
                data = json.load(f)

            for entry in data:
                if not entry.get('success'):
                    continue

                op_name = entry['op_name']
                if op_name in summary['successful_operators']:
                    result[op_name] = {
                        'speedup_vs_cuda': summary['successful_operators'][op_name],
                        'code': entry.get('code', ''),
                        'success': True
                    }

        self.log(f"Loaded {len(result)} successful operators", "success")
        return result

    def load_flaggems_excel(self) -> Dict[str, float]:
        """
        从 Excel 加载 FlagGems 性能数据

        返回:
            {
                "op_name": average_speedup_vs_cuda
            }
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required. Install with: pip install pandas openpyxl")

        self.log("Loading FlagGems data from Excel...", "info")

        df = pd.read_excel(self.flaggems_excel, sheet_name='Speedup')

        # 过滤掉 NaN 行
        df = df[df['op_name'].notna()]

        result = {}
        for _, row in df.iterrows():
            op_name = row['op_name']
            avg_speedup = row['AVERAGE']

            # 只保留有效数据
            if pd.notna(avg_speedup) and avg_speedup > 0:
                result[op_name] = avg_speedup

        self.log(f"Loaded {len(result)} operators from FlagGems", "success")
        return result

    def process_batch1(self) -> Dict[str, Any]:
        """
        处理 Batch 1: 比较 GPT vs FlagGems

        筛选条件: gpt_speedup / flaggems_speedup >= threshold

        说明:
        - speedup 值越大越好
        - gpt_speedup / flaggems_speedup > 1 表示 GPT 比 FlagGems 快
        """
        # 1. 加载数据
        gpt_data = self.load_gpt_data()
        flaggems_data = self.load_flaggems_excel()

        # 2. 匹配和计算
        self.log("Calculating relative speedup...", "info")
        operators = {}

        for op_name in gpt_data.keys():
            if op_name not in flaggems_data:
                self.log(f"⚠️  {op_name} not found in FlagGems, skipping", "warning")
                continue

            gpt_speedup = gpt_data[op_name]['speedup_vs_cuda']
            fg_speedup = flaggems_data[op_name]

            if fg_speedup == 0:
                self.log(f"⚠️  {op_name} FlagGems speedup is 0, skipping", "warning")
                continue

            # 计算相对加速比: GPT / FlagGems
            # 当 > 1 时表示 GPT 比 FlagGems 快
            relative_speedup = gpt_speedup / fg_speedup

            operators[op_name] = {
                'gpt_speedup_vs_cuda': gpt_speedup,
                'flaggems_speedup_vs_cuda': fg_speedup,
                'speedup_vs_flaggems': relative_speedup,
                'code': gpt_data[op_name]['code'],
                'has_code': bool(gpt_data[op_name]['code'])
            }

        # 3. 筛选
        self.log(f"Filtering operators with relative speedup >= {self.threshold}...", "info")
        selected = {
            op_name: data
            for op_name, data in operators.items()
            if data['speedup_vs_flaggems'] >= self.threshold
        }

        # 按加速比排序
        selected = dict(sorted(
            selected.items(),
            key=lambda x: x[1]['speedup_vs_flaggems'],
            reverse=True
        ))

        self.log(f"Selected {len(selected)} operators (out of {len(operators)} total)", "success")

        # 4. 生成输出
        output = {
            'batch': self.batch,
            'threshold': self.threshold,
            'criterion': self.criterion,
            'total_operators': len(operators),
            'selected_operators': len(selected),
            'operators': selected
        }

        return output

    def process_batch2(self) -> Dict[str, Any]:
        """
        处理 Batch 2: GPT vs CUDA

        筛选条件: gpt_speedup >= threshold
        """
        # 1. 加载数据
        gpt_data = self.load_gpt_data()

        # 2. 筛选
        self.log(f"Filtering operators with speedup >= {self.threshold}...", "info")
        selected = {
            op_name: {
                'gpt_speedup_vs_cuda': data['speedup_vs_cuda'],
                'code': data['code'],
                'has_code': bool(data['code'])
            }
            for op_name, data in gpt_data.items()
            if data['speedup_vs_cuda'] >= self.threshold
        }

        # 按加速比排序
        selected = dict(sorted(
            selected.items(),
            key=lambda x: x[1]['gpt_speedup_vs_cuda'],
            reverse=True
        ))

        self.log(f"Selected {len(selected)} operators (out of {len(gpt_data)} total)", "success")

        # 3. 生成输出
        output = {
            'batch': self.batch,
            'threshold': self.threshold,
            'criterion': self.criterion,
            'total_operators': len(gpt_data),
            'selected_operators': len(selected),
            'operators': selected
        }

        return output

    def run(self, output_path: Path):
        """运行筛选并保存结果"""
        self.log("="*70, "info")
        self.log(f"Operator Filter - Batch {self.batch}", "info")
        self.log(f"Threshold: {self.threshold}", "info")
        self.log("="*70, "info")

        # 处理
        if self.batch == 1:
            result = self.process_batch1()
        else:
            result = self.process_batch2()

        # 保存结果
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        self.log(f"Results saved to: {output_path}", "success")

        # 打印摘要
        self.print_summary(result)

    def print_summary(self, result: Dict[str, Any]):
        """打印筛选摘要"""
        print("\n" + "="*70)
        print("📊 Filter Summary")
        print("="*70)
        print(f"Batch:              {result['batch']}")
        print(f"Threshold:          {result['threshold']}")
        print(f"Criterion:          {result['criterion']}")
        print(f"Total operators:    {result['total_operators']}")
        print(f"Selected:           {result['selected_operators']}")
        print(f"Selection rate:     {result['selected_operators']/result['total_operators']*100:.1f}%")

        if result['selected_operators'] > 0:
            print(f"\n🏆 Top 10 operators:")
            operators = list(result['operators'].items())[:10]
            for i, (op_name, data) in enumerate(operators, 1):
                if self.batch == 1:
                    speedup = data['speedup_vs_flaggems']
                    print(f"  {i:2d}. {op_name:<30s} {speedup:>6.4f}x")
                else:
                    speedup = data['gpt_speedup_vs_cuda']
                    print(f"  {i:2d}. {op_name:<30s} {speedup:>6.4f}")

        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Filter operators for experimental import",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--batch",
        type=int,
        choices=[1, 2],
        required=True,
        help="Batch number: 1=existing ops, 2=new ops"
    )
    parser.add_argument(
        "--gpt-data-dir",
        type=Path,
        required=True,
        help="Path to GPT data directory (e.g., eval_perf_gpt5_pass_10_20251117-114806)"
    )
    parser.add_argument(
        "--flaggems-excel",
        type=Path,
        help="Path to FlagGems Excel file (required for batch 1, e.g., vendor-test-1106.xlsx)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Custom threshold (batch 1 default: 1.30, batch 2 default: 0.80)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for filtered results"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    try:
        filter_obj = OperatorFilter(
            batch=args.batch,
            gpt_data_dir=args.gpt_data_dir,
            flaggems_excel=args.flaggems_excel,
            threshold=args.threshold,
            verbose=not args.quiet
        )
        filter_obj.run(args.output)

    except Exception as e:
        print(f"❌ Error: {e}", file=__import__('sys').stderr)
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
