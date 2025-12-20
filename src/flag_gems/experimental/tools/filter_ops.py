#!/usr/bin/env python3
"""
Filter and select operators for experimental import based on performance criteria

Usage:
    # Filter batch 1: Existing FlagGems ops with 30% speedup
    python filter_ops.py --batch 1 \
        --your-data <your_perf.json> \
        --flaggems-data <flaggems_perf.json> \
        --output selected_batch1.json

    # Filter batch 2: New ops with 80% CUDA performance
    python filter_ops.py --batch 2 \
        --your-data <your_perf.json> \
        --output selected_batch2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class OpPerformance:
    """Performance data for an operator"""
    op_name: str
    shape: tuple
    dtype: str
    your_time: float  # ms
    cuda_time: float  # ms
    flaggems_time: Optional[float] = None  # ms, only for batch 1
    speedup_vs_flaggems: Optional[float] = None
    relative_to_cuda: Optional[float] = None

    def __post_init__(self):
        """Calculate derived metrics"""
        # Relative to CUDA (lower is better)
        self.relative_to_cuda = self.your_time / self.cuda_time

        # Speedup vs FlagGems (higher is better)
        if self.flaggems_time:
            self.speedup_vs_flaggems = self.flaggems_time / self.your_time


class OperatorFilter:
    """Filter operators based on performance criteria"""

    def __init__(self, batch: int, verbose: bool = True):
        """
        Initialize filter

        Args:
            batch: 1 for existing ops, 2 for new ops
            verbose: Print progress information
        """
        self.batch = batch
        self.verbose = verbose

        # Criteria
        if batch == 1:
            self.threshold = 1.30  # 30% speedup vs FlagGems
            self.criterion = "speedup_vs_flaggems"
        elif batch == 2:
            self.threshold = 1.25  # 80% of CUDA = 1/0.8 = 1.25
            self.criterion = "relative_to_cuda"
        else:
            raise ValueError(f"Invalid batch number: {batch}")

    def load_your_data(self, file_path: Path) -> Dict[str, Any]:
        """Load your performance data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if self.verbose:
            print(f"✓ Loaded your data: {len(data)} operators")

        return data

    def load_flaggems_data(self, file_path: Path) -> Dict[str, Any]:
        """Load FlagGems performance data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if self.verbose:
            print(f"✓ Loaded FlagGems data: {len(data)} operators")

        return data

    def parse_performance_data(
        self,
        your_data: Dict[str, Any],
        flaggems_data: Optional[Dict[str, Any]] = None
    ) -> List[OpPerformance]:
        """
        Parse and combine performance data

        Args:
            your_data: Your performance results
            flaggems_data: FlagGems performance results (for batch 1)

        Returns:
            List of OpPerformance objects
        """
        performances = []

        # TODO: This depends on your data format
        # You'll need to adjust this based on actual JSON structure

        # Example structure (adjust as needed):
        # {
        #   "op_name": {
        #     "configs": [
        #       {
        #         "shape": [256, 256],
        #         "dtype": "float32",
        #         "your_time": 0.5,
        #         "cuda_time": 1.0
        #       }
        #     ]
        #   }
        # }

        for op_name, op_data in your_data.items():
            configs = op_data.get('configs', [])

            for config in configs:
                shape = tuple(config['shape'])
                dtype = config['dtype']
                your_time = config['your_time']
                cuda_time = config['cuda_time']

                # Get FlagGems time if available
                flaggems_time = None
                if flaggems_data and op_name in flaggems_data:
                    # Find matching config in FlagGems data
                    for fg_config in flaggems_data[op_name].get('configs', []):
                        if (tuple(fg_config['shape']) == shape and
                            fg_config['dtype'] == dtype):
                            flaggems_time = fg_config['flaggems_time']
                            break

                perf = OpPerformance(
                    op_name=op_name,
                    shape=shape,
                    dtype=dtype,
                    your_time=your_time,
                    cuda_time=cuda_time,
                    flaggems_time=flaggems_time
                )

                performances.append(perf)

        return performances

    def filter_operators(
        self,
        performances: List[OpPerformance]
    ) -> List[OpPerformance]:
        """
        Filter operators based on criteria

        Args:
            performances: List of operator performances

        Returns:
            Filtered list meeting criteria
        """
        selected = []

        for perf in performances:
            if self.batch == 1:
                # Need speedup vs FlagGems >= 1.30
                if perf.speedup_vs_flaggems and perf.speedup_vs_flaggems >= self.threshold:
                    selected.append(perf)
            elif self.batch == 2:
                # Need relative to CUDA <= 1.25 (i.e., >= 80% of CUDA)
                if perf.relative_to_cuda and perf.relative_to_cuda <= self.threshold:
                    selected.append(perf)

        if self.verbose:
            total = len(performances)
            selected_count = len(selected)
            print(f"\n📊 Filtering Results:")
            print(f"   Total configs: {total}")
            print(f"   Selected: {selected_count} ({selected_count/total*100:.1f}%)")

            if self.batch == 1:
                print(f"   Criterion: speedup vs FlagGems >= {self.threshold:.2f}x")
            else:
                print(f"   Criterion: relative to CUDA <= {self.threshold:.2f} (≥80%)")

        return selected

    def generate_summary_report(
        self,
        selected: List[OpPerformance],
        output_path: Path
    ):
        """Generate summary report"""
        # Group by operator name
        ops_summary = {}
        for perf in selected:
            if perf.op_name not in ops_summary:
                ops_summary[perf.op_name] = {
                    'configs': [],
                    'avg_speedup_vs_flaggems': [],
                    'avg_relative_to_cuda': []
                }

            ops_summary[perf.op_name]['configs'].append({
                'shape': list(perf.shape),
                'dtype': perf.dtype,
                'your_time': perf.your_time,
                'cuda_time': perf.cuda_time,
                'flaggems_time': perf.flaggems_time,
                'speedup_vs_flaggems': perf.speedup_vs_flaggems,
                'relative_to_cuda': perf.relative_to_cuda
            })

            if perf.speedup_vs_flaggems:
                ops_summary[perf.op_name]['avg_speedup_vs_flaggems'].append(
                    perf.speedup_vs_flaggems
                )
            if perf.relative_to_cuda:
                ops_summary[perf.op_name]['avg_relative_to_cuda'].append(
                    perf.relative_to_cuda
                )

        # Calculate averages
        for op_name, data in ops_summary.items():
            if data['avg_speedup_vs_flaggems']:
                avg = sum(data['avg_speedup_vs_flaggems']) / len(data['avg_speedup_vs_flaggems'])
                data['avg_speedup_vs_flaggems'] = round(avg, 3)
            else:
                data['avg_speedup_vs_flaggems'] = None

            if data['avg_relative_to_cuda']:
                avg = sum(data['avg_relative_to_cuda']) / len(data['avg_relative_to_cuda'])
                data['avg_relative_to_cuda'] = round(avg, 3)
            else:
                data['avg_relative_to_cuda'] = None

        # Write report
        report = {
            'batch': self.batch,
            'threshold': self.threshold,
            'criterion': self.criterion,
            'total_operators': len(ops_summary),
            'total_configs': len(selected),
            'operators': ops_summary
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"\n✅ Summary report saved to: {output_path}")
            print(f"   Total operators selected: {len(ops_summary)}")

            # Print top performers
            if self.batch == 1:
                sorted_ops = sorted(
                    ops_summary.items(),
                    key=lambda x: x[1]['avg_speedup_vs_flaggems'] or 0,
                    reverse=True
                )
                print(f"\n🏆 Top 5 performers (speedup vs FlagGems):")
                for op_name, data in sorted_ops[:5]:
                    speedup = data['avg_speedup_vs_flaggems']
                    print(f"   {op_name}: {speedup:.2f}x")
            else:
                sorted_ops = sorted(
                    ops_summary.items(),
                    key=lambda x: x[1]['avg_relative_to_cuda'] or float('inf')
                )
                print(f"\n🏆 Top 5 performers (closest to CUDA):")
                for op_name, data in sorted_ops[:5]:
                    relative = data['avg_relative_to_cuda']
                    percentage = (1 / relative) * 100
                    print(f"   {op_name}: {percentage:.1f}% of CUDA")


def main():
    parser = argparse.ArgumentParser(
        description="Filter operators for experimental import"
    )
    parser.add_argument(
        "--batch",
        type=int,
        choices=[1, 2],
        required=True,
        help="Batch number: 1=existing ops, 2=new ops"
    )
    parser.add_argument(
        "--your-data",
        type=Path,
        required=True,
        help="Path to your performance data JSON"
    )
    parser.add_argument(
        "--flaggems-data",
        type=Path,
        help="Path to FlagGems performance data JSON (required for batch 1)"
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

    # Validate inputs
    if args.batch == 1 and not args.flaggems_data:
        parser.error("--flaggems-data is required for batch 1")

    # Initialize filter
    filter_obj = OperatorFilter(batch=args.batch, verbose=not args.quiet)

    # Load data
    your_data = filter_obj.load_your_data(args.your_data)
    flaggems_data = None
    if args.flaggems_data:
        flaggems_data = filter_obj.load_flaggems_data(args.flaggems_data)

    # Parse performance data
    performances = filter_obj.parse_performance_data(your_data, flaggems_data)

    # Filter operators
    selected = filter_obj.filter_operators(performances)

    # Generate summary report
    filter_obj.generate_summary_report(selected, args.output)


if __name__ == "__main__":
    main()
