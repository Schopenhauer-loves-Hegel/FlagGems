#!/usr/bin/env python3
"""
Validate data format for operator import

Usage:
    python validate_data.py --your-data your_perf.json
    python validate_data.py --your-data your_perf.json --flaggems-data flaggems_perf.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Set


class DataValidator:
    """Validate performance data format"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.errors = []
        self.warnings = []

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

    def validate_structure(self, data: Dict[str, Any], data_type: str) -> bool:
        """Validate basic structure"""
        self.log(f"Validating {data_type} structure...", "info")

        if not isinstance(data, dict):
            self.errors.append(f"{data_type}: Root should be a dictionary")
            return False

        if not data:
            self.warnings.append(f"{data_type}: Empty data")
            return True

        # Check first operator as example
        op_name = list(data.keys())[0]
        op_data = data[op_name]

        if 'configs' not in op_data:
            self.errors.append(f"{data_type}: Missing 'configs' field in operator '{op_name}'")
            return False

        if not isinstance(op_data['configs'], list):
            self.errors.append(f"{data_type}: 'configs' should be a list in operator '{op_name}'")
            return False

        if not op_data['configs']:
            self.warnings.append(f"{data_type}: Empty configs for operator '{op_name}'")
            return True

        # Check first config
        config = op_data['configs'][0]
        required_fields = ['shape', 'dtype']

        if data_type == "your_data":
            required_fields.extend(['your_time', 'cuda_time'])
        elif data_type == "flaggems_data":
            required_fields.extend(['flaggems_time', 'cuda_time'])

        for field in required_fields:
            if field not in config:
                self.errors.append(
                    f"{data_type}: Missing '{field}' in config for operator '{op_name}'"
                )
                return False

        self.log(f"{data_type} structure is valid", "success")
        return True

    def validate_your_data(self, data: Dict[str, Any]) -> bool:
        """Validate your performance data"""
        if not self.validate_structure(data, "your_data"):
            return False

        # Additional checks
        total_configs = 0
        for op_name, op_data in data.items():
            total_configs += len(op_data['configs'])

            for config in op_data['configs']:
                # Check types
                if not isinstance(config['shape'], list):
                    self.errors.append(f"Shape should be a list in {op_name}")
                    return False

                if not isinstance(config['your_time'], (int, float)):
                    self.errors.append(f"your_time should be a number in {op_name}")
                    return False

                if not isinstance(config['cuda_time'], (int, float)):
                    self.errors.append(f"cuda_time should be a number in {op_name}")
                    return False

                # Check values
                if config['your_time'] <= 0:
                    self.warnings.append(f"your_time <= 0 in {op_name}: {config}")

                if config['cuda_time'] <= 0:
                    self.warnings.append(f"cuda_time <= 0 in {op_name}: {config}")

        self.log(f"Total operators: {len(data)}", "info")
        self.log(f"Total configs: {total_configs}", "info")

        return True

    def validate_flaggems_data(self, data: Dict[str, Any]) -> bool:
        """Validate FlagGems performance data"""
        if not self.validate_structure(data, "flaggems_data"):
            return False

        total_configs = 0
        for op_name, op_data in data.items():
            total_configs += len(op_data['configs'])

            for config in op_data['configs']:
                if not isinstance(config['flaggems_time'], (int, float)):
                    self.errors.append(f"flaggems_time should be a number in {op_name}")
                    return False

                if config['flaggems_time'] <= 0:
                    self.warnings.append(f"flaggems_time <= 0 in {op_name}: {config}")

        self.log(f"Total operators: {len(data)}", "info")
        self.log(f"Total configs: {total_configs}", "info")

        return True

    def check_compatibility(
        self,
        your_data: Dict[str, Any],
        flaggems_data: Dict[str, Any]
    ):
        """Check compatibility between datasets"""
        self.log("\nChecking dataset compatibility...", "info")

        # Find common operators
        your_ops = set(your_data.keys())
        fg_ops = set(flaggems_data.keys())

        common_ops = your_ops & fg_ops
        only_yours = your_ops - fg_ops
        only_fg = fg_ops - your_ops

        self.log(f"Common operators: {len(common_ops)}", "info")
        if only_yours:
            self.log(f"Only in your data: {len(only_yours)}", "warning")
            if self.verbose and len(only_yours) <= 10:
                for op in list(only_yours)[:10]:
                    self.log(f"  - {op}", "warning")

        if only_fg:
            self.log(f"Only in FlagGems data: {len(only_fg)}", "warning")
            if self.verbose and len(only_fg) <= 10:
                for op in list(only_fg)[:10]:
                    self.log(f"  - {op}", "warning")

        # Check config compatibility for common operators
        total_matched = 0
        total_unmatched = 0

        for op_name in common_ops:
            your_configs = {
                (tuple(c['shape']), c['dtype'])
                for c in your_data[op_name]['configs']
            }
            fg_configs = {
                (tuple(c['shape']), c['dtype'])
                for c in flaggems_data[op_name]['configs']
            }

            matched = your_configs & fg_configs
            total_matched += len(matched)
            total_unmatched += len(your_configs - matched)

        self.log(f"\nConfig matching:", "info")
        self.log(f"  Matched configs: {total_matched}", "success")
        if total_unmatched > 0:
            self.log(f"  Unmatched configs: {total_unmatched}", "warning")
            self.warnings.append(
                f"{total_unmatched} configs in your data have no matching FlagGems baseline"
            )

        return len(common_ops) > 0 and total_matched > 0

    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*70)
        print("📊 Validation Summary")
        print("="*70)

        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"   {error}")

        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ All checks passed!")
        elif not self.errors:
            print(f"\n⚠️  Validation passed with {len(self.warnings)} warnings")
        else:
            print(f"\n❌ Validation failed with {len(self.errors)} errors")

        print("="*70 + "\n")

        return len(self.errors) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate data format for operator import"
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
        help="Path to FlagGems performance data JSON"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )

    args = parser.parse_args()

    validator = DataValidator(verbose=not args.quiet)

    # Load your data
    try:
        with open(args.your_data, 'r', encoding='utf-8') as f:
            your_data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load your data: {e}")
        return 1

    # Validate your data
    if not validator.validate_your_data(your_data):
        validator.print_summary()
        return 1

    # Load and validate FlagGems data if provided
    if args.flaggems_data:
        try:
            with open(args.flaggems_data, 'r', encoding='utf-8') as f:
                flaggems_data = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load FlagGems data: {e}")
            return 1

        if not validator.validate_flaggems_data(flaggems_data):
            validator.print_summary()
            return 1

        # Check compatibility
        if not validator.check_compatibility(your_data, flaggems_data):
            validator.warnings.append(
                "No compatible configs found between datasets"
            )

    # Print summary
    success = validator.print_summary()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
