#!/usr/bin/env python3
"""
Sort registrations in FlagGems codebase to maintain alphabetical order.

This script automatically sorts:
1. operators.yaml entries by operator id
2. ops/__init__.py __all__ list
3. __init__.py _FULL_CONFIG tuples

Usage:
    python sort_registrations.py [--check]

    --check: Only check if files are sorted, exit 1 if not (for CI)
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from ruamel.yaml import YAML


def sort_operators_yaml(yaml_path: Path, check_only: bool = False) -> bool:
    """Sort operators.yaml entries by id field using structured YAML parsing."""
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 4096  # prevent line wrapping

    data = yaml.load(yaml_path)

    if "ops" not in data or not data["ops"]:
        print(f"Warning: 'ops' section not found or empty in {yaml_path}")
        return True

    ops = data["ops"]
    ids = [op["id"] for op in ops]

    # Detect duplicates
    dupes = {k: v for k, v in Counter(ids).items() if v > 1}
    if dupes:
        print(f"❌ Error: Found {len(dupes)} duplicate operator ID(s) in {yaml_path}:")
        for dup_id, count in sorted(dupes.items()):
            print(f"    - '{dup_id}' appears {count} times")
        print("    Please fix duplicates manually before sorting.")
        return False

    sorted_ids = sorted(ids)
    is_sorted = ids == sorted_ids

    if check_only:
        if not is_sorted:
            print(f"❌ {yaml_path} is not sorted")
            for i, (u, s) in enumerate(zip(ids, sorted_ids)):
                if u != s:
                    print(f"  First mismatch at position {i}: '{u}' should be '{s}'")
                    break
        return is_sorted

    if is_sorted:
        print(f"✅ {yaml_path} already sorted")
        return True

    data["ops"].sort(key=lambda op: op["id"])

    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
    print(f"✅ Sorted {yaml_path}")
    return True


def sort_ops_init_all(init_path: Path, check_only: bool = False) -> bool:
    """Sort ops/__init__.py __all__ list."""
    content = init_path.read_text()

    # Find __all__ list
    all_match = re.search(r"(__all__\s*=\s*\[)(.*?)(\n\])", content, re.DOTALL)

    if not all_match:
        print(f"Warning: __all__ not found in {init_path}")
        return True

    prefix = all_match.group(1)
    all_body = all_match.group(2)
    suffix = all_match.group(3)

    # Parse entries
    entries = []
    for line in all_body.split("\n"):
        stripped = line.strip()
        if stripped and stripped.startswith('"'):
            match = re.match(r'"([^"]+)"', stripped)
            if match:
                entries.append(match.group(1))

    sorted_entries = sorted(entries)
    is_sorted = sorted_entries == entries

    if check_only:
        if not is_sorted:
            print(f"❌ {init_path} __all__ is not sorted")
            for i, (u, s) in enumerate(zip(entries, sorted_entries)):
                if u != s:
                    print(f"  First mismatch at position {i}: '{u}' should be '{s}'")
                    break
        return is_sorted

    if is_sorted:
        print(f"✅ {init_path} __all__ already sorted")
        return True

    # Reconstruct with proper formatting
    new_all_body = '\n    "' + '",\n    "'.join(sorted_entries) + '",'
    new_content = (
        content[: all_match.start()]
        + prefix
        + new_all_body
        + suffix
        + content[all_match.end() :]
    )

    init_path.write_text(new_content)
    print(f"✅ Sorted {init_path} __all__")
    return True


def sort_full_config(init_path: Path, check_only: bool = False) -> bool:
    """Sort _FULL_CONFIG in __init__.py."""
    content = init_path.read_text()

    # Find _FULL_CONFIG section (it's a tuple)
    config_match = re.search(r"(_FULL_CONFIG\s*=\s*\()(.*?)(\n\))", content, re.DOTALL)

    if not config_match:
        print(f"Warning: _FULL_CONFIG not found in {init_path}")
        return True

    prefix = config_match.group(1)
    config_body = config_match.group(2)
    suffix = config_match.group(3)

    # Parse entries (handles single-line tuples, multi-line tuples, and comments)
    entries = (
        []
    )  # (sort_key, original_text) — sort_key is aten_name or None for comments
    lines = config_body.split("\n")
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue

        if stripped.startswith("#"):
            # Comment line — extract aten_name if it looks like a commented-out tuple
            comment_match = re.search(r'\("([^"]+)"', stripped)
            sort_key = comment_match.group(1) if comment_match else None
            entries.append((sort_key, lines[i].rstrip()))
            i += 1
            continue

        if stripped.startswith("(") and stripped.endswith("),"):
            # Single-line tuple: ("aten_name", func_name),
            match = re.match(r'\("([^"]+)"', stripped)
            if match:
                entries.append((match.group(1), lines[i].rstrip()))
            i += 1
            continue

        if stripped.startswith("("):
            # Multi-line tuple — collect until closing '),'
            block_lines = [lines[i]]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("),"):
                block_lines.append(lines[i])
                i += 1
            if i < len(lines):
                block_lines.append(lines[i])  # the '),' line
                i += 1
            block_text = "\n".join(line.rstrip() for line in block_lines)
            match = re.search(r'"([^"]+)"', block_text)
            sort_key = match.group(1) if match else None
            entries.append((sort_key, block_text))
            continue

        i += 1

    # Fail if any entry could not be parsed — don't silently reorder them
    sortable = [(key, text) for key, text in entries if key is not None]
    unsortable = [(key, text) for key, text in entries if key is None]
    if unsortable:
        print(
            f"❌ Error: {len(unsortable)} entries in _FULL_CONFIG could not be parsed:"
        )
        for _, text in unsortable:
            first_line = text.split("\n")[0].strip()
            print(f"    - {first_line}")
        print("    Please fix them manually before sorting.")
        return False

    sorted_entries = sorted(sortable, key=lambda x: x[0])
    is_sorted = sorted_entries == sortable

    if check_only:
        if not is_sorted:
            print(f"❌ {init_path} _FULL_CONFIG is not sorted")
            for i, (u, s) in enumerate(zip(sortable, sorted_entries)):
                if u[0] != s[0]:
                    print(
                        f"  First mismatch at position {i}: '{u[0]}' should be '{s[0]}'"
                    )
                    break
        return is_sorted

    if is_sorted:
        print(f"✅ {init_path} _FULL_CONFIG already sorted")
        return True

    # Reconstruct — preserve original indentation and multi-line format
    new_lines = [text for _, text in sorted_entries]

    new_config_body = "\n" + "\n".join(new_lines)
    new_content = (
        content[: config_match.start()]
        + prefix
        + new_config_body
        + suffix
        + content[config_match.end() :]
    )

    init_path.write_text(new_content)
    print(f"✅ Sorted {init_path} _FULL_CONFIG")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Sort FlagGems registrations alphabetically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--check", action="store_true", help="Only check, do not modify"
    )
    args = parser.parse_args()

    # Find repo root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent  # tools/sort_registrations.py -> repo root

    files_to_check = [
        (repo_root / "conf" / "operators.yaml", sort_operators_yaml, "operators.yaml"),
        (
            repo_root / "src" / "flag_gems" / "ops" / "__init__.py",
            sort_ops_init_all,
            "ops/__init__.py __all__",
        ),
        (
            repo_root / "src" / "flag_gems" / "__init__.py",
            sort_full_config,
            "__init__.py _FULL_CONFIG",
        ),
    ]

    all_sorted = True
    for file_path, sort_func, name in files_to_check:
        if not file_path.exists():
            print(f"Warning: {file_path} not found")
            continue

        try:
            sorted_ok = sort_func(file_path, check_only=args.check)
            if not sorted_ok:
                all_sorted = False
                if not args.check:
                    # Stop on first error when actually modifying files
                    print("\n❌ Stopped due to errors. Fix issues and re-run.")
                    sys.exit(1)
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")
            import traceback

            traceback.print_exc()
            all_sorted = False
            sys.exit(1)

    if args.check:
        if all_sorted:
            print("\n✅ All files are properly sorted")
            sys.exit(0)
        else:
            print("\n❌ Some files are not sorted. Run without --check to fix.")
            sys.exit(1)
    else:
        print("\n✅ Done")


if __name__ == "__main__":
    main()
