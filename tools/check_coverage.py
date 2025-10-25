#!/usr/bin/env python3
"""
NPP Coverage Analysis Tool

Analyzes implementation and test coverage against API function list.
"""

import os
import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple


class CoverageAnalyzer:
    """Analyzes implementation and test coverage"""

    def __init__(self, api_csv: Path, src_dir: Path, test_dir: Path):
        self.api_csv = api_csv
        self.src_dir = src_dir
        self.test_dir = test_dir

        self.api_functions = {}  # {func_name: {module, return_type, ...}}
        self.implementations = {}  # {func_name: file_path}
        self.tests = {}  # {func_name: file_path}

    def load_api_functions(self):
        """Load API functions from CSV"""
        with open(self.api_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                func_name = row['function_name']
                self.api_functions[func_name] = {
                    'module': row['module'],
                    'return_type': row['return_type'],
                    'params': row['params']
                }

    def scan_implementations(self):
        """Scan source code for function implementations"""
        # Scan .cu files
        for cu_file in self.src_dir.rglob('*.cu'):
            self._scan_impl_file(cu_file)

        # Scan .cpp files
        for cpp_file in self.src_dir.rglob('*.cpp'):
            self._scan_impl_file(cpp_file)

    def _scan_impl_file(self, file_path: Path):
        """Scan implementation file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Pattern 1: nppiFunc_impl or nppiFunc_Ctx_impl
            impl_pattern = r'NppStatus\s+(nppi\w+|npps\w+)(?:_Ctx)?_impl\s*\('

            # Pattern 2: Direct wrapper: NppStatus nppiFunc(...)
            wrapper_pattern = r'NppStatus\s+(nppi\w+|npps\w+)\s*\('

            for pattern in [impl_pattern, wrapper_pattern]:
                for match in re.finditer(pattern, content):
                    func_base = match.group(1)

                    # Try to find matching API function
                    if func_base in self.api_functions:
                        self.implementations[func_base] = str(file_path.relative_to(self.src_dir.parent))
                    elif f"{func_base}_Ctx" in self.api_functions:
                        self.implementations[f"{func_base}_Ctx"] = str(file_path.relative_to(self.src_dir.parent))

        except Exception as e:
            print(f"Warning: Error scanning {file_path}: {e}")

    def scan_tests(self):
        """Scan test files for function calls"""
        for test_file in self.test_dir.rglob('test_*.cpp'):
            self._scan_test_file(test_file)

    def _scan_test_file(self, file_path: Path):
        """Scan test file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Look for direct API calls
            pattern = r'\b(nppi\w+|npps\w+)\s*\('

            for match in re.finditer(pattern, content):
                func_name = match.group(1)
                if func_name in self.api_functions:
                    self.tests[func_name] = str(file_path.relative_to(self.test_dir.parent))

        except Exception as e:
            print(f"Warning: Error scanning {file_path}: {e}")

    def generate_coverage_csv(self, output_path: Path):
        """Generate coverage CSV"""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'function_name',
                'module',
                'implemented',
                'impl_file',
                'tested',
                'test_file'
            ])

            # Sort by module then function name
            functions = sorted(self.api_functions.items(),
                             key=lambda x: (x[1]['module'], x[0]))

            for func_name, func_info in functions:
                implemented = 'yes' if func_name in self.implementations else 'no'
                impl_file = self.implementations.get(func_name, '')

                tested = 'yes' if func_name in self.tests else 'no'
                test_file = self.tests.get(func_name, '')

                writer.writerow([
                    func_name,
                    func_info['module'],
                    implemented,
                    impl_file,
                    tested,
                    test_file
                ])

    def generate_summary(self, output_path: Path):
        """Generate summary report"""
        total = len(self.api_functions)
        implemented = len(self.implementations)
        tested = len(self.tests)

        impl_pct = (implemented / total * 100) if total > 0 else 0
        test_pct = (tested / total * 100) if total > 0 else 0

        lines = ["# Coverage Summary\n"]
        lines.append(f"- Total API Functions: {total}")
        lines.append(f"- Implemented: {implemented} ({impl_pct:.1f}%)")
        lines.append(f"- Tested: {tested} ({test_pct:.1f}%)")
        lines.append(f"- Implemented but not tested: {implemented - len(set(self.implementations.keys()) & set(self.tests.keys()))}")
        lines.append("")

        # By module
        by_module = defaultdict(lambda: {'total': 0, 'impl': 0, 'test': 0})

        for func_name, func_info in self.api_functions.items():
            module = func_info['module']
            by_module[module]['total'] += 1

            if func_name in self.implementations:
                by_module[module]['impl'] += 1

            if func_name in self.tests:
                by_module[module]['test'] += 1

        lines.append("## By Module\n")

        # Group by nppi/npps
        nppi = {k: v for k, v in by_module.items() if k.startswith('nppi')}
        npps = {k: v for k, v in by_module.items() if k.startswith('npps')}

        if nppi:
            lines.append("### NPPI\n")
            for module in sorted(nppi.keys()):
                stats = nppi[module]
                impl_pct = (stats['impl'] / stats['total'] * 100) if stats['total'] > 0 else 0
                test_pct = (stats['test'] / stats['total'] * 100) if stats['total'] > 0 else 0
                lines.append(f"**{module}**")
                lines.append(f"- Total: {stats['total']}, Impl: {stats['impl']} ({impl_pct:.1f}%), Test: {stats['test']} ({test_pct:.1f}%)")
                lines.append("")

        if npps:
            lines.append("### NPPS\n")
            for module in sorted(npps.keys()):
                stats = npps[module]
                impl_pct = (stats['impl'] / stats['total'] * 100) if stats['total'] > 0 else 0
                test_pct = (stats['test'] / stats['total'] * 100) if stats['total'] > 0 else 0
                lines.append(f"**{module}**")
                lines.append(f"- Total: {stats['total']}, Impl: {stats['impl']} ({impl_pct:.1f}%), Test: {stats['test']} ({test_pct:.1f}%)")
                lines.append("")

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='Analyze NPP implementation coverage')
    parser.add_argument('--api-csv', type=Path,
                        default=Path('api_analysis/api_functions.csv'),
                        help='Path to API functions CSV')
    parser.add_argument('--src-dir', type=Path, default=Path('src'),
                        help='Source code directory')
    parser.add_argument('--test-dir', type=Path, default=Path('test'),
                        help='Test directory')
    parser.add_argument('--output', type=Path, default=Path('api_analysis/coverage.csv'),
                        help='Output coverage CSV path')
    parser.add_argument('--summary', type=Path, default=Path('api_analysis/coverage_summary.md'),
                        help='Output summary report path')

    args = parser.parse_args()

    if not args.api_csv.exists():
        print(f"Error: API functions CSV not found at {args.api_csv}")
        print("Run analyze_api.py first to generate API function list")
        return 1

    print(f"Loading API functions from {args.api_csv}...")
    analyzer = CoverageAnalyzer(args.api_csv, args.src_dir, args.test_dir)
    analyzer.load_api_functions()
    print(f"Loaded {len(analyzer.api_functions)} API functions")

    print("Scanning implementations...")
    analyzer.scan_implementations()
    print(f"Found {len(analyzer.implementations)} implemented functions")

    print("Scanning tests...")
    analyzer.scan_tests()
    print(f"Found {len(analyzer.tests)} tested functions")

    print(f"Generating coverage CSV to {args.output}...")
    analyzer.generate_coverage_csv(args.output)

    print(f"Generating summary to {args.summary}...")
    analyzer.generate_summary(args.summary)

    print("Done!")
    return 0


if __name__ == '__main__':
    exit(main())
