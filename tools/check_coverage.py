#!/usr/bin/env python3
"""
NPP Coverage Analysis Tool

Analyzes implementation and test coverage against API inventory.
"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple


class CoverageAnalyzer:
    """Analyzes implementation and test coverage"""

    def __init__(self, api_inventory_path: Path, src_dir: Path, test_dir: Path):
        self.api_inventory_path = api_inventory_path
        self.src_dir = src_dir
        self.test_dir = test_dir

        self.api_functions = {}  # All API functions
        self.implemented_functions = set()  # Functions with implementation
        self.tested_functions = set()  # Functions with tests

    def load_api_inventory(self, version: str = '12.8'):
        """Load API function inventory"""
        with open(self.api_inventory_path, 'r') as f:
            data = json.load(f)

        if version not in data:
            available = [k for k in data.keys() if k != 'comparisons']
            raise ValueError(f"Version {version} not found. Available: {available}")

        # Extract function names from inventory
        for key, func_info in data[version]['inventory'].items():
            func_name = func_info['name']
            self.api_functions[func_name] = func_info

    def scan_implementations(self):
        """Scan source code for function implementations"""

        # Scan .cu files for extern "C" implementations
        for cu_file in self.src_dir.rglob('*.cu'):
            self._scan_cu_file(cu_file)

        # Scan .cpp files for wrapper implementations
        for cpp_file in self.src_dir.rglob('*.cpp'):
            self._scan_cpp_file(cpp_file)

    def _scan_cu_file(self, cu_path: Path):
        """Scan CUDA file for implementations"""
        try:
            with open(cu_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Find extern "C" blocks and functions inside
            # Pattern: NppStatus func_name_Ctx_impl(...) or func_name_impl(...)
            pattern = r'NppStatus\s+(nppi\w+|npps\w+)(?:_Ctx)?_impl\s*\('

            for match in re.finditer(pattern, content):
                func_base = match.group(1)
                # Remove _impl suffix to get API function name
                api_name = func_base

                # Try to find matching API function
                if api_name in self.api_functions:
                    self.implemented_functions.add(api_name)
                # Also check with _Ctx suffix
                elif f"{api_name}_Ctx" in self.api_functions:
                    self.implemented_functions.add(f"{api_name}_Ctx")

        except Exception as e:
            print(f"Warning: Error scanning {cu_path}: {e}")

    def _scan_cpp_file(self, cpp_path: Path):
        """Scan C++ wrapper file for implementations"""
        try:
            with open(cpp_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Pattern: NppStatus nppiFunc(...) { or NppStatus nppiFunc_Ctx(...)
            pattern = r'NppStatus\s+(nppi\w+|npps\w+)\s*\('

            for match in re.finditer(pattern, content):
                func_name = match.group(1)
                if func_name in self.api_functions:
                    self.implemented_functions.add(func_name)

        except Exception as e:
            print(f"Warning: Error scanning {cpp_path}: {e}")

    def scan_tests(self):
        """Scan test files for test coverage"""

        for test_file in self.test_dir.rglob('test_*.cpp'):
            self._scan_test_file(test_file)

    def _scan_test_file(self, test_path: Path):
        """Scan test file for function calls"""
        try:
            with open(test_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Look for direct API calls: nppiFunc(...) or nppiFunc_Ctx(...)
            pattern = r'\b(nppi\w+|npps\w+)\s*\('

            for match in re.finditer(pattern, content):
                func_name = match.group(1)
                if func_name in self.api_functions:
                    self.tested_functions.add(func_name)

        except Exception as e:
            print(f"Warning: Error scanning {test_path}: {e}")

    def generate_report(self, output_path: Path, version: str = '12.8'):
        """Generate coverage report"""

        lines = ["# NPP Implementation Coverage Report\n"]

        total = len(self.api_functions)
        implemented = len(self.implemented_functions)
        tested = len(self.tested_functions)

        impl_pct = (implemented / total * 100) if total > 0 else 0
        test_pct = (tested / total * 100) if total > 0 else 0

        lines.append(f"**API Version:** {version}\n")
        lines.append("## Summary\n")
        lines.append(f"- Total API Functions: {total}")
        lines.append(f"- Implemented: {implemented} ({impl_pct:.1f}%)")
        lines.append(f"- Tested: {tested} ({test_pct:.1f}%)")
        lines.append("")

        # Group by module
        by_module = self._group_by_module()

        lines.append("## Coverage by Module\n")

        # Group modules by nppi/npps
        nppi_modules = {k: v for k, v in by_module.items() if k.startswith('nppi')}
        npps_modules = {k: v for k, v in by_module.items() if k.startswith('npps')}

        if nppi_modules:
            lines.append("### NPPI Modules\n")
            for module in sorted(nppi_modules.keys()):
                stats = nppi_modules[module]
                lines.append(f"**{module}**")
                lines.append(f"- Total: {stats['total']}")
                lines.append(f"- Implemented: {stats['implemented']} ({stats['impl_pct']:.1f}%)")
                lines.append(f"- Tested: {stats['tested']} ({stats['test_pct']:.1f}%)")
                lines.append("")

        if npps_modules:
            lines.append("### NPPS Modules\n")
            for module in sorted(npps_modules.keys()):
                stats = npps_modules[module]
                lines.append(f"**{module}**")
                lines.append(f"- Total: {stats['total']}")
                lines.append(f"- Implemented: {stats['implemented']} ({stats['impl_pct']:.1f}%)")
                lines.append(f"- Tested: {stats['tested']} ({stats['test_pct']:.1f}%)")
                lines.append("")

        # Implemented but not tested
        impl_not_tested = self.implemented_functions - self.tested_functions
        if impl_not_tested:
            lines.append("## Implemented but Not Tested\n")
            lines.append(f"Total: {len(impl_not_tested)}\n")
            for func in sorted(list(impl_not_tested)[:20]):
                lines.append(f"- {func}")
            if len(impl_not_tested) > 20:
                lines.append(f"- ... and {len(impl_not_tested) - 20} more")
            lines.append("")

        # Not implemented (sample)
        not_implemented = set(self.api_functions.keys()) - self.implemented_functions
        if not_implemented:
            lines.append("## Not Implemented (Sample)\n")
            lines.append(f"Total: {len(not_implemented)}\n")

            # Group by module for better organization
            not_impl_by_module = defaultdict(list)
            for func in not_implemented:
                module = self.api_functions[func]['file']
                not_impl_by_module[module].append(func)

            # Show samples from each module
            for module in sorted(not_impl_by_module.keys())[:5]:
                lines.append(f"\n**{module}** ({len(not_impl_by_module[module])} functions):")
                for func in sorted(not_impl_by_module[module][:5]):
                    lines.append(f"- {func}")
                if len(not_impl_by_module[module]) > 5:
                    lines.append(f"- ... and {len(not_impl_by_module[module]) - 5} more")

            if len(not_impl_by_module) > 5:
                lines.append(f"\n... and {len(not_impl_by_module) - 5} more modules")
            lines.append("")

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

    def _group_by_module(self) -> Dict:
        """Group coverage statistics by module"""
        by_module = defaultdict(lambda: {
            'total': 0,
            'implemented': 0,
            'tested': 0
        })

        for func_name, func_info in self.api_functions.items():
            module = func_info['file']
            by_module[module]['total'] += 1

            if func_name in self.implemented_functions:
                by_module[module]['implemented'] += 1

            if func_name in self.tested_functions:
                by_module[module]['tested'] += 1

        # Calculate percentages
        for stats in by_module.values():
            total = stats['total']
            stats['impl_pct'] = (stats['implemented'] / total * 100) if total > 0 else 0
            stats['test_pct'] = (stats['tested'] / total * 100) if total > 0 else 0

        return dict(by_module)


def main():
    parser = argparse.ArgumentParser(description='Analyze NPP implementation coverage')
    parser.add_argument('--api-inventory', type=Path,
                        default=Path('api_analysis/api_inventory.json'),
                        help='Path to API inventory JSON')
    parser.add_argument('--src-dir', type=Path, default=Path('src'),
                        help='Source code directory')
    parser.add_argument('--test-dir', type=Path, default=Path('test'),
                        help='Test directory')
    parser.add_argument('--version', default='12.2',
                        help='API version to check against')
    parser.add_argument('--output', type=Path, default=Path('api_analysis/coverage_report.md'),
                        help='Output report path')

    args = parser.parse_args()

    if not args.api_inventory.exists():
        print(f"Error: API inventory not found at {args.api_inventory}")
        print("Run analyze_api.py first to generate API inventory")
        return 1

    print(f"Loading API inventory (version {args.version})...")
    analyzer = CoverageAnalyzer(args.api_inventory, args.src_dir, args.test_dir)

    try:
        analyzer.load_api_inventory(args.version)
        print(f"Loaded {len(analyzer.api_functions)} API functions")
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print("Scanning implementations...")
    analyzer.scan_implementations()
    print(f"Found {len(analyzer.implemented_functions)} implemented functions")

    print("Scanning tests...")
    analyzer.scan_tests()
    print(f"Found {len(analyzer.tested_functions)} tested functions")

    print(f"Generating report to {args.output}...")
    analyzer.generate_report(args.output, args.version)
    print("Done!")

    return 0


if __name__ == '__main__':
    exit(main())
