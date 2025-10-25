#!/usr/bin/env python3
"""
NPP API Analysis Tool

Analyzes NPP API headers across different versions to extract:
- Complete function inventory
- Version differences
- Module and file-level statistics
"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple


class FunctionSignature:
    """Represents a complete function signature"""

    def __init__(self, name: str, return_type: str, params: str, file: str, line: int):
        self.name = name
        self.return_type = return_type.strip()
        self.params = self._normalize_params(params)
        self.file = file
        self.line = line

    def _normalize_params(self, params: str) -> str:
        """Normalize parameter string for comparison"""
        params = re.sub(r'\s+', ' ', params.strip())
        return params.strip()

    def signature(self) -> str:
        """Return full signature for comparison"""
        return f"{self.return_type} {self.name}({self.params})"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'name': self.name,
            'return_type': self.return_type,
            'params': self.params,
            'file': self.file,
            'line': self.line,
            'signature': self.signature()
        }


class APIAnalyzer:
    """Analyzes NPP API headers"""

    def __init__(self, api_root: Path):
        self.api_root = api_root
        self.functions = {}

    def analyze(self) -> Dict[str, FunctionSignature]:
        """Analyze all header files in API directory"""
        for header_file in self.api_root.glob('*.h'):
            if header_file.name.startswith(('nppi', 'npps')) and \
               header_file.name not in ('nppcore.h', 'nppdefs.h'):
                self._parse_header(header_file)
        return self.functions

    def _parse_header(self, header_path: Path):
        """Parse a single header file"""
        with open(header_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Remove comments
        content = self._remove_comments(content)

        lines = content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            # Case 1: Return type alone on one line
            ret_type_match = re.match(r'^(NppStatus|void|int|size_t|Npp\w+)\s*$', line)
            if ret_type_match:
                return_type = ret_type_match.group(1)
                i += 1

                if i < len(lines):
                    next_line = lines[i].strip()
                    func_match = re.match(r'^(nppi\w+|npps\w+)\s*\(', next_line)

                    if func_match:
                        func_name = func_match.group(1)
                        full_decl = next_line

                        # Collect multi-line declaration
                        while not full_decl.rstrip().endswith(';'):
                            i += 1
                            if i >= len(lines):
                                break
                            full_decl += ' ' + lines[i].strip()

                        params_match = re.search(r'\((.*?)\)\s*;', full_decl, re.DOTALL)
                        if params_match:
                            self._add_function(func_name, return_type, params_match.group(1),
                                             header_path.name, i + 1)
                continue

            # Case 2: Return type and function name on same line
            same_line_match = re.match(r'^(NppStatus|void|int|size_t|Npp\w+)\s+(nppi\w+|npps\w+)\s*\(', line)
            if same_line_match:
                return_type = same_line_match.group(1)
                func_name = same_line_match.group(2)
                full_decl = line

                # Collect multi-line declaration
                while not full_decl.rstrip().endswith(';'):
                    i += 1
                    if i >= len(lines):
                        break
                    full_decl += ' ' + lines[i].strip()

                params_match = re.search(r'\((.*?)\)\s*;', full_decl, re.DOTALL)
                if params_match:
                    self._add_function(func_name, return_type, params_match.group(1),
                                     header_path.name, i + 1)

            i += 1

    def _add_function(self, name: str, return_type: str, params: str, file: str, line: int):
        """Add function to inventory"""
        sig = FunctionSignature(name, return_type, params, file, line)
        sig_key = f"{name}::{sig.signature()}"
        self.functions[sig_key] = sig

    def _remove_comments(self, content: str) -> str:
        """Remove C/C++ comments"""
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
        return content

    def get_functions_by_module(self) -> Dict[str, List[FunctionSignature]]:
        """Group functions by module (file)"""
        by_module = defaultdict(list)
        for sig in self.functions.values():
            by_module[sig.file].append(sig)
        return dict(by_module)


def analyze_version(api_path: Path, version: str) -> Tuple[Dict, Dict]:
    """Analyze a single API version"""
    analyzer = APIAnalyzer(api_path)
    functions = analyzer.analyze()
    by_module = analyzer.get_functions_by_module()

    inventory = {key: sig.to_dict() for key, sig in functions.items()}

    module_stats = {}
    for module, sigs in by_module.items():
        module_stats[module] = {
            'count': len(sigs),
            'functions': [sig.name for sig in sorted(sigs, key=lambda x: x.name)]
        }

    return inventory, module_stats


def compare_versions(v1_funcs: Dict, v2_funcs: Dict) -> Dict:
    """Compare two versions and find differences"""
    v1_keys = set(v1_funcs.keys())
    v2_keys = set(v2_funcs.keys())

    v1_names = {k.split('::')[0] for k in v1_keys}
    v2_names = {k.split('::')[0] for k in v2_keys}

    added_keys = v2_keys - v1_keys
    removed_keys = v1_keys - v2_keys

    changed = []
    for name in v1_names & v2_names:
        v1_sigs = [k for k in v1_keys if k.startswith(name + '::')]
        v2_sigs = [k for k in v2_keys if k.startswith(name + '::')]

        if v1_sigs != v2_sigs:
            changed.append({
                'name': name,
                'old_sig': v1_funcs[v1_sigs[0]]['signature'] if v1_sigs else None,
                'new_sig': v2_funcs[v2_sigs[0]]['signature'] if v2_sigs else None
            })

    return {
        'added': [v2_funcs[k] for k in sorted(added_keys)],
        'removed': [v1_funcs[k] for k in sorted(removed_keys)],
        'changed': changed
    }


def generate_csv(version_data: Dict, output_path: Path):
    """Generate CSV file with all API functions"""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(['function_name', 'module', 'return_type', 'params', 'signature'])

        # Sort functions by module then name
        functions = []
        for key, func_info in version_data['inventory'].items():
            functions.append(func_info)

        functions.sort(key=lambda x: (x['file'], x['name']))

        # Write data
        for func in functions:
            writer.writerow([
                func['name'],
                func['file'],
                func['return_type'],
                func['params'],
                func['signature']
            ])


def generate_report(results: Dict, output_path: Path):
    """Generate markdown report"""
    lines = ["# NPP API Analysis Report\n"]

    versions = sorted([v for v in results.keys() if v != 'comparisons'])

    lines.append("## Version Overview\n")
    for version in versions:
        total = len(results[version]['inventory'])
        lines.append(f"**{version}**: {total} functions")
    lines.append("")

    if 'comparisons' in results:
        lines.append("## Version Differences\n")

        for comp_key, diff in results['comparisons'].items():
            v1, v2 = comp_key.split(' -> ')
            lines.append(f"### {v1} -> {v2}\n")

            added = len(diff['added'])
            removed = len(diff['removed'])
            changed = len(diff['changed'])

            lines.append(f"- Added: {added}")
            lines.append(f"- Removed: {removed}")
            lines.append(f"- Changed: {changed}")
            lines.append("")

            if added > 0:
                lines.append("**Added:**")
                for func in diff['added'][:10]:
                    lines.append(f"- {func['name']}")
                if added > 10:
                    lines.append(f"- ... and {added - 10} more")
                lines.append("")

            if changed > 0:
                lines.append("**Changed:**")
                for func in diff['changed'][:5]:
                    lines.append(f"- {func['name']}")
                if changed > 5:
                    lines.append(f"- ... and {changed - 5} more")
                lines.append("")

    latest_version = versions[-1] if versions else None
    if latest_version:
        lines.append(f"## Module Statistics ({latest_version})\n")

        modules = results[latest_version]['modules']

        nppi_modules = {k: v for k, v in modules.items() if k.startswith('nppi')}
        npps_modules = {k: v for k, v in modules.items() if k.startswith('npps')}

        if nppi_modules:
            lines.append("### NPPI\n")
            total_nppi = sum(v['count'] for v in nppi_modules.values())
            lines.append(f"Total: {total_nppi}\n")
            for module in sorted(nppi_modules.keys()):
                count = nppi_modules[module]['count']
                lines.append(f"- {module}: {count}")
            lines.append("")

        if npps_modules:
            lines.append("### NPPS\n")
            total_npps = sum(v['count'] for v in npps_modules.values())
            lines.append(f"Total: {total_npps}\n")
            for module in sorted(npps_modules.keys()):
                count = npps_modules[module]['count']
                lines.append(f"- {module}: {count}")
            lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='Analyze NPP API headers')
    parser.add_argument('--api-dir', type=Path, default=Path('.'),
                        help='Root directory containing API-* folders')
    parser.add_argument('--versions', nargs='+', default=['11.4', '12.2', '12.8'],
                        help='Versions to analyze')
    parser.add_argument('--output', type=Path, default=Path('api_analysis'),
                        help='Output directory')

    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)

    results = {}

    for version in args.versions:
        api_path = args.api_dir / f'API-{version}'

        if not api_path.exists():
            print(f"Warning: {api_path} not found, skipping")
            continue

        print(f"Analyzing {version}...")
        inventory, modules = analyze_version(api_path, version)

        results[version] = {
            'inventory': inventory,
            'modules': modules
        }

    if len(results) > 1:
        comparisons = {}
        versions = sorted([v for v in results.keys() if v != 'comparisons'])

        for i in range(len(versions) - 1):
            v1, v2 = versions[i], versions[i + 1]
            if v1 in results and v2 in results:
                comp_key = f"{v1} -> {v2}"
                print(f"Comparing {comp_key}...")
                comparisons[comp_key] = compare_versions(
                    results[v1]['inventory'],
                    results[v2]['inventory']
                )

        results['comparisons'] = comparisons

    json_path = args.output / 'api_inventory.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved inventory to {json_path}")

    report_path = args.output / 'api_report.md'
    generate_report(results, report_path)
    print(f"Saved report to {report_path}")

    # Generate CSV for latest version
    if results:
        latest = sorted([v for v in results.keys() if v != 'comparisons'])[-1]
        csv_path = args.output / f'api_functions_{latest}.csv'
        generate_csv(results[latest], csv_path)
        print(f"Saved function list to {csv_path}")


if __name__ == '__main__':
    main()
