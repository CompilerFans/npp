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
        # Remove parameter names, keep only types
        params = re.sub(r'\s+', ' ', params.strip())
        # Remove trailing whitespace
        params = params.strip()
        return params

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
        self.functions = {}  # {func_name: FunctionSignature}

    def analyze(self) -> Dict[str, FunctionSignature]:
        """Analyze all header files in API directory"""

        # Focus on nppi and npps headers (skip core/defs)
        for header_file in self.api_root.glob('*.h'):
            if header_file.name.startswith(('nppi', 'npps')) and \
               not header_file.name in ('nppcore.h', 'nppdefs.h'):
                self._parse_header(header_file)

        return self.functions

    def _parse_header(self, header_path: Path):
        """Parse a single header file"""

        with open(header_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Remove comments
        content = self._remove_comments(content)

        # Extract function declarations
        # Pattern: return_type function_name(params);
        # Return type can be NppStatus, void, int, etc.
        pattern = r'^\s*([\w]+)\s+(nppi\w+|npps\w+)\s*\('

        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]

            # Check if line starts a function declaration
            match = re.match(pattern, line)
            if match:
                return_type = match.group(1)
                func_name = match.group(2)

                # Collect full declaration (may span multiple lines)
                full_decl = line
                while not full_decl.rstrip().endswith(';'):
                    i += 1
                    if i >= len(lines):
                        break
                    full_decl += ' ' + lines[i].strip()

                # Extract params from complete declaration
                params_match = re.search(r'\((.*?)\)\s*;', full_decl)
                if params_match:
                    params = params_match.group(1)

                    # Create signature
                    sig = FunctionSignature(
                        name=func_name,
                        return_type=return_type,
                        params=params,
                        file=header_path.name,
                        line=i + 1
                    )

                    # Use full signature as key to detect parameter changes
                    sig_key = f"{func_name}::{sig.signature()}"
                    self.functions[sig_key] = sig

            i += 1

    def _remove_comments(self, content: str) -> str:
        """Remove C/C++ comments"""
        # Remove /* */ comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        # Remove // comments
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

    # Convert to serializable format
    inventory = {}
    for key, sig in functions.items():
        inventory[key] = sig.to_dict()

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

    # Get function names (without signature)
    v1_names = {k.split('::')[0] for k in v1_keys}
    v2_names = {k.split('::')[0] for k in v2_keys}

    added_keys = v2_keys - v1_keys
    removed_keys = v1_keys - v2_keys

    # Detect signature changes (same name, different signature)
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


def generate_report(results: Dict, output_path: Path):
    """Generate markdown report"""

    lines = []
    lines.append("# NPP API Analysis Report\n")

    # Get version list (exclude 'comparisons')
    versions = sorted([v for v in results.keys() if v != 'comparisons'])

    # Version overview
    lines.append("## Version Overview\n")
    for version in versions:
        total = len(results[version]['inventory'])
        lines.append(f"**{version}**: {total} functions")
    lines.append("")

    # Version comparisons
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
                lines.append("**Added functions:**")
                for func in diff['added'][:10]:  # Show first 10
                    lines.append(f"- {func['name']}")
                if added > 10:
                    lines.append(f"- ... and {added - 10} more")
                lines.append("")

            if changed > 0:
                lines.append("**Changed signatures:**")
                for func in diff['changed'][:5]:  # Show first 5
                    lines.append(f"- {func['name']}")
                if changed > 5:
                    lines.append(f"- ... and {changed - 5} more")
                lines.append("")

    # Module statistics for latest version
    latest_version = versions[-1] if versions else None
    if latest_version:
        lines.append(f"## Module Statistics ({latest_version})\n")

        modules = results[latest_version]['modules']

        # Group by nppi/npps
        nppi_modules = {k: v for k, v in modules.items() if k.startswith('nppi_')}
        npps_modules = {k: v for k, v in modules.items() if k.startswith('npps_')}

        if nppi_modules:
            lines.append("### NPPI Modules\n")
            for module in sorted(nppi_modules.keys()):
                count = nppi_modules[module]['count']
                lines.append(f"- {module}: {count}")
            lines.append("")

        if npps_modules:
            lines.append("### NPPS Modules\n")
            for module in sorted(npps_modules.keys()):
                count = npps_modules[module]['count']
                lines.append(f"- {module}: {count}")
            lines.append("")

    # Write report
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

    # Create output directory
    args.output.mkdir(exist_ok=True)

    results = {}

    # Analyze each version
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

    # Compare versions
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

    # Save results
    json_path = args.output / 'api_inventory.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved inventory to {json_path}")

    # Generate report
    report_path = args.output / 'api_report.md'
    generate_report(results, report_path)
    print(f"Saved report to {report_path}")


if __name__ == '__main__':
    main()
