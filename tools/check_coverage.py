#!/usr/bin/env python3
"""
NPP Coverage Analysis Tool

Analyzes implementation and test coverage against the API function list.

Coverage semantics:
- implemented/tested mean real implementation and enabled test coverage
- stubs and disabled-only references are tracked separately
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


class CoverageAnalyzer:
    """Analyzes implementation and test coverage."""

    API_REF_PATTERNS = (
        r'\b(nppi\w+|npps\w+)\s*\(',
        r'\b(nppi\w+|npps\w+)\s*[,\)]',
    )

    TEST_MACRO_PATTERN = re.compile(
        r'\bTEST(?:_F|_P)?\s*\(\s*([^,]+)\s*,\s*([^)]+)\)\s*\{',
        re.MULTILINE,
    )

    NPPSTATUS_FUNCTION_PATTERN = re.compile(r'NppStatus\s+(nppi\w+|npps\w+)\s*\(')
    MALLOC_FUNCTION_PATTERN = re.compile(r'Npp\w+\s*\*\s*(nppiMalloc_\w+|nppsMalloc_\w+)\s*\(')
    FREE_FUNCTION_PATTERN = re.compile(r'void\s+(nppiFree|nppsFree)\s*\(')

    def __init__(self, api_csv: Path, src_dir: Path, test_dir: Path):
        self.api_csv = api_csv
        self.src_dir = src_dir
        self.test_dir = test_dir

        self.api_functions = {}  # {func_name: {module, return_type, params}}
        self.implementations = {}  # real implementations: {func_name: file_path}
        self.stub_implementations = {}  # stub implementations: {func_name: file_path}
        self.tests = {}  # enabled tests: {func_name: file_path}
        self.disabled_tests = {}  # disabled-only references: {func_name: file_path}

    def load_api_functions(self):
        """Load API functions from CSV."""
        with open(self.api_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                func_name = row['function_name']
                self.api_functions[func_name] = {
                    'module': row['module'],
                    'return_type': row['return_type'],
                    'params': row['params'],
                }

    def scan_implementations(self):
        """Scan source code for function implementations."""
        for cu_file in self.src_dir.rglob('*.cu'):
            self._scan_impl_file(cu_file)

        for cpp_file in self.src_dir.rglob('*.cpp'):
            self._scan_impl_file(cpp_file)

    def scan_tests(self):
        """Scan test files for function calls."""
        for test_file in self.test_dir.rglob('test_*.cpp'):
            self._scan_test_file(test_file)

    def _scan_impl_file(self, file_path: Path):
        """Scan implementation file."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            rel_path = self._relative_project_path(file_path)

            for raw_name, body in self._iter_function_definitions(content, self.NPPSTATUS_FUNCTION_PATTERN):
                api_name = self._normalize_api_name(raw_name)
                if api_name not in self.api_functions:
                    continue

                kind = 'stub' if self._is_stub_body(body) else 'real'
                self._record_implementation(api_name, rel_path, kind)

            for raw_name, _body in self._iter_function_definitions(content, self.MALLOC_FUNCTION_PATTERN):
                if raw_name in self.api_functions:
                    self._record_implementation(raw_name, rel_path, 'real')

            for raw_name, _body in self._iter_function_definitions(content, self.FREE_FUNCTION_PATTERN):
                if raw_name in self.api_functions:
                    self._record_implementation(raw_name, rel_path, 'real')

            self._capture_copy_wrappers(content, rel_path)
            self._capture_swapchannels_wrappers(content, rel_path)
            self._capture_warpperspective_wrappers(content, rel_path)

        except Exception as e:
            print(f"Warning: Error scanning {file_path}: {e}")

    def _scan_test_file(self, file_path: Path):
        """Scan test file."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            rel_path = self._relative_project_path(file_path)

            disabled_spans = self._extract_disabled_test_spans(content)
            enabled_content = self._remove_spans(content, disabled_spans)
            disabled_content = ''.join(content[start:end] for start, end in disabled_spans)

            for func_name in self._extract_api_references(enabled_content):
                self._record_test(func_name, rel_path, 'enabled')

            for func_name in self._extract_api_references(disabled_content):
                self._record_test(func_name, rel_path, 'disabled')

        except Exception as e:
            print(f"Warning: Error scanning {file_path}: {e}")

    def _relative_project_path(self, file_path: Path) -> str:
        return str(file_path.relative_to(self.src_dir.parent))

    def _normalize_api_name(self, raw_name: str) -> str:
        if raw_name.endswith('_impl'):
            return raw_name[:-len('_impl')]
        return raw_name

    def _is_stub_body(self, body: str) -> bool:
        return 'return NPP_NOT_IMPLEMENTED_ERROR;' in body

    def _record_implementation(self, api_name: str, rel_path: str, kind: str):
        if kind == 'real':
            self.implementations[api_name] = rel_path
            self.stub_implementations.pop(api_name, None)
            return

        if api_name not in self.implementations and api_name not in self.stub_implementations:
            self.stub_implementations[api_name] = rel_path

    def _record_test(self, api_name: str, rel_path: str, kind: str):
        if api_name not in self.api_functions:
            return

        if kind == 'enabled':
            self.tests[api_name] = rel_path
            self.disabled_tests.pop(api_name, None)
            return

        if api_name not in self.tests and api_name not in self.disabled_tests:
            self.disabled_tests[api_name] = rel_path

    def _capture_copy_wrappers(self, content: str, rel_path: str):
        """Detect macro-generated copy wrappers."""

        def record(name: str):
            if name in self.api_functions:
                self._record_implementation(name, rel_path, 'real')

        simple_pattern = r'NPPI_COPY_WRAPPER\s*\(\s*[^,]+,\s*([^,]+),\s*([^\)\s]+)\s*\)'
        for match in re.finditer(simple_pattern, content):
            suffix, channel = match.group(1).strip(), match.group(2).strip()
            record(f'nppiCopy_{suffix}_{channel}_Ctx')
            record(f'nppiCopy_{suffix}_{channel}')

        cxpxr_pattern = r'NPPI_COPY_CXPXR_WRAPPER\s*\(\s*[^,]+,\s*([^,]+),\s*(\d+)\s*\)'
        for match in re.finditer(cxpxr_pattern, content):
            suffix, channels = match.group(1).strip(), match.group(2).strip()
            record(f'nppiCopy_{suffix}_C{channels}P{channels}R_Ctx')
            record(f'nppiCopy_{suffix}_C{channels}P{channels}R')

        pxcxr_pattern = r'NPPI_COPY_PXCXR_WRAPPER\s*\(\s*[^,]+,\s*([^,]+),\s*(\d+)\s*\)'
        for match in re.finditer(pxcxr_pattern, content):
            suffix, channels = match.group(1).strip(), match.group(2).strip()
            record(f'nppiCopy_{suffix}_P{channels}C{channels}R_Ctx')
            record(f'nppiCopy_{suffix}_P{channels}C{channels}R')

    def _capture_swapchannels_wrappers(self, content: str, rel_path: str):
        """Detect macro-generated swapChannels wrappers."""

        def record(name: str):
            if name in self.api_functions:
                self._record_implementation(name, rel_path, 'real')

        all_pattern = r'GENERATE_ALL_SWAP_CHANNELS\s*\(\s*([^) \t]+)\s*\)'
        suffixes = ['C3R', 'C4R', 'C3IR', 'C4IR', 'C4C3R', 'C3C4R', 'AC4R']

        for match in re.finditer(all_pattern, content):
            data_type = match.group(1).strip()
            for suffix in suffixes:
                record(f'nppiSwapChannels_{data_type}_{suffix}')
                record(f'nppiSwapChannels_{data_type}_{suffix}_Ctx')

    def _capture_warpperspective_wrappers(self, content: str, rel_path: str):
        """Detect macro-generated warpPerspective wrappers."""

        def record(name: str):
            if name in self.api_functions:
                self._record_implementation(name, rel_path, 'real')

        ac4_pattern = r'DEFINE_WARP_PERSPECTIVE_AC4\s*\(\s*[^,]+,\s*([^,]+),'
        planar_pattern = r'DEFINE_WARP_PERSPECTIVE_PLANAR\s*\(\s*[^,]+,\s*([^,]+),\s*(\d+)\s*\)'
        back_ac4_pattern = r'DEFINE_WARP_PERSPECTIVE_BACK_AC4\s*\(\s*[^,]+,\s*([^,]+),'
        back_planar_pattern = r'DEFINE_WARP_PERSPECTIVE_BACK_PLANAR\s*\(\s*[^,]+,\s*([^,]+),\s*(\d+)\s*\)'
        quad_packed_pattern = r'DEFINE_WARP_PERSPECTIVE_QUAD_PACKED\s*\(\s*[^,]+,\s*([^,]+),\s*([^,]+),'
        quad_planar_pattern = r'DEFINE_WARP_PERSPECTIVE_QUAD_PLANAR\s*\(\s*[^,]+,\s*([^,]+),\s*(\d+)\s*\)'
        batch_pattern = r'DEFINE_WARP_PERSPECTIVE_BATCH\s*\(\s*[^,]+,\s*([^,]+),\s*([^,]+),'

        for match in re.finditer(ac4_pattern, content):
            suffix = match.group(1).strip()
            record(f'nppiWarpPerspective_{suffix}_AC4R')
            record(f'nppiWarpPerspective_{suffix}_AC4R_Ctx')

        for match in re.finditer(planar_pattern, content):
            suffix, planes = match.group(1).strip(), match.group(2).strip()
            record(f'nppiWarpPerspective_{suffix}_P{planes}R')
            record(f'nppiWarpPerspective_{suffix}_P{planes}R_Ctx')

        for match in re.finditer(back_ac4_pattern, content):
            suffix = match.group(1).strip()
            record(f'nppiWarpPerspectiveBack_{suffix}_AC4R')
            record(f'nppiWarpPerspectiveBack_{suffix}_AC4R_Ctx')

        for match in re.finditer(back_planar_pattern, content):
            suffix, planes = match.group(1).strip(), match.group(2).strip()
            record(f'nppiWarpPerspectiveBack_{suffix}_P{planes}R')
            record(f'nppiWarpPerspectiveBack_{suffix}_P{planes}R_Ctx')

        for match in re.finditer(quad_packed_pattern, content):
            suffix, variant = match.group(1).strip(), match.group(2).strip()
            record(f'nppiWarpPerspectiveQuad_{suffix}_{variant}')
            record(f'nppiWarpPerspectiveQuad_{suffix}_{variant}_Ctx')

        for match in re.finditer(quad_planar_pattern, content):
            suffix, planes = match.group(1).strip(), match.group(2).strip()
            record(f'nppiWarpPerspectiveQuad_{suffix}_P{planes}R')
            record(f'nppiWarpPerspectiveQuad_{suffix}_P{planes}R_Ctx')

        for match in re.finditer(batch_pattern, content):
            suffix, variant = match.group(1).strip(), match.group(2).strip()
            record(f'nppiWarpPerspectiveBatch_{suffix}_{variant}')
            record(f'nppiWarpPerspectiveBatch_{suffix}_{variant}_Ctx')

        if 'nppiWarpPerspectiveBatchInit_Ctx' in content:
            record('nppiWarpPerspectiveBatchInit_Ctx')
        if 'nppiWarpPerspectiveBatchInit(' in content:
            record('nppiWarpPerspectiveBatchInit')

    def _iter_function_definitions(self, content: str, pattern: re.Pattern):
        for match in pattern.finditer(content):
            raw_name = match.group(1)
            paren_start = match.end() - 1
            paren_end = self._find_matching_delimiter(content, paren_start, '(', ')')
            if paren_end == -1:
                continue

            body_start = self._skip_whitespace_and_comments(content, paren_end + 1)
            if body_start >= len(content) or content[body_start] != '{':
                continue

            body_end = self._find_matching_delimiter(content, body_start, '{', '}')
            if body_end == -1:
                continue

            yield raw_name, content[body_start:body_end + 1]

    def _extract_disabled_test_spans(self, content: str) -> List[Tuple[int, int]]:
        spans = []
        for match in self.TEST_MACRO_PATTERN.finditer(content):
            suite_name = match.group(1).strip()
            test_name = match.group(2).strip()
            if not (suite_name.startswith('DISABLED_') or test_name.startswith('DISABLED_')):
                continue

            brace_start = match.end() - 1
            brace_end = self._find_matching_delimiter(content, brace_start, '{', '}')
            if brace_end == -1:
                continue

            spans.append((match.start(), brace_end + 1))

        return spans

    def _remove_spans(self, content: str, spans: List[Tuple[int, int]]) -> str:
        if not spans:
            return content

        pieces = []
        cursor = 0
        for start, end in spans:
            pieces.append(content[cursor:start])
            cursor = end
        pieces.append(content[cursor:])
        return ''.join(pieces)

    def _extract_api_references(self, content: str) -> List[str]:
        refs = set()
        for pattern in self.API_REF_PATTERNS:
            for match in re.finditer(pattern, content):
                func_name = match.group(1)
                if func_name in self.api_functions:
                    refs.add(func_name)
        return sorted(refs)

    def _skip_whitespace_and_comments(self, content: str, index: int) -> int:
        i = index
        while i < len(content):
            if content[i].isspace():
                i += 1
                continue

            if content.startswith('//', i):
                newline = content.find('\n', i)
                return len(content) if newline == -1 else self._skip_whitespace_and_comments(content, newline + 1)

            if content.startswith('/*', i):
                comment_end = content.find('*/', i + 2)
                return len(content) if comment_end == -1 else self._skip_whitespace_and_comments(content, comment_end + 2)

            return i

        return i

    def _find_matching_delimiter(self, content: str, start_idx: int, open_char: str, close_char: str) -> int:
        depth = 0
        in_string = None
        in_line_comment = False
        in_block_comment = False
        escaped = False
        i = start_idx

        while i < len(content):
            ch = content[i]
            nxt = content[i + 1] if i + 1 < len(content) else ''

            if in_line_comment:
                if ch == '\n':
                    in_line_comment = False
                i += 1
                continue

            if in_block_comment:
                if ch == '*' and nxt == '/':
                    in_block_comment = False
                    i += 2
                else:
                    i += 1
                continue

            if in_string:
                if escaped:
                    escaped = False
                elif ch == '\\':
                    escaped = True
                elif ch == in_string:
                    in_string = None
                i += 1
                continue

            if ch == '/' and nxt == '/':
                in_line_comment = True
                i += 2
                continue

            if ch == '/' and nxt == '*':
                in_block_comment = True
                i += 2
                continue

            if ch in ('"', "'"):
                in_string = ch
                i += 1
                continue

            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    return i

            i += 1

        return -1

    def generate_coverage_csv(self, output_path: Path):
        """Generate coverage CSV."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'function_name',
                'module',
                'implemented',
                'impl_file',
                'tested',
                'test_file',
                'impl_status',
                'stub_file',
                'test_status',
                'disabled_test_file',
            ])

            functions = sorted(self.api_functions.items(), key=lambda x: (x[1]['module'], x[0]))

            for func_name, func_info in functions:
                implemented = 'yes' if func_name in self.implementations else 'no'
                impl_file = self.implementations.get(func_name, '')
                impl_status = 'real' if func_name in self.implementations else 'stub' if func_name in self.stub_implementations else 'missing'
                stub_file = self.stub_implementations.get(func_name, '')

                tested = 'yes' if func_name in self.tests else 'no'
                test_file = self.tests.get(func_name, '')
                test_status = 'enabled' if func_name in self.tests else 'disabled' if func_name in self.disabled_tests else 'missing'
                disabled_test_file = self.disabled_tests.get(func_name, '')

                writer.writerow([
                    func_name,
                    func_info['module'],
                    implemented,
                    impl_file,
                    tested,
                    test_file,
                    impl_status,
                    stub_file,
                    test_status,
                    disabled_test_file,
                ])

    def generate_summary(self, output_path: Path):
        """Generate summary report."""
        total = len(self.api_functions)
        implemented = len(self.implementations)
        stubbed = len(self.stub_implementations)
        tested = len(self.tests)
        disabled_only = len(self.disabled_tests)

        impl_pct = (implemented / total * 100) if total > 0 else 0
        stub_pct = (stubbed / total * 100) if total > 0 else 0
        test_pct = (tested / total * 100) if total > 0 else 0
        disabled_pct = (disabled_only / total * 100) if total > 0 else 0

        lines = ["# Coverage Summary\n"]
        lines.append(f"- Total API Functions: {total}")
        lines.append(f"- Implemented: {implemented} ({impl_pct:.1f}%)")
        lines.append(f"- Stubbed Only: {stubbed} ({stub_pct:.1f}%)")
        lines.append(f"- Tested: {tested} ({test_pct:.1f}%)")
        lines.append(f"- Referenced only in disabled tests: {disabled_only} ({disabled_pct:.1f}%)")
        lines.append(f"- Implemented but not tested: {implemented - len(set(self.implementations) & set(self.tests))}")
        lines.append("")

        by_module = defaultdict(lambda: {'total': 0, 'impl': 0, 'stub': 0, 'test': 0, 'disabled': 0})

        for func_name, func_info in self.api_functions.items():
            module = func_info['module']
            by_module[module]['total'] += 1

            if func_name in self.implementations:
                by_module[module]['impl'] += 1
            elif func_name in self.stub_implementations:
                by_module[module]['stub'] += 1

            if func_name in self.tests:
                by_module[module]['test'] += 1
            elif func_name in self.disabled_tests:
                by_module[module]['disabled'] += 1

        lines.append("## By Module\n")

        nppi = {k: v for k, v in by_module.items() if k.startswith('nppi')}
        npps = {k: v for k, v in by_module.items() if k.startswith('npps')}

        if nppi:
            lines.append("### NPPI\n")
            for module in sorted(nppi.keys()):
                stats = nppi[module]
                impl_pct = (stats['impl'] / stats['total'] * 100) if stats['total'] > 0 else 0
                stub_pct = (stats['stub'] / stats['total'] * 100) if stats['total'] > 0 else 0
                test_pct = (stats['test'] / stats['total'] * 100) if stats['total'] > 0 else 0
                disabled_pct = (stats['disabled'] / stats['total'] * 100) if stats['total'] > 0 else 0
                lines.append(f"**{module}**")
                lines.append(
                    f"- Total: {stats['total']}, Impl: {stats['impl']} ({impl_pct:.1f}%), "
                    f"Stub: {stats['stub']} ({stub_pct:.1f}%), "
                    f"Test: {stats['test']} ({test_pct:.1f}%), "
                    f"Disabled-only: {stats['disabled']} ({disabled_pct:.1f}%)"
                )
                lines.append("")

        if npps:
            lines.append("### NPPS\n")
            for module in sorted(npps.keys()):
                stats = npps[module]
                impl_pct = (stats['impl'] / stats['total'] * 100) if stats['total'] > 0 else 0
                stub_pct = (stats['stub'] / stats['total'] * 100) if stats['total'] > 0 else 0
                test_pct = (stats['test'] / stats['total'] * 100) if stats['total'] > 0 else 0
                disabled_pct = (stats['disabled'] / stats['total'] * 100) if stats['total'] > 0 else 0
                lines.append(f"**{module}**")
                lines.append(
                    f"- Total: {stats['total']}, Impl: {stats['impl']} ({impl_pct:.1f}%), "
                    f"Stub: {stats['stub']} ({stub_pct:.1f}%), "
                    f"Test: {stats['test']} ({test_pct:.1f}%), "
                    f"Disabled-only: {stats['disabled']} ({disabled_pct:.1f}%)"
                )
                lines.append("")

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='Analyze NPP implementation coverage')
    parser.add_argument('--api-csv', type=Path, default=Path('api_analysis/api_functions_12.2.csv'),
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
    print(f"Found {len(analyzer.implementations)} real implementations and {len(analyzer.stub_implementations)} stubs")

    print("Scanning tests...")
    analyzer.scan_tests()
    print(f"Found {len(analyzer.tests)} enabled tests and {len(analyzer.disabled_tests)} disabled-only references")

    print(f"Generating coverage CSV to {args.output}...")
    analyzer.generate_coverage_csv(args.output)

    print(f"Generating summary to {args.summary}...")
    analyzer.generate_summary(args.summary)

    print("Done!")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
