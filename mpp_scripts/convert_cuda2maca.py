#!/usr/bin/env python3
"""
Convert CUDA Runtime API to MACA Runtime API (Generic Version)

This script uses generic prefix/suffix replacement rules to minimize hardcoded mappings.
Only exceptions to the general patterns are explicitly listed.

General Rules:
- cuda* -> mc* (function names, types)
- CUDA* -> MC* (macros, constants)
- cudaError* -> mcError*
- cudaStream* -> mcStream*
- etc.

Exceptions (explicit mappings required):
- cudaStreamLegacy -> mcStreamAllThreads (not mcStreamLegacy)
- cudaGetDeviceAttribute -> mcDeviceGetAttribute (word order change)
- cudaDeviceProp -> mcDeviceProp_t (suffix addition)
- cudaDevAttr* -> mcDeviceAttr* (expanded prefix)

Usage:
  python3 convert_cuda2maca.py --check     # Preview changes
  python3 convert_cuda2maca.py --backup    # Create backup
  python3 convert_cuda2maca.py --apply     # Apply conversion
  python3 convert_cuda2maca.py --restore   # Restore from backup
"""

import os
import re
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Callable

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent

# Directories to process
PROCESS_DIRS = ['src', 'test', 'examples']
API_DIR = 'API'
API_VERSION_DIRS = ['API-11.4', 'API-12.2', 'API-12.8']
SKIP_DIRS = ['.git', 'build', 'build-nvidia', '.npp_backup', '.cuda_backup', 'wrapper', 'maca_test', 'cuda-npp-samples'] + API_VERSION_DIRS
FILE_EXTENSIONS = ['.cu', '.cpp', '.c', '.h', '.hpp', '.cuh']
BACKUP_DIR = PROJECT_DIR / '.cuda_backup'


# =============================================================================
# Exception Mappings (patterns that don't follow general rules)
# =============================================================================

# Patterns to SKIP (project-defined macros that should NOT be converted)
# These are not CUDA API, they are project-specific version detection macros
SKIP_PATTERNS = [
    r'\bCUDA_SDK_AT_LEAST\b',
    r'\bCUDA_SDK_VERSION_MAJOR\b',
    r'\bCUDA_SDK_VERSION_MINOR\b',
    r'\bCUDA_SDK_VERSION\b',
]

# These are exceptions where simple prefix replacement doesn't work
EXCEPTION_MAPPINGS = [
    # Word order changes
    ('cudaGetDeviceAttribute', 'mcDeviceGetAttribute'),

    # Stream legacy special case
    ('cudaStreamLegacy', 'mcStreamAllThreads'),

    # Type suffix differences
    ('cudaDeviceProp', 'mcDeviceProp_t'),

    # Expanded prefix (cudaDevAttr -> mcDeviceAttribute)
    ('cudaDevAttrComputeCapabilityMajor', 'mcDeviceAttributeComputeCapabilityMajor'),
    ('cudaDevAttrComputeCapabilityMinor', 'mcDeviceAttributeComputeCapabilityMinor'),
    ('cudaDevAttrMaxThreadsPerBlock', 'mcDeviceAttributeMaxThreadsPerBlock'),
    ('cudaDevAttrMaxBlockDimX', 'mcDeviceAttributeMaxBlockDimX'),
    ('cudaDevAttrMaxBlockDimY', 'mcDeviceAttributeMaxBlockDimY'),
    ('cudaDevAttrMaxBlockDimZ', 'mcDeviceAttributeMaxBlockDimZ'),
    ('cudaDevAttrMaxGridDimX', 'mcDeviceAttributeMaxGridDimX'),
    ('cudaDevAttrMaxGridDimY', 'mcDeviceAttributeMaxGridDimY'),
    ('cudaDevAttrMaxGridDimZ', 'mcDeviceAttributeMaxGridDimZ'),
    ('cudaDevAttrMaxSharedMemoryPerBlock', 'mcDeviceAttributeMaxSharedMemoryPerBlock'),
    ('cudaDevAttrTotalConstantMemory', 'mcDeviceAttributeTotalConstantMemory'),
    ('cudaDevAttrWarpSize', 'mcDeviceAttributeWarpSize'),
    ('cudaDevAttrMaxPitch', 'mcDeviceAttributeMaxPitch'),
    ('cudaDevAttrMaxRegistersPerBlock', 'mcDeviceAttributeMaxRegistersPerBlock'),
    ('cudaDevAttrClockRate', 'mcDeviceAttributeClockRate'),
    ('cudaDevAttrTextureAlignment', 'mcDeviceAttributeTextureAlignment'),
    ('cudaDevAttrMultiProcessorCount', 'mcDeviceAttributeMultiProcessorCount'),
    ('cudaDevAttrMaxThreadsPerMultiProcessor', 'mcDeviceAttributeMaxThreadsPerMultiProcessor'),

    # StreamContext field names (nCuda* -> nMc*)
    ('nCudaDeviceId', 'nMcDeviceId'),
    ('nCudaDevAttrComputeCapabilityMajor', 'nMcDevAttrComputeCapabilityMajor'),
    ('nCudaDevAttrComputeCapabilityMinor', 'nMcDevAttrComputeCapabilityMinor'),
]

# Header file mappings (special cases)
HEADER_MAPPINGS = [
    ('cuda_runtime.h', 'mc_runtime.h'),
    ('cuda_runtime_api.h', 'mc_runtime_api.h'),
    ('cuda.h', 'maca.h'),
    ('cuda_fp16.h', 'maca_fp16.h'),
    ('cuda_bf16.h', 'maca_bfloat16.h'),
]


# =============================================================================
# Generic Prefix Replacement Rules
# =============================================================================

class GenericReplacer:
    """Handles generic prefix-based replacements with exception support."""

    def __init__(self):
        # Build exception lookup for fast checking
        self.exceptions = {old: new for old, new in EXCEPTION_MAPPINGS}

        # Compile skip patterns for efficiency
        self.skip_patterns = [re.compile(p) for p in SKIP_PATTERNS]

        # Generic prefix rules: (pattern_prefix, replacement_prefix)
        # Order matters - more specific patterns first
        self.prefix_rules = [
            # These are applied in order, more specific first
            (r'cudaError', 'mcError'),
            (r'cudaStream', 'mcStream'),
            (r'cudaEvent', 'mcEvent'),
            (r'cudaMemcpy', 'mcMemcpy'),
            (r'cudaMalloc', 'mcMalloc'),
            (r'cudaFree', 'mcFree'),
            (r'cudaMemset', 'mcMemset'),
            (r'cudaDevice', 'mcDevice'),
            (r'cudaGet', 'mcGet'),
            (r'cudaSet', 'mcSet'),
            (r'cudaPeek', 'mcPeek'),
            (r'cuda', 'mc'),  # Fallback for remaining cuda* patterns
        ]

    def should_skip(self, text: str) -> bool:
        """Check if text matches any skip pattern."""
        for pattern in self.skip_patterns:
            if pattern.match(text):
                return True
        return False

    def replace(self, content: str) -> Tuple[str, List[str]]:
        """Apply all replacements and return (new_content, list_of_changes)."""
        changes = []

        # Step 1: Apply header mappings
        for old_header, new_header in HEADER_MAPPINGS:
            pattern = rf'#include\s*[<"]({re.escape(old_header)})[>"]'
            if re.search(pattern, content):
                content = re.sub(pattern, f'#include <{new_header}>', content)
                changes.append(f'Header: {old_header} -> {new_header}')

        # Step 2: Apply exception mappings first (highest priority)
        for old, new in EXCEPTION_MAPPINGS:
            pattern = rf'\b{re.escape(old)}\b'
            if re.search(pattern, content):
                content = re.sub(pattern, new, content)
                changes.append(f'Exception: {old} -> {new}')

        # Step 3: Apply generic cuda* -> mc* pattern
        # This handles any remaining cuda followed by capital letter
        def replace_cuda_generic(match):
            full_match = match.group(0)
            # Check if should skip (project-defined macros)
            if self.should_skip(full_match):
                return full_match  # Don't convert
            # Check if already handled by exceptions
            if full_match in self.exceptions:
                return full_match  # Already replaced above
            # Generic replacement: cuda -> mc
            suffix = match.group(1)
            new_name = 'mc' + suffix
            changes.append(f'Generic: {full_match} -> {new_name}')
            return new_name

        # Match cuda followed by capital letter or digit (API naming convention)
        content = re.sub(r'\bcuda([A-Z][a-zA-Z0-9_]*)\b', replace_cuda_generic, content)

        # Step 4: Apply CUDA_* -> MC_* for macros/constants
        # But skip project-defined macros like CUDA_SDK_*
        def replace_cuda_macro(match):
            full_match = match.group(0)  # e.g., CUDA_SDK_AT_LEAST or CUDA_SUCCESS
            if self.should_skip(full_match):
                return full_match  # Don't convert project macros
            new_name = 'MC_' + match.group(1)
            changes.append(f'Macro: {full_match} -> {new_name}')
            return new_name

        # Match CUDA_ followed by identifier characters
        content = re.sub(r'\bCUDA_([A-Z][A-Z0-9_]*)\b', replace_cuda_macro, content)

        return content, changes


def get_all_files(base_dir: Path, include_api: bool = True) -> List[Path]:
    """Get all source files to process."""
    files = []

    for process_dir in PROCESS_DIRS:
        dir_path = base_dir / process_dir
        if not dir_path.exists():
            continue
        for root, dirs, filenames in os.walk(dir_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for filename in filenames:
                if any(filename.endswith(ext) for ext in FILE_EXTENSIONS):
                    files.append(Path(root) / filename)

    if include_api:
        api_dir = base_dir / API_DIR
        if api_dir.is_symlink():
            link_target = os.readlink(api_dir)
            api_dir = base_dir / link_target
        if api_dir.exists():
            for filename in os.listdir(api_dir):
                if any(filename.endswith(ext) for ext in FILE_EXTENSIONS):
                    files.append(api_dir / filename)

    return files


def create_backup():
    """Create backup of files before conversion."""
    print(f"Creating backup in {BACKUP_DIR}")

    if BACKUP_DIR.exists():
        print(f"  Backup already exists. Remove {BACKUP_DIR} first.")
        return False

    BACKUP_DIR.mkdir(parents=True)
    files = get_all_files(PROJECT_DIR)
    backed_up = 0

    for file_path in files:
        rel_path = file_path.relative_to(PROJECT_DIR)
        backup_path = BACKUP_DIR / rel_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, backup_path)
        backed_up += 1

    print(f"  Backed up {backed_up} files")
    return True


def restore_backup():
    """Restore files from backup."""
    if not BACKUP_DIR.exists():
        print(f"Error: Backup not found: {BACKUP_DIR}")
        return False

    print(f"Restoring from {BACKUP_DIR}")
    restored = 0

    for root, dirs, filenames in os.walk(BACKUP_DIR):
        for filename in filenames:
            backup_path = Path(root) / filename
            rel_path = backup_path.relative_to(BACKUP_DIR)
            target_path = PROJECT_DIR / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_path, target_path)
            restored += 1

    print(f"  Restored {restored} files")
    return True


def check_changes():
    """Preview changes without applying."""
    print("Checking for CUDA -> MACA conversions (Generic Rules)...")
    print()

    replacer = GenericReplacer()
    files = get_all_files(PROJECT_DIR)
    total_changes = 0
    files_with_changes = 0

    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
            continue

        _, changes = replacer.replace(content)

        if changes:
            files_with_changes += 1
            rel_path = file_path.relative_to(PROJECT_DIR)
            print(f"{rel_path}:")

            # Deduplicate and count
            change_counts: Dict[str, int] = {}
            for change in changes:
                change_counts[change] = change_counts.get(change, 0) + 1

            for change, count in sorted(change_counts.items()):
                if count > 1:
                    print(f"  {change} (x{count})")
                else:
                    print(f"  {change}")
                total_changes += count
            print()

    print(f"Summary: {total_changes} changes in {files_with_changes} files")
    print()
    print("Generic rules applied:")
    print("  - cuda* -> mc* (functions, types)")
    print("  - CUDA_* -> MC_* (macros)")
    print(f"  - {len(EXCEPTION_MAPPINGS)} exception mappings")
    return total_changes > 0


def apply_changes():
    """Apply CUDA to MACA conversion."""
    print("Applying CUDA -> MACA conversion (Generic Rules)...")
    print()

    replacer = GenericReplacer()
    files = get_all_files(PROJECT_DIR)
    total_changes = 0
    files_modified = 0

    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
            continue

        new_content, changes = replacer.replace(content)

        if changes and new_content != content:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                files_modified += 1
                rel_path = file_path.relative_to(PROJECT_DIR)
                print(f"  Modified: {rel_path} ({len(changes)} changes)")
                total_changes += len(changes)
            except Exception as e:
                print(f"  Error writing {file_path}: {e}")

    print()
    print(f"Conversion complete: {total_changes} changes in {files_modified} files")
    return files_modified > 0


def main():
    parser = argparse.ArgumentParser(
        description='Convert CUDA Runtime API to MACA (Generic Rules)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script uses generic prefix replacement rules instead of exhaustive mappings.
Only exceptions to the general cuda* -> mc* pattern are explicitly listed.

Examples:
  python3 convert_cuda2maca.py --check     # Preview changes
  python3 convert_cuda2maca.py --backup    # Create backup
  python3 convert_cuda2maca.py --apply     # Apply conversion
  python3 convert_cuda2maca.py --restore   # Restore from backup
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true', help='Preview changes')
    group.add_argument('--backup', action='store_true', help='Create backup')
    group.add_argument('--apply', action='store_true', help='Apply conversion')
    group.add_argument('--restore', action='store_true', help='Restore from backup')

    args = parser.parse_args()
    os.chdir(PROJECT_DIR)

    if args.check:
        check_changes()
    elif args.backup:
        create_backup()
    elif args.apply:
        apply_changes()
    elif args.restore:
        restore_backup()


if __name__ == '__main__':
    main()
