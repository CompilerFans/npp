#!/usr/bin/env python3
"""
Convert CUDA Runtime API to MACA Runtime API

This script converts CUDA-specific code to MACA equivalents:
- cuda_runtime.h -> mc_runtime.h
- cudaError_t -> mcError_t
- cudaStream_t -> mcStream_t
- cudaDeviceProp -> mcDeviceProp_t
- cudaMalloc -> mcMalloc
- cudaMemcpy -> mcMemcpy
- cudaMemcpyHostToDevice -> mcMemcpyHostToDevice
- etc.

Special naming differences between CUDA and MACA:
- cudaStreamLegacy -> mcStreamAllThreads (not mcStreamLegacy)
- cudaGetDeviceAttribute -> mcDeviceGetAttribute (different word order)

Note: For MppStreamContext (converted from NppStreamContext), the field mappings are:
  - hStream: mcStream_t (from mcStream_t)
  - nMcDeviceId: int (from mcGetDevice())
  - nMultiProcessorCount: int (from mcDeviceProp_t.multiProcessorCount)
  - nMaxThreadsPerMultiProcessor: int (from mcDeviceProp_t.maxThreadsPerMultiProcessor)
  - nMaxThreadsPerBlock: int (from mcDeviceProp_t.maxThreadsPerBlock)
  - nSharedMemPerBlock: size_t (from mcDeviceProp_t.sharedMemPerBlock)
  - nMcDevAttrComputeCapabilityMajor: int (from mcDeviceProp_t.major)
  - nMcDevAttrComputeCapabilityMinor: int (from mcDeviceProp_t.minor)
  - nStreamFlags: unsigned int (from mcStreamGetFlags())

Usage:
  python3 convert_cuda_to_maca.py --check     # Preview changes without applying
  python3 convert_cuda_to_maca.py --backup    # Create backup in .cuda_backup/
  python3 convert_cuda_to_maca.py --apply     # Apply conversion
  python3 convert_cuda_to_maca.py --restore   # Restore from backup
"""

import os
import re
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent

# Directories to process (source code)
PROCESS_DIRS = ['src', 'test', 'examples']

# API directory to process (headers with CUDA types)
API_DIR = 'API'

# Original API version directories (read-only reference, not modified)
API_VERSION_DIRS = ['API-11.4', 'API-12.2', 'API-12.8']

# Directories to skip
SKIP_DIRS = ['.git', 'build', 'build-nvidia', '.npp_backup', '.cuda_backup'] + API_VERSION_DIRS

# File extensions to process
FILE_EXTENSIONS = ['.cu', '.cpp', '.c', '.h', '.hpp', '.cuh']

# Backup directory
BACKUP_DIR = PROJECT_DIR / '.cuda_backup'

# =============================================================================
# CUDA to MACA mappings
# =============================================================================

# Header file mappings
HEADER_MAPPINGS = [
    ('cuda_runtime.h', 'mc_runtime.h'),
    ('cuda_runtime_api.h', 'mc_runtime_api.h'),
    ('cuda.h', 'maca.h'),
    ('cuda_fp16.h', 'maca_fp16.h'),
    ('cuda_bf16.h', 'maca_bfloat16.h'),
    ('device_launch_parameters.h', 'device_launch_parameters.h'),
]

# Type mappings (order matters - more specific first)
TYPE_MAPPINGS = [
    # Error types
    ('cudaError_t', 'mcError_t'),
    ('cudaError', 'mcError'),
    # Stream types
    ('cudaStream_t', 'mcStream_t'),
    # Event types
    ('cudaEvent_t', 'mcEvent_t'),
    # Device properties (MACA uses _t suffix)
    ('cudaDeviceProp', 'mcDeviceProp_t'),
    # Memory types
    ('cudaMemcpyKind', 'mcMemcpyKind'),
    # Dim types
    ('dim3', 'dim3'),  # Same in MACA
]

# Error code mappings
ERROR_MAPPINGS = [
    ('cudaSuccess', 'mcSuccess'),
    ('cudaErrorInvalidValue', 'mcErrorInvalidValue'),
    ('cudaErrorMemoryAllocation', 'mcErrorMemoryAllocation'),
    ('cudaErrorInvalidDevicePointer', 'mcErrorInvalidDevicePointer'),
    ('cudaErrorInvalidMemcpyDirection', 'mcErrorInvalidMemcpyDirection'),
    ('cudaErrorNotReady', 'mcErrorNotReady'),
    ('cudaErrorNoDevice', 'mcErrorNoDevice'),
    ('cudaErrorInvalidDevice', 'mcErrorInvalidDevice'),
]

# Memory copy direction mappings
MEMCPY_DIRECTION_MAPPINGS = [
    ('cudaMemcpyHostToHost', 'mcMemcpyHostToHost'),
    ('cudaMemcpyHostToDevice', 'mcMemcpyHostToDevice'),
    ('cudaMemcpyDeviceToHost', 'mcMemcpyDeviceToHost'),
    ('cudaMemcpyDeviceToDevice', 'mcMemcpyDeviceToDevice'),
    ('cudaMemcpyDefault', 'mcMemcpyDefault'),
]

# Stream flag mappings
STREAM_FLAG_MAPPINGS = [
    ('cudaStreamDefault', 'mcStreamDefault'),
    ('cudaStreamNonBlocking', 'mcStreamNonBlocking'),
    ('cudaStreamLegacy', 'mcStreamAllThreads'),
    ('cudaStreamPerThread', 'mcStreamPerThread'),
]

# Function mappings (cuda* -> mc*)
# These are converted using a pattern, but we list common ones for reference
FUNCTION_MAPPINGS = [
    # Memory management
    ('cudaMalloc', 'mcMalloc'),
    ('cudaMallocPitch', 'mcMallocPitch'),
    ('cudaMallocHost', 'mcMallocHost'),
    ('cudaMallocManaged', 'mcMallocManaged'),
    ('cudaFree', 'mcFree'),
    ('cudaFreeHost', 'mcFreeHost'),
    ('cudaMemcpy', 'mcMemcpy'),
    ('cudaMemcpyAsync', 'mcMemcpyAsync'),
    ('cudaMemcpy2D', 'mcMemcpy2D'),
    ('cudaMemcpy2DAsync', 'mcMemcpy2DAsync'),
    ('cudaMemset', 'mcMemset'),
    ('cudaMemsetAsync', 'mcMemsetAsync'),
    ('cudaMemset2D', 'mcMemset2D'),
    ('cudaMemset2DAsync', 'mcMemset2DAsync'),
    # Device management
    ('cudaGetDevice', 'mcGetDevice'),
    ('cudaSetDevice', 'mcSetDevice'),
    ('cudaGetDeviceCount', 'mcGetDeviceCount'),
    ('cudaGetDeviceProperties', 'mcGetDeviceProperties'),
    ('cudaGetDeviceAttribute', 'mcDeviceGetAttribute'),  # Note: different naming convention
    ('cudaDeviceSynchronize', 'mcDeviceSynchronize'),
    ('cudaDeviceReset', 'mcDeviceReset'),
    # Stream management
    ('cudaStreamCreate', 'mcStreamCreate'),
    ('cudaStreamCreateWithFlags', 'mcStreamCreateWithFlags'),
    ('cudaStreamDestroy', 'mcStreamDestroy'),
    ('cudaStreamSynchronize', 'mcStreamSynchronize'),
    ('cudaStreamWaitEvent', 'mcStreamWaitEvent'),
    ('cudaStreamGetFlags', 'mcStreamGetFlags'),
    ('cudaStreamQuery', 'mcStreamQuery'),
    # Event management
    ('cudaEventCreate', 'mcEventCreate'),
    ('cudaEventCreateWithFlags', 'mcEventCreateWithFlags'),
    ('cudaEventDestroy', 'mcEventDestroy'),
    ('cudaEventRecord', 'mcEventRecord'),
    ('cudaEventSynchronize', 'mcEventSynchronize'),
    ('cudaEventElapsedTime', 'mcEventElapsedTime'),
    ('cudaEventQuery', 'mcEventQuery'),
    # Error handling
    ('cudaGetLastError', 'mcGetLastError'),
    ('cudaPeekAtLastError', 'mcPeekAtLastError'),
    ('cudaGetErrorString', 'mcGetErrorString'),
    ('cudaGetErrorName', 'mcGetErrorName'),
]

# Variable name patterns to convert (cuda prefix in variable names)
# These are common patterns but we use regex for flexibility
VARIABLE_PATTERNS = [
    (r'\bcudaStatus\b', 'mcStatus'),
    (r'\bcudaErr\b', 'mcErr'),
    (r'\bcudaResult\b', 'mcResult'),
    (r'\bcudaRet\b', 'mcRet'),
]

# StreamContext field name mappings (for NppStreamContext/MppStreamContext)
# These convert CUDA-specific field names to MACA equivalents
STREAM_CONTEXT_FIELD_MAPPINGS = [
    ('nCudaDeviceId', 'nMcDeviceId'),
    ('nCudaDevAttrComputeCapabilityMajor', 'nMcDevAttrComputeCapabilityMajor'),
    ('nCudaDevAttrComputeCapabilityMinor', 'nMcDevAttrComputeCapabilityMinor'),
]


def get_all_files(base_dir: Path, include_api: bool = True) -> List[Path]:
    """Get all source files to process."""
    files = []

    # Process source directories
    for process_dir in PROCESS_DIRS:
        dir_path = base_dir / process_dir
        if not dir_path.exists():
            continue
        for root, dirs, filenames in os.walk(dir_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for filename in filenames:
                if any(filename.endswith(ext) for ext in FILE_EXTENSIONS):
                    files.append(Path(root) / filename)

    # Process API directory
    if include_api:
        api_dir = base_dir / API_DIR
        if api_dir.exists():
            for filename in os.listdir(api_dir):
                if any(filename.endswith(ext) for ext in FILE_EXTENSIONS):
                    files.append(api_dir / filename)

    return files


def convert_content(content: str) -> Tuple[str, List[str]]:
    """Convert CUDA API calls to MACA equivalents."""
    changes = []
    original = content

    # Convert headers
    for cuda_header, maca_header in HEADER_MAPPINGS:
        pattern = rf'#include\s*[<"]({re.escape(cuda_header)})[>"]'
        if re.search(pattern, content):
            content = re.sub(pattern, f'#include <{maca_header}>', content)
            changes.append(f'Header: {cuda_header} -> {maca_header}')

    # Convert types (must be done before functions to avoid partial matches)
    for cuda_type, maca_type in TYPE_MAPPINGS:
        pattern = rf'\b{re.escape(cuda_type)}\b'
        if re.search(pattern, content):
            content = re.sub(pattern, maca_type, content)
            changes.append(f'Type: {cuda_type} -> {maca_type}')

    # Convert error codes
    for cuda_err, maca_err in ERROR_MAPPINGS:
        pattern = rf'\b{re.escape(cuda_err)}\b'
        if re.search(pattern, content):
            content = re.sub(pattern, maca_err, content)
            changes.append(f'Error: {cuda_err} -> {maca_err}')

    # Convert memcpy directions
    for cuda_dir, maca_dir in MEMCPY_DIRECTION_MAPPINGS:
        pattern = rf'\b{re.escape(cuda_dir)}\b'
        if re.search(pattern, content):
            content = re.sub(pattern, maca_dir, content)
            changes.append(f'MemcpyKind: {cuda_dir} -> {maca_dir}')

    # Convert stream flags
    for cuda_flag, maca_flag in STREAM_FLAG_MAPPINGS:
        pattern = rf'\b{re.escape(cuda_flag)}\b'
        if re.search(pattern, content):
            content = re.sub(pattern, maca_flag, content)
            changes.append(f'StreamFlag: {cuda_flag} -> {maca_flag}')

    # Convert functions
    for cuda_func, maca_func in FUNCTION_MAPPINGS:
        pattern = rf'\b{re.escape(cuda_func)}\b'
        if re.search(pattern, content):
            content = re.sub(pattern, maca_func, content)
            changes.append(f'Function: {cuda_func} -> {maca_func}')

    # Convert common variable name patterns
    for pattern, replacement in VARIABLE_PATTERNS:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            changes.append(f'Variable: {pattern} -> {replacement}')

    # Convert StreamContext field names (nCudaDeviceId -> nMcDeviceId, etc.)
    for cuda_field, maca_field in STREAM_CONTEXT_FIELD_MAPPINGS:
        pattern = rf'\b{re.escape(cuda_field)}\b'
        if re.search(pattern, content):
            content = re.sub(pattern, maca_field, content)
            changes.append(f'Field: {cuda_field} -> {maca_field}')

    # Generic pattern: convert any remaining cuda* function calls to mc*
    # This catches functions not explicitly listed
    def replace_cuda_func(match):
        func_name = match.group(0)
        if func_name.startswith('cuda'):
            new_name = 'mc' + func_name[4:]
            changes.append(f'Function (auto): {func_name} -> {new_name}')
            return new_name
        return func_name

    # Match cuda followed by capital letter (function naming convention)
    content = re.sub(r'\bcuda([A-Z][a-zA-Z0-9_]*)\b',
                     lambda m: 'mc' + m.group(1),
                     content)

    return content, changes


def create_backup():
    """Create backup of files before conversion."""
    print(f"Creating backup in {BACKUP_DIR}")

    if BACKUP_DIR.exists():
        print(f"  Backup already exists. Remove {BACKUP_DIR} first if you want a fresh backup.")
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
        print(f"Error: Backup directory not found: {BACKUP_DIR}")
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
    print("Checking for CUDA -> MACA conversions...")
    print()

    files = get_all_files(PROJECT_DIR)
    total_changes = 0
    files_with_changes = 0
    api_files_with_changes = 0
    api_dir = PROJECT_DIR / API_DIR

    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
            continue

        _, changes = convert_content(content)

        if changes:
            files_with_changes += 1
            if file_path.parent == api_dir:
                api_files_with_changes += 1
            rel_path = file_path.relative_to(PROJECT_DIR)
            print(f"{rel_path}:")

            # Deduplicate and count changes
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
    print(f"  - Source files: {files_with_changes - api_files_with_changes}")
    print(f"  - API headers: {api_files_with_changes}")
    return total_changes > 0


def apply_changes():
    """Apply CUDA to MACA conversion."""
    print("Applying CUDA -> MACA conversion...")
    print()

    files = get_all_files(PROJECT_DIR)
    total_changes = 0
    files_modified = 0
    api_files_modified = 0
    api_dir = PROJECT_DIR / API_DIR

    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
            continue

        new_content, changes = convert_content(content)

        if changes and new_content != content:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                files_modified += 1
                if file_path.parent == api_dir:
                    api_files_modified += 1
                rel_path = file_path.relative_to(PROJECT_DIR)
                print(f"  Modified: {rel_path} ({len(changes)} changes)")
                total_changes += len(changes)
            except Exception as e:
                print(f"  Error writing {file_path}: {e}")

    print()
    print(f"Conversion complete: {total_changes} changes in {files_modified} files")
    print(f"  - Source files: {files_modified - api_files_modified}")
    print(f"  - API headers: {api_files_modified}")
    return files_modified > 0


def main():
    parser = argparse.ArgumentParser(
        description='Convert CUDA Runtime API to MACA Runtime API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 convert_cuda_to_maca.py --check     # Preview changes
  python3 convert_cuda_to_maca.py --backup    # Create backup first
  python3 convert_cuda_to_maca.py --apply     # Apply conversion
  python3 convert_cuda_to_maca.py --restore   # Restore from backup
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true',
                       help='Preview changes without applying')
    group.add_argument('--backup', action='store_true',
                       help='Create backup of source files')
    group.add_argument('--apply', action='store_true',
                       help='Apply CUDA to MACA conversion')
    group.add_argument('--restore', action='store_true',
                       help='Restore files from backup')

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
