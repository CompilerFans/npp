#!/usr/bin/env python3
"""
Convert NPP to MPP (Generic Version)

This script uses generic prefix/suffix replacement rules to minimize hardcoded mappings.
Only exceptions to the general patterns are explicitly listed.

General Rules:
- npp* -> mpp* (function names, lowercase prefix)
- Npp* -> Mpp* (type names, capitalized prefix)
- NPP* -> MPP* (macros, constants, uppercase prefix)
- nppi* -> mppi* (image processing)
- npps* -> mpps* (signal processing)
- NPPI* -> MPPI*, NPPS* -> MPPS*
- Nppi* -> Mppi*, Npps* -> Mpps*

File/Directory naming:
- npp_*.h -> mpp_*.h
- nppi_* -> mppi_*
- npps_* -> mpps_*

Exceptions: Any mapping that doesn't follow the simple prefix replacement pattern.

Usage:
  python3 convert_npp2mpp.py --preview     # Preview changes
  python3 convert_npp2mpp.py --backup      # Create backup
  python3 convert_npp2mpp.py --apply       # Apply conversion
  python3 convert_npp2mpp.py --restore     # Restore from backup
"""

import os
import re
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Set

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent

# Directories
PROCESS_DIRS = ['src', 'test', 'cmake', 'examples']
API_DIR = 'API'
API_VERSION_DIRS = ['API-11.4', 'API-12.2', 'API-12.8']
SKIP_DIRS = ['build', 'build-nvidia', '.git', 'third_party', 'wrapper', '.npp_backup', '.cuda_backup', 'maca_test', 'cuda-npp-samples', 'cuda-npp-samples-11.6'] + API_VERSION_DIRS
ROOT_FILES = ['CMakeLists.txt', 'build.sh', 'build_maca.sh']
CODE_EXTENSIONS = {'.cpp', '.cu', '.h', '.hpp', '.cuh', '.c'}
BUILD_EXTENSIONS = {'.cmake', '.txt', '.sh', '.md'}
ALL_EXTENSIONS = CODE_EXTENSIONS | BUILD_EXTENSIONS
BACKUP_DIR = '.npp_backup'


# =============================================================================
# Exception Mappings (patterns that don't follow general rules)
# =============================================================================

# Currently no exceptions - all NPP->MPP mappings follow the simple prefix rule
# Add exceptions here if any are discovered
EXCEPTION_MAPPINGS = [
    # Example: ('nppSpecialFunction', 'mppDifferentName'),
]


# =============================================================================
# Generic Prefix Replacement Rules
# =============================================================================

class GenericNppReplacer:
    """
    Handles generic prefix-based replacements for NPP -> MPP conversion.

    The key insight is that NPP naming is consistent:
    - Lowercase: npp, nppi, npps -> mpp, mppi, mpps
    - Capitalized: Npp, Nppi, Npps -> Mpp, Mppi, Mpps
    - Uppercase: NPP, NPPI, NPPS -> MPP, MPPI, MPPS
    """

    def __init__(self):
        self.exceptions = {old: new for old, new in EXCEPTION_MAPPINGS}
        self.changes: List[str] = []

    def _add_change(self, category: str, old: str, new: str):
        """Record a change for reporting."""
        self.changes.append(f'{category}: {old} -> {new}')

    def replace_in_content(self, content: str) -> str:
        """Apply all generic replacements to content."""
        self.changes = []

        # Apply exceptions first
        for old, new in EXCEPTION_MAPPINGS:
            pattern = rf'\b{re.escape(old)}\b'
            if re.search(pattern, content):
                content = re.sub(pattern, new, content)
                self._add_change('Exception', old, new)

        # Generic prefix replacements
        # Order: longer/more specific prefixes first to avoid partial replacements

        # 1. UPPERCASE macros/constants (NPPI_, NPPS_, NPP_)
        content = self._replace_prefix(content, r'\bNPPI_', 'MPPI_', 'Macro')
        content = self._replace_prefix(content, r'\bNPPS_', 'MPPS_', 'Macro')
        content = self._replace_prefix(content, r'\bNPP_', 'MPP_', 'Macro')

        # 2. Capitalized types (Nppi*, Npps*, Npp*)
        content = self._replace_pattern(content, r'\bNppi([A-Z][a-zA-Z0-9_]*)\b', 'Mppi', 'Type')
        content = self._replace_pattern(content, r'\bNpps([A-Z][a-zA-Z0-9_]*)\b', 'Mpps', 'Type')
        content = self._replace_pattern(content, r'\bNpp([A-Z][a-zA-Z0-9_]*)\b', 'Mpp', 'Type')

        # 3. Basic Npp types (Npp8u, Npp16s, Npp16f_2, etc.)
        # Extended pattern to handle variants like Npp16f_2
        content = self._replace_pattern(content, r'\bNpp(\d+[usfUFS]c?(?:_\d+)?)\b', 'Mpp', 'Type')

        # 4. Lowercase functions (nppi*, npps*, npp*)
        content = self._replace_pattern(content, r'\bnppi([A-Z0-9][a-zA-Z0-9_]*)\b', 'mppi', 'Function')
        content = self._replace_pattern(content, r'\bnpps([A-Z0-9][a-zA-Z0-9_]*)\b', 'mpps', 'Function')
        content = self._replace_pattern(content, r'\bnpp([A-Z][a-zA-Z0-9_]*)\b', 'mpp', 'Function')

        # 5. Library names (nppc, nppial, etc. - lowercase without following capital)
        lib_mappings = [
            (r'\bnppc\b', 'mppc'),
            (r'\bnppial\b', 'mppial'),
            (r'\bnppicc\b', 'mppicc'),
            (r'\bnppidei\b', 'mppidei'),
            (r'\bnppif\b', 'mppif'),
            (r'\bnppig\b', 'mppig'),
            (r'\bnppim\b', 'mppim'),
            (r'\bnppist\b', 'mppist'),
            (r'\bnppisu\b', 'mppisu'),
            (r'\bnppitc\b', 'mppitc'),
            (r'\bnpps\b', 'mpps'),
        ]
        for pattern, replacement in lib_mappings:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                self._add_change('Library', pattern.strip(r'\b'), replacement)

        # 6. CMake function names (npp_xxx -> mpp_xxx)
        content = self._replace_pattern(content, r'\bnpp_([a-z_]+)\b', 'mpp_', 'CMakeFunc')

        # 7. Include headers
        # Handle: #include <nppi_xxx.h> or #include "nppi_xxx.h"
        content = re.sub(r'#include\s*([<"])nppi_([a-z_]+)\.h([>"])',
                        r'#include \1mppi_\2.h\3', content)
        content = re.sub(r'#include\s*([<"])npps_([a-z_]+)\.h([>"])',
                        r'#include \1mpps_\2.h\3', content)
        content = re.sub(r'#include\s*([<"])npp_([a-z_]+)\.h([>"])',
                        r'#include \1mpp_\2.h\3', content)
        content = re.sub(r'#include\s*([<"])nppi\.h([>"])',
                        r'#include \1mppi.h\2', content)
        content = re.sub(r'#include\s*([<"])npps\.h([>"])',
                        r'#include \1mpps.h\2', content)
        content = re.sub(r'#include\s*([<"])npp\.h([>"])',
                        r'#include \1mpp.h\2', content)
        content = re.sub(r'#include\s*([<"])nppdefs\.h([>"])',
                        r'#include \1mppdefs.h\2', content)
        content = re.sub(r'#include\s*([<"])nppcore\.h([>"])',
                        r'#include \1mppcore.h\2', content)

        # 8. Directory paths in includes/cmake
        content = re.sub(r'/nppi_([a-z_]+)/', r'/mppi_\1/', content)
        content = re.sub(r'/npps_([a-z_]+)/', r'/mpps_\1/', content)
        content = re.sub(r'/nppi/', '/mppi/', content)
        content = re.sub(r'/npps/', '/mpps/', content)
        content = re.sub(r'/nppcore/', '/mppcore/', content)

        # 9. Header guards - handle all NV_NPP* patterns
        content = re.sub(r'\bNV_NPP([A-Z_]*_H)\b', r'MX_MPP\1', content)
        content = re.sub(r'\b_NPP_H_\b', '_MPP_H_', content)
        content = re.sub(r'\b_NPPDEFS_H_\b', '_MPPDEFS_H_', content)

        # 10. Uppercase NPP followed by digits (NPP16F, NPP32F, etc.)
        # Also handle cases inside macro names like MPP_HALF_TO_NPP16F -> MPP_HALF_TO_MPP16F
        content = re.sub(r'\bNPP(\d+[A-Z])\b', r'MPP\1', content)
        content = re.sub(r'_NPP(\d+[A-Z])\b', r'_MPP\1', content)

        # 11. Filenames in comments (\file nppdefs.h, \file nppcore.h, \file npp.h, etc.)
        content = re.sub(r'(\\file\s+)npp([a-z_]*)\.h', r'\1mpp\2.h', content)
        content = re.sub(r'\bnppdefs\.h\b', 'mppdefs.h', content)
        content = re.sub(r'\bnppcore\.h\b', 'mppcore.h', content)
        content = re.sub(r'\bnpp\.h\b', 'mpp.h', content)

        # 12. Doxygen group names (@ingroup nppi, @defgroup core_npp, etc.)
        content = re.sub(r'@ingroup\s+nppi\b', '@ingroup mppi', content)
        content = re.sub(r'@ingroup\s+npps\b', '@ingroup mpps', content)
        content = re.sub(r'@ingroup\s+npp\b', '@ingroup mpp', content)
        content = re.sub(r'\\defgroup\s+([a-z_]*)npp([a-z_]*)\s', r'\\defgroup \1mpp\2 ', content)
        content = re.sub(r'Module LabelCoreNPP', 'Module LabelCoreMPP', content)

        # 13. NPP in documentation/comments - comprehensive patterns
        # Patterns that should be converted
        npp_doc_patterns = [
            (r'\bNPP library\b', 'MPP library'),
            (r'\bNPP function\b', 'MPP function'),
            (r'\bNPP functionality\b', 'MPP functionality'),
            (r'\bNPP primitives\b', 'MPP primitives'),
            (r'\bNPP release\b', 'MPP release'),
            (r'\bNPP releases\b', 'MPP releases'),
            (r'\bNPP stream\b', 'MPP stream'),
            (r'\bNPP Type\b', 'MPP Type'),
            (r'\bNPP Data\b', 'MPP Data'),
            (r'\bNPP Batch\b', 'MPP Batch'),
            (r'\bNPP version\b', 'MPP version'),
            (r'\bNPP CUDA\b', 'MPP MACA'),
            (r'\bNPP managed\b', 'MPP managed'),
            (r'\bNPP enables\b', 'MPP enables'),
            (r'\bNPP maintained\b', 'MPP maintained'),
            (r'\bNPP supports\b', 'MPP supports'),
            (r'\bNPP computes\b', 'MPP computes'),
            (r'\bNPP Image\b', 'MPP Image'),
            (r'\bNPP Signal\b', 'MPP Signal'),
            (r'\bNPP Core\b', 'MPP Core'),
            (r'\bthe NPP\b', 'the MPP'),
            (r'\bby NPP\b', 'by MPP'),
            (r'\bto NPP\b', 'to MPP'),
            (r'\bfor NPP\b', 'for MPP'),
            (r'\bof NPP\b', 'of MPP'),
            (r'\bfrom NPP\b', 'from MPP'),
            (r'\ban NPP\b', 'an MPP'),
            (r'NPP converts', 'MPP converts'),
            (r'NPP only', 'MPP only'),
            (r'NPP does', 'MPP does'),
            (r', NPP ', ', MPP '),
            (r'Basic NPP', 'Basic MPP'),
        ]
        for pattern, replacement in npp_doc_patterns:
            content = re.sub(pattern, replacement, content)

        # 14. Doxygen @defgroup with nppi/npps
        content = re.sub(r'@defgroup\s+nppi\s+NPP\s+Image', '@defgroup mppi MPP Image', content)
        content = re.sub(r'@defgroup\s+nppi\s+MPP\s+Image', '@defgroup mppi MPP Image', content)
        content = re.sub(r'@defgroup\s+mpps\s+NPP\s+Signal', '@defgroup mpps MPP Signal', content)

        # 15. Parameter names containing Npp (pNppStreamContext -> pMppStreamContext)
        content = re.sub(r'\bpNpp([A-Z][a-zA-Z]*)\b', r'pMpp\1', content)

        # 16. Internal include paths
        content = re.sub(r'#include\s*"internal/nppi_', r'#include "internal/mppi_', content)
        content = re.sub(r'#include\s*"internal/npps_', r'#include "internal/mpps_', content)

        return content

    def _replace_prefix(self, content: str, pattern: str, replacement: str, category: str) -> str:
        """Replace prefix pattern in content."""
        matches = re.findall(pattern, content)
        if matches:
            content = re.sub(pattern, replacement, content)
            self._add_change(category, pattern.strip(r'\b'), replacement)
        return content

    def _replace_pattern(self, content: str, pattern: str, new_prefix: str, category: str) -> str:
        """Replace pattern keeping the suffix, using new prefix."""
        def replacer(match):
            suffix = match.group(1)
            old_name = match.group(0)
            new_name = new_prefix + suffix
            self._add_change(category, old_name, new_name)
            return new_name

        return re.sub(pattern, replacer, content)

    def get_changes(self) -> List[str]:
        """Get list of changes made."""
        return self.changes


class NppToMppConverter:
    """Main converter class for NPP to MPP."""

    def __init__(self, project_root: str, dry_run: bool = True, verbose: bool = False):
        self.project_root = Path(project_root)
        self.dry_run = dry_run
        self.verbose = verbose
        self.replacer = GenericNppReplacer()
        self.all_changes: List[Dict] = []

    def should_skip_dir(self, dir_path: Path) -> bool:
        """Check if directory should be skipped."""
        for skip in SKIP_DIRS:
            if skip in dir_path.parts:
                return True
        return False

    def should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed for content replacement."""
        return file_path.suffix.lower() in ALL_EXTENSIONS

    def process_file_content(self, file_path: Path) -> None:
        """Process a single file's content."""
        if not self.should_process_file(file_path):
            return

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"  Warning: Could not read {file_path}: {e}")
            return

        new_content = self.replacer.replace_in_content(content)
        changes = self.replacer.get_changes()

        if new_content != content:
            self.all_changes.append({
                'type': 'content',
                'path': str(file_path),
                'changes': len(changes)
            })
            if self.verbose:
                print(f"  Content: {file_path} ({len(changes)} changes)")

            if not self.dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

    def get_new_filename(self, filename: str) -> str:
        """Get new filename after NPP->MPP replacement."""
        # Generic rules for filenames
        replacements = [
            ('nppi_', 'mppi_'),
            ('npps_', 'mpps_'),
            ('npp_', 'mpp_'),
            ('nppcore', 'mppcore'),
            ('nppdefs', 'mppdefs'),
            ('nppi.', 'mppi.'),
            ('npps.', 'mpps.'),
            ('npp.', 'mpp.'),
        ]
        new_name = filename
        for old, new in replacements:
            new_name = new_name.replace(old, new)
        return new_name

    def get_new_dirname(self, dirname: str) -> str:
        """Get new directory name after NPP->MPP replacement."""
        # Generic rules for directory names
        replacements = [
            ('nppi_', 'mppi_'),
            ('npps_', 'mpps_'),
            ('nppcore', 'mppcore'),
            ('nppi', 'mppi'),
            ('npps', 'mpps'),
        ]
        new_name = dirname
        for old, new in replacements:
            if old in new_name:
                new_name = new_name.replace(old, new)
                break  # Only apply first match
        return new_name

    def collect_renames(self, dir_path: Path) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
        """Collect all files and directories that need renaming."""
        file_renames = []
        dir_renames = []

        for root, dirs, files in os.walk(dir_path, topdown=False):
            root_path = Path(root)
            if self.should_skip_dir(root_path):
                continue

            # File renames
            for filename in files:
                new_name = self.get_new_filename(filename)
                if new_name != filename:
                    old_path = root_path / filename
                    new_path = root_path / new_name
                    file_renames.append((old_path, new_path))

            # Directory renames
            for dirname in dirs:
                if dirname in SKIP_DIRS:
                    continue
                new_name = self.get_new_dirname(dirname)
                if new_name != dirname:
                    old_path = root_path / dirname
                    new_path = root_path / new_name
                    dir_renames.append((old_path, new_path))

        return file_renames, dir_renames

    def process_api_directory(self) -> None:
        """Process API directory."""
        api_dir = self.project_root / API_DIR

        if api_dir.is_symlink():
            link_target = os.readlink(api_dir)
            api_dir = self.project_root / link_target
        if not api_dir.exists():
            return

        print(f"  Processing API: {api_dir.name}/")

        # Process content
        for header_file in api_dir.glob('*.h'):
            self.process_file_content(header_file)

        # Collect and apply renames
        for header_file in api_dir.glob('*.h'):
            new_name = self.get_new_filename(header_file.name)
            if new_name != header_file.name:
                new_path = api_dir / new_name
                self.all_changes.append({
                    'type': 'api_rename',
                    'old': str(header_file),
                    'new': str(new_path)
                })
                if self.verbose:
                    print(f"    Header: {header_file.name} -> {new_name}")
                if not self.dry_run:
                    if new_path.exists():
                        new_path.unlink()
                    header_file.rename(new_path)

    def run(self) -> None:
        """Run the conversion process."""
        mode = '(Preview)' if self.dry_run else '(Apply)'
        print(f"\n{'='*60}")
        print(f"NPP to MPP Conversion {mode} - Generic Rules")
        print(f"{'='*60}")
        print(f"Project root: {self.project_root}")
        print()

        # Process file contents
        print("Step 1: Processing file contents...")
        for proc_dir in PROCESS_DIRS:
            dir_path = self.project_root / proc_dir
            if not dir_path.exists():
                continue
            print(f"  Directory: {proc_dir}/")
            for root, dirs, files in os.walk(dir_path):
                root_path = Path(root)
                if self.should_skip_dir(root_path):
                    continue
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                for filename in files:
                    self.process_file_content(root_path / filename)

        # Root files
        for filename in ROOT_FILES:
            file_path = self.project_root / filename
            if file_path.exists():
                self.process_file_content(file_path)

        # API directory
        print("\nStep 2: Processing API headers...")
        self.process_api_directory()

        # Collect renames
        print("\nStep 3: Collecting renames...")
        all_file_renames = []
        all_dir_renames = []
        for proc_dir in PROCESS_DIRS:
            dir_path = self.project_root / proc_dir
            if not dir_path.exists():
                continue
            file_renames, dir_renames = self.collect_renames(dir_path)
            all_file_renames.extend(file_renames)
            all_dir_renames.extend(dir_renames)

        # Apply file renames
        print("\nStep 4: Renaming files...")
        for old_path, new_path in all_file_renames:
            if not old_path.exists():
                continue
            self.all_changes.append({
                'type': 'file_rename',
                'old': str(old_path),
                'new': str(new_path)
            })
            if self.verbose:
                print(f"  File: {old_path.name} -> {new_path.name}")
            if not self.dry_run:
                if new_path.exists():
                    new_path.unlink()
                old_path.rename(new_path)

        # Apply directory renames
        print("\nStep 5: Renaming directories...")
        all_dir_renames.sort(key=lambda x: len(x[0].parts), reverse=True)
        for old_path, new_path in all_dir_renames:
            if not old_path.exists():
                continue
            self.all_changes.append({
                'type': 'dir_rename',
                'old': str(old_path),
                'new': str(new_path)
            })
            if self.verbose:
                print(f"  Dir: {old_path.name} -> {new_path.name}")
            if not self.dry_run:
                if new_path.exists():
                    shutil.rmtree(new_path)
                old_path.rename(new_path)

        # Summary
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        content_changes = [c for c in self.all_changes if c['type'] == 'content']
        file_renames = [c for c in self.all_changes if c['type'] == 'file_rename']
        dir_renames = [c for c in self.all_changes if c['type'] == 'dir_rename']
        api_renames = [c for c in self.all_changes if c['type'] == 'api_rename']

        print(f"  Files with content changes: {len(content_changes)}")
        print(f"  Files to rename: {len(file_renames)}")
        print(f"  Directories to rename: {len(dir_renames)}")
        print(f"  API headers to rename: {len(api_renames)}")
        print()
        print("Generic rules applied:")
        print("  - npp*/Npp*/NPP* -> mpp*/Mpp*/MPP*")
        print("  - nppi*/Nppi*/NPPI* -> mppi*/Mppi*/MPPI*")
        print("  - npps*/Npps*/NPPS* -> mpps*/Mpps*/MPPS*")
        print(f"  - {len(EXCEPTION_MAPPINGS)} exception mappings")

        if self.dry_run:
            print(f"\n{'='*60}")
            print("Preview mode. Use --apply to make changes.")
            print(f"{'='*60}")


def create_backup(project_root: Path) -> str:
    """Create a backup."""
    backup_dir = project_root / BACKUP_DIR
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = backup_dir / timestamp

    print(f"Creating backup: {backup_path}")
    backup_path.mkdir(parents=True, exist_ok=True)

    for proc_dir in PROCESS_DIRS:
        src_dir = project_root / proc_dir
        if src_dir.exists():
            dst_dir = backup_path / proc_dir
            if not dst_dir.exists():
                shutil.copytree(src_dir, dst_dir)
                print(f"  Backed up: {proc_dir}/")

    # Backup API (handle symlink)
    api_dir = project_root / API_DIR
    if api_dir.exists() or api_dir.is_symlink():
        if api_dir.is_symlink():
            link_target = os.readlink(api_dir)
            real_api_dir = project_root / link_target
            if real_api_dir.exists():
                dst_dir = backup_path / link_target
                if not dst_dir.exists():
                    shutil.copytree(real_api_dir, dst_dir)
                symlink_info_file = backup_path / 'API_SYMLINK'
                with open(symlink_info_file, 'w') as f:
                    f.write(link_target)
                print(f"  Backed up: {link_target}/ (API)")
        else:
            dst_dir = backup_path / API_DIR
            if not dst_dir.exists():
                shutil.copytree(api_dir, dst_dir)
                print(f"  Backed up: {API_DIR}/")

    for filename in ROOT_FILES:
        src_file = project_root / filename
        if src_file.exists():
            shutil.copy2(src_file, backup_path / filename)
            print(f"  Backed up: {filename}")

    return str(backup_path)


def restore_backup(project_root: Path, backup_name: str = None) -> None:
    """Restore from backup."""
    backup_dir = project_root / BACKUP_DIR
    if not backup_dir.exists():
        print("No backups found.")
        return

    backups = sorted([d for d in backup_dir.iterdir() if d.is_dir()], reverse=True)
    if not backups:
        print("No backups found.")
        return

    if backup_name:
        backup_path = backup_dir / backup_name
        if not backup_path.exists():
            print(f"Backup '{backup_name}' not found.")
            return
    else:
        backup_path = backups[0]
        print(f"Restoring from: {backup_path.name}")

    symlink_info_file = backup_path / 'API_SYMLINK'
    api_symlink_target = None
    if symlink_info_file.exists():
        with open(symlink_info_file, 'r') as f:
            api_symlink_target = f.read().strip()

    for item in backup_path.iterdir():
        if item.name == 'API_SYMLINK':
            continue
        dst = project_root / item.name
        if item.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(item, dst)
            print(f"  Restored: {item.name}/")
        else:
            shutil.copy2(item, dst)
            print(f"  Restored: {item.name}")


def list_backups(project_root: Path) -> None:
    """List available backups."""
    backup_dir = project_root / BACKUP_DIR
    if not backup_dir.exists():
        print("No backups found.")
        return

    backups = sorted([d for d in backup_dir.iterdir() if d.is_dir()], reverse=True)
    if not backups:
        print("No backups found.")
        return

    print("Available backups:")
    for b in backups:
        print(f"  {b.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert NPP to MPP (Generic Rules)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script uses generic prefix replacement rules.
All npp*/Npp*/NPP* patterns are converted to mpp*/Mpp*/MPP*.

Examples:
  %(prog)s --preview           Preview all changes
  %(prog)s --preview --verbose Detailed preview
  %(prog)s --backup            Create backup
  %(prog)s --apply             Apply changes
  %(prog)s --restore           Restore from backup
        """
    )

    parser.add_argument('--preview', action='store_true', help='Preview changes')
    parser.add_argument('--apply', action='store_true', help='Apply changes')
    parser.add_argument('--backup', action='store_true', help='Create backup')
    parser.add_argument('--restore', nargs='?', const='__latest__', metavar='NAME',
                        help='Restore from backup')
    parser.add_argument('--list-backups', action='store_true', help='List backups')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--project-root', type=str, default=str(PROJECT_DIR),
                        help='Project root directory')

    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()

    if not (project_root / 'src').exists():
        print(f"Error: '{project_root}' is not the project root.")
        sys.exit(1)

    if args.list_backups:
        list_backups(project_root)
        return

    if args.restore:
        backup_name = None if args.restore == '__latest__' else args.restore
        restore_backup(project_root, backup_name)
        return

    if args.backup:
        create_backup(project_root)
        if not args.apply:
            return

    dry_run = not args.apply
    converter = NppToMppConverter(project_root, dry_run=dry_run, verbose=args.verbose)
    converter.run()


if __name__ == '__main__':
    main()
