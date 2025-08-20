#!/bin/bash

# NPP to MPP Conversion Script
# 一键转换整个NPP工程为MetaX MPP工程
# Author: Claude Code Assistant
# Date: $(date)

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Usage information
show_usage() {
    cat << EOF
NPP to MPP Conversion Script

Usage: $0 [options] <source_directory> [target_directory]

Options:
    -h, --help          Show this help message
    -b, --backup        Create backup of original files (default: true)
    -t, --test-only     Only convert test files
    -s, --source-only   Only convert source files
    -f, --headers-only  Only convert header files
    -d, --dry-run       Show what would be converted without making changes
    --no-backup         Do not create backup files

Arguments:
    source_directory    Directory containing NPP source code
    target_directory    Optional: Target directory for converted code (default: source_directory)

Examples:
    $0 /path/to/npp/project                    # Convert entire project in-place
    $0 /path/to/npp/project /path/to/mpp/      # Convert to new directory
    $0 --headers-only /path/to/npp/api/        # Convert only header files
    $0 --dry-run /path/to/npp/project          # Preview changes without applying

EOF
}

# Default options
CREATE_BACKUP=true
CONVERT_TESTS=true
CONVERT_SOURCE=true  
CONVERT_HEADERS=true
DRY_RUN=false
SOURCE_DIR=""
TARGET_DIR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -b|--backup)
            CREATE_BACKUP=true
            ;;
        --no-backup)
            CREATE_BACKUP=false
            ;;
        -t|--test-only)
            CONVERT_TESTS=true
            CONVERT_SOURCE=false
            CONVERT_HEADERS=false
            ;;
        -s|--source-only)
            CONVERT_TESTS=false
            CONVERT_SOURCE=true
            CONVERT_HEADERS=false
            ;;
        -f|--headers-only)
            CONVERT_TESTS=false
            CONVERT_SOURCE=false
            CONVERT_HEADERS=true
            ;;
        -d|--dry-run)
            DRY_RUN=true
            ;;
        -*)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            if [[ -z "$SOURCE_DIR" ]]; then
                SOURCE_DIR="$1"
            elif [[ -z "$TARGET_DIR" ]]; then
                TARGET_DIR="$1"
            else
                print_error "Too many arguments"
                show_usage
                exit 1
            fi
            ;;
    esac
    shift
done

# Validate arguments
if [[ -z "$SOURCE_DIR" ]]; then
    print_error "Source directory is required"
    show_usage
    exit 1
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
    print_error "Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

if [[ -z "$TARGET_DIR" ]]; then
    TARGET_DIR="$SOURCE_DIR"
fi

# Ensure target directory exists
if [[ "$TARGET_DIR" != "$SOURCE_DIR" ]]; then
    mkdir -p "$TARGET_DIR"
    # Copy source to target if different
    if [[ "$DRY_RUN" == "false" ]]; then
        print_status "Copying source to target directory..."
        cp -r "$SOURCE_DIR"/* "$TARGET_DIR"/
    fi
fi

print_status "NPP to MPP Conversion Script"
print_status "Source Directory: $SOURCE_DIR"
print_status "Target Directory: $TARGET_DIR"
print_status "Create Backup: $CREATE_BACKUP"
print_status "Dry Run: $DRY_RUN"
print_status ""

# File extension patterns for different types
HEADER_PATTERNS="*.h *.hpp *.hxx"
SOURCE_PATTERNS="*.cpp *.cxx *.cc *.cu"
CMAKE_PATTERNS="CMakeLists.txt *.cmake"
TEST_PATTERNS="test_*.cpp test_*.cc test_*.cu *_test.cpp *_test.cc *_test.cu"

# Conversion function for file content
convert_file_content() {
    local file="$1"
    local temp_file="${file}.tmp"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "Would convert: $file"
        return 0
    fi
    
    print_status "Converting: $file"
    
    # Create backup if requested
    if [[ "$CREATE_BACKUP" == "true" ]]; then
        cp "$file" "${file}.bak"
    fi
    
    # Perform all conversions
    sed -e 's/\bNPP_/MPP_/g' \
        -e 's/\bnpp\b/mpp/g' \
        -e 's/\bNpp/Mpp/g' \
        -e 's/\bnppi\b/mppi/g' \
        -e 's/\bNppi/Mppi/g' \
        -e 's/\bnpps\b/mpps/g' \
        -e 's/\bNpps/Mpps/g' \
        -e 's/\bNVIDIA/MetaX/g' \
        -e 's/\bnVidia/MetaX/g' \
        -e 's/\bCUDA/MACA/g' \
        -e 's/\bcuda/maca/g' \
        -e 's/\bCuda/Maca/g' \
        -e 's/#include "npp\.h"/#include "mpp.h"/g' \
        -e 's/#include "nppcore\.h"/#include "mppcore.h"/g' \
        -e 's/#include "nppdefs\.h"/#include "mppdefs.h"/g' \
        -e 's/#include "nppi\.h"/#include "mppi.h"/g' \
        -e 's/#include "npps\.h"/#include "mpps.h"/g' \
        -e 's/#include <npp\.h>/#include <mpp.h>/g' \
        -e 's/#include <nppcore\.h>/#include <mppcore.h>/g' \
        -e 's/#include <nppdefs\.h>/#include <mppdefs.h>/g' \
        -e 's/#include <nppi\.h>/#include <mppi.h>/g' \
        -e 's/#include <npps\.h>/#include <mpps.h>/g' \
        -e 's/\bNPP_H/MPP_H/g' \
        -e 's/\bNPPCORE_H/MPPCORE_H/g' \
        -e 's/\bNPPDEFS_H/MPPDEFS_H/g' \
        -e 's/\bNPPI_H/MPPI_H/g' \
        -e 's/\bNPPS_H/MPPS_H/g' \
        -e 's/MC_NPP_/MC_MPP_/g' \
        -e 's/MC_NPPI_/MC_MPPI_/g' \
        -e 's/MC_NPPS_/MC_MPPS_/g' \
        -e 's/nppi_/mppi_/g' \
        -e 's/npps_/mpps_/g' \
        -e 's/NPP library/MPP library/g' \
        -e 's/NPP Library/MPP Library/g' \
        -e 's/nppial library/mppial library/g' \
        -e 's/nppist library/mppist library/g' \
        -e 's/npps library/mpps library/g' \
        -e 's/nppial/mppial/g' \
        -e 's/nppist/mppist/g' \
        -e 's/mcStreamSynchronize/macaStreamSynchronize/g' \
        -e 's/mcDeviceSynchronize/macaDeviceSynchronize/g' \
        -e 's/mcStream_t/macaStream_t/g' \
        -e 's/MetaX CUDA/MetaX MACA/g' \
        -e 's/\bCUDA_/MACA_/g' \
        -e 's/cuda_runtime/maca_runtime/g' \
        -e 's/cudaError/macaError/g' \
        -e 's/cudaSuccess/macaSuccess/g' \
        -e 's/cudaMemcpy/macaMemcpy/g' \
        -e 's/mppStreamCtx/mppStreamCtx/g' \
        -e 's/nppStreamCtx/mppStreamCtx/g' \
        -e 's/\bnppGet/mppGet/g' \
        -e 's/\bnppSet/mppSet/g' \
        -e 's/\bnppiAdd/mppiAdd/g' \
        -e 's/\bnppiSub/mppiSub/g' \
        -e 's/\bnppiMul/mppiMul/g' \
        -e 's/\bnppiDiv/mppiDiv/g' \
        -e 's/\bnppiAbs/mppiAbs/g' \
        -e 's/\bnppiSqr/mppiSqr/g' \
        -e 's/\bnppiSqrt/mppiSqrt/g' \
        -e 's/\bnppiLn/mppiLn/g' \
        -e 's/\bnppiExp/mppiExp/g' \
        -e 's/\bnppi\([A-Z]\)/mppi\1/g' \
        -e 's/\bnpps\([A-Z]\)/mpps\1/g' \
        "$file" > "$temp_file"
    
    # Replace original file
    mv "$temp_file" "$file"
}

# Rename files function
rename_files() {
    local dir="$1"
    local pattern="$2"
    
    find "$dir" -name "$pattern" | while read -r file; do
        local dirname=$(dirname "$file")
        local basename=$(basename "$file")
        local extension="${basename##*.}"
        local filename="${basename%.*}"
        
        # Convert filename
        local new_filename=""
        if [[ "$filename" == npp* ]]; then
            new_filename="mpp${filename#npp}"
        elif [[ "$filename" == *npp* ]]; then
            new_filename="${filename/npp/mpp}"
        else
            continue  # Skip if no npp in filename
        fi
        
        local new_file="$dirname/${new_filename}.${extension}"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            print_status "Would rename: $file -> $new_file"
        else
            print_status "Renaming: $file -> $new_file"
            mv "$file" "$new_file"
        fi
    done
}

# Process files by type
process_files() {
    local dir="$1"
    local patterns="$2"
    local description="$3"
    
    print_status "Processing $description in $dir..."
    
    local file_count=0
    for pattern in $patterns; do
        find "$dir" -name "$pattern" | while read -r file; do
            if [[ -f "$file" ]]; then
                convert_file_content "$file"
                ((file_count++))
            fi
        done
    done
    
    print_success "Completed $description processing"
}

# Main conversion process
main() {
    cd "$TARGET_DIR"
    
    print_status "Starting NPP to MPP conversion..."
    
    # Rename files first
    if [[ "$CONVERT_HEADERS" == "true" ]]; then
        print_status "Renaming header files..."
        for pattern in $HEADER_PATTERNS; do
            rename_files "." "$pattern"
        done
    fi
    
    if [[ "$CONVERT_SOURCE" == "true" ]]; then
        print_status "Renaming source files..."
        for pattern in $SOURCE_PATTERNS; do
            rename_files "." "$pattern"
        done
    fi
    
    # Convert file contents
    if [[ "$CONVERT_HEADERS" == "true" ]]; then
        process_files "." "$HEADER_PATTERNS" "header files"
    fi
    
    if [[ "$CONVERT_SOURCE" == "true" ]]; then
        process_files "." "$SOURCE_PATTERNS" "source files"
    fi
    
    if [[ "$CONVERT_TESTS" == "true" ]]; then
        process_files "." "$TEST_PATTERNS" "test files"
    fi
    
    # Convert CMake files
    print_status "Processing build files..."
    process_files "." "$CMAKE_PATTERNS" "CMake files"
    
    print_success "NPP to MPP conversion completed successfully!"
    
    if [[ "$CREATE_BACKUP" == "true" && "$DRY_RUN" == "false" ]]; then
        print_warning "Backup files created with .bak extension"
        print_warning "You can remove them with: find . -name '*.bak' -delete"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_warning "This was a dry run. No files were actually modified."
        print_warning "Run without --dry-run to apply changes."
    fi
}

# Trap to clean up on exit
cleanup() {
    if [[ -f "$temp_file" ]]; then
        rm -f "$temp_file"
    fi
}
trap cleanup EXIT

# Run main function
main

print_success "All done! 🎉"