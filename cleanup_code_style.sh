#!/bin/bash

# Code style cleanup script
# Removes AI-style comments and vendor-specific naming
# IMPORTANT: Does NOT modify API header files

set -e

echo "=== Code Style Cleanup Script ==="
echo "This script will:"
echo "1. Remove AI-style verbose comments"
echo "2. Replace vendor-specific naming with generic terms"
echo "3. Simplify comment structure"
echo "4. Preserve API header files (no changes)"

# Define directories to process (excluding API directory)
DIRS_TO_PROCESS=(
    "src"
    "test"
)

# File extensions to process
FILE_EXTENSIONS=("*.cpp" "*.cu" "*.h")

# Backup original files
echo "Creating backup..."
mkdir -p backup_$(date +%Y%m%d_%H%M%S)
cp -r src test backup_$(date +%Y%m%d_%H%M%S)/

echo "Starting cleanup..."

# Function to process files
process_files() {
    local dir=$1
    echo "Processing directory: $dir"
    
    for ext in "${FILE_EXTENSIONS[@]}"; do
        find "$dir" -name "$ext" -type f | while read -r file; do
            echo "  Processing: $file"
            
            # Skip if it's in API directory (safety check)
            if [[ "$file" == *"/API/"* ]]; then
                echo "    SKIPPED (API file)"
                continue
            fi
            
            # Create temp file for modifications
            temp_file=$(mktemp)
            
            # Process the file
            sed -e 's/_cuda(/_impl(/g' \
                -e 's/CUDA/GPU/g' \
                -e 's/NVIDIA/vendor/g' \
                -e 's/nvidia/vendor/g' \
                -e 's/Nvidia/Vendor/g' \
                -e '/^\/\*\*/,/\*\//c\
// Implementation file' \
                -e 's/\/\/ Helper function/\/\/ Function/g' \
                -e 's/\/\/ Forward declarations for CUDA kernels/\/\/ Kernel declarations/g' \
                -e 's/\/\/ Forward declarations for .* kernels/\/\/ Kernel declarations/g' \
                -e 's/\/\/ CUDA kernel/\/\/ Kernel/g' \
                -e 's/\/\/ GPU kernel/\/\/ Kernel/g' \
                -e 's/\/\*\*$/\/\//g' \
                -e 's/^ \* / \/\/ /g' \
                -e 's/^ \*$/\/\//g' \
                -e 's/参数验证/Parameter validation/g' \
                -e 's/\/\/ 参数验证/\/\/ Parameter validation/g' \
                "$file" > "$temp_file"
            
            # Replace original file if changes were made
            if ! cmp -s "$file" "$temp_file"; then
                mv "$temp_file" "$file"
                echo "    MODIFIED"
            else
                rm "$temp_file"
                echo "    NO CHANGES"
            fi
        done
    done
}

# Process each directory
for dir in "${DIRS_TO_PROCESS[@]}"; do
    if [ -d "$dir" ]; then
        process_files "$dir"
    else
        echo "Warning: Directory $dir not found"
    fi
done

# Additional cleanup for specific patterns
echo "Applying additional cleanup..."

# Remove AI-style file headers in test files
find test -name "*.cpp" -type f -exec sed -i '
/^\/\*\*$/,/^\*\/$/{
    /^\/\*\*$/c\
// Test file
    /^\*\/$/d
    /^ \* @file/d
    /^ \* @brief/d
    /^ \* Test.*cover/d
    /^ \* Tests.*operations/d
    /^ \* -.*API/d
    /^ \*$/d
}' {} \;

# Clean up comment formatting
find src test -name "*.cpp" -o -name "*.cu" -o -name "*.h" | while read -r file; do
    if [[ "$file" != *"/API/"* ]]; then
        # Remove multiple consecutive comment lines
        sed -i '/^\/\/ *$/N;/^\/\/ *\n\/\/ *$/d' "$file"
        
        # Fix spacing in comments
        sed -i 's/\/\/  */\/\/ /g' "$file"
        
        # Remove empty comment blocks
        sed -i '/^\/\/ *$/d' "$file"
    fi
done

echo "Cleanup completed successfully!"
echo ""
echo "Summary of changes:"
echo "- Replaced '_cuda(' with '_impl('"
echo "- Replaced 'CUDA/NVIDIA' with generic terms"
echo "- Simplified comment structure"
echo "- Removed AI-style verbose comments"
echo "- Converted Chinese comments to English"
echo ""
echo "To verify changes, run: git diff"
echo "To build and test: ./build.sh && cd build && ./unit_tests"