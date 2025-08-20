#!/bin/bash

# MPP Headers Validation Script
# 验证转换后的头文件语法正确性

set -e

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_status "Validating MPP headers..."

# Check if we're in the right directory
if [[ ! -f "mpp.h" ]]; then
    print_error "mpp.h not found. Please run this script from the mc_api directory."
    exit 1
fi

# Check for basic C++ compilation
print_status "Checking basic compilation..."

# Create a simple test program
cat > test_compilation.cpp << 'EOF'
#include "mpp.h"

int main() {
    MppLibraryVersion version;
    MppStatus status = MPP_NO_ERROR;
    MppiSize size = {640, 480};
    MppiPoint point = {100, 200};
    MppiRect rect = {0, 0, 640, 480};
    
    // Test basic types
    Mpp8u val8u = 255;
    Mpp16s val16s = -1000;
    Mpp32f val32f = 3.14f;
    Mpp64f val64f = 3.141592653589793;
    
    // Test complex types
    Mpp32fc complex32 = {1.0f, 2.0f};
    Mpp64fc complex64 = {1.0, 2.0};
    
    return 0;
}
EOF

# Try to compile with g++
if command -v g++ >/dev/null 2>&1; then
    print_status "Testing with g++..."
    if g++ -I. -c test_compilation.cpp -o test_compilation.o 2>/dev/null; then
        print_success "g++ compilation successful"
        rm -f test_compilation.o
    else
        print_error "g++ compilation failed"
        g++ -I. -c test_compilation.cpp -o test_compilation.o
        exit 1
    fi
else
    print_status "g++ not found, skipping compilation test"
fi

# Clean up
rm -f test_compilation.cpp test_compilation.o

# Check for common conversion issues
print_status "Checking for conversion issues..."

error_count=0

# Check for remaining NPP references
print_status "Checking for remaining NPP references..."
if grep -r "NPP" *.h | grep -v "MPP" >/dev/null; then
    print_error "Found remaining NPP references:"
    grep -r "NPP" *.h | grep -v "MPP" | head -5
    ((error_count++))
else
    print_success "No remaining NPP references found"
fi

# Check for remaining NVIDIA references  
print_status "Checking for remaining NVIDIA references..."
if grep -r "NVIDIA" *.h >/dev/null; then
    print_error "Found remaining NVIDIA references:"
    grep -r "NVIDIA" *.h | head -5
    ((error_count++))
else
    print_success "No remaining NVIDIA references found"
fi

# Check for remaining CUDA references
print_status "Checking for remaining CUDA references..."
if grep -r "CUDA" *.h | grep -v "MACA" >/dev/null; then
    print_error "Found remaining CUDA references:"
    grep -r "CUDA" *.h | grep -v "MACA" | head -5
    ((error_count++))
else
    print_success "No remaining CUDA references found"
fi

# Check for header guard consistency
print_status "Checking header guards..."
for file in *.h; do
    if ! grep -q "#ifndef MC_MPP" "$file" && ! grep -q "#ifndef MC_MPPI" "$file" && ! grep -q "#ifndef MC_MPPS" "$file"; then
        print_error "Header guard issue in $file"
        ((error_count++))
    fi
done

if [[ $error_count -eq 0 ]]; then
    print_success "Header guard check passed"
fi

# Check for include consistency
print_status "Checking include statements..."
include_errors=0
for file in *.h; do
    if grep -q '#include "npp' "$file"; then
        print_error "Found old NPP include in $file"
        ((include_errors++))
    fi
    if grep -q '#include <npp' "$file"; then
        print_error "Found old NPP system include in $file" 
        ((include_errors++))
    fi
done

if [[ $include_errors -eq 0 ]]; then
    print_success "Include statement check passed"
else
    ((error_count += include_errors))
fi

# Check file count
print_status "Checking file count..."
header_count=$(ls *.h | wc -l)
expected_count=21

if [[ $header_count -eq $expected_count ]]; then
    print_success "Found $header_count header files (expected $expected_count)"
else
    print_error "Found $header_count header files, expected $expected_count"
    ((error_count++))
fi

# Summary
echo
print_status "Validation Summary:"
echo "=================="

if [[ $error_count -eq 0 ]]; then
    print_success "All checks passed! Headers appear to be correctly converted."
    echo
    print_status "Header files ready for use:"
    ls -1 *.h | while read file; do
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        printf "  %-40s %8d bytes\n" "$file" "$size"
    done
    echo
    print_success "Validation completed successfully! 🎉"
    exit 0
else
    print_error "Found $error_count issues that need to be addressed."
    exit 1
fi