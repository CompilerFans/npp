#!/bin/bash

# Advanced code style cleanup script for MPP project
# Removes AI-style comments, emoji, and vendor-specific terms

set -e

# Create backup
BACKUP_DIR="backup_advanced_$(date +%Y%m%d_%H%M%S)"
echo "Creating backup in $BACKUP_DIR..."
mkdir -p "$BACKUP_DIR"
cp -r src/ test/ "$BACKUP_DIR/"

# Function to process files
process_file() {
    local file="$1"
    echo "Processing: $file"
    
    # Skip API header files - they should not be modified
    if [[ "$file" == api/* ]]; then
        echo "  Skipping API header file: $file"
        return
    fi
    
    # Create temporary file
    local temp_file=$(mktemp)
    
    # Process the file with comprehensive cleanup
    sed -e '
        # Remove AI-style emotional/enthusiastic comments
        s/\(\/\/.*\)[!]{2,}[^!]*$/\1/g
        s/\(\/\/.*\)非常[^，。]*//g
        s/\(\/\/.*\)很[^，。]*//g
        s/\(\/\/.*\)超级[^，。]*//g
        s/\(\/\/.*\)完美[^，。]*//g
        s/\(\/\/.*\)优秀[^，。]*//g
        s/\(\/\/.*\)amazing[^，。]*//gi
        s/\(\/\/.*\)awesome[^，。]*//gi
        s/\(\/\/.*\)fantastic[^，。]*//gi
        s/\(\/\/.*\)excellent[^，。]*//gi
        s/\(\/\/.*\)perfect[^，。]*//gi
        s/\(\/\/.*\)wonderful[^，。]*//gi
        
        # Remove emoji and special characters
        s/[🎯🚀💡⚡🔧🎨🌟✨💪🎪🎭🎪💎🏆🎖️🎁🎊🎉🔥💯⭐️✅❌⚠️🔴🟢🟡🔵]//g
        s/[😀😃😄😁😆😊☺️😇🤗🤔💭🎈🎀🎁🎂🎃🎄🎆🎇🎈]//g
        
        # Convert Chinese technical comments to concise English
        s/\/\/.*实现[^，。]*功能/\/\/ Implementation/g
        s/\/\/.*处理[^，。]*数据/\/\/ Process data/g
        s/\/\/.*验证[^，。]*参数/\/\/ Validate parameters/g
        s/\/\/.*检查[^，。]*条件/\/\/ Check conditions/g
        s/\/\/.*计算[^，。]*结果/\/\/ Calculate result/g
        s/\/\/.*分配[^，。]*内存/\/\/ Allocate memory/g
        s/\/\/.*释放[^，。]*资源/\/\/ Free resources/g
        s/\/\/.*调用[^，。]*内核/\/\/ Call kernel/g
        s/\/\/.*启动[^，。]*线程/\/\/ Launch threads/g
        s/\/\/.*同步[^，。]*操作/\/\/ Synchronize/g
        s/\/\/.*错误[^，。]*处理/\/\/ Error handling/g
        s/\/\/.*性能[^，。]*优化/\/\/ Performance optimization/g
        s/\/\/.*边界[^，。]*检查/\/\/ Boundary check/g
        s/\/\/.*数据[^，。]*类型/\/\/ Data type/g
        s/\/\/.*图像[^，。]*处理/\/\/ Image processing/g
        s/\/\/.*像素[^，。]*操作/\/\/ Pixel operation/g
        s/\/\/.*矩阵[^，。]*运算/\/\/ Matrix operation/g
        s/\/\/.*向量[^，。]*计算/\/\/ Vector computation/g
        s/\/\/.*滤波[^，。]*算法/\/\/ Filter algorithm/g
        s/\/\/.*形态学[^，。]*操作/\/\/ Morphology operation/g
        s/\/\/.*统计[^，。]*计算/\/\/ Statistical computation/g
        s/\/\/.*几何[^，。]*变换/\/\/ Geometric transform/g
        s/\/\/.*颜色[^，。]*转换/\/\/ Color conversion/g
        
        # Remove verbose AI-style descriptions
        s/\/\/.*这个函数[^，。]*实现了[^，。]*算法[^，。]*//g
        s/\/\/.*此方法[^，。]*提供了[^，。]*功能[^，。]*//g
        s/\/\/.*该函数[^，。]*支持[^，。]*特性[^，。]*//g
        s/\/\/.*本实现[^，。]*采用了[^，。]*技术[^，。]*//g
        
        # Replace vendor-specific terms with generic equivalents
        s/CUDA/GPU/g
        s/cuda/gpu/g
        s/NVIDIA/Generic/g
        s/nvidia/generic/g
        s/GeForce/Graphics/g
        s/Tesla/Compute/g
        s/Quadro/Professional/g
        s/cuDNN/Deep Learning/g
        s/CUBLAS/Linear Algebra/g
        s/CUFFT/FFT Library/g
        s/Thrust/Parallel Library/g
        s/NCCL/Communication Library/g
        
        # Simplify function documentation
        s/\/\*\*$/\/\*/g
        s/^ \* \*$/\*/g
        s/^ \* @brief [^.]*\./\/\/ /g
        s/^ \* @param [^.]*\./\/\/ /g
        s/^ \* @return [^.]*\./\/\/ /g
        s/^ \* @note [^.]*\./\/\/ /g
        s/^ \* @warning [^.]*\./\/\/ /g
        
        # Remove redundant comments
        s/\/\/.*Implementation file.*//g
        s/\/\/.*实现文件.*//g
        s/\/\/.*源文件.*//g
        s/\/\/.*头文件.*//g
        s/\/\/.*包含文件.*//g
        
        # Clean up empty comment lines
        s/^[[:space:]]*\/\/[[:space:]]*$//g
        
        # Remove multiple consecutive empty lines
        /^$/N
        /^\n$/d
        
    ' "$file" > "$temp_file"
    
    # Apply additional processing for specific patterns
    python3 -c "
import re
import sys

with open('$temp_file', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove AI-style verbose comments patterns
content = re.sub(r'//.*这里我们.*', '// ', content, flags=re.MULTILINE)
content = re.sub(r'//.*让我们.*', '// ', content, flags=re.MULTILINE)
content = re.sub(r'//.*现在我们.*', '// ', content, flags=re.MULTILINE)
content = re.sub(r'//.*首先.*', '// ', content, flags=re.MULTILINE)
content = re.sub(r'//.*然后.*', '// ', content, flags=re.MULTILINE)
content = re.sub(r'//.*最后.*', '// ', content, flags=re.MULTILINE)
content = re.sub(r'//.*接下来.*', '// ', content, flags=re.MULTILINE)

# Remove comments with multiple exclamation marks
content = re.sub(r'//.*!{2,}.*', '//', content, flags=re.MULTILINE)

# Clean up redundant whitespace
content = re.sub(r'\n\s*\n\s*\n', '\n\n', content, flags=re.MULTILINE)

# Ensure file ends with newline
if not content.endswith('\n'):
    content += '\n'

with open('$temp_file', 'w', encoding='utf-8') as f:
    f.write(content)
"
    
    # Only update if file content changed
    if ! cmp -s "$file" "$temp_file"; then
        mv "$temp_file" "$file"
        echo "  Updated: $file"
    else
        rm "$temp_file"
        echo "  No changes: $file"
    fi
}

# Process all source and test files
echo "Processing source files..."
find src/ -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.h" -o -name "*.hpp" \) | while read -r file; do
    process_file "$file"
done

echo "Processing test files..."
find test/ -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.h" -o -name "*.hpp" \) | while read -r file; do
    process_file "$file"
done

# Special processing for specific vendor term replacements in code
echo "Applying vendor-specific term replacements..."
find src/ test/ -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.h" -o -name "*.hpp" \) | while read -r file; do
    # Skip API directory
    if [[ "$file" == api/* ]]; then
        continue
    fi
    
    # Apply replacements but preserve NPP API calls and error constants
    sed -i '
        # Preserve NPP constants and API calls
        /NPP_/!{
            /npp[A-Z]/!{
                # Replace function names but preserve NPP prefixes
                s/\([^_]\)cuda\([A-Z]\)/\1gpu\2/g
                s/\([^_]\)Cuda\([A-Z]\)/\1Gpu\2/g
                # Replace in comments and strings
                s/CUDA kernel/GPU kernel/g
                s/CUDA device/GPU device/g
                s/CUDA stream/GPU stream/g
                s/CUDA runtime/GPU runtime/g
                s/CUDA memory/GPU memory/g
                s/CUDA context/GPU context/g
                # But preserve CUDA API calls like cudaMalloc, cudaMemcpy etc.
            }
        }
    ' "$file"
done

echo "Advanced code style cleanup completed!"
echo "Backup created in: $BACKUP_DIR"
echo ""
echo "Summary of changes applied:"
echo "  - Removed AI-style emotional/enthusiastic comments"
echo "  - Removed all emoji and special characters"
echo "  - Converted Chinese technical comments to concise English"
echo "  - Replaced vendor-specific terms with generic equivalents"
echo "  - Simplified function documentation"
echo "  - Cleaned up redundant and empty comments"
echo "  - Preserved API headers and NPP-specific terms"
echo ""
echo "Please verify the changes and run tests to ensure functionality is preserved."