#!/bin/bash

echo "批量恢复函数间空行..."

# 处理所有C++和CUDA文件，在函数定义间添加空行
find src -name "*.cpp" -o -name "*.cu" | while read file; do
    echo "处理文件: $file"
    
    # 使用临时文件处理
    temp_file=$(mktemp)
    
    # 逐行处理，在关键位置添加空行
    awk '
    BEGIN { prev_was_brace = 0; prev_was_empty = 0 }
    {
        current_line = $0
        
        # 如果当前行是函数定义开始（包含参数和{），并且前一行不是空行
        if (current_line ~ /^[A-Za-z_][A-Za-z0-9_*]*.*\(.*\).*{$/ && !prev_was_empty && NR > 1) {
            print ""
        }
        
        # 如果当前行是简单函数结束}，后面跟着函数定义
        if (prev_was_brace && current_line ~ /^[A-Za-z_][A-Za-z0-9_*]*.*\(/ && !prev_was_empty) {
            print ""
        }
        
        # 如果当前行是extern "C"开始
        if (current_line ~ /^extern "C" {$/ && !prev_was_empty) {
            print ""
        }
        
        print current_line
        
        # 更新状态
        prev_was_brace = (current_line ~ /^}$/)
        prev_was_empty = (current_line ~ /^$/)
    }
    ' "$file" > "$temp_file"
    
    # 替换原文件
    mv "$temp_file" "$file"
done

echo "批量空行恢复完成！"