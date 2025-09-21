#!/bin/bash

echo "恢复代码块间的必要空行..."

# 为头文件恢复适当的空行
find src -name "*.h" -exec sed -i '
# 在include组后添加空行
/^#include.*$/,/^[^#]/s/^[^#]/\n&/
# 在ifndef/define后添加空行  
/^#ifndef.*$/a\\

/^#define.*$/a\\

# 在extern "C"块前后添加空行
/^#ifdef __cplusplus$/i\\

/^#ifdef __cplusplus$/a\\

/^extern "C" {$/a\\

/^}$/i\\

/^#endif$/i\\

' {} \;

# 为实现文件恢复函数间空行
find src -name "*.cpp" -o -name "*.cu" -exec sed -i '
# 在include组后添加空行
/^#include.*$/,/^[^#]/s/^[^#]/\n&/

# 在函数定义前添加空行（匹配返回类型开头的行）
/^[A-Za-z_][A-Za-z0-9_]*\s*[A-Za-z_*][A-Za-z0-9_*]*.*{$/i\\

# 在extern "C"块内函数间添加空行
/^[A-Za-z_][A-Za-z0-9_]*.*Ctx_impl.*{$/i\\

# 在}后面如果不是末尾则添加空行
/^}$/a\\

# 在static函数前添加空行
/^static .*{$/i\\

# 在template定义前添加空行
/^template.*$/i\\

' {} \;

echo "空行恢复完成！"