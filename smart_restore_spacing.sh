#!/bin/bash

echo "智能恢复代码块间空行..."

# 处理.h头文件
find src -name "*.h" -exec sed -i '
# 在include块后添加空行（如果后面紧跟非include行）
/^#include.*$/{
N
if (/\n[^#]/) {
    s/\n/\n\n/
}
}

# 在预处理器指令后添加空行
/^#define.*$/{
N
if (/\n[^#]/) {
    s/\n/\n\n/
}
}

# 在extern "C"前后添加空行
/^#ifdef __cplusplus$/{
N
if (/\nextern "C"/) {
    s/\n/\n\n/
}
}
' {} \;

# 处理.cpp和.cu文件的函数间空行
find src -name "*.cpp" -o -name "*.cu" -exec sed -i '
# 在include块后添加空行
/^#include.*$/{
N
if (/\n[^#]/) {
    s/\n/\n\n/
}
}

# 在函数结束}后添加空行（如果后面不是文件结束、另一个}或空行）
/^}$/{
N
if (/\n[A-Za-z_]/ || /\ntemplate/ || /\nstatic/ || /\n\/\// || /\nextern/) {
    s/\n/\n\n/
}
}

# 在static函数前添加空行
/^static.*{$/{
i\

}

# 在template定义前添加空行
/^template.*$/{
i\

}
' {} \;

echo "智能空行恢复完成！"