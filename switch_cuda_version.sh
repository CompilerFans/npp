#!/bin/bash

# CUDA SDK版本切换脚本

set -e

# 显示当前版本
show_current() {
    if [ -L "API" ]; then
        echo "当前版本: $(readlink API)"
    else
        echo "API软链接不存在"
    fi
}

# 列出可用版本
list_versions() {
    echo "可用版本:"
    for dir in API-*; do
        if [ -d "$dir" ]; then
            echo "  ${dir#API-}"
        fi
    done
}

# 切换版本
switch_version() {
    local version="$1"
    local target="API-$version"
    
    if [ ! -d "$target" ]; then
        echo "错误: 版本 $version 不存在"
        echo
        show_help
        exit 1
    fi
    
    rm -f API
    ln -sf "$target" API
    echo "已切换到版本: $version"
}

# 显示帮助信息
show_help() {
    echo "用法: $0 <版本号|current|list>"
    echo "示例: $0 12.8"
    echo
    show_current
    echo
    list_versions
}

# 主逻辑
case "${1:-}" in
    "")
        show_help
        ;;
    "current")
        show_current
        ;;
    "list")
        list_versions
        ;;
    *)
        switch_version "$1"
        ;;
esac