/**
 * @file nppcore_unimplemented.cpp
 * @brief NPP Core未实现函数的空实现
 * 
 * 为测试中调用但尚未实现的核心函数提供空实现，
 * 返回NPP_FUNCTION_NOT_IMPLEMENTED错误码并打印警告信息。
 */

#include "npp.h"
#include <stdio.h>

// ==================== GPU信息函数 (未实现) ====================

NppStatus nppGetGpuComputeCapability(int * pMajor, int * pMinor)
{
    if (pMajor == nullptr || pMinor == nullptr) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    // 打印未实现警告
    fprintf(stderr, "WARNING: nppGetGpuComputeCapability is not implemented in this NPP library build\n");
    
    // 设置默认值以避免未初始化内存
    *pMajor = 0;
    *pMinor = 0;
    
    return NPP_FUNCTION_NOT_IMPLEMENTED;
}

// ==================== 流上下文函数 (未实现的签名) ====================

NppStatus nppSetStreamContext(NppStreamContext nppStreamContext)
{
    // 打印未实现警告
    fprintf(stderr, "WARNING: nppSetStreamContext is not implemented in this NPP library build\n");
    
    return NPP_FUNCTION_NOT_IMPLEMENTED;
}