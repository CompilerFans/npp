/**
 * @file opennpp_loader.h
 * @brief OpenNPP动态库加载器 - 用于对比测试
 * 
 * 提供统一的OpenNPP库动态加载和函数符号解析功能，
 * 实现真正独立的OpenNPP vs NVIDIA NPP对比测试
 */

#ifndef TEST_FRAMEWORK_OPENNPP_LOADER_H
#define TEST_FRAMEWORK_OPENNPP_LOADER_H

#include <dlfcn.h>
#include <string>
#include <unordered_map>
#include <memory>
#include <iostream>
#include "npp.h"

namespace test_framework {

/**
 * @brief OpenNPP库加载器单例类
 * 
 * 负责动态加载OpenNPP库，管理函数指针，
 * 提供统一的接口供对比测试使用
 */
class OpenNppLoader {
public:
    // 获取单例实例
    static OpenNppLoader& getInstance() {
        static OpenNppLoader instance;
        return instance;
    }
    
    // 禁用拷贝和移动
    OpenNppLoader(const OpenNppLoader&) = delete;
    OpenNppLoader& operator=(const OpenNppLoader&) = delete;
    OpenNppLoader(OpenNppLoader&&) = delete;
    OpenNppLoader& operator=(OpenNppLoader&&) = delete;
    
    // 检查OpenNPP是否可用
    bool isAvailable() const { return available_; }
    
    // 获取错误信息
    const std::string& getErrorMessage() const { return errorMessage_; }
    
    // ==================== NPP Core Functions ====================
    
    // Device management
    typedef int (*nppGetGpuNumSMs_func)();
    typedef int (*nppGetMaxThreadsPerBlock_func)();
    typedef int (*nppGetMaxThreadsPerSM_func)();
    typedef const char* (*nppGetGpuName_func)();
    typedef int (*nppGetGpuDeviceProperties_func)(int*, int*, int*);
    
    nppGetGpuNumSMs_func op_nppGetGpuNumSMs = nullptr;
    nppGetMaxThreadsPerBlock_func op_nppGetMaxThreadsPerBlock = nullptr;
    nppGetMaxThreadsPerSM_func op_nppGetMaxThreadsPerSM = nullptr;
    nppGetGpuName_func op_nppGetGpuName = nullptr;
    nppGetGpuDeviceProperties_func op_nppGetGpuDeviceProperties = nullptr;
    
    // Stream management
    typedef cudaStream_t (*nppGetStream_func)();
    typedef NppStatus (*nppSetStream_func)(cudaStream_t);
    typedef NppStatus (*nppGetStreamContext_func)(NppStreamContext*);
    
    nppGetStream_func op_nppGetStream = nullptr;
    nppSetStream_func op_nppSetStream = nullptr;
    nppGetStreamContext_func op_nppGetStreamContext = nullptr;
    
    // ==================== NPPI Support Functions ====================
    
    // Memory allocation - 8u
    typedef Npp8u* (*nppiMalloc_8u_C1_func)(int, int, int*);
    typedef Npp8u* (*nppiMalloc_8u_C2_func)(int, int, int*);
    typedef Npp8u* (*nppiMalloc_8u_C3_func)(int, int, int*);
    typedef Npp8u* (*nppiMalloc_8u_C4_func)(int, int, int*);
    
    nppiMalloc_8u_C1_func op_nppiMalloc_8u_C1 = nullptr;
    nppiMalloc_8u_C2_func op_nppiMalloc_8u_C2 = nullptr;
    nppiMalloc_8u_C3_func op_nppiMalloc_8u_C3 = nullptr;
    nppiMalloc_8u_C4_func op_nppiMalloc_8u_C4 = nullptr;
    
    // Memory allocation - 16u
    typedef Npp16u* (*nppiMalloc_16u_C1_func)(int, int, int*);
    typedef Npp16u* (*nppiMalloc_16u_C2_func)(int, int, int*);
    typedef Npp16u* (*nppiMalloc_16u_C3_func)(int, int, int*);
    typedef Npp16u* (*nppiMalloc_16u_C4_func)(int, int, int*);
    
    nppiMalloc_16u_C1_func op_nppiMalloc_16u_C1 = nullptr;
    nppiMalloc_16u_C2_func op_nppiMalloc_16u_C2 = nullptr;
    nppiMalloc_16u_C3_func op_nppiMalloc_16u_C3 = nullptr;
    nppiMalloc_16u_C4_func op_nppiMalloc_16u_C4 = nullptr;
    
    // Memory allocation - 32f
    typedef Npp32f* (*nppiMalloc_32f_C1_func)(int, int, int*);
    typedef Npp32f* (*nppiMalloc_32f_C2_func)(int, int, int*);
    typedef Npp32f* (*nppiMalloc_32f_C3_func)(int, int, int*);
    typedef Npp32f* (*nppiMalloc_32f_C4_func)(int, int, int*);
    
    nppiMalloc_32f_C1_func op_nppiMalloc_32f_C1 = nullptr;
    nppiMalloc_32f_C2_func op_nppiMalloc_32f_C2 = nullptr;
    nppiMalloc_32f_C3_func op_nppiMalloc_32f_C3 = nullptr;
    nppiMalloc_32f_C4_func op_nppiMalloc_32f_C4 = nullptr;
    
    // Memory free
    typedef void (*nppiFree_func)(void*);
    nppiFree_func op_nppiFree = nullptr;
    
    // ==================== NPPI Arithmetic Functions ====================
    
    // AddC functions
    typedef NppStatus (*nppiAddC_8u_C1RSfs_Ctx_func)(const Npp8u*, int, Npp8u, Npp8u*, int, NppiSize, int, NppStreamContext);
    typedef NppStatus (*nppiAddC_16u_C1RSfs_Ctx_func)(const Npp16u*, int, Npp16u, Npp16u*, int, NppiSize, int, NppStreamContext);
    typedef NppStatus (*nppiAddC_32f_C1R_Ctx_func)(const Npp32f*, int, Npp32f, Npp32f*, int, NppiSize, NppStreamContext);
    
    nppiAddC_8u_C1RSfs_Ctx_func op_nppiAddC_8u_C1RSfs_Ctx = nullptr;
    nppiAddC_16u_C1RSfs_Ctx_func op_nppiAddC_16u_C1RSfs_Ctx = nullptr;
    nppiAddC_32f_C1R_Ctx_func op_nppiAddC_32f_C1R_Ctx = nullptr;
    
    // SubC functions
    typedef NppStatus (*nppiSubC_8u_C1RSfs_Ctx_func)(const Npp8u*, int, Npp8u, Npp8u*, int, NppiSize, int, NppStreamContext);
    typedef NppStatus (*nppiSubC_16u_C1RSfs_Ctx_func)(const Npp16u*, int, Npp16u, Npp16u*, int, NppiSize, int, NppStreamContext);
    typedef NppStatus (*nppiSubC_32f_C1R_Ctx_func)(const Npp32f*, int, Npp32f, Npp32f*, int, NppiSize, NppStreamContext);
    
    nppiSubC_8u_C1RSfs_Ctx_func op_nppiSubC_8u_C1RSfs_Ctx = nullptr;
    nppiSubC_16u_C1RSfs_Ctx_func op_nppiSubC_16u_C1RSfs_Ctx = nullptr;
    nppiSubC_32f_C1R_Ctx_func op_nppiSubC_32f_C1R_Ctx = nullptr;
    
    // MulC functions
    typedef NppStatus (*nppiMulC_8u_C1RSfs_Ctx_func)(const Npp8u*, int, Npp8u, Npp8u*, int, NppiSize, int, NppStreamContext);
    typedef NppStatus (*nppiMulC_16u_C1RSfs_Ctx_func)(const Npp16u*, int, Npp16u, Npp16u*, int, NppiSize, int, NppStreamContext);
    typedef NppStatus (*nppiMulC_32f_C1R_Ctx_func)(const Npp32f*, int, Npp32f, Npp32f*, int, NppiSize, NppStreamContext);
    
    nppiMulC_8u_C1RSfs_Ctx_func op_nppiMulC_8u_C1RSfs_Ctx = nullptr;
    nppiMulC_16u_C1RSfs_Ctx_func op_nppiMulC_16u_C1RSfs_Ctx = nullptr;
    nppiMulC_32f_C1R_Ctx_func op_nppiMulC_32f_C1R_Ctx = nullptr;
    
    // DivC functions
    typedef NppStatus (*nppiDivC_8u_C1RSfs_Ctx_func)(const Npp8u*, int, Npp8u, Npp8u*, int, NppiSize, int, NppStreamContext);
    typedef NppStatus (*nppiDivC_16u_C1RSfs_Ctx_func)(const Npp16u*, int, Npp16u, Npp16u*, int, NppiSize, int, NppStreamContext);
    typedef NppStatus (*nppiDivC_32f_C1R_Ctx_func)(const Npp32f*, int, Npp32f, Npp32f*, int, NppiSize, NppStreamContext);
    
    nppiDivC_8u_C1RSfs_Ctx_func op_nppiDivC_8u_C1RSfs_Ctx = nullptr;
    nppiDivC_16u_C1RSfs_Ctx_func op_nppiDivC_16u_C1RSfs_Ctx = nullptr;
    nppiDivC_32f_C1R_Ctx_func op_nppiDivC_32f_C1R_Ctx = nullptr;
    
    // Dual-source arithmetic functions
    typedef NppStatus (*nppiAdd_32f_C1R_Ctx_func)(const Npp32f*, int, const Npp32f*, int, Npp32f*, int, NppiSize, NppStreamContext);
    typedef NppStatus (*nppiSub_32f_C1R_Ctx_func)(const Npp32f*, int, const Npp32f*, int, Npp32f*, int, NppiSize, NppStreamContext);
    typedef NppStatus (*nppiMul_32f_C1R_Ctx_func)(const Npp32f*, int, const Npp32f*, int, Npp32f*, int, NppiSize, NppStreamContext);
    typedef NppStatus (*nppiDiv_32f_C1R_Ctx_func)(const Npp32f*, int, const Npp32f*, int, Npp32f*, int, NppiSize, NppStreamContext);
    
    nppiAdd_32f_C1R_Ctx_func op_nppiAdd_32f_C1R_Ctx = nullptr;
    nppiSub_32f_C1R_Ctx_func op_nppiSub_32f_C1R_Ctx = nullptr;
    nppiMul_32f_C1R_Ctx_func op_nppiMul_32f_C1R_Ctx = nullptr;
    nppiDiv_32f_C1R_Ctx_func op_nppiDiv_32f_C1R_Ctx = nullptr;

private:
    // 私有构造函数
    OpenNppLoader() {
        loadLibrary();
    }
    
    // 析构函数
    ~OpenNppLoader() {
        unloadLibrary();
    }
    
    // 加载库
    void loadLibrary() {
        // 尝试加载OpenNPP共享库
        const char* libraryPath = "./libopennpp.so";
        
        handle_ = dlopen(libraryPath, RTLD_LAZY);
        if (!handle_) {
            errorMessage_ = "Failed to load OpenNPP library: ";
            errorMessage_ += dlerror();
            available_ = false;
            return;
        }
        
        std::cout << "Successfully loaded OpenNPP library from " << libraryPath << std::endl;
        
        // 加载所有函数符号
        if (!loadSymbols()) {
            unloadLibrary();
            available_ = false;
            return;
        }
        
        available_ = true;
        errorMessage_ = "OpenNPP loaded successfully";
    }
    
    // 卸载库
    void unloadLibrary() {
        if (handle_) {
            dlclose(handle_);
            handle_ = nullptr;
        }
    }
    
    // 辅助宏：加载符号
    #define LOAD_SYMBOL(var, name) \
        do { \
            var = (decltype(var))dlsym(handle_, name); \
            if (!var) { \
                errorMessage_ = "Failed to load symbol: " name " - " + std::string(dlerror() ? dlerror() : "unknown error"); \
                return false; \
            } \
        } while(0)
    
    // 加载函数符号
    bool loadSymbols() {
        // 加载NPP Core函数
        LOAD_SYMBOL(op_nppGetGpuNumSMs, "nppGetGpuNumSMs");
        LOAD_SYMBOL(op_nppGetMaxThreadsPerBlock, "nppGetMaxThreadsPerBlock");
        LOAD_SYMBOL(op_nppGetMaxThreadsPerSM, "nppGetMaxThreadsPerSM");
        LOAD_SYMBOL(op_nppGetGpuName, "nppGetGpuName");
        LOAD_SYMBOL(op_nppGetGpuDeviceProperties, "nppGetGpuDeviceProperties");
        
        LOAD_SYMBOL(op_nppGetStream, "nppGetStream");
        LOAD_SYMBOL(op_nppSetStream, "nppSetStream");
        LOAD_SYMBOL(op_nppGetStreamContext, "nppGetStreamContext");
        
        // 加载NPPI Support函数
        LOAD_SYMBOL(op_nppiMalloc_8u_C1, "nppiMalloc_8u_C1");
        LOAD_SYMBOL(op_nppiMalloc_16u_C1, "nppiMalloc_16u_C1");
        LOAD_SYMBOL(op_nppiMalloc_32f_C1, "nppiMalloc_32f_C1");
        LOAD_SYMBOL(op_nppiFree, "nppiFree");
        
        // 加载NPPI Arithmetic函数
        LOAD_SYMBOL(op_nppiAddC_8u_C1RSfs_Ctx, "nppiAddC_8u_C1RSfs_Ctx");
        LOAD_SYMBOL(op_nppiAddC_16u_C1RSfs_Ctx, "nppiAddC_16u_C1RSfs_Ctx");
        LOAD_SYMBOL(op_nppiAddC_32f_C1R_Ctx, "nppiAddC_32f_C1R_Ctx");
        
        LOAD_SYMBOL(op_nppiSubC_8u_C1RSfs_Ctx, "nppiSubC_8u_C1RSfs_Ctx");
        LOAD_SYMBOL(op_nppiSubC_16u_C1RSfs_Ctx, "nppiSubC_16u_C1RSfs_Ctx");
        LOAD_SYMBOL(op_nppiSubC_32f_C1R_Ctx, "nppiSubC_32f_C1R_Ctx");
        
        LOAD_SYMBOL(op_nppiMulC_8u_C1RSfs_Ctx, "nppiMulC_8u_C1RSfs_Ctx");
        LOAD_SYMBOL(op_nppiMulC_16u_C1RSfs_Ctx, "nppiMulC_16u_C1RSfs_Ctx");
        LOAD_SYMBOL(op_nppiMulC_32f_C1R_Ctx, "nppiMulC_32f_C1R_Ctx");
        
        LOAD_SYMBOL(op_nppiDivC_8u_C1RSfs_Ctx, "nppiDivC_8u_C1RSfs_Ctx");
        LOAD_SYMBOL(op_nppiDivC_16u_C1RSfs_Ctx, "nppiDivC_16u_C1RSfs_Ctx");
        LOAD_SYMBOL(op_nppiDivC_32f_C1R_Ctx, "nppiDivC_32f_C1R_Ctx");
        
        // 加载双源算术函数
        LOAD_SYMBOL(op_nppiAdd_32f_C1R_Ctx, "nppiAdd_32f_C1R_Ctx");
        LOAD_SYMBOL(op_nppiSub_32f_C1R_Ctx, "nppiSub_32f_C1R_Ctx");
        LOAD_SYMBOL(op_nppiMul_32f_C1R_Ctx, "nppiMul_32f_C1R_Ctx");
        LOAD_SYMBOL(op_nppiDiv_32f_C1R_Ctx, "nppiDiv_32f_C1R_Ctx");
        
        return true;
    }
    
private:
    void* handle_ = nullptr;
    bool available_ = false;
    std::string errorMessage_;
};

} // namespace test_framework

#endif // TEST_FRAMEWORK_OPENNPP_LOADER_H