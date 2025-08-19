/**
 * @file nvidia_npp_loader.h
 * @brief NVIDIA NPP动态库加载器 - 全局测试框架组件
 * 
 * 提供统一的NVIDIA NPP库动态加载和函数符号解析功能，
 * 避免符号冲突，实现真正的OpenNPP vs NVIDIA NPP对比测试
 */

#ifndef TEST_FRAMEWORK_NVIDIA_NPP_LOADER_H
#define TEST_FRAMEWORK_NVIDIA_NPP_LOADER_H

#include <dlfcn.h>
#include <string>
#include <unordered_map>
#include <memory>
#include <iostream>
#include "npp.h"

namespace test_framework {

/**
 * @brief NVIDIA NPP库加载器单例类
 * 
 * 负责动态加载NVIDIA NPP库，管理函数指针，
 * 提供统一的接口供测试使用
 */
class NvidiaNppLoader {
public:
    // 获取单例实例
    static NvidiaNppLoader& getInstance() {
        static NvidiaNppLoader instance;
        return instance;
    }
    
    // 禁用拷贝和移动
    NvidiaNppLoader(const NvidiaNppLoader&) = delete;
    NvidiaNppLoader& operator=(const NvidiaNppLoader&) = delete;
    NvidiaNppLoader(NvidiaNppLoader&&) = delete;
    NvidiaNppLoader& operator=(NvidiaNppLoader&&) = delete;
    
    // 检查NVIDIA NPP是否可用
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
    
    nppGetGpuNumSMs_func nv_nppGetGpuNumSMs = nullptr;
    nppGetMaxThreadsPerBlock_func nv_nppGetMaxThreadsPerBlock = nullptr;
    nppGetMaxThreadsPerSM_func nv_nppGetMaxThreadsPerSM = nullptr;
    nppGetGpuName_func nv_nppGetGpuName = nullptr;
    nppGetGpuDeviceProperties_func nv_nppGetGpuDeviceProperties = nullptr;
    
    // Stream management
    typedef cudaStream_t (*nppGetStream_func)();
    typedef NppStatus (*nppSetStream_func)(cudaStream_t);
    typedef NppStatus (*nppGetStreamContext_func)(NppStreamContext*);
    
    nppGetStream_func nv_nppGetStream = nullptr;
    nppSetStream_func nv_nppSetStream = nullptr;
    nppGetStreamContext_func nv_nppGetStreamContext = nullptr;
    
    // ==================== NPPI Support Functions ====================
    
    // Memory allocation - 8u
    typedef Npp8u* (*nppiMalloc_8u_C1_func)(int, int, int*);
    typedef Npp8u* (*nppiMalloc_8u_C2_func)(int, int, int*);
    typedef Npp8u* (*nppiMalloc_8u_C3_func)(int, int, int*);
    typedef Npp8u* (*nppiMalloc_8u_C4_func)(int, int, int*);
    
    nppiMalloc_8u_C1_func nv_nppiMalloc_8u_C1 = nullptr;
    nppiMalloc_8u_C2_func nv_nppiMalloc_8u_C2 = nullptr;
    nppiMalloc_8u_C3_func nv_nppiMalloc_8u_C3 = nullptr;
    nppiMalloc_8u_C4_func nv_nppiMalloc_8u_C4 = nullptr;
    
    // Memory allocation - 16u
    typedef Npp16u* (*nppiMalloc_16u_C1_func)(int, int, int*);
    typedef Npp16u* (*nppiMalloc_16u_C2_func)(int, int, int*);
    typedef Npp16u* (*nppiMalloc_16u_C3_func)(int, int, int*);
    typedef Npp16u* (*nppiMalloc_16u_C4_func)(int, int, int*);
    
    nppiMalloc_16u_C1_func nv_nppiMalloc_16u_C1 = nullptr;
    nppiMalloc_16u_C2_func nv_nppiMalloc_16u_C2 = nullptr;
    nppiMalloc_16u_C3_func nv_nppiMalloc_16u_C3 = nullptr;
    nppiMalloc_16u_C4_func nv_nppiMalloc_16u_C4 = nullptr;
    
    // Memory allocation - 32f
    typedef Npp32f* (*nppiMalloc_32f_C1_func)(int, int, int*);
    typedef Npp32f* (*nppiMalloc_32f_C2_func)(int, int, int*);
    typedef Npp32f* (*nppiMalloc_32f_C3_func)(int, int, int*);
    typedef Npp32f* (*nppiMalloc_32f_C4_func)(int, int, int*);
    
    nppiMalloc_32f_C1_func nv_nppiMalloc_32f_C1 = nullptr;
    nppiMalloc_32f_C2_func nv_nppiMalloc_32f_C2 = nullptr;
    nppiMalloc_32f_C3_func nv_nppiMalloc_32f_C3 = nullptr;
    nppiMalloc_32f_C4_func nv_nppiMalloc_32f_C4 = nullptr;
    
    // Memory free
    typedef void (*nppiFree_func)(void*);
    nppiFree_func nv_nppiFree = nullptr;
    
    // ==================== NPPI Arithmetic Functions ====================
    
    // AddC functions
    typedef NppStatus (*nppiAddC_8u_C1RSfs_func)(const Npp8u*, int, const Npp8u, Npp8u*, int, NppiSize, int);
    typedef NppStatus (*nppiAddC_8u_C1RSfs_Ctx_func)(const Npp8u*, int, const Npp8u, Npp8u*, int, NppiSize, int, NppStreamContext);
    typedef NppStatus (*nppiAddC_16u_C1RSfs_func)(const Npp16u*, int, const Npp16u, Npp16u*, int, NppiSize, int);
    typedef NppStatus (*nppiAddC_16u_C1RSfs_Ctx_func)(const Npp16u*, int, const Npp16u, Npp16u*, int, NppiSize, int, NppStreamContext);
    typedef NppStatus (*nppiAddC_32f_C1R_func)(const Npp32f*, int, const Npp32f, Npp32f*, int, NppiSize);
    typedef NppStatus (*nppiAddC_32f_C1R_Ctx_func)(const Npp32f*, int, const Npp32f, Npp32f*, int, NppiSize, NppStreamContext);
    
    nppiAddC_8u_C1RSfs_func nv_nppiAddC_8u_C1RSfs = nullptr;
    nppiAddC_8u_C1RSfs_Ctx_func nv_nppiAddC_8u_C1RSfs_Ctx = nullptr;
    nppiAddC_16u_C1RSfs_func nv_nppiAddC_16u_C1RSfs = nullptr;
    nppiAddC_16u_C1RSfs_Ctx_func nv_nppiAddC_16u_C1RSfs_Ctx = nullptr;
    nppiAddC_32f_C1R_func nv_nppiAddC_32f_C1R = nullptr;
    nppiAddC_32f_C1R_Ctx_func nv_nppiAddC_32f_C1R_Ctx = nullptr;
    
    // SubC functions
    typedef NppStatus (*nppiSubC_8u_C1RSfs_func)(const Npp8u*, int, const Npp8u, Npp8u*, int, NppiSize, int);
    typedef NppStatus (*nppiSubC_32f_C1R_func)(const Npp32f*, int, const Npp32f, Npp32f*, int, NppiSize);
    
    nppiSubC_8u_C1RSfs_func nv_nppiSubC_8u_C1RSfs = nullptr;
    nppiSubC_32f_C1R_func nv_nppiSubC_32f_C1R = nullptr;
    
    // MulC functions
    typedef NppStatus (*nppiMulC_8u_C1RSfs_func)(const Npp8u*, int, const Npp8u, Npp8u*, int, NppiSize, int);
    typedef NppStatus (*nppiMulC_32f_C1R_func)(const Npp32f*, int, const Npp32f, Npp32f*, int, NppiSize);
    
    nppiMulC_8u_C1RSfs_func nv_nppiMulC_8u_C1RSfs = nullptr;
    nppiMulC_32f_C1R_func nv_nppiMulC_32f_C1R = nullptr;
    
    // DivC functions
    typedef NppStatus (*nppiDivC_8u_C1RSfs_func)(const Npp8u*, int, const Npp8u, Npp8u*, int, NppiSize, int);
    typedef NppStatus (*nppiDivC_32f_C1R_func)(const Npp32f*, int, const Npp32f, Npp32f*, int, NppiSize);
    
    nppiDivC_8u_C1RSfs_func nv_nppiDivC_8u_C1RSfs = nullptr;
    nppiDivC_32f_C1R_func nv_nppiDivC_32f_C1R = nullptr;
    
    // Add (dual source) functions
    typedef NppStatus (*nppiAdd_8u_C1RSfs_func)(const Npp8u*, int, const Npp8u*, int, Npp8u*, int, NppiSize, int);
    typedef NppStatus (*nppiAdd_32f_C1R_func)(const Npp32f*, int, const Npp32f*, int, Npp32f*, int, NppiSize);
    
    nppiAdd_8u_C1RSfs_func nv_nppiAdd_8u_C1RSfs = nullptr;
    nppiAdd_32f_C1R_func nv_nppiAdd_32f_C1R = nullptr;
    
    // Sub (dual source) functions
    typedef NppStatus (*nppiSub_8u_C1RSfs_func)(const Npp8u*, int, const Npp8u*, int, Npp8u*, int, NppiSize, int);
    typedef NppStatus (*nppiSub_32f_C1R_func)(const Npp32f*, int, const Npp32f*, int, Npp32f*, int, NppiSize);
    
    nppiSub_8u_C1RSfs_func nv_nppiSub_8u_C1RSfs = nullptr;
    nppiSub_32f_C1R_func nv_nppiSub_32f_C1R = nullptr;
    
    // Mul (dual source) functions
    typedef NppStatus (*nppiMul_8u_C1RSfs_func)(const Npp8u*, int, const Npp8u*, int, Npp8u*, int, NppiSize, int);
    typedef NppStatus (*nppiMul_32f_C1R_func)(const Npp32f*, int, const Npp32f*, int, Npp32f*, int, NppiSize);
    
    nppiMul_8u_C1RSfs_func nv_nppiMul_8u_C1RSfs = nullptr;
    nppiMul_32f_C1R_func nv_nppiMul_32f_C1R = nullptr;
    
    // Div (dual source) functions
    typedef NppStatus (*nppiDiv_8u_C1RSfs_func)(const Npp8u*, int, const Npp8u*, int, Npp8u*, int, NppiSize, int);
    typedef NppStatus (*nppiDiv_32f_C1R_func)(const Npp32f*, int, const Npp32f*, int, Npp32f*, int, NppiSize);
    
    nppiDiv_8u_C1RSfs_func nv_nppiDiv_8u_C1RSfs = nullptr;
    nppiDiv_32f_C1R_func nv_nppiDiv_32f_C1R = nullptr;
    
    // 获取加载的库句柄（用于调试）
    void* getNppcHandle() const { return nppcLib_; }
    void* getNppisuHandle() const { return nppisuLib_; }
    void* getNppialHandle() const { return nppialLib_; }
    
private:
    NvidiaNppLoader() {
        loadLibraries();
        if (available_) {
            loadFunctions();
        }
    }
    
    ~NvidiaNppLoader() {
        if (nppcLib_) dlclose(nppcLib_);
        if (nppisuLib_) dlclose(nppisuLib_);
        if (nppialLib_) dlclose(nppialLib_);
    }
    
    void loadLibraries() {
        // 尝试加载NPP核心库
        nppcLib_ = dlopen("libnppc.so.12", RTLD_NOW | RTLD_LOCAL);
        if (!nppcLib_) {
            nppcLib_ = dlopen("/usr/local/cuda/lib64/libnppc.so.12", RTLD_NOW | RTLD_LOCAL);
        }
        if (!nppcLib_) {
            nppcLib_ = dlopen("/usr/local/cuda/lib64/libnppc.so", RTLD_NOW | RTLD_LOCAL);
        }
        
        // 尝试加载NPPI支持函数库
        nppisuLib_ = dlopen("libnppisu.so.12", RTLD_NOW | RTLD_LOCAL);
        if (!nppisuLib_) {
            nppisuLib_ = dlopen("/usr/local/cuda/lib64/libnppisu.so.12", RTLD_NOW | RTLD_LOCAL);
        }
        if (!nppisuLib_) {
            nppisuLib_ = dlopen("/usr/local/cuda/lib64/libnppisu.so", RTLD_NOW | RTLD_LOCAL);
        }
        
        // 尝试加载NPPI算术运算库
        nppialLib_ = dlopen("libnppial.so.12", RTLD_NOW | RTLD_LOCAL);
        if (!nppialLib_) {
            nppialLib_ = dlopen("/usr/local/cuda/lib64/libnppial.so.12", RTLD_NOW | RTLD_LOCAL);
        }
        if (!nppialLib_) {
            nppialLib_ = dlopen("/usr/local/cuda/lib64/libnppial.so", RTLD_NOW | RTLD_LOCAL);
        }
        
        if (!nppcLib_ || !nppisuLib_ || !nppialLib_) {
            available_ = false;
            errorMessage_ = "Failed to load NVIDIA NPP libraries: ";
            if (!nppcLib_) errorMessage_ += "nppc ";
            if (!nppisuLib_) errorMessage_ += "nppisu ";
            if (!nppialLib_) errorMessage_ += "nppial ";
            const char* dlerr = dlerror();
            if (dlerr) {
                errorMessage_ += "\nDetails: ";
                errorMessage_ += dlerr;
            }
        } else {
            available_ = true;
            std::cout << "Successfully loaded NVIDIA NPP libraries" << std::endl;
        }
    }
    
    void loadFunctions() {
        // 加载NPP Core函数
        loadFunction(nppcLib_, "nppGetGpuNumSMs", nv_nppGetGpuNumSMs);
        loadFunction(nppcLib_, "nppGetMaxThreadsPerBlock", nv_nppGetMaxThreadsPerBlock);
        loadFunction(nppcLib_, "nppGetMaxThreadsPerSM", nv_nppGetMaxThreadsPerSM);
        loadFunction(nppcLib_, "nppGetGpuName", nv_nppGetGpuName);
        loadFunction(nppcLib_, "nppGetGpuDeviceProperties", nv_nppGetGpuDeviceProperties);
        loadFunction(nppcLib_, "nppGetStream", nv_nppGetStream);
        loadFunction(nppcLib_, "nppSetStream", nv_nppSetStream);
        loadFunction(nppcLib_, "nppGetStreamContext", nv_nppGetStreamContext);
        
        // 加载NPPI支持函数
        loadFunction(nppisuLib_, "nppiMalloc_8u_C1", nv_nppiMalloc_8u_C1);
        loadFunction(nppisuLib_, "nppiMalloc_8u_C2", nv_nppiMalloc_8u_C2);
        loadFunction(nppisuLib_, "nppiMalloc_8u_C3", nv_nppiMalloc_8u_C3);
        loadFunction(nppisuLib_, "nppiMalloc_8u_C4", nv_nppiMalloc_8u_C4);
        loadFunction(nppisuLib_, "nppiMalloc_16u_C1", nv_nppiMalloc_16u_C1);
        loadFunction(nppisuLib_, "nppiMalloc_16u_C2", nv_nppiMalloc_16u_C2);
        loadFunction(nppisuLib_, "nppiMalloc_16u_C3", nv_nppiMalloc_16u_C3);
        loadFunction(nppisuLib_, "nppiMalloc_16u_C4", nv_nppiMalloc_16u_C4);
        loadFunction(nppisuLib_, "nppiMalloc_32f_C1", nv_nppiMalloc_32f_C1);
        loadFunction(nppisuLib_, "nppiMalloc_32f_C2", nv_nppiMalloc_32f_C2);
        loadFunction(nppisuLib_, "nppiMalloc_32f_C3", nv_nppiMalloc_32f_C3);
        loadFunction(nppisuLib_, "nppiMalloc_32f_C4", nv_nppiMalloc_32f_C4);
        loadFunction(nppisuLib_, "nppiFree", nv_nppiFree);
        
        // 加载NPPI算术运算函数
        loadFunction(nppialLib_, "nppiAddC_8u_C1RSfs", nv_nppiAddC_8u_C1RSfs);
        loadFunction(nppialLib_, "nppiAddC_8u_C1RSfs_Ctx", nv_nppiAddC_8u_C1RSfs_Ctx);
        loadFunction(nppialLib_, "nppiAddC_16u_C1RSfs", nv_nppiAddC_16u_C1RSfs);
        loadFunction(nppialLib_, "nppiAddC_16u_C1RSfs_Ctx", nv_nppiAddC_16u_C1RSfs_Ctx);
        loadFunction(nppialLib_, "nppiAddC_32f_C1R", nv_nppiAddC_32f_C1R);
        loadFunction(nppialLib_, "nppiAddC_32f_C1R_Ctx", nv_nppiAddC_32f_C1R_Ctx);
        
        loadFunction(nppialLib_, "nppiSubC_8u_C1RSfs", nv_nppiSubC_8u_C1RSfs);
        loadFunction(nppialLib_, "nppiSubC_32f_C1R", nv_nppiSubC_32f_C1R);
        
        loadFunction(nppialLib_, "nppiMulC_8u_C1RSfs", nv_nppiMulC_8u_C1RSfs);
        loadFunction(nppialLib_, "nppiMulC_32f_C1R", nv_nppiMulC_32f_C1R);
        
        loadFunction(nppialLib_, "nppiDivC_8u_C1RSfs", nv_nppiDivC_8u_C1RSfs);
        loadFunction(nppialLib_, "nppiDivC_32f_C1R", nv_nppiDivC_32f_C1R);
        
        loadFunction(nppialLib_, "nppiAdd_8u_C1RSfs", nv_nppiAdd_8u_C1RSfs);
        loadFunction(nppialLib_, "nppiAdd_32f_C1R", nv_nppiAdd_32f_C1R);
        
        loadFunction(nppialLib_, "nppiSub_8u_C1RSfs", nv_nppiSub_8u_C1RSfs);
        loadFunction(nppialLib_, "nppiSub_32f_C1R", nv_nppiSub_32f_C1R);
        
        loadFunction(nppialLib_, "nppiMul_8u_C1RSfs", nv_nppiMul_8u_C1RSfs);
        loadFunction(nppialLib_, "nppiMul_32f_C1R", nv_nppiMul_32f_C1R);
        
        loadFunction(nppialLib_, "nppiDiv_8u_C1RSfs", nv_nppiDiv_8u_C1RSfs);
        loadFunction(nppialLib_, "nppiDiv_32f_C1R", nv_nppiDiv_32f_C1R);
    }
    
    template<typename T>
    void loadFunction(void* lib, const char* name, T& funcPtr) {
        if (!lib) return;
        funcPtr = reinterpret_cast<T>(dlsym(lib, name));
        if (!funcPtr) {
            std::cerr << "Warning: Failed to load function " << name << ": " << dlerror() << std::endl;
        }
    }
    
private:
    void* nppcLib_ = nullptr;      // NPP Core库
    void* nppisuLib_ = nullptr;    // NPPI支持函数库
    void* nppialLib_ = nullptr;    // NPPI算术运算库
    bool available_ = false;
    std::string errorMessage_;
};

} // namespace test_framework

#endif // TEST_FRAMEWORK_NVIDIA_NPP_LOADER_H