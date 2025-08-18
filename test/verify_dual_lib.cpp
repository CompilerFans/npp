/**
 * 验证是否真的链接了两个不同的NPP库
 */
#include <iostream>
#include <dlfcn.h>
#include "npp.h"

#ifdef HAVE_NVIDIA_NPP
#include <nppi.h>
#endif

int main() {
    std::cout << "=== 符号地址验证 ===" << std::endl;
    
    // 获取我们的nppiMalloc_8u_C1地址
    void* our_func = (void*)&nppiMalloc_8u_C1;
    std::cout << "OpenNPP nppiMalloc_8u_C1 地址: " << our_func << std::endl;
    
#ifdef HAVE_NVIDIA_NPP
    // 获取NVIDIA的nppiMalloc_8u_C1地址
    void* nv_func = (void*)&::nppiMalloc_8u_C1;
    std::cout << "NVIDIA nppiMalloc_8u_C1 地址: " << nv_func << std::endl;
    
    if (our_func == nv_func) {
        std::cout << "❌ 警告：两个函数地址相同！可能链接了同一个实现！" << std::endl;
        return 1;
    } else {
        std::cout << "✓ 两个函数地址不同，确实是不同的实现" << std::endl;
    }
#endif
    
    // 测试实际调用
    std::cout << "\n=== 实际调用测试 ===" << std::endl;
    
    int step1, step2;
    Npp8u* ptr1 = nppiMalloc_8u_C1(640, 480, &step1);
    std::cout << "OpenNPP分配: ptr=" << ptr1 << ", step=" << step1 << std::endl;
    
#ifdef HAVE_NVIDIA_NPP
    Npp8u* ptr2 = ::nppiMalloc_8u_C1(640, 480, &step2);
    std::cout << "NVIDIA分配: ptr=" << ptr2 << ", step=" << step2 << std::endl;
    
    if (step1 == step2) {
        std::cout << "⚠ Step值相同，可能是巧合或相同的实现" << std::endl;
    } else {
        std::cout << "✓ Step值不同，说明是不同的实现策略" << std::endl;
    }
    
    ::nppiFree(ptr2);
#endif
    
    nppiFree(ptr1);
    
    // 使用动态加载来检查
    std::cout << "\n=== 动态符号查找 ===" << std::endl;
    
    // 获取当前进程的符号
    void* handle = dlopen(NULL, RTLD_NOW);
    if (handle) {
        void* sym = dlsym(handle, "nppiMalloc_8u_C1");
        if (sym) {
            std::cout << "找到符号 nppiMalloc_8u_C1: " << sym << std::endl;
            std::cout << "与OpenNPP地址比较: " << (sym == our_func ? "相同" : "不同") << std::endl;
        }
        dlclose(handle);
    }
    
    return 0;
}