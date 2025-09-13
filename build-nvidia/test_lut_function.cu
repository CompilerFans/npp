#include <cuda_runtime.h>
#include <npp.h>
#include <iostream>

int main() {
    // 测试nppiLUT_Linear_8u_C1R函数是否存在
    std::cout << "Testing if nppiLUT_Linear_8u_C1R exists..." << std::endl;
    
    // 尝试获取函数指针
    auto func_ptr = &nppiLUT_Linear_8u_C1R;
    std::cout << "Function pointer: " << (void*)func_ptr << std::endl;
    
    // 测试简单调用
    NppiSize roi = {1, 1};
    Npp32s levels[2] = {0, 255};
    Npp32s values[2] = {0, 255};
    
    NppStatus status = nppiLUT_Linear_8u_C1R(nullptr, 0, nullptr, 0, roi, values, levels, 2);
    std::cout << "Function call status: " << status << std::endl;
    
    return 0;
}