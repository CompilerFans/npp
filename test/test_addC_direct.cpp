#include <iostream>
#include <cuda_runtime.h>
#include "npp.h"

/**
 * 直接测试AddC函数，不使用复杂框架
 */

// 简单的测试函数
bool test_addC_8u() {
    const int width = 32;
    const int height = 32;
    const Npp8u constant = 50;
    const int scale_factor = 1;
    
    // 分配内存
    Npp8u* h_src = new Npp8u[width * height];
    Npp8u* h_dst = new Npp8u[width * height];
    Npp8u* h_ref = new Npp8u[width * height];
    
    Npp8u* d_src = nullptr;
    Npp8u* d_dst = nullptr;
    
    cudaMalloc(&d_src, width * height * sizeof(Npp8u));
    cudaMalloc(&d_dst, width * height * sizeof(Npp8u));
    
    // 初始化数据
    for (int i = 0; i < width * height; i++) {
        h_src[i] = static_cast<Npp8u>(i % 128);
    }
    
    // 计算参考结果
    for (int i = 0; i < width * height; i++) {
        int result = static_cast<int>(h_src[i]) + static_cast<int>(constant);
        result = result >> scale_factor;
        result = std::max(0, std::min(255, result));
        h_ref[i] = static_cast<Npp8u>(result);
    }
    
    // 复制到设备
    cudaMemcpy(d_src, h_src, width * height * sizeof(Npp8u), cudaMemcpyHostToDevice);
    
    // 运行我们的实现
    NppiSize roi = {width, height};
    NppStatus status = nppiAddC_8u_C1RSfs(d_src, width, constant, d_dst, width, roi, scale_factor);
    
    // 验证结果
    bool passed = true;
    
    if (status != NPP_NO_ERROR) {
        printf("AddC failed with status: %d\n", status);
        passed = false;
    } else {
        // 复制结果回主机
        cudaMemcpy(h_dst, d_dst, width * height * sizeof(Npp8u), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < width * height; i++) {
            if (h_dst[i] != h_ref[i]) {
                printf("Mismatch at %d: got %u, expected %u\n", i, h_dst[i], h_ref[i]);
                passed = false;
                break;
            }
        }
    }
    
    printf("AddC 8u test: %s\n", passed ? "PASS" : "FAIL");
    
    delete[] h_src;
    delete[] h_dst;
    delete[] h_ref;
    cudaFree(d_src);
    cudaFree(d_dst);
    
    return passed;
}

int main() {
    printf("Direct AddC Test\n");
    printf("================\n");
    
    // 检查CUDA设备
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        printf("Error: No CUDA devices found!\n");
        return 1;
    }
    
    cudaSetDevice(0);
    
    // 运行测试
    bool all_passed = true;
    all_passed &= test_addC_8u();
    
    printf("\nDirect test completed: %s\n", all_passed ? "ALL PASSED" : "SOME FAILED");
    
    return all_passed ? 0 : 1;
}