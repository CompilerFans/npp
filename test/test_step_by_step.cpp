#include <iostream>
#include <cuda_runtime.h>
#include "npp.h"

// 逐步测试每个组件

int main() {
    printf("Step 1: Basic initialization\n");
    
    // 检查CUDA设备
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("Error: No CUDA devices found!\n");
        return 1;
    }
    cudaSetDevice(0);
    printf("CUDA initialized\n");
    
    printf("Step 2: Check NPP version\n");
    const NppLibraryVersion* npp_version = nppGetLibVersion();
    if (npp_version) {
        printf("NPP version: %d.%d.%d\n", 
               npp_version->major, npp_version->minor, npp_version->build);
    }
    
    printf("Step 3: Test simple memory allocation\n");
    Npp8u* test_ptr = nullptr;
    cudaError_t result = cudaMalloc(&test_ptr, 1024);
    if (result == cudaSuccess) {
        printf("CUDA malloc successful\n");
        cudaFree(test_ptr);
    } else {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(result));
        return 1;
    }
    
    printf("Step 4: Test NPP function call\n");
    Npp8u* d_src = nullptr;
    Npp8u* d_dst = nullptr;
    const int size = 32 * 32;
    
    cudaMalloc(&d_src, size);
    cudaMalloc(&d_dst, size);
    
    // 初始化数据
    Npp8u* h_data = new Npp8u[size];
    for (int i = 0; i < size; i++) {
        h_data[i] = static_cast<Npp8u>(i % 100);
    }
    cudaMemcpy(d_src, h_data, size, cudaMemcpyHostToDevice);
    
    NppiSize roi = {32, 32};
    NppStatus status = nppiAddC_8u_C1RSfs(d_src, 32, 50, d_dst, 32, roi, 1);
    
    if (status == NPP_NO_ERROR) {
        printf("NPP function call successful\n");
    } else {
        printf("NPP function call failed: %d\n", status);
    }
    
    delete[] h_data;
    cudaFree(d_src);
    cudaFree(d_dst);
    
    printf("All steps completed successfully!\n");
    return 0;
}