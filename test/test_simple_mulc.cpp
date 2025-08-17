#include <iostream>
#include <cuda_runtime.h>
#include "npp.h"

int main() {
    // 检查CUDA设备
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("Error: No CUDA devices found!\n");
        return 1;
    }
    
    printf("Testing nppiMulC functions...\n");
    
    // 测试数据设置
    const int width = 32;
    const int height = 32;
    const int size = width * height;
    
    // 8位无符号数测试
    printf("\n8u MulC test...\n");
    Npp8u* h_src_8u = new Npp8u[size];
    Npp8u* h_dst_8u = new Npp8u[size];
    Npp8u* d_src_8u = nullptr;
    Npp8u* d_dst_8u = nullptr;
    
    // 初始化源数据
    for (int i = 0; i < size; i++) {
        h_src_8u[i] = static_cast<Npp8u>(10 + (i % 20)); // 较小的值避免溢出
    }
    
    cudaMalloc(&d_src_8u, size);
    cudaMalloc(&d_dst_8u, size);
    cudaMemcpy(d_src_8u, h_src_8u, size, cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiMulC_8u_C1RSfs(d_src_8u, width, 3, d_dst_8u, width, roi, 2);
    
    if (status == NPP_NO_ERROR) {
        printf("8u MulC: SUCCESS\n");
        
        // 检查结果
        cudaMemcpy(h_dst_8u, d_dst_8u, size, cudaMemcpyDeviceToHost);
        printf("First 5 results: %d %d %d %d %d\n", 
               h_dst_8u[0], h_dst_8u[1], h_dst_8u[2], h_dst_8u[3], h_dst_8u[4]);
    } else {
        printf("8u MulC: FAILED with status %d\n", status);
    }
    
    // 32位浮点测试
    printf("\n32f MulC test...\n");
    Npp32f* h_src_32f = new Npp32f[size];
    Npp32f* h_dst_32f = new Npp32f[size];
    Npp32f* d_src_32f = nullptr;
    Npp32f* d_dst_32f = nullptr;
    
    // 初始化浮点数据
    for (int i = 0; i < size; i++) {
        h_src_32f[i] = 10.0f + static_cast<float>(i % 20);
    }
    
    cudaMalloc(&d_src_32f, size * sizeof(Npp32f));
    cudaMalloc(&d_dst_32f, size * sizeof(Npp32f));
    cudaMemcpy(d_src_32f, h_src_32f, size * sizeof(Npp32f), cudaMemcpyHostToDevice);
    
    NppiSize roi_32f = {width, height};
    status = nppiMulC_32f_C1R(d_src_32f, width * sizeof(Npp32f), 2.5f, 
                              d_dst_32f, width * sizeof(Npp32f), roi_32f);
    
    if (status == NPP_NO_ERROR) {
        printf("32f MulC: SUCCESS\n");
        
        // 检查结果
        cudaMemcpy(h_dst_32f, d_dst_32f, size * sizeof(Npp32f), cudaMemcpyDeviceToHost);
        printf("First 5 results: %.2f %.2f %.2f %.2f %.2f\n", 
               h_dst_32f[0], h_dst_32f[1], h_dst_32f[2], h_dst_32f[3], h_dst_32f[4]);
    } else {
        printf("32f MulC: FAILED with status %d\n", status);
    }
    
    // 清理内存
    delete[] h_src_8u;
    delete[] h_dst_8u;
    delete[] h_src_32f;
    delete[] h_dst_32f;
    
    cudaFree(d_src_8u);
    cudaFree(d_dst_8u);
    cudaFree(d_src_32f);
    cudaFree(d_dst_32f);
    
    printf("\nMulC test completed!\n");
    return 0;
}