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
    
    printf("Testing nppiDivC functions...\n");
    
    // 测试数据设置
    const int width = 32;
    const int height = 32;
    const int size = width * height;
    
    // 8位无符号数测试
    printf("\n8u DivC test...\n");
    Npp8u* h_src_8u = new Npp8u[size];
    Npp8u* h_dst_8u = new Npp8u[size];
    Npp8u* d_src_8u = nullptr;
    Npp8u* d_dst_8u = nullptr;
    
    // 初始化源数据
    for (int i = 0; i < size; i++) {
        h_src_8u[i] = static_cast<Npp8u>(100 + (i % 100)); // 100-199范围的值
    }
    
    cudaMalloc(&d_src_8u, size);
    cudaMalloc(&d_dst_8u, size);
    cudaMemcpy(d_src_8u, h_src_8u, size, cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiDivC_8u_C1RSfs(d_src_8u, width, 4, d_dst_8u, width, roi, 2);
    
    if (status == NPP_NO_ERROR) {
        printf("8u DivC: SUCCESS\n");
        
        // 检查结果
        cudaMemcpy(h_dst_8u, d_dst_8u, size, cudaMemcpyDeviceToHost);
        printf("First 5 results: %d %d %d %d %d\n", 
               h_dst_8u[0], h_dst_8u[1], h_dst_8u[2], h_dst_8u[3], h_dst_8u[4]);
    } else {
        printf("8u DivC: FAILED with status %d\n", status);
    }
    
    // 测试除以零的情况
    printf("\n8u DivC division by zero test...\n");
    status = nppiDivC_8u_C1RSfs(d_src_8u, width, 0, d_dst_8u, width, roi, 2);
    if (status == NPP_DIVIDE_BY_ZERO_ERROR) {
        printf("8u DivC: Division by zero correctly detected\n");
    } else {
        printf("8u DivC: Unexpected status for division by zero: %d\n", status);
    }
    
    // 32位浮点测试
    printf("\n32f DivC test...\n");
    Npp32f* h_src_32f = new Npp32f[size];
    Npp32f* h_dst_32f = new Npp32f[size];
    Npp32f* d_src_32f = nullptr;
    Npp32f* d_dst_32f = nullptr;
    
    // 初始化浮点数据
    for (int i = 0; i < size; i++) {
        h_src_32f[i] = 100.0f + static_cast<float>(i % 100);
    }
    
    cudaMalloc(&d_src_32f, size * sizeof(Npp32f));
    cudaMalloc(&d_dst_32f, size * sizeof(Npp32f));
    cudaMemcpy(d_src_32f, h_src_32f, size * sizeof(Npp32f), cudaMemcpyHostToDevice);
    
    NppiSize roi_32f = {width, height};
    status = nppiDivC_32f_C1R(d_src_32f, width * sizeof(Npp32f), 4.0f, 
                              d_dst_32f, width * sizeof(Npp32f), roi_32f);
    
    if (status == NPP_NO_ERROR) {
        printf("32f DivC: SUCCESS\n");
        
        // 检查结果
        cudaMemcpy(h_dst_32f, d_dst_32f, size * sizeof(Npp32f), cudaMemcpyDeviceToHost);
        printf("First 5 results: %.2f %.2f %.2f %.2f %.2f\n", 
               h_dst_32f[0], h_dst_32f[1], h_dst_32f[2], h_dst_32f[3], h_dst_32f[4]);
    } else {
        printf("32f DivC: FAILED with status %d\n", status);
    }
    
    // 32f除以零测试 (应该产生inf)
    printf("\n32f DivC division by zero test...\n");
    status = nppiDivC_32f_C1R(d_src_32f, width * sizeof(Npp32f), 0.0f, 
                              d_dst_32f, width * sizeof(Npp32f), roi_32f);
    if (status == NPP_NO_ERROR) {
        printf("32f DivC: Division by zero allowed (produces inf/nan)\n");
        
        // 检查结果
        cudaMemcpy(h_dst_32f, d_dst_32f, size * sizeof(Npp32f), cudaMemcpyDeviceToHost);
        printf("First 5 results: %.2f %.2f %.2f %.2f %.2f\n", 
               h_dst_32f[0], h_dst_32f[1], h_dst_32f[2], h_dst_32f[3], h_dst_32f[4]);
    } else {
        printf("32f DivC: Division by zero failed with status %d\n", status);
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
    
    printf("\nDivC test completed!\n");
    return 0;
}