/**
 * @file debug_test.cpp
 * @brief 调试AddC函数问题的简单测试
 */

#include "test/functional/framework/npp_test_base.h"

using namespace npp_functional_test;

int main() {
    // 初始化CUDA设备
    cudaError_t cudaErr = cudaSetDevice(0);
    if (cudaErr != cudaSuccess) {
        std::cout << "Failed to set CUDA device: " << cudaGetErrorString(cudaErr) << std::endl;
        return 1;
    }
    
    // 获取流上下文并查看详情
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);
    
    std::cout << "Stream context info:" << std::endl;
    std::cout << "  Stream: " << ctx.hStream << std::endl;
    std::cout << "  Device ID: " << ctx.nCudaDeviceId << std::endl;
    std::cout << "  SM Count: " << ctx.nMultiProcessorCount << std::endl;
    
    // 测试简单的数据
    const int width = 4;
    const int height = 4;
    
    // 分配主机内存
    std::vector<Npp8u> src_data(width * height, 50);
    std::vector<Npp8u> dst_data(width * height, 0);
    
    // 分配GPU内存
    int step;
    Npp8u* d_src = nppiMalloc_8u_C1(width, height, &step);
    Npp8u* d_dst = nppiMalloc_8u_C1(width, height, &step);
    
    if (!d_src || !d_dst) {
        std::cout << "Failed to allocate GPU memory" << std::endl;
        return 1;
    }
    
    // 拷贝数据到GPU
    cudaErr = cudaMemcpy2D(d_src, step, src_data.data(), width,
                          width, height, cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess) {
        std::cout << "Failed to copy to GPU: " << cudaGetErrorString(cudaErr) << std::endl;
        return 1;
    }
    
    // 调用AddC函数
    std::cout << "Calling nppiAddC_8u_C1RSfs..." << std::endl;
    NppiSize roi = {width, height};
    NppStatus status = nppiAddC_8u_C1RSfs(d_src, step, 30, d_dst, step, roi, 0);
    
    std::cout << "nppiAddC_8u_C1RSfs returned status: " << status << std::endl;
    
    // 检查CUDA错误
    cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        std::cout << "CUDA error after AddC: " << cudaGetErrorString(cudaErr) << std::endl;
    }
    
    // 同步设备并检查
    cudaErr = cudaDeviceSynchronize();
    if (cudaErr != cudaSuccess) {
        std::cout << "CUDA sync error: " << cudaGetErrorString(cudaErr) << std::endl;
    }
    
    if (status == NPP_NO_ERROR) {
        // 拷贝结果回主机
        cudaErr = cudaMemcpy2D(dst_data.data(), width, d_dst, step,
                              width, height, cudaMemcpyDeviceToHost);
        if (cudaErr != cudaSuccess) {
            std::cout << "Failed to copy from GPU: " << cudaGetErrorString(cudaErr) << std::endl;
        } else {
            std::cout << "Results: ";
            for (int i = 0; i < 8; i++) {
                std::cout << (int)dst_data[i] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    // 清理
    nppiFree(d_src);
    nppiFree(d_dst);
    
    return 0;
}