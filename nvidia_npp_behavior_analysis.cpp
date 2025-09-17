#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <iomanip>
#include <npp.h>
#include <cuda_runtime.h>

int main() {
    std::cout << "NVIDIA NPP行为分析程序" << std::endl;
    std::cout << "分析Ln和Sqrt函数的实际行为模式" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 初始化CUDA
    if (cudaSetDevice(0) != cudaSuccess) {
        std::cerr << "无法设置CUDA设备" << std::endl;
        return -1;
    }
    
    // 测试数据
    const int width = 8;
    const int height = 1;
    const int step = width * sizeof(Npp8u);
    
    // 创建测试数据 - 8位无符号整数
    std::vector<Npp8u> src_8u = {1, 2, 4, 8, 16, 32, 64, 128};
    std::vector<Npp8u> dst_8u(width);
    
    // 分配GPU内存
    Npp8u* d_src_8u, *d_dst_8u;
    cudaMalloc(&d_src_8u, width * sizeof(Npp8u));
    cudaMalloc(&d_dst_8u, width * sizeof(Npp8u));
    
    // 复制数据到GPU
    cudaMemcpy(d_src_8u, src_8u.data(), width * sizeof(Npp8u), cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    
    std::cout << "\n=== 8位Ln函数行为分析 ===" << std::endl;
    
    // 测试不同的缩放因子
    for (int scaleFactor = 0; scaleFactor <= 3; scaleFactor++) {
        std::cout << "\n缩放因子 " << scaleFactor << ":" << std::endl;
        
        NppStatus status = nppiLn_8u_C1RSfs(d_src_8u, step, d_dst_8u, step, roi, scaleFactor);
        
        if (status == NPP_SUCCESS) {
            // 复制结果回CPU
            cudaMemcpy(dst_8u.data(), d_dst_8u, width * sizeof(Npp8u), cudaMemcpyDeviceToHost);
            
            std::cout << "输入 -> NVIDIA输出 (数学期望)" << std::endl;
            for (int i = 0; i < width; i++) {
                float mathematical_result = std::log((float)src_8u[i]) * (1 << scaleFactor);
                int mathematical_rounded = (int)(mathematical_result + 0.5f);
                mathematical_rounded = std::max(0, std::min(255, mathematical_rounded));
                
                std::cout << std::setw(3) << (int)src_8u[i] 
                         << " -> " << std::setw(3) << (int)dst_8u[i]
                         << " (" << std::setw(3) << mathematical_rounded << ")" << std::endl;
            }
        } else {
            std::cout << "错误状态: " << status << std::endl;
        }
    }
    
    // 32位浮点数测试
    std::vector<Npp32f> src_32f = {0.0f, -1.0f, 1.0f, 2.718281828f, 10.0f, 100.0f};
    std::vector<Npp32f> dst_32f(6);
    const int width_32f = 6;
    const int step_32f = width_32f * sizeof(Npp32f);
    
    Npp32f* d_src_32f, *d_dst_32f;
    cudaMalloc(&d_src_32f, width_32f * sizeof(Npp32f));
    cudaMalloc(&d_dst_32f, width_32f * sizeof(Npp32f));
    
    cudaMemcpy(d_src_32f, src_32f.data(), width_32f * sizeof(Npp32f), cudaMemcpyHostToDevice);
    
    NppiSize roi_32f = {width_32f, height};
    
    std::cout << "\n=== 32位浮点Ln函数行为分析 ===" << std::endl;
    
    NppStatus status_32f = nppiLn_32f_C1R(d_src_32f, step_32f, d_dst_32f, step_32f, roi_32f);
    
    if (status_32f == NPP_SUCCESS) {
        cudaMemcpy(dst_32f.data(), d_dst_32f, width_32f * sizeof(Npp32f), cudaMemcpyDeviceToHost);
        
        std::cout << "输入 -> NVIDIA输出 (数学期望)" << std::endl;
        for (int i = 0; i < width_32f; i++) {
            float mathematical_result = (src_32f[i] <= 0.0f) ? 
                std::numeric_limits<float>::quiet_NaN() : std::log(src_32f[i]);
            
            std::cout << std::setw(10) << src_32f[i] 
                     << " -> " << std::setw(10) << dst_32f[i]
                     << " (" << std::setw(10) << mathematical_result << ")" << std::endl;
        }
    } else {
        std::cout << "错误状态: " << status_32f << std::endl;
    }
    
    // Sqrt函数测试
    std::cout << "\n=== 8位Sqrt函数行为分析 ===" << std::endl;
    
    for (int scaleFactor = 0; scaleFactor <= 3; scaleFactor++) {
        std::cout << "\n缩放因子 " << scaleFactor << ":" << std::endl;
        
        NppStatus status = nppiSqrt_8u_C1RSfs(d_src_8u, step, d_dst_8u, step, roi, scaleFactor);
        
        if (status == NPP_SUCCESS) {
            cudaMemcpy(dst_8u.data(), d_dst_8u, width * sizeof(Npp8u), cudaMemcpyDeviceToHost);
            
            std::cout << "输入 -> NVIDIA输出 (数学期望)" << std::endl;
            for (int i = 0; i < width; i++) {
                float mathematical_result = std::sqrt((float)src_8u[i]) * (1 << scaleFactor);
                int mathematical_rounded = (int)(mathematical_result + 0.5f);
                mathematical_rounded = std::max(0, std::min(255, mathematical_rounded));
                
                std::cout << std::setw(3) << (int)src_8u[i] 
                         << " -> " << std::setw(3) << (int)dst_8u[i]
                         << " (" << std::setw(3) << mathematical_rounded << ")" << std::endl;
            }
        } else {
            std::cout << "错误状态: " << status << std::endl;
        }
    }
    
    // 32位浮点Sqrt测试
    std::cout << "\n=== 32位浮点Sqrt函数行为分析 ===" << std::endl;
    
    status_32f = nppiSqrt_32f_C1R(d_src_32f, step_32f, d_dst_32f, step_32f, roi_32f);
    
    if (status_32f == NPP_SUCCESS) {
        cudaMemcpy(dst_32f.data(), d_dst_32f, width_32f * sizeof(Npp32f), cudaMemcpyDeviceToHost);
        
        std::cout << "输入 -> NVIDIA输出 (数学期望)" << std::endl;
        for (int i = 0; i < width_32f; i++) {
            float mathematical_result = (src_32f[i] < 0.0f) ? 
                std::numeric_limits<float>::quiet_NaN() : std::sqrt(src_32f[i]);
            
            std::cout << std::setw(10) << src_32f[i] 
                     << " -> " << std::setw(10) << dst_32f[i]
                     << " (" << std::setw(10) << mathematical_result << ")" << std::endl;
        }
    } else {
        std::cout << "错误状态: " << status_32f << std::endl;
    }
    
    // 清理
    cudaFree(d_src_8u);
    cudaFree(d_dst_8u);
    cudaFree(d_src_32f);
    cudaFree(d_dst_32f);
    
    std::cout << "\n分析完成" << std::endl;
    return 0;
}