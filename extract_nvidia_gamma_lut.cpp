#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <npp.h>
#include <cuda_runtime.h>

// 从NVIDIA NPP中提取完整的Gamma LUT
// 这将帮助我们理解NVIDIA的确切实现

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

void checkNppError(NppStatus status, const char* msg) {
    if (status != NPP_SUCCESS) {
        std::cerr << "NPP Error: " << msg << " - Status: " << status << std::endl;
        exit(1);
    }
}

int main() {
    // 初始化CUDA
    checkCudaError(cudaSetDevice(0), "cudaSetDevice");

    // 为每个可能的输入值创建测试
    const int width = 256;
    const int height = 1;
    const int channels = 3;

    // 分配主机和设备内存
    Npp8u* h_input = new Npp8u[width * height * channels];
    Npp8u* h_output_fwd = new Npp8u[width * height * channels];
    Npp8u* h_output_inv = new Npp8u[width * height * channels];

    Npp8u* d_input;
    Npp8u* d_output;

    checkCudaError(cudaMalloc(&d_input, width * height * channels * sizeof(Npp8u)), "malloc input");
    checkCudaError(cudaMalloc(&d_output, width * height * channels * sizeof(Npp8u)), "malloc output");

    // 创建输入：每个像素的所有通道都设置为相同值 (0, 1, 2, ..., 255)
    for (int i = 0; i < width; i++) {
        h_input[i * 3 + 0] = i;  // R
        h_input[i * 3 + 1] = i;  // G
        h_input[i * 3 + 2] = i;  // B
    }

    // ========== 测试 Forward Gamma ==========
    checkCudaError(cudaMemcpy(d_input, h_input, width * height * channels * sizeof(Npp8u),
                               cudaMemcpyHostToDevice), "memcpy input fwd");

    NppiSize oSizeROI = {width, height};
    int nStep = width * channels * sizeof(Npp8u);

    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    checkCudaError(cudaGetDevice(&nppStreamCtx.nCudaDeviceId), "get device");
    cudaDeviceProp prop;
    checkCudaError(cudaGetDeviceProperties(&prop, nppStreamCtx.nCudaDeviceId), "get properties");
    nppStreamCtx.nMultiProcessorCount = prop.multiProcessorCount;
    nppStreamCtx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    nppStreamCtx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    nppStreamCtx.nSharedMemPerBlock = prop.sharedMemPerBlock;

    NppStatus status = nppiGammaFwd_8u_C3R_Ctx(d_input, nStep, d_output, nStep, oSizeROI, nppStreamCtx);
    checkNppError(status, "nppiGammaFwd_8u_C3R_Ctx");

    checkCudaError(cudaMemcpy(h_output_fwd, d_output, width * height * channels * sizeof(Npp8u),
                               cudaMemcpyDeviceToHost), "memcpy output fwd");

    // ========== 测试 Inverse Gamma ==========
    checkCudaError(cudaMemcpy(d_input, h_input, width * height * channels * sizeof(Npp8u),
                               cudaMemcpyHostToDevice), "memcpy input inv");

    status = nppiGammaInv_8u_C3R_Ctx(d_input, nStep, d_output, nStep, oSizeROI, nppStreamCtx);
    checkNppError(status, "nppiGammaInv_8u_C3R_Ctx");

    checkCudaError(cudaMemcpy(h_output_inv, d_output, width * height * channels * sizeof(Npp8u),
                               cudaMemcpyDeviceToHost), "memcpy output inv");

    // ========== 输出结果 ==========
    std::cout << "NVIDIA NPP Gamma LUT Extraction" << std::endl;
    std::cout << "================================" << std::endl;
    std::cout << std::endl;

    std::cout << "Forward Gamma (Linear -> sRGB):" << std::endl;
    std::cout << "Input -> Output" << std::endl;
    for (int i = 0; i < 256; i++) {
        int output = h_output_fwd[i * 3];  // 取R通道
        std::cout << std::setw(3) << i << " -> " << std::setw(3) << output << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Inverse Gamma (sRGB -> Linear):" << std::endl;
    std::cout << "Input -> Output" << std::endl;
    for (int i = 0; i < 256; i++) {
        int output = h_output_inv[i * 3];  // 取R通道
        std::cout << std::setw(3) << i << " -> " << std::setw(3) << output << std::endl;
    }

    // ========== 写入CSV文件 ==========
    std::ofstream csv("nvidia_gamma_lut.csv");
    csv << "Input,Forward_Output,Inverse_Output" << std::endl;
    for (int i = 0; i < 256; i++) {
        csv << i << "," << (int)h_output_fwd[i * 3] << "," << (int)h_output_inv[i * 3] << std::endl;
    }
    csv.close();

    std::cout << std::endl;
    std::cout << "LUT saved to nvidia_gamma_lut.csv" << std::endl;

    // ========== 分析Forward LUT，寻找最佳拟合gamma ==========
    std::cout << std::endl;
    std::cout << "Analyzing Forward LUT for best-fit gamma..." << std::endl;
    std::cout << "===========================================" << std::endl;

    // 测试不同的gamma值
    double best_gamma = 1.87;
    int best_max_error = 255;

    for (double gamma = 1.70; gamma <= 2.00; gamma += 0.001) {
        int max_error = 0;
        for (int i = 1; i < 256; i++) {  // 跳过0（总是0）
            float normalized = i / 255.0f;
            float result = powf(normalized, 1.0f / gamma);
            int predicted = (int)(result * 255.0f + 0.5f);
            int actual = h_output_fwd[i * 3];
            int error = abs(predicted - actual);
            if (error > max_error) {
                max_error = error;
            }
        }

        if (max_error < best_max_error) {
            best_max_error = max_error;
            best_gamma = gamma;
        }
    }

    std::cout << "Best single gamma: " << std::fixed << std::setprecision(3) << best_gamma << std::endl;
    std::cout << "Max error: " << best_max_error << std::endl;

    // 显示误差分布
    std::cout << std::endl;
    std::cout << "Error distribution for gamma " << best_gamma << ":" << std::endl;
    for (int i = 0; i < 256; i += 16) {
        float normalized = i / 255.0f;
        float result = powf(normalized, 1.0f / best_gamma);
        int predicted = (int)(result * 255.0f + 0.5f);
        int actual = h_output_fwd[i * 3];
        int error = predicted - actual;
        std::cout << "Input " << std::setw(3) << i << ": predicted=" << std::setw(3) << predicted
                  << ", actual=" << std::setw(3) << actual << ", error=" << std::setw(2) << error << std::endl;
    }

    // 清理
    delete[] h_input;
    delete[] h_output_fwd;
    delete[] h_output_inv;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
