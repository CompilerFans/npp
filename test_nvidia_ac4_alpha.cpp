#include <iostream>
#include <npp.h>
#include <cuda_runtime.h>

// 测试NVIDIA NPP的AC4格式Alpha通道处理行为

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
    checkCudaError(cudaSetDevice(0), "cudaSetDevice");

    const int width = 4;
    const int height = 1;
    const int channels = 4;

    // 分配主机内存
    Npp8u h_input[width * channels] = {
        100, 110, 120, 200,  // Pixel 0: RGB=(100,110,120), Alpha=200
        150, 160, 170, 150,  // Pixel 1: RGB=(150,160,170), Alpha=150
        50,  60,  70,  100,  // Pixel 2: RGB=(50,60,70),    Alpha=100
        200, 210, 220, 255   // Pixel 3: RGB=(200,210,220), Alpha=255
    };

    Npp8u h_output[width * channels];

    // 分配设备内存
    Npp8u *d_input, *d_output;
    checkCudaError(cudaMalloc(&d_input, width * channels * sizeof(Npp8u)), "malloc input");
    checkCudaError(cudaMalloc(&d_output, width * channels * sizeof(Npp8u)), "malloc output");

    // 拷贝输入到设备
    checkCudaError(cudaMemcpy(d_input, h_input, width * channels * sizeof(Npp8u),
                               cudaMemcpyHostToDevice), "memcpy input");

    // 设置NPP参数
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

    // 调用NVIDIA NPP的AC4 Forward Gamma
    NppStatus status = nppiGammaFwd_8u_AC4R_Ctx(d_input, nStep, d_output, nStep, oSizeROI, nppStreamCtx);
    checkNppError(status, "nppiGammaFwd_8u_AC4R_Ctx");

    // 拷贝结果回主机
    checkCudaError(cudaMemcpy(h_output, d_output, width * channels * sizeof(Npp8u),
                               cudaMemcpyDeviceToHost), "memcpy output");

    // 打印结果
    std::cout << "NVIDIA NPP AC4 Alpha Channel Behavior Test" << std::endl;
    std::cout << "===========================================" << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < width; i++) {
        std::cout << "Pixel " << i << ":" << std::endl;
        std::cout << "  Input:  R=" << (int)h_input[i*4+0]
                  << " G=" << (int)h_input[i*4+1]
                  << " B=" << (int)h_input[i*4+2]
                  << " A=" << (int)h_input[i*4+3] << std::endl;
        std::cout << "  Output: R=" << (int)h_output[i*4+0]
                  << " G=" << (int)h_output[i*4+1]
                  << " B=" << (int)h_output[i*4+2]
                  << " A=" << (int)h_output[i*4+3] << std::endl;

        if (h_input[i*4+3] != h_output[i*4+3]) {
            std::cout << "  ⚠️  Alpha CHANGED: " << (int)h_input[i*4+3]
                      << " -> " << (int)h_output[i*4+3] << std::endl;
        } else {
            std::cout << "  ✅ Alpha preserved" << std::endl;
        }
        std::cout << std::endl;
    }

    // 清理
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
