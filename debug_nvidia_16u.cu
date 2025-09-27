#include <iostream>
#include <vector>
#include "npp.h"

int main() {
    // Test 16u AC4R pixel alpha
    const int width = 2;
    const int height = 1;
    const int channels = 4;
    
    std::vector<Npp16u> hostSrc = {50000, 30000, 20000, 40000,  // Pixel 1
                                   25000, 35000, 15000, 20000}; // Pixel 2
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp16u *d_src = nppiMalloc_16u_C4(width, height, &srcStep);
    Npp16u *d_dst = nppiMalloc_16u_C4(width, height, &dstStep);
    
    // Copy data to GPU
    int hostStep = width * channels * sizeof(Npp16u);
    cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
    
    // Execute operation
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiAlphaPremul_16u_AC4R(d_src, srcStep, d_dst, dstStep, oSizeROI);
    
    std::cout << "Status: " << status << std::endl;
    
    // Copy result back
    std::vector<Npp16u> hostResult(width * height * channels);
    cudaMemcpy2D(hostResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);
    
    std::cout << "Input:  ";
    for (int i = 0; i < 8; ++i) {
        std::cout << hostSrc[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Output: ";
    for (int i = 0; i < 8; ++i) {
        std::cout << hostResult[i] << " ";
    }
    std::cout << std::endl;
    
    // Manual calculation to understand NVIDIA's method
    std::cout << "Expected (>>16): ";
    for (int p = 0; p < 2; ++p) {
        int idx = p * 4;
        Npp16u alpha = hostSrc[idx + 3];
        for (int c = 0; c < 3; ++c) {
            int64_t result = (static_cast<int64_t>(hostSrc[idx + c]) * alpha) >> 16;
            std::cout << result << " ";
        }
        std::cout << alpha << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Expected (/65535): ";
    for (int p = 0; p < 2; ++p) {
        int idx = p * 4;
        Npp16u alpha = hostSrc[idx + 3];
        for (int c = 0; c < 3; ++c) {
            int64_t result = (static_cast<int64_t>(hostSrc[idx + c]) * alpha) / 65535;
            std::cout << result << " ";
        }
        std::cout << alpha << " ";
    }
    std::cout << std::endl;
    
    nppiFree(d_src);
    nppiFree(d_dst);
    
    return 0;
}