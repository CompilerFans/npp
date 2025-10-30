#include "linear_resize_nvidia_compatible.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <npp.h>

int main() {
    // Test case: 2x2 -> 3x3
    int srcW = 2, srcH = 2;
    int dstW = 3, dstH = 3;
    int channels = 3;
    
    std::vector<unsigned char> src = {
        0, 0, 0,      100, 100, 100,
        50, 50, 50,   150, 150, 150
    };
    
    // CPU reference
    std::vector<unsigned char> cpuResult(dstW * dstH * channels);
    LinearResizeNvidiaCompatible<unsigned char>::resize(
        src.data(), srcW * channels, srcW, srcH,
        cpuResult.data(), dstW * channels, dstW, dstH, channels);
    
    // NVIDIA NPP
    unsigned char *d_src, *d_dst;
    cudaMalloc(&d_src, srcW * srcH * channels);
    cudaMalloc(&d_dst, dstW * dstH * channels);
    cudaMemcpy(d_src, src.data(), srcW * srcH * channels, cudaMemcpyHostToDevice);
    
    NppiSize srcSize = {srcW, srcH};
    NppiSize dstSize = {dstW, dstH};
    NppiRect srcROI = {0, 0, srcW, srcH};
    NppiRect dstROI = {0, 0, dstW, dstH};
    
    nppiResize_8u_C3R(d_src, srcW * channels, srcSize, srcROI,
                      d_dst, dstW * channels, dstSize, dstROI,
                      NPPI_INTER_LINEAR);
    
    std::vector<unsigned char> nppResult(dstW * dstH * channels);
    cudaMemcpy(nppResult.data(), d_dst, dstW * dstH * channels, cudaMemcpyDeviceToHost);
    
    // Compare
    int matches = 0;
    int total = dstW * dstH * channels;
    
    std::cout << "NVIDIA Compatible CPU Reference Validation:\n\n";
    
    for (int y = 0; y < dstH; y++) {
        for (int x = 0; x < dstW; x++) {
            int idx = (y * dstW + x) * channels;
            bool match = (cpuResult[idx] == nppResult[idx]);
            if (match) matches++;
            
            std::cout << "Pixel[" << x << "," << y << "]: "
                      << "CPU=" << (int)cpuResult[idx]
                      << ", NPP=" << (int)nppResult[idx]
                      << (match ? " ✓" : " ✗") << "\n";
        }
    }
    
    std::cout << "\nResult: " << matches << " / " << total 
              << " (" << (100.0f * matches / total) << "%)\n";
    
    cudaFree(d_src);
    cudaFree(d_dst);
    
    return (matches == total) ? 0 : 1;
}
