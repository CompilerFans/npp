#include "linear_resize_nvidia_compatible.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <npp.h>

int main() {
    // Original test case that showed 100% match
    int srcW = 2, srcH = 2;
    int dstW = 3, dstH = 3;
    int channels = 1;
    
    // Simple test pattern (all 0)
    std::vector<unsigned char> src1 = {0, 100, 50, 150};
    
    // Gradient pattern
    std::vector<unsigned char> src2(srcW * srcH * channels);
    for (int y = 0; y < srcH; y++) {
        for (int x = 0; x < srcW; x++) {
            int idx = y * srcW + x;
            src2[idx] = (unsigned char)((x * 255 / (srcW - 1)) * 0.5f + 
                                        (y * 255 / (srcH - 1)) * 0.5f);
        }
    }
    
    auto testPattern = [&](const std::vector<unsigned char>& src, const std::string& name) {
        std::cout << "\nTesting pattern: " << name << "\n";
        std::cout << "Source: ";
        for (auto v : src) std::cout << (int)v << " ";
        std::cout << "\n\n";
        
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
        
        nppiResize_8u_C1R(d_src, srcW * channels, srcSize, srcROI,
                          d_dst, dstW * channels, dstSize, dstROI,
                          NPPI_INTER_LINEAR);
        
        std::vector<unsigned char> nppResult(dstW * dstH * channels);
        cudaMemcpy(nppResult.data(), d_dst, dstW * dstH * channels, cudaMemcpyDeviceToHost);
        
        // Display results
        std::cout << "CPU Result: ";
        for (auto v : cpuResult) std::cout << (int)v << " ";
        std::cout << "\nNPP Result: ";
        for (auto v : nppResult) std::cout << (int)v << " ";
        std::cout << "\n";
        
        // Compare
        int matches = 0;
        for (size_t i = 0; i < cpuResult.size(); i++) {
            if (cpuResult[i] == nppResult[i]) matches++;
            else {
                std::cout << "Mismatch at " << i << ": CPU=" << (int)cpuResult[i] 
                          << ", NPP=" << (int)nppResult[i] << "\n";
            }
        }
        
        std::cout << "Match: " << matches << "/" << cpuResult.size() 
                  << " (" << (100.0f * matches / cpuResult.size()) << "%)\n";
        
        cudaFree(d_src);
        cudaFree(d_dst);
    };
    
    testPattern(src1, "Simple [0,100,50,150]");
    testPattern(src2, "Gradient");
    
    return 0;
}
