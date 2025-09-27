#include "npp.h"
#include <iostream>
#include <vector>

int main() {
    // Test case from AlphaPremulC_8u_C3R_MultiChannel
    const int width = 2;
    const int height = 2;
    const int channels = 3;
    const int totalPixels = width * height * channels;

    std::vector<Npp8u> hostSrc = {255, 128, 64,    192, 96, 32,    // Pixel 1, 2
                                  128, 255, 0,     64, 32, 255};   // Pixel 3, 4
    Npp8u alpha = 128; // 0.5 in 8-bit

    // Allocate GPU memory
    int srcStep, dstStep;
    Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
    Npp8u *d_dst = nppiMalloc_8u_C3(width, height, &dstStep);

    // Copy data to GPU
    int hostStep = width * channels * sizeof(Npp8u);
    cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

    // Execute NVIDIA NPP operation
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiAlphaPremulC_8u_C3R(d_src, srcStep, alpha, d_dst, dstStep, oSizeROI);
    std::cout << "NPP Status: " << status << std::endl;

    // Copy result back
    std::vector<Npp8u> nvidiaResult(totalPixels);
    cudaMemcpy2D(nvidiaResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);

    std::cout << "Input data:" << std::endl;
    for (int i = 0; i < totalPixels; ++i) {
        std::cout << "src[" << i << "] = " << static_cast<int>(hostSrc[i]) << std::endl;
    }

    std::cout << "\nNVIDIA NPP results (alpha=" << static_cast<int>(alpha) << "):" << std::endl;
    for (int i = 0; i < totalPixels; ++i) {
        std::cout << "result[" << i << "] = " << static_cast<int>(nvidiaResult[i]);
        
        // My simple calculation: src * alpha / 255
        int myResult = (hostSrc[i] * alpha) / 255;
        std::cout << ", my_calc = " << myResult;
        
        // Rounded calculation: (src * alpha + 127) / 255
        int roundedResult = (hostSrc[i] * alpha + 127) / 255;
        std::cout << ", rounded = " << roundedResult;
        
        // Exact division: src * alpha / 255.0 (rounded)
        double exactResult = (static_cast<double>(hostSrc[i]) * alpha) / 255.0;
        int exactRounded = static_cast<int>(exactResult + 0.5);
        std::cout << ", exact_rounded = " << exactRounded;
        
        std::cout << std::endl;
    }

    // Test AC4 case
    std::cout << "\n=== AC4 Test ===" << std::endl;
    
    const int ac4_channels = 4;
    std::vector<Npp8u> hostSrcAC4 = {255, 128, 64, 192,    // Pixel 1: R=255, G=128, B=64, A=192
                                     128, 255, 32, 128};   // Pixel 2: R=128, G=255, B=32, A=128

    Npp8u *d_srcAC4 = nppiMalloc_8u_C4(2, 1, &srcStep);
    Npp8u *d_dstAC4 = nppiMalloc_8u_C4(2, 1, &dstStep);

    int hostStepAC4 = 2 * ac4_channels * sizeof(Npp8u);
    cudaMemcpy2D(d_srcAC4, srcStep, hostSrcAC4.data(), hostStepAC4, hostStepAC4, 1, cudaMemcpyHostToDevice);

    NppiSize ac4SizeROI = {2, 1};
    status = nppiAlphaPremul_8u_AC4R(d_srcAC4, srcStep, d_dstAC4, dstStep, ac4SizeROI);
    std::cout << "AC4 NPP Status: " << status << std::endl;

    std::vector<Npp8u> nvidiaResultAC4(8);
    cudaMemcpy2D(nvidiaResultAC4.data(), hostStepAC4, d_dstAC4, dstStep, hostStepAC4, 1, cudaMemcpyDeviceToHost);

    std::cout << "AC4 Input data (RGBA):" << std::endl;
    for (int p = 0; p < 2; ++p) {
        int idx = p * 4;
        std::cout << "Pixel " << p << ": R=" << static_cast<int>(hostSrcAC4[idx]) 
                  << " G=" << static_cast<int>(hostSrcAC4[idx+1])
                  << " B=" << static_cast<int>(hostSrcAC4[idx+2])
                  << " A=" << static_cast<int>(hostSrcAC4[idx+3]) << std::endl;
    }

    std::cout << "AC4 NVIDIA NPP results:" << std::endl;
    for (int p = 0; p < 2; ++p) {
        int idx = p * 4;
        Npp8u pixelAlpha = hostSrcAC4[idx + 3];
        std::cout << "Pixel " << p << ": ";
        
        for (int c = 0; c < 4; ++c) {
            std::cout << "Ch" << c << "=" << static_cast<int>(nvidiaResultAC4[idx + c]);
            if (c < 3) { // RGB channels
                int myResult = (hostSrcAC4[idx + c] * pixelAlpha) / 255;
                int roundedResult = (hostSrcAC4[idx + c] * pixelAlpha + 127) / 255;
                double exactResult = (static_cast<double>(hostSrcAC4[idx + c]) * pixelAlpha) / 255.0;
                int exactRounded = static_cast<int>(exactResult + 0.5);
                std::cout << "(my:" << myResult << ",round:" << roundedResult << ",exact:" << exactRounded << ")";
            }
            std::cout << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    nppiFree(d_src);
    nppiFree(d_dst);
    nppiFree(d_srcAC4);
    nppiFree(d_dstAC4);

    return 0;
}