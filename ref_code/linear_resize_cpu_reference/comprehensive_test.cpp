#include "linear_resize_nvidia_compatible.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>
#include <npp.h>

struct TestCase {
    int srcW, srcH;
    int dstW, dstH;
    std::string description;
};

void runTest(const TestCase& tc) {
    int channels = 1;
    
    // Generate gradient test pattern
    std::vector<unsigned char> src(tc.srcW * tc.srcH * channels);
    for (int y = 0; y < tc.srcH; y++) {
        for (int x = 0; x < tc.srcW; x++) {
            int idx = (y * tc.srcW + x) * channels;
            // Gradient pattern for better interpolation visibility
            src[idx] = (unsigned char)((x * 255 / std::max(1, tc.srcW - 1)) * 0.5f + 
                                       (y * 255 / std::max(1, tc.srcH - 1)) * 0.5f);
        }
    }
    
    // CPU reference
    std::vector<unsigned char> cpuResult(tc.dstW * tc.dstH * channels);
    LinearResizeNvidiaCompatible<unsigned char>::resize(
        src.data(), tc.srcW * channels, tc.srcW, tc.srcH,
        cpuResult.data(), tc.dstW * channels, tc.dstW, tc.dstH, channels);
    
    // NVIDIA NPP
    unsigned char *d_src, *d_dst;
    cudaMalloc(&d_src, tc.srcW * tc.srcH * channels);
    cudaMalloc(&d_dst, tc.dstW * tc.dstH * channels);
    cudaMemcpy(d_src, src.data(), tc.srcW * tc.srcH * channels, cudaMemcpyHostToDevice);
    
    NppiSize srcSize = {tc.srcW, tc.srcH};
    NppiSize dstSize = {tc.dstW, tc.dstH};
    NppiRect srcROI = {0, 0, tc.srcW, tc.srcH};
    NppiRect dstROI = {0, 0, tc.dstW, tc.dstH};
    
    NppStatus status = nppiResize_8u_C1R(d_src, tc.srcW * channels, srcSize, srcROI,
                                          d_dst, tc.dstW * channels, dstSize, dstROI,
                                          NPPI_INTER_LINEAR);
    
    if (status != NPP_SUCCESS) {
        std::cerr << "NPP error: " << status << std::endl;
        cudaFree(d_src);
        cudaFree(d_dst);
        return;
    }
    
    std::vector<unsigned char> nppResult(tc.dstW * tc.dstH * channels);
    cudaMemcpy(nppResult.data(), d_dst, tc.dstW * tc.dstH * channels, cudaMemcpyDeviceToHost);
    
    // Compare
    int matches = 0;
    int total = tc.dstW * tc.dstH * channels;
    int maxDiff = 0;
    int totalDiff = 0;
    
    for (int i = 0; i < total; i++) {
        int diff = std::abs((int)cpuResult[i] - (int)nppResult[i]);
        if (diff == 0) matches++;
        maxDiff = std::max(maxDiff, diff);
        totalDiff += diff;
    }
    
    float matchPercent = 100.0f * matches / total;
    float avgDiff = (float)totalDiff / total;
    float scaleX = (float)tc.srcW / tc.dstW;
    float scaleY = (float)tc.srcH / tc.dstH;
    
    std::cout << std::setw(25) << std::left << tc.description
              << " | " << std::setw(9) << (std::to_string(tc.srcW) + "x" + std::to_string(tc.srcH))
              << " -> " << std::setw(9) << (std::to_string(tc.dstW) + "x" + std::to_string(tc.dstH))
              << " | Scale: " << std::fixed << std::setprecision(3) << scaleX << "x" << scaleY
              << " | Match: " << std::setw(6) << std::setprecision(2) << matchPercent << "%"
              << " | MaxDiff: " << std::setw(2) << maxDiff
              << " | AvgDiff: " << std::setprecision(3) << avgDiff
              << " | " << (matchPercent == 100.0f ? "✓ PERFECT" : (matchPercent >= 95.0f ? "✓ GOOD" : "✗ POOR"))
              << "\n";
    
    cudaFree(d_src);
    cudaFree(d_dst);
}

int main() {
    std::cout << "Comprehensive NVIDIA NPP Linear Interpolation Algorithm Test\n";
    std::cout << "Testing linear_resize_nvidia_compatible.h against NVIDIA NPP 12.6\n\n";
    
    std::vector<TestCase> tests;
    
    // Integer upscales
    tests.push_back({2, 2, 4, 4, "2x upscale"});
    tests.push_back({2, 2, 6, 6, "3x upscale"});
    tests.push_back({2, 2, 8, 8, "4x upscale"});
    tests.push_back({3, 3, 9, 9, "3x upscale (3->9)"});
    tests.push_back({4, 4, 8, 8, "2x upscale (4->8)"});
    tests.push_back({4, 4, 12, 12, "3x upscale (4->12)"});
    
    // Fractional upscales
    tests.push_back({2, 2, 3, 3, "1.5x upscale (2->3)"});
    tests.push_back({3, 3, 4, 4, "1.33x upscale (3->4)"});
    tests.push_back({3, 3, 5, 5, "1.67x upscale (3->5)"});
    tests.push_back({4, 4, 5, 5, "1.25x upscale (4->5)"});
    tests.push_back({4, 4, 6, 6, "1.5x upscale (4->6)"});
    tests.push_back({5, 5, 7, 7, "1.4x upscale (5->7)"});
    
    // Integer downscales
    tests.push_back({4, 4, 2, 2, "2x downscale"});
    tests.push_back({6, 6, 2, 2, "3x downscale"});
    tests.push_back({8, 8, 2, 2, "4x downscale"});
    tests.push_back({8, 8, 4, 4, "2x downscale (8->4)"});
    tests.push_back({9, 9, 3, 3, "3x downscale (9->3)"});
    tests.push_back({12, 12, 4, 4, "3x downscale (12->4)"});
    
    // Fractional downscales
    tests.push_back({3, 3, 2, 2, "1.5x downscale (3->2)"});
    tests.push_back({4, 4, 3, 3, "1.33x downscale (4->3)"});
    tests.push_back({5, 5, 3, 3, "1.67x downscale (5->3)"});
    tests.push_back({5, 5, 4, 4, "1.25x downscale (5->4)"});
    tests.push_back({6, 6, 4, 4, "1.5x downscale (6->4)"});
    tests.push_back({7, 7, 5, 5, "1.4x downscale (7->5)"});
    
    // Non-uniform scaling
    tests.push_back({2, 4, 4, 8, "2x upscale non-uniform"});
    tests.push_back({4, 2, 8, 4, "2x upscale non-uniform"});
    tests.push_back({2, 3, 4, 6, "2x upscale non-uniform"});
    tests.push_back({3, 2, 6, 4, "2x upscale non-uniform"});
    tests.push_back({4, 8, 2, 4, "2x downscale non-uniform"});
    tests.push_back({8, 4, 4, 2, "2x downscale non-uniform"});
    
    // Large images
    tests.push_back({10, 10, 20, 20, "2x upscale (large)"});
    tests.push_back({20, 20, 10, 10, "2x downscale (large)"});
    tests.push_back({16, 16, 32, 32, "2x upscale (16->32)"});
    tests.push_back({32, 32, 16, 16, "2x downscale (32->16)"});
    
    // Edge cases
    tests.push_back({1, 1, 2, 2, "1x1 -> 2x2"});
    tests.push_back({1, 1, 3, 3, "1x1 -> 3x3"});
    tests.push_back({2, 2, 1, 1, "2x2 -> 1x1"});
    tests.push_back({3, 3, 1, 1, "3x3 -> 1x1"});
    
    // Mixed scales (upscale in one dimension, downscale in other)
    tests.push_back({2, 4, 4, 2, "Up 2x, Down 2x"});
    tests.push_back({4, 2, 2, 4, "Down 2x, Up 2x"});
    tests.push_back({3, 6, 6, 3, "Up 2x, Down 2x"});
    
    std::cout << std::string(160, '-') << "\n";
    
    int perfectCount = 0;
    int goodCount = 0;
    int totalCount = 0;
    
    for (auto& test : tests) {
        runTest(test);
        totalCount++;
    }
    
    std::cout << std::string(160, '-') << "\n";
    std::cout << "\nTotal tests: " << totalCount << "\n";
    
    return 0;
}
