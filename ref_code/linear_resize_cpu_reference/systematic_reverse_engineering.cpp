#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include <npp.h>

struct DataPoint {
    float srcX, srcY;
    float fx, fy;
    int p00, p10, p01, p11;
    int npp_result;
    int pixel_idx;
};

void testPattern(const std::vector<unsigned char>& src, int srcW, int srcH, int dstW, int dstH, 
                 const std::string& name) {
    std::cout << "\n=== Testing: " << name << " ===\n";
    std::cout << "Source (" << srcW << "x" << srcH << "): ";
    for (auto v : src) std::cout << (int)v << " ";
    std::cout << "\n\n";
    
    int channels = 1;
    
    // Get NVIDIA NPP result
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
    
    // Analyze each pixel
    float scaleX = (float)srcW / dstW;
    float scaleY = (float)srcH / dstH;
    
    std::vector<DataPoint> dataPoints;
    
    for (int dy = 0; dy < dstH; dy++) {
        for (int dx = 0; dx < dstW; dx++) {
            int idx = dy * dstW + dx;
            
            float srcX = (dx + 0.5f) * scaleX - 0.5f;
            float srcY = (dy + 0.5f) * scaleY - 0.5f;
            
            srcX = std::max(0.0f, std::min((float)(srcW - 1), srcX));
            srcY = std::max(0.0f, std::min((float)(srcH - 1), srcY));
            
            int x0 = (int)std::floor(srcX);
            int y0 = (int)std::floor(srcY);
            int x1 = std::min(x0 + 1, srcW - 1);
            int y1 = std::min(y0 + 1, srcH - 1);
            
            float fx = srcX - x0;
            float fy = srcY - y0;
            
            int p00 = src[y0 * srcW + x0];
            int p10 = src[y0 * srcW + x1];
            int p01 = src[y1 * srcW + x0];
            int p11 = src[y1 * srcW + x1];
            
            DataPoint dp;
            dp.srcX = srcX;
            dp.srcY = srcY;
            dp.fx = fx;
            dp.fy = fy;
            dp.p00 = p00;
            dp.p10 = p10;
            dp.p01 = p01;
            dp.p11 = p11;
            dp.npp_result = nppResult[idx];
            dp.pixel_idx = idx;
            
            dataPoints.push_back(dp);
        }
    }
    
    // Analyze non-trivial interpolations (fx > 0 and fy > 0)
    std::cout << "Non-trivial interpolations (fx > 0 or fy > 0):\n";
    std::cout << std::string(120, '-') << "\n";
    std::cout << "Idx | fx    | fy    | p00 p10 p01 p11 | NPP | StdBilin | Diff\n";
    std::cout << std::string(120, '-') << "\n";
    
    for (auto& dp : dataPoints) {
        if (dp.fx > 0.001f || dp.fy > 0.001f) {
            // Standard bilinear
            float v0_std = dp.p00 * (1 - dp.fx) + dp.p10 * dp.fx;
            float v1_std = dp.p01 * (1 - dp.fx) + dp.p11 * dp.fx;
            float result_std = v0_std * (1 - dp.fy) + v1_std * dp.fy;
            int std_rounded = (int)std::floor(result_std + 0.5f);
            
            std::cout << std::setw(3) << dp.pixel_idx << " | "
                      << std::fixed << std::setprecision(3) 
                      << std::setw(5) << dp.fx << " | "
                      << std::setw(5) << dp.fy << " | "
                      << std::setw(3) << dp.p00 << " "
                      << std::setw(3) << dp.p10 << " "
                      << std::setw(3) << dp.p01 << " "
                      << std::setw(3) << dp.p11 << " | "
                      << std::setw(3) << dp.npp_result << " | "
                      << std::setw(8) << result_std << " | "
                      << std::setw(4) << (result_std - dp.npp_result)
                      << "\n";
        }
    }
    
    cudaFree(d_src);
    cudaFree(d_dst);
}

int main() {
    std::cout << "Systematic Reverse Engineering of NVIDIA NPP Linear Interpolation\n";
    std::cout << "==================================================================\n";
    
    // Test multiple patterns
    testPattern({0, 100, 50, 150}, 2, 2, 3, 3, "Pattern 1: [0,100,50,150]");
    testPattern({0, 127, 127, 255}, 2, 2, 3, 3, "Pattern 2: [0,127,127,255]");
    testPattern({0, 255, 0, 255}, 2, 2, 3, 3, "Pattern 3: [0,255,0,255]");
    testPattern({100, 200, 100, 200}, 2, 2, 3, 3, "Pattern 4: [100,200,100,200]");
    
    return 0;
}
