#include <iostream>
#include <vector>
#include "npp.h"
#include <cuda_runtime.h>

// Test basic functionality of newly implemented functions
int main() {
    std::cout << "Testing newly implemented NPP functions..." << std::endl;
    
    // Test Resize functions
    {
        std::cout << "Testing nppiResize functions..." << std::endl;
        
        // Test data for 16u
        std::vector<Npp16u> srcData16u(64 * 64, 1000);
        std::vector<Npp16u> dstData16u(32 * 32, 0);
        
        Npp16u* d_src16u = nullptr;
        Npp16u* d_dst16u = nullptr;
        
        cudaMalloc(&d_src16u, srcData16u.size() * sizeof(Npp16u));
        cudaMalloc(&d_dst16u, dstData16u.size() * sizeof(Npp16u));
        
        cudaMemcpy(d_src16u, srcData16u.data(), srcData16u.size() * sizeof(Npp16u), cudaMemcpyHostToDevice);
        
        NppiSize srcSize = {64, 64};
        NppiSize dstSize = {32, 32};
        NppiRect srcROI = {0, 0, 64, 64};
        NppiRect dstROI = {0, 0, 32, 32};
        
        NppStreamContext ctx;
        ctx.hStream = 0;
        
        NppStatus status = nppiResize_16u_C1R_Ctx(d_src16u, 64 * sizeof(Npp16u), srcSize, srcROI,
                                                  d_dst16u, 32 * sizeof(Npp16u), dstSize, dstROI,
                                                  0, ctx);
        
        if (status == NPP_SUCCESS) {
            std::cout << "nppiResize_16u_C1R_Ctx: SUCCESS" << std::endl;
        } else {
            std::cout << "nppiResize_16u_C1R_Ctx: FAILED with status " << status << std::endl;
        }
        
        cudaFree(d_src16u);
        cudaFree(d_dst16u);
    }
    
    // Test NV12ToBGR functions
    {
        std::cout << "Testing nppiNV12ToBGR functions..." << std::endl;
        
        int width = 64, height = 64;
        std::vector<Npp8u> yData(width * height, 128);
        std::vector<Npp8u> uvData(width * height / 2, 128);
        std::vector<Npp8u> bgrData(width * height * 3, 0);
        
        Npp8u* d_y = nullptr;
        Npp8u* d_uv = nullptr;
        Npp8u* d_bgr = nullptr;
        
        cudaMalloc(&d_y, yData.size());
        cudaMalloc(&d_uv, uvData.size());
        cudaMalloc(&d_bgr, bgrData.size());
        
        cudaMemcpy(d_y, yData.data(), yData.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_uv, uvData.data(), uvData.size(), cudaMemcpyHostToDevice);
        
        const Npp8u* pSrc[2] = {d_y, d_uv};
        NppiSize roi = {width, height};
        
        NppStreamContext ctx;
        ctx.hStream = 0;
        
        NppStatus status = nppiNV12ToBGR_8u_P2C3R_Ctx(pSrc, width, d_bgr, width * 3, roi, ctx);
        
        if (status == NPP_SUCCESS || status == NPP_NO_ERROR) {
            std::cout << "nppiNV12ToBGR_8u_P2C3R_Ctx: SUCCESS" << std::endl;
        } else {
            std::cout << "nppiNV12ToBGR_8u_P2C3R_Ctx: FAILED with status " << status << std::endl;
        }
        
        cudaFree(d_y);
        cudaFree(d_uv);
        cudaFree(d_bgr);
    }
    
    // Test Set functions
    {
        std::cout << "Testing nppiSet functions..." << std::endl;
        
        int width = 32, height = 32;
        std::vector<Npp8u> dstData8u(width * height, 0);
        
        Npp8u* d_dst8u = nullptr;
        cudaMalloc(&d_dst8u, dstData8u.size());
        
        NppiSize roi = {width, height};
        NppStreamContext ctx;
        ctx.hStream = 0;
        
        // Test single channel set
        NppStatus status = nppiSet_8u_C1R_Ctx(100, d_dst8u, width, roi, ctx);
        
        if (status == NPP_SUCCESS) {
            std::cout << "nppiSet_8u_C1R_Ctx: SUCCESS" << std::endl;
        } else {
            std::cout << "nppiSet_8u_C1R_Ctx: FAILED with status " << status << std::endl;
        }
        
        cudaFree(d_dst8u);
        
        // Test three channel set
        std::vector<Npp8u> dstData8u_C3(width * height * 3, 0);
        Npp8u* d_dst8u_C3 = nullptr;
        cudaMalloc(&d_dst8u_C3, dstData8u_C3.size());
        
        Npp8u values[3] = {50, 100, 150};
        status = nppiSet_8u_C3R_Ctx(values, d_dst8u_C3, width * 3, roi, ctx);
        
        if (status == NPP_SUCCESS) {
            std::cout << "nppiSet_8u_C3R_Ctx: SUCCESS" << std::endl;
        } else {
            std::cout << "nppiSet_8u_C3R_Ctx: FAILED with status " << status << std::endl;
        }
        
        cudaFree(d_dst8u_C3);
    }
    
    std::cout << "Testing completed!" << std::endl;
    return 0;
}