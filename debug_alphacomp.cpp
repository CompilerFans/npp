#include "npp.h"
#include <iostream>
#include <vector>

int main() {
    // Test data from the failing test
    std::vector<Npp8u> hostSrc1 = {100, 150, 200, 50, 75, 125};
    std::vector<Npp8u> hostSrc2 = {50, 75, 100, 200, 150, 125};
    Npp8u alpha1 = 128; // 0.5 in 8-bit
    Npp8u alpha2 = 192; // 0.75 in 8-bit
    
    std::cout << "Input data:" << std::endl;
    std::cout << "Src1: ";
    for (auto v : hostSrc1) std::cout << (int)v << " ";
    std::cout << std::endl;
    
    std::cout << "Src2: ";
    for (auto v : hostSrc2) std::cout << (int)v << " ";
    std::cout << std::endl;
    
    std::cout << "Alpha1: " << (int)alpha1 << " (" << (alpha1/255.0) << ")" << std::endl;
    std::cout << "Alpha2: " << (int)alpha2 << " (" << (alpha2/255.0) << ")" << std::endl;
    
    const int width = 3;
    const int height = 2;
    
    // Allocate GPU memory
    int src1Step, src2Step, dstStep;
    Npp8u *d_src1 = nppiMalloc_8u_C1(width, height, &src1Step);
    Npp8u *d_src2 = nppiMalloc_8u_C1(width, height, &src2Step);
    Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
    
    // Copy data to GPU
    int hostStep = width * sizeof(Npp8u);
    cudaMemcpy2D(d_src1, src1Step, hostSrc1.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_src2, src2Step, hostSrc2.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
    
    // Test ALPHA_IN operation
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiAlphaCompC_8u_C1R(d_src1, src1Step, alpha1, 
                                             d_src2, src2Step, alpha2,
                                             d_dst, dstStep, oSizeROI, NPPI_OP_ALPHA_IN);
    
    std::cout << "ALPHA_IN Status: " << status << std::endl;
    
    // Copy result back
    std::vector<Npp8u> hostResult(6);
    cudaMemcpy2D(hostResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);
    
    std::cout << "NVIDIA ALPHA_IN result: ";
    for (auto v : hostResult) std::cout << (int)v << " ";
    std::cout << std::endl;
    
    // Test ALPHA_OVER_PREMUL operation
    status = nppiAlphaCompC_8u_C1R(d_src1, src1Step, alpha1, 
                                   d_src2, src2Step, alpha2,
                                   d_dst, dstStep, oSizeROI, NPPI_OP_ALPHA_OVER_PREMUL);
    
    std::cout << "ALPHA_OVER_PREMUL Status: " << status << std::endl;
    
    if (status == NPP_SUCCESS) {
        cudaMemcpy2D(hostResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);
        std::cout << "NVIDIA ALPHA_OVER_PREMUL result: ";
        for (auto v : hostResult) std::cout << (int)v << " ";
        std::cout << std::endl;
    }
    
    // Test all operations to understand the pattern
    std::vector<NppiAlphaOp> ops = {NPPI_OP_ALPHA_OVER, NPPI_OP_ALPHA_IN, NPPI_OP_ALPHA_OUT, 
                                    NPPI_OP_ALPHA_ATOP, NPPI_OP_ALPHA_XOR, NPPI_OP_ALPHA_PLUS};
    std::vector<std::string> opNames = {"OVER", "IN", "OUT", "ATOP", "XOR", "PLUS"};
    
    for (size_t i = 0; i < ops.size(); ++i) {
        status = nppiAlphaCompC_8u_C1R(d_src1, src1Step, alpha1, 
                                       d_src2, src2Step, alpha2,
                                       d_dst, dstStep, oSizeROI, ops[i]);
        if (status == NPP_SUCCESS) {
            cudaMemcpy2D(hostResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);
            std::cout << "NVIDIA " << opNames[i] << ": ";
            for (auto v : hostResult) std::cout << (int)v << " ";
            std::cout << std::endl;
        } else {
            std::cout << "NVIDIA " << opNames[i] << ": FAILED (status=" << status << ")" << std::endl;
        }
    }
    
    // Cleanup
    nppiFree(d_src1);
    nppiFree(d_src2);
    nppiFree(d_dst);
    
    return 0;
}