#include "npp.h"
#include <iostream>
#include <vector>

int main() {
    // Simple test case
    std::vector<Npp8u> hostSrc1 = {100};
    std::vector<Npp8u> hostSrc2 = {200};
    Npp8u alpha1 = 128; // 0.5
    Npp8u alpha2 = 192; // 0.75
    
    const int width = 1;
    const int height = 1;
    
    // Allocate GPU memory
    int src1Step, src2Step, dstStep;
    Npp8u *d_src1 = nppiMalloc_8u_C1(width, height, &src1Step);
    Npp8u *d_src2 = nppiMalloc_8u_C1(width, height, &src2Step);
    Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
    
    // Copy data to GPU
    int hostStep = width * sizeof(Npp8u);
    cudaMemcpy2D(d_src1, src1Step, hostSrc1.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_src2, src2Step, hostSrc2.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
    
    NppiSize oSizeROI = {width, height};
    std::vector<Npp8u> hostResult(1);
    
    std::cout << "Input: src1=" << (int)hostSrc1[0] << ", src2=" << (int)hostSrc2[0] << std::endl;
    std::cout << "Alpha1=" << (int)alpha1 << " (" << (alpha1/255.0) << "), Alpha2=" << (int)alpha2 << " (" << (alpha2/255.0) << ")" << std::endl;
    std::cout << std::endl;
    
    // Test all operations
    std::vector<NppiAlphaOp> ops = {
        NPPI_OP_ALPHA_OVER, NPPI_OP_ALPHA_IN, NPPI_OP_ALPHA_OUT, 
        NPPI_OP_ALPHA_ATOP, NPPI_OP_ALPHA_XOR, NPPI_OP_ALPHA_PLUS,
        NPPI_OP_ALPHA_OVER_PREMUL, NPPI_OP_ALPHA_IN_PREMUL, NPPI_OP_ALPHA_OUT_PREMUL,
        NPPI_OP_ALPHA_ATOP_PREMUL, NPPI_OP_ALPHA_XOR_PREMUL, NPPI_OP_ALPHA_PLUS_PREMUL
    };
    
    std::vector<std::string> opNames = {
        "OVER", "IN", "OUT", "ATOP", "XOR", "PLUS",
        "OVER_PREMUL", "IN_PREMUL", "OUT_PREMUL", "ATOP_PREMUL", "XOR_PREMUL", "PLUS_PREMUL"
    };
    
    for (size_t i = 0; i < ops.size(); ++i) {
        NppStatus status = nppiAlphaCompC_8u_C1R(d_src1, src1Step, alpha1, 
                                                 d_src2, src2Step, alpha2,
                                                 d_dst, dstStep, oSizeROI, ops[i]);
        if (status == NPP_SUCCESS) {
            cudaMemcpy2D(hostResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);
            std::cout << opNames[i] << ": " << (int)hostResult[0] << std::endl;
        } else {
            std::cout << opNames[i] << ": FAILED (status=" << status << ")" << std::endl;
        }
    }
    
    // Cleanup
    nppiFree(d_src1);
    nppiFree(d_src2);
    nppiFree(d_dst);
    
    return 0;
}