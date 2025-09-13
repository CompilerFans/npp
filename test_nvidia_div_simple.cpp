#include <iostream>
#include <vector>
#include <npp.h>
#include <cuda_runtime.h>

int main() {
    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaSetDevice failed! Error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }
    
    std::cout << "=== Simple NVIDIA NPP Division Test ===" << std::endl;
    
    // Very simple test: 2x2 image
    const int width = 2;
    const int height = 2;
    NppiSize roi = {width, height};
    
    // Simple test data
    std::vector<Npp8u> srcData1 = {20, 40, 60, 80};  // Divisors
    std::vector<Npp8u> srcData2 = {4, 8, 12, 16};    // Dividends
    // Expected results: 20/4=5, 40/8=5, 60/12=5, 80/16=5
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp8u* d_src1 = nppiMalloc_8u_C1(width, height, &srcStep);
    Npp8u* d_src2 = nppiMalloc_8u_C1(width, height, &srcStep);
    Npp8u* d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
    
    if (!d_src1 || !d_src2 || !d_dst) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return 1;
    }
    
    std::cout << "Step sizes: src=" << srcStep << ", dst=" << dstStep << std::endl;
    std::cout << "Input data:" << std::endl;
    std::cout << "  Src1 (dividend): ";
    for (int val : srcData1) std::cout << (int)val << " ";
    std::cout << std::endl;
    std::cout << "  Src2 (divisor): ";
    for (int val : srcData2) std::cout << (int)val << " ";
    std::cout << std::endl;
    
    // Copy input to GPU row by row to handle step properly
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src1 + y * srcStep, srcData1.data() + y * width, width * sizeof(Npp8u), cudaMemcpyHostToDevice);
        cudaMemcpy((char*)d_src2 + y * srcStep, srcData2.data() + y * width, width * sizeof(Npp8u), cudaMemcpyHostToDevice);
    }
    
    // Test division with different scale factors
    for (int scaleFactor = 0; scaleFactor <= 3; scaleFactor++) {
        std::cout << "\nScale Factor: " << scaleFactor << " (result will be divided by 2^" << scaleFactor << ")" << std::endl;
        
        // Execute NVIDIA NPP function
        NppStatus status = nppiDiv_8u_C1RSfs(d_src1, srcStep, d_src2, srcStep, d_dst, dstStep, roi, scaleFactor);
        
        if (status != NPP_SUCCESS) {
            std::cout << "  ERROR: nppiDiv returned status " << status << std::endl;
            continue;
        }
        
        // Copy result back row by row
        std::vector<Npp8u> resultData(width * height);
        for (int y = 0; y < height; y++) {
            cudaMemcpy(resultData.data() + y * width, (char*)d_dst + y * dstStep, width * sizeof(Npp8u), cudaMemcpyDeviceToHost);
        }
        
        std::cout << "  Results: ";
        for (int i = 0; i < width * height; i++) {
            float expected = (float)srcData1[i] / srcData2[i];
            if (scaleFactor > 0) expected /= (1 << scaleFactor);
            std::cout << (int)resultData[i] << " (expected ~" << expected << ") ";
        }
        std::cout << std::endl;
    }
    
    nppiFree(d_src1);
    nppiFree(d_src2);
    nppiFree(d_dst);
    
    return 0;
}