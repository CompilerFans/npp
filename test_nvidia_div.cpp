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
    
    std::cout << "=== Testing NVIDIA NPP Division Behavior ===" << std::endl;
    
    const int width = 4;
    const int height = 1;
    NppiSize roi = {width, height};
    
    // Test different combinations
    std::vector<std::pair<Npp8u, Npp8u>> test_cases = {
        {100, 5},   // Expected: 20
        {200, 10},  // Expected: 20  
        {50, 2},    // Expected: 25
        {128, 4},   // Expected: 32
    };
    
    for (auto& test_case : test_cases) {
        Npp8u src1_val = test_case.first;
        Npp8u src2_val = test_case.second;
        
        std::vector<Npp8u> srcData1(width, src1_val);
        std::vector<Npp8u> srcData2(width, src2_val);
        
        // Allocate GPU memory
        int srcStep, dstStep;
        Npp8u* d_src1 = nppiMalloc_8u_C1(width, height, &srcStep);
        Npp8u* d_src2 = nppiMalloc_8u_C1(width, height, &srcStep);
        Npp8u* d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
        
        if (!d_src1 || !d_src2 || !d_dst) {
            std::cerr << "Memory allocation failed!" << std::endl;
            continue;
        }
        
        // Copy input to GPU
        cudaError_t copyStatus1 = cudaMemcpy(d_src1, srcData1.data(), width * sizeof(Npp8u), cudaMemcpyHostToDevice);
        cudaError_t copyStatus2 = cudaMemcpy(d_src2, srcData2.data(), width * sizeof(Npp8u), cudaMemcpyHostToDevice);
        
        if (copyStatus1 != cudaSuccess || copyStatus2 != cudaSuccess) {
            std::cerr << "  CUDA copy failed!" << std::endl;
            nppiFree(d_src1); nppiFree(d_src2); nppiFree(d_dst);
            continue;
        }
        
        // Test different scale factors
        for (int scaleFactor = 0; scaleFactor <= 2; scaleFactor++) {
            // Execute NVIDIA NPP function
            NppStatus status = nppiDiv_8u_C1RSfs(d_src1, srcStep, d_src2, srcStep, d_dst, dstStep, roi, scaleFactor);
            
            if (status != NPP_SUCCESS) {
                std::cout << "  ERROR: " << status << " for scale factor " << scaleFactor << std::endl;
                continue;
            }
            
            // Copy result back
            std::vector<Npp8u> resultData(width);
            cudaMemcpy(resultData.data(), d_dst, width * sizeof(Npp8u), cudaMemcpyDeviceToHost);
            
            std::cout << "Src1=" << (int)src1_val << ", Src2=" << (int)src2_val 
                      << ", ScaleFactor=" << scaleFactor
                      << ", Result=" << (int)resultData[0] 
                      << ", Expected=" << (int)(src1_val / src2_val) << std::endl;
        }
        
        nppiFree(d_src1);
        nppiFree(d_src2);
        nppiFree(d_dst);
    }
    
    return 0;
}
