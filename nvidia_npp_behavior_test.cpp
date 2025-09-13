#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <npp.h>

int main() {
    // Initialize CUDA
    cudaSetDevice(0);
    
    std::cout << "=== NVIDIA NPP Behavior Testing ===" << std::endl;
    
    // Test nppiExp_8u_C1RSfs behavior
    std::cout << "\n--- Testing nppiExp_8u_C1RSfs ---" << std::endl;
    
    const int width = 6;
    const int height = 1;
    NppiSize roi = {width, height};
    
    // Create test input data
    std::vector<Npp8u> srcData = {0, 1, 2, 3, 4, 5};
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C1(width, height, &srcStep);
    Npp8u* d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
    
    if (!d_src || !d_dst) {
        std::cerr << "GPU memory allocation failed!" << std::endl;
        return -1;
    }
    
    // Copy input to GPU
    cudaMemcpy(d_src, srcData.data(), width * sizeof(Npp8u), cudaMemcpyHostToDevice);
    
    // Test different scale factors
    for (int scaleFactor = 0; scaleFactor <= 4; scaleFactor++) {
        std::cout << "Scale factor: " << scaleFactor << std::endl;
        
        // Execute NVIDIA NPP function
        NppStatus status = nppiExp_8u_C1RSfs(d_src, srcStep, d_dst, dstStep, roi, scaleFactor);
        
        if (status != NPP_SUCCESS) {
            std::cout << "  ERROR: " << status << std::endl;
            continue;
        }
        
        // Copy result back
        std::vector<Npp8u> resultData(width);
        cudaMemcpy(resultData.data(), d_dst, width * sizeof(Npp8u), cudaMemcpyDeviceToHost);
        
        // Print results
        std::cout << "  Input:  ";
        for (int i = 0; i < width; i++) std::cout << (int)srcData[i] << " ";
        std::cout << std::endl;
        
        std::cout << "  Output: ";
        for (int i = 0; i < width; i++) std::cout << (int)resultData[i] << " ";
        std::cout << std::endl;
        
        std::cout << "  Expected (e^x >> " << scaleFactor << "): ";
        for (int i = 0; i < width; i++) {
            double expected = exp(srcData[i]) / (1 << scaleFactor);
            std::cout << (int)std::min(255.0, expected) << " ";
        }
        std::cout << std::endl << std::endl;
    }
    
    // Test nppiExp_32f_C1R behavior
    std::cout << "\n--- Testing nppiExp_32f_C1R ---" << std::endl;
    
    std::vector<Npp32f> srcData32f = {0.0f, 1.0f, 2.0f, 3.0f, -1.0f, -2.0f};
    
    Npp32f* d_src32f = nppiMalloc_32f_C1(width, height, &srcStep);
    Npp32f* d_dst32f = nppiMalloc_32f_C1(width, height, &dstStep);
    
    if (d_src32f && d_dst32f) {
        cudaMemcpy(d_src32f, srcData32f.data(), width * sizeof(Npp32f), cudaMemcpyHostToDevice);
        
        NppStatus status = nppiExp_32f_C1R(d_src32f, srcStep, d_dst32f, dstStep, roi);
        
        if (status == NPP_SUCCESS) {
            std::vector<Npp32f> resultData32f(width);
            cudaMemcpy(resultData32f.data(), d_dst32f, width * sizeof(Npp32f), cudaMemcpyDeviceToHost);
            
            std::cout << "Input:  ";
            for (int i = 0; i < width; i++) std::cout << srcData32f[i] << " ";
            std::cout << std::endl;
            
            std::cout << "Output: ";
            for (int i = 0; i < width; i++) std::cout << resultData32f[i] << " ";
            std::cout << std::endl;
            
            std::cout << "Expected: ";
            for (int i = 0; i < width; i++) std::cout << exp(srcData32f[i]) << " ";
            std::cout << std::endl;
        } else {
            std::cout << "ERROR: " << status << std::endl;
        }
        
        nppiFree(d_src32f);
        nppiFree(d_dst32f);
    }
    
    // Test nppiLn_8u_C1RSfs behavior
    std::cout << "\n--- Testing nppiLn_8u_C1RSfs ---" << std::endl;
    
    std::vector<Npp8u> lnSrcData = {1, 2, 3, 4, 5, 6}; // ln(0) is undefined, so start from 1
    
    // Copy input to GPU
    cudaMemcpy(d_src, lnSrcData.data(), width * sizeof(Npp8u), cudaMemcpyHostToDevice);
    
    // Test different scale factors for Ln
    for (int scaleFactor = 0; scaleFactor <= 3; scaleFactor++) {
        std::cout << "Ln Scale factor: " << scaleFactor << std::endl;
        
        // Execute NVIDIA NPP function
        NppStatus status = nppiLn_8u_C1RSfs(d_src, srcStep, d_dst, dstStep, roi, scaleFactor);
        
        if (status != NPP_SUCCESS) {
            std::cout << "  ERROR: " << status << std::endl;
            continue;
        }
        
        // Copy result back
        std::vector<Npp8u> lnResultData(width);
        cudaMemcpy(lnResultData.data(), d_dst, width * sizeof(Npp8u), cudaMemcpyDeviceToHost);
        
        // Print results
        std::cout << "  Input:  ";
        for (int i = 0; i < width; i++) std::cout << (int)lnSrcData[i] << " ";
        std::cout << std::endl;
        
        std::cout << "  Output: ";
        for (int i = 0; i < width; i++) std::cout << (int)lnResultData[i] << " ";
        std::cout << std::endl;
        
        std::cout << "  Expected (ln(x) << " << scaleFactor << "): ";
        for (int i = 0; i < width; i++) {
            double expected = log(lnSrcData[i]) * (1 << scaleFactor);
            std::cout << (int)std::max(0.0, std::min(255.0, expected)) << " ";
        }
        std::cout << std::endl << std::endl;
    }
    
    // Test nppiLn_32f_C1R behavior
    std::cout << "\n--- Testing nppiLn_32f_C1R ---" << std::endl;
    
    std::vector<Npp32f> lnSrcData32f = {1.0f, 2.0f, 3.0f, 4.0f, 0.5f, 10.0f};
    
    if (d_src32f && d_dst32f) {
        cudaMemcpy(d_src32f, lnSrcData32f.data(), width * sizeof(Npp32f), cudaMemcpyHostToDevice);
        
        NppStatus status = nppiLn_32f_C1R(d_src32f, srcStep, d_dst32f, dstStep, roi);
        
        if (status == NPP_SUCCESS) {
            std::vector<Npp32f> lnResultData32f(width);
            cudaMemcpy(lnResultData32f.data(), d_dst32f, width * sizeof(Npp32f), cudaMemcpyDeviceToHost);
            
            std::cout << "Input:  ";
            for (int i = 0; i < width; i++) std::cout << lnSrcData32f[i] << " ";
            std::cout << std::endl;
            
            std::cout << "Output: ";
            for (int i = 0; i < width; i++) std::cout << lnResultData32f[i] << " ";
            std::cout << std::endl;
            
            std::cout << "Expected: ";
            for (int i = 0; i < width; i++) std::cout << log(lnSrcData32f[i]) << " ";
            std::cout << std::endl;
        } else {
            std::cout << "ERROR: " << status << std::endl;
        }
    }
    
    // Clean up
    nppiFree(d_src);
    nppiFree(d_dst);
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    return 0;
}