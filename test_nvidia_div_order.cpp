#include <iostream>
#include <vector>
#include <npp.h>
#include <cuda_runtime.h>

int main() {
    cudaSetDevice(0);
    std::cout << "=== Testing Parameter Order for NVIDIA NPP Division ===" << std::endl;
    
    const int width = 1;
    const int height = 1;
    NppiSize roi = {width, height};
    
    // Test both parameter orders to see which one works
    Npp8u val1 = 100;  // Large value 
    Npp8u val2 = 5;    // Small value
    
    std::vector<Npp8u> data1 = {val1};
    std::vector<Npp8u> data2 = {val2};
    
    // Allocate GPU memory
    int step;
    Npp8u* d_val1 = nppiMalloc_8u_C1(width, height, &step);
    Npp8u* d_val2 = nppiMalloc_8u_C1(width, height, &step);
    Npp8u* d_result = nppiMalloc_8u_C1(width, height, &step);
    
    if (!d_val1 || !d_val2 || !d_result) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return 1;
    }
    
    // Copy data
    cudaMemcpy(d_val1, data1.data(), sizeof(Npp8u), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val2, data2.data(), sizeof(Npp8u), cudaMemcpyHostToDevice);
    
    // Test 1: val1 / val2 (100 / 5 = 20)
    std::cout << "\nTest 1: nppiDiv_8u_C1RSfs(100, step, 5, step, result, step, roi, 0)" << std::endl;
    NppStatus status1 = nppiDiv_8u_C1RSfs(d_val1, step, d_val2, step, d_result, step, roi, 0);
    
    if (status1 == NPP_SUCCESS) {
        Npp8u result1;
        cudaMemcpy(&result1, d_result, sizeof(Npp8u), cudaMemcpyDeviceToHost);
        std::cout << "  Status: SUCCESS, Result: " << (int)result1 << std::endl;
    } else {
        std::cout << "  Status: ERROR " << status1 << std::endl;
    }
    
    // Test 2: val2 / val1 (5 / 100 = 0.05)
    std::cout << "\nTest 2: nppiDiv_8u_C1RSfs(5, step, 100, step, result, step, roi, 0)" << std::endl;
    NppStatus status2 = nppiDiv_8u_C1RSfs(d_val2, step, d_val1, step, d_result, step, roi, 0);
    
    if (status2 == NPP_SUCCESS) {
        Npp8u result2;
        cudaMemcpy(&result2, d_result, sizeof(Npp8u), cudaMemcpyDeviceToHost);
        std::cout << "  Status: SUCCESS, Result: " << (int)result2 << std::endl;
    } else {
        std::cout << "  Status: ERROR " << status2 << std::endl;
    }
    
    // Test 3: Try 32f division to see if it works
    std::cout << "\nTest 3: Trying nppiDiv_32f_C1R for comparison" << std::endl;
    
    Npp32f* d_val1_32f = nppiMalloc_32f_C1(width, height, &step);
    Npp32f* d_val2_32f = nppiMalloc_32f_C1(width, height, &step);
    Npp32f* d_result_32f = nppiMalloc_32f_C1(width, height, &step);
    
    if (d_val1_32f && d_val2_32f && d_result_32f) {
        float fval1 = 100.0f, fval2 = 5.0f;
        cudaMemcpy(d_val1_32f, &fval1, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_val2_32f, &fval2, sizeof(float), cudaMemcpyHostToDevice);
        
        NppStatus status3 = nppiDiv_32f_C1R(d_val1_32f, step, d_val2_32f, step, d_result_32f, step, roi);
        
        if (status3 == NPP_SUCCESS) {
            float result3;
            cudaMemcpy(&result3, d_result_32f, sizeof(float), cudaMemcpyDeviceToHost);
            std::cout << "  32f division result: " << result3 << " (expected: 20.0)" << std::endl;
        } else {
            std::cout << "  32f division failed with status: " << status3 << std::endl;
        }
    }
    
    // Cleanup
    nppiFree(d_val1);
    nppiFree(d_val2);
    nppiFree(d_result);
    if (d_val1_32f) nppiFree(d_val1_32f);
    if (d_val2_32f) nppiFree(d_val2_32f);
    if (d_result_32f) nppiFree(d_result_32f);
    
    return 0;
}