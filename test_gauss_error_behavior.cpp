#include <iostream>
#include "npp.h"

int main() {
    std::cout << "Testing NVIDIA NPP Gaussian filter error handling behavior..." << std::endl;
    
    const int width = 16, height = 16;
    
    // Allocate memory
    int step;
    Npp8u* d_src = nppiMalloc_8u_C1(width, height, &step);
    Npp8u* d_dst = nppiMalloc_8u_C1(width, height, &step);
    
    NppiSize roi = {width, height};
    
    // Test 1: Null pointer
    std::cout << "\nTest 1: Null source pointer" << std::endl;
    NppStatus status = nppiFilterGauss_8u_C1R(nullptr, step, d_dst, step, roi, NPP_MASK_SIZE_3_X_3);
    std::cout << "Status: " << status << " (Expected: != 0)" << std::endl;
    
    // Test 2: Null destination pointer
    std::cout << "\nTest 2: Null destination pointer" << std::endl;
    status = nppiFilterGauss_8u_C1R(d_src, step, nullptr, step, roi, NPP_MASK_SIZE_3_X_3);
    std::cout << "Status: " << status << " (Expected: != 0)" << std::endl;
    
    // Test 3: Zero size ROI
    std::cout << "\nTest 3: Zero size ROI" << std::endl;
    NppiSize zeroRoi = {0, 0};
    status = nppiFilterGauss_8u_C1R(d_src, step, d_dst, step, zeroRoi, NPP_MASK_SIZE_3_X_3);
    std::cout << "Status: " << status << " (Expected: 0 for success)" << std::endl;
    
    // Test 4: Zero width ROI
    std::cout << "\nTest 4: Zero width ROI" << std::endl;
    NppiSize zeroWidthRoi = {0, height};
    status = nppiFilterGauss_8u_C1R(d_src, step, d_dst, step, zeroWidthRoi, NPP_MASK_SIZE_3_X_3);
    std::cout << "Status: " << status << " (Expected: 0 for success)" << std::endl;
    
    // Test 5: Zero height ROI
    std::cout << "\nTest 5: Zero height ROI" << std::endl;
    NppiSize zeroHeightRoi = {width, 0};
    status = nppiFilterGauss_8u_C1R(d_src, step, d_dst, step, zeroHeightRoi, NPP_MASK_SIZE_3_X_3);
    std::cout << "Status: " << status << " (Expected: 0 for success)" << std::endl;
    
    // Test 6: Unsupported mask size
    std::cout << "\nTest 6: Unsupported mask size" << std::endl;
    status = nppiFilterGauss_8u_C1R(d_src, step, d_dst, step, roi, NPP_MASK_SIZE_1_X_3);
    std::cout << "Status: " << status << " (Expected: != 0)" << std::endl;
    
    // Test 7: Valid operation
    std::cout << "\nTest 7: Valid operation" << std::endl;
    status = nppiFilterGauss_8u_C1R(d_src, step, d_dst, step, roi, NPP_MASK_SIZE_3_X_3);
    std::cout << "Status: " << status << " (Expected: 0 for success)" << std::endl;
    
    // Cleanup
    nppiFree(d_src);
    nppiFree(d_dst);
    
    std::cout << "\nAll tests completed!" << std::endl;
    return 0;
}