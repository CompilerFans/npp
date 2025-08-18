#include "npp.h"
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

/**
 * Test NPPI Support Functions (Memory Management)
 */

void testNppi8uMemoryAllocation() {
    std::cout << "=== Testing 8u Memory Allocation ===" << std::endl;
    
    const int width = 640;
    const int height = 480;
    int stepBytes;
    int widthBytes;

    // Test 8u C1
    Npp8u* ptr_8u_c1 = nppiMalloc_8u_C1(width, height, &stepBytes);
    assert(ptr_8u_c1 != nullptr);
    widthBytes = width * sizeof(Npp8u);
    assert(stepBytes >= widthBytes);
    std::cout << "8u C1: ptr=" << (void*)ptr_8u_c1 << ", widthBytes=" << widthBytes << ", step=" << stepBytes << std::endl;
    nppiFree(ptr_8u_c1);
    
    // Test 8u C2
    Npp8u* ptr_8u_c2 = nppiMalloc_8u_C2(width, height, &stepBytes);
    assert(ptr_8u_c2 != nullptr);
    widthBytes = width * 2 * sizeof(Npp8u);
    assert(stepBytes >= widthBytes);
    std::cout << "8u C2: ptr=" << (void*)ptr_8u_c2 << ", widthBytes=" << widthBytes << ", step=" << stepBytes << std::endl;
    nppiFree(ptr_8u_c2);
    
    // Test 8u C3
    Npp8u* ptr_8u_c3 = nppiMalloc_8u_C3(width, height, &stepBytes);
    assert(ptr_8u_c3 != nullptr);
    widthBytes = width * 3 * sizeof(Npp8u);
    assert(stepBytes >= widthBytes);
    std::cout << "8u C3: ptr=" << (void*)ptr_8u_c3 << ", widthBytes=" << widthBytes << ", step=" << stepBytes << std::endl;
    nppiFree(ptr_8u_c3);
    
    // Test 8u C4
    Npp8u* ptr_8u_c4 = nppiMalloc_8u_C4(width, height, &stepBytes);
    assert(ptr_8u_c4 != nullptr);
    widthBytes = width * 4* sizeof(Npp8u);
    assert(stepBytes >= widthBytes);
    std::cout << "8u C4: ptr=" << (void*)ptr_8u_c4 << ", widthBytes=" << widthBytes << ", step=" << stepBytes << std::endl;
    nppiFree(ptr_8u_c4);
    
    std::cout << "✓ 8u Memory allocation test PASSED!" << std::endl;
}

void testNppi16uMemoryAllocation() {
    std::cout << "\n=== Testing 16u Memory Allocation ===" << std::endl;
    
    const int width = 320;
    const int height = 240;
    int stepBytes;
    int widthBytes;

    // Test 16u C1
    Npp16u* ptr_16u_c1 = nppiMalloc_16u_C1(width, height, &stepBytes);
    assert(ptr_16u_c1 != nullptr);
    widthBytes = width * 1 * sizeof(Npp16u);
    assert(stepBytes >= widthBytes);
    std::cout << "16u C1: ptr=" << (void*)ptr_16u_c1 << ", widthBytes=" << widthBytes << ", step=" << stepBytes << std::endl;
    nppiFree(ptr_16u_c1);
    
    // Test 16u C4
    Npp16u* ptr_16u_c4 = nppiMalloc_16u_C4(width, height, &stepBytes);
    assert(ptr_16u_c4 != nullptr);
    widthBytes = width * 4 * sizeof(Npp16u);
    assert(stepBytes >= widthBytes);
    std::cout << "16u C4: ptr=" << (void*)ptr_16u_c4 << ", widthBytes=" << widthBytes << ", step=" << stepBytes << std::endl;
    nppiFree(ptr_16u_c4);
    
    std::cout << "✓ 16u Memory allocation test PASSED!" << std::endl;
}

void testNppi32fMemoryAllocation() {
    std::cout << "\n=== Testing 32f Memory Allocation ===" << std::endl;
    
    const int width = 256;
    const int height = 256;
    int stepBytes;
    int widthBytes;
    
    // Test 32f C1
    Npp32f* ptr_32f_c1 = nppiMalloc_32f_C1(width, height, &stepBytes);
    assert(ptr_32f_c1 != nullptr);
    widthBytes = width * 1 * sizeof(Npp32f);
    assert(stepBytes >= widthBytes);
    std::cout << "32f C1: ptr=" << (void*)ptr_32f_c1 << ", widthBytes=" << widthBytes << ", step=" << stepBytes << std::endl;
    nppiFree(ptr_32f_c1);
    
    // Test 32f C3
    Npp32f* ptr_32f_c3 = nppiMalloc_32f_C3(width, height, &stepBytes);
    assert(ptr_32f_c3 != nullptr);
    widthBytes = width * 3 * sizeof(Npp32f);
    assert(stepBytes >= widthBytes);
    std::cout << "32f C3: ptr=" << (void*)ptr_32f_c3 << ", widthBytes=" << widthBytes << ", step=" << stepBytes << std::endl;
    nppiFree(ptr_32f_c3);
    
    std::cout << "✓ 32f Memory allocation test PASSED!" << std::endl;
}

void testNppiComplexMemoryAllocation() {
    std::cout << "\n=== Testing Complex Memory Allocation ===" << std::endl;
    
    const int width = 128;
    const int height = 128;
    int stepBytes;
    int widthBytes;
    
    // Test 16sc C1
    Npp16sc* ptr_16sc_c1 = nppiMalloc_16sc_C1(width, height, &stepBytes);
    assert(ptr_16sc_c1 != nullptr);
    widthBytes = width * 1 * sizeof(Npp16sc);
    assert(stepBytes >= widthBytes);
    std::cout << "16sc C1: ptr=" << (void*)ptr_16sc_c1 << ", widthBytes=" << widthBytes << ", step=" << stepBytes << std::endl;
    nppiFree(ptr_16sc_c1);
    
    // Test 32fc C1
    Npp32fc* ptr_32fc_c1 = nppiMalloc_32fc_C1(width, height, &stepBytes);
    assert(ptr_32fc_c1 != nullptr);
    widthBytes = width * 1 * sizeof(Npp32fc);
    assert(stepBytes >= widthBytes);
    std::cout << "32fc C1: ptr=" << (void*)ptr_32fc_c1 << ", widthBytes=" << widthBytes << ", step=" << stepBytes << std::endl;
    nppiFree(ptr_32fc_c1);
    
    // Test 32fc C3
    Npp32fc* ptr_32fc_c3 = nppiMalloc_32fc_C3(width, height, &stepBytes);
    assert(ptr_32fc_c3 != nullptr);
    widthBytes = width * 3 * sizeof(Npp32fc);
    assert(stepBytes >= widthBytes);
    std::cout << "32fc C3: ptr=" << (void*)ptr_32fc_c3  << ", widthBytes=" << widthBytes << ", step=" << stepBytes << std::endl;
    nppiFree(ptr_32fc_c3);
    
    std::cout << "✓ Complex Memory allocation test PASSED!" << std::endl;
}

void testErrorConditions() {
    std::cout << "\n=== Testing Error Conditions ===" << std::endl;
    
    int stepBytes;
    int widthBytes;
    
    // Test invalid width
    Npp8u* ptr = nppiMalloc_8u_C1(0, 480, &stepBytes);
    assert(ptr == nullptr);
    
    // Test invalid height
    ptr = nppiMalloc_8u_C1(640, 0, &stepBytes);
    assert(ptr == nullptr);
    
    // Test null stepBytes pointer
    ptr = nppiMalloc_8u_C1(640, 480, nullptr);
    assert(ptr == nullptr);
    
    // Test negative values
    ptr = nppiMalloc_8u_C1(-10, 480, &stepBytes);
    assert(ptr == nullptr);
    
    // Test free with null pointer (should not crash)
    nppiFree(nullptr);
    
    std::cout << "✓ Error conditions test PASSED!" << std::endl;
}

void testMemoryPattern() {
    std::cout << "\n=== Testing Memory Access Pattern ===" << std::endl;
    
    const int width = 100;
    const int height = 100;
    int stepBytes;
    
    // Allocate memory
    Npp8u* ptr = nppiMalloc_8u_C1(width, height, &stepBytes);
    assert(ptr != nullptr);
    
    // Create test pattern on host
    Npp8u* hostData = new Npp8u[stepBytes * height];
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            hostData[y * stepBytes + x] = static_cast<Npp8u>((x + y) % 256);
        }
    }
    
    // Copy to device
    cudaError_t result = cudaMemcpy(ptr, hostData, stepBytes * height, cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);
    
    // Copy back from device
    Npp8u* hostResult = new Npp8u[stepBytes * height];
    result = cudaMemcpy(hostResult, ptr, stepBytes * height, cudaMemcpyDeviceToHost);
    assert(result == cudaSuccess);
    
    // Verify data integrity
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            assert(hostResult[y * stepBytes + x] == hostData[y * stepBytes + x]);
        }
    }
    
    // Cleanup
    delete[] hostData;
    delete[] hostResult;
    nppiFree(ptr);
    
    std::cout << "✓ Memory access pattern test PASSED!" << std::endl;
}

int main() {
    // Initialize CUDA
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return -1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Testing on device: " << prop.name << std::endl;
    
    try {
        testNppi8uMemoryAllocation();
        testNppi16uMemoryAllocation();
        testNppi32fMemoryAllocation();
        testNppiComplexMemoryAllocation();
        testErrorConditions();
        testMemoryPattern();
        
        std::cout << "\n🎉 All NPPI Support Functions tests PASSED!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception!" << std::endl;
        return -1;
    }
}