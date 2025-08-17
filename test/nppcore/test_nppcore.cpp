#include "npp.h"
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

/**
 * Test NPP Core functions
 */

void testNppLibraryVersion() {
    std::cout << "=== Testing NPP Library Version ===" << std::endl;
    
    const NppLibraryVersion* version = nppGetLibVersion();
    assert(version != nullptr);
    
    std::cout << "NPP Version: " << version->major << "." << version->minor 
              << " Build: " << version->build << std::endl;
    
    // Verify version values match the header definitions
    assert(version->major == NPP_VER_MAJOR);
    assert(version->minor == NPP_VER_MINOR);
    assert(version->build == NPP_VER_BUILD);
    
    std::cout << "✓ NPP Library Version test PASSED!" << std::endl;
}

void testNppGpuProperties() {
    std::cout << "\n=== Testing NPP GPU Properties ===" << std::endl;
    
    // Test individual property functions
    int numSMs = nppGetGpuNumSMs();
    int maxThreadsPerBlock = nppGetMaxThreadsPerBlock();
    int maxThreadsPerSM = nppGetMaxThreadsPerSM();
    
    assert(numSMs > 0);
    assert(maxThreadsPerBlock > 0);
    assert(maxThreadsPerSM > 0);
    
    std::cout << "GPU SMs: " << numSMs << std::endl;
    std::cout << "Max threads per block: " << maxThreadsPerBlock << std::endl;
    std::cout << "Max threads per SM: " << maxThreadsPerSM << std::endl;
    
    // Test combined properties function
    int propSMs, propMaxThreadsBlock, propMaxThreadsSM;
    int result = nppGetGpuDeviceProperties(&propMaxThreadsSM, &propMaxThreadsBlock, &propSMs);
    assert(result == 0);
    assert(propSMs == numSMs);
    assert(propMaxThreadsBlock == maxThreadsPerBlock);
    assert(propMaxThreadsSM == maxThreadsPerSM);
    
    // Test GPU name
    const char* gpuName = nppGetGpuName();
    assert(gpuName != nullptr);
    std::cout << "GPU Name: " << gpuName << std::endl;
    
    std::cout << "✓ NPP GPU Properties test PASSED!" << std::endl;
}

void testNppStreamManagement() {
    std::cout << "\n=== Testing NPP Stream Management ===" << std::endl;
    
    // Test default stream
    cudaStream_t defaultStream = nppGetStream();
    assert(defaultStream == 0);
    std::cout << "Default stream: " << defaultStream << std::endl;
    
    // Test stream context
    NppStreamContext streamCtx = {0};
    NppStatus status = nppGetStreamContext(&streamCtx);
    assert(status == NPP_NO_ERROR);
    assert(streamCtx.hStream == 0);
    assert(streamCtx.nCudaDeviceId >= 0);
    assert(streamCtx.nMultiProcessorCount > 0);
    
    std::cout << "Stream context - Device ID: " << streamCtx.nCudaDeviceId 
              << ", SMs: " << streamCtx.nMultiProcessorCount << std::endl;
    
    // Test stream-specific functions
    unsigned int streamSMs = nppGetStreamNumSMs();
    unsigned int streamMaxThreads = nppGetStreamMaxThreadsPerSM();
    assert(streamSMs > 0);
    assert(streamMaxThreads > 0);
    
    std::cout << "Stream SMs: " << streamSMs << ", Max threads per SM: " << streamMaxThreads << std::endl;
    
    // Test setting new stream
    cudaStream_t newStream;
    cudaStreamCreate(&newStream);
    
    status = nppSetStream(newStream);
    assert(status == NPP_NO_ERROR);
    
    cudaStream_t currentStream = nppGetStream();
    assert(currentStream == newStream);
    
    // Reset to default stream
    status = nppSetStream(0);
    assert(status == NPP_NO_ERROR);
    
    cudaStreamDestroy(newStream);
    
    std::cout << "✓ NPP Stream Management test PASSED!" << std::endl;
}

void testErrorHandling() {
    std::cout << "\n=== Testing Error Handling ===" << std::endl;
    
    // Test null pointer handling
    int result = nppGetGpuDeviceProperties(nullptr, nullptr, nullptr);
    assert(result == -1);
    
    int validValue;
    result = nppGetGpuDeviceProperties(&validValue, nullptr, nullptr);
    assert(result == -1);
    
    std::cout << "✓ Error handling test PASSED!" << std::endl;
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
        testNppLibraryVersion();
        testNppGpuProperties();
        testNppStreamManagement();
        testErrorHandling();
        
        std::cout << "\n🎉 All NPP Core tests PASSED!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception!" << std::endl;
        return -1;
    }
}