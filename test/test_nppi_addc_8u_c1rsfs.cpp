#include <cassert>
#include <cstring>
#include <cstdio>
#include <cuda_runtime.h>
#include <ctime>
#include <cmath>

#include "nppdefs.h"
#include "npp_image_arithmetic.h"

/**
 * Test framework for nppiAddC_8u_C1RSfs_Ctx
 * 
 * This test framework validates our source implementation against the
 * NVIDIA closed-source implementation and ensures correctness.
 */

class TestImage {
public:
    TestImage(int width, int height) : width_(width), height_(height) {
        size_ = width_ * height_;
        host_data_ = new Npp8u[size_];
        device_data_ = nullptr;
        
        // Allocate device memory
        cudaMalloc(&device_data_, size_);
        cudaMemset(device_data_, 0, size_);
    }
    
    ~TestImage() {
        delete[] host_data_;
        if (device_data_) {
            cudaFree(device_data_);
        }
    }
    
    void fillRandom() {
        for (int i = 0; i < size_; i++) {
            host_data_[i] = static_cast<Npp8u>(rand() % 256);
        }
    }
    
    void fillPattern() {
        for (int y = 0; y < height_; y++) {
            for (int x = 0; x < width_; x++) {
                host_data_[y * width_ + x] = static_cast<Npp8u>((x + y) % 256);
            }
        }
    }
    
    void copyToDevice() {
        cudaMemcpy(device_data_, host_data_, size_, cudaMemcpyHostToDevice);
    }
    
    void copyFromDevice() {
        cudaMemcpy(host_data_, device_data_, size_, cudaMemcpyDeviceToHost);
    }
    
    Npp8u* hostData() { return host_data_; }
    Npp8u* deviceData() { return device_data_; }
    int width() const { return width_; }
    int height() const { return height_; }
    int size() const { return size_; }
    
    bool compare(const TestImage& other, double tolerance = 0.0) const {
        if (width_ != other.width_ || height_ != other.height_) {
            return false;
        }
        
        for (int i = 0; i < size_; i++) {
            double diff = abs(static_cast<int>(host_data_[i]) - static_cast<int>(other.host_data_[i]));
            if (diff > tolerance) {
                return false;
            }
        }
        return true;
    }
    
    void printInfo(const char* name) const {
        printf("%s: %dx%d, min=%u, max=%u, avg=%.2f\n", 
               name, width_, height_, 
               getMin(), getMax(), getAverage());
    }
    
private:
    Npp8u getMin() const {
        Npp8u min = 255;
        for (int i = 0; i < size_; i++) {
            if (host_data_[i] < min) min = host_data_[i];
        }
        return min;
    }
    
    Npp8u getMax() const {
        Npp8u max = 0;
        for (int i = 0; i < size_; i++) {
            if (host_data_[i] > max) max = host_data_[i];
        }
        return max;
    }
    
    double getAverage() const {
        double sum = 0;
        for (int i = 0; i < size_; i++) {
            sum += host_data_[i];
        }
        return sum / size_;
    }
    
    int width_, height_, size_;
    Npp8u* host_data_;
    Npp8u* device_data_;
};

/**
 * Test runner class
 */
class TestRunner {
public:
    TestRunner() {
        srand(time(nullptr));
        
        // Initialize CUDA stream context
        cudaStreamCreate(&stream_);
        
        // Setup stream context
        stream_ctx_.hStream = stream_;
        stream_ctx_.nCudaDeviceId = 0;
        
        // Get device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        stream_ctx_.nMultiProcessorCount = prop.multiProcessorCount;
        stream_ctx_.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
        stream_ctx_.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
        stream_ctx_.nSharedMemPerBlock = prop.sharedMemPerBlock;
        stream_ctx_.nCudaDevAttrComputeCapabilityMajor = prop.major;
        stream_ctx_.nCudaDevAttrComputeCapabilityMinor = prop.minor;
        stream_ctx_.nStreamFlags = 0;
    }
    
    ~TestRunner() {
        cudaStreamDestroy(stream_);
    }
    
    bool runTest(const char* testName, int width, int height, Npp8u constant, int scaleFactor) {
        printf("Running test: %s\n", testName);
        printf("  Parameters: %dx%d, constant=%u, scaleFactor=%d\n", width, height, constant, scaleFactor);
        
        // Create test images
        TestImage src(width, height);
        TestImage dst_reference(width, height);
        TestImage dst_cpu(width, height);
        TestImage dst_cuda(width, height);
        
        // Fill with test data
        src.fillRandom();
        src.copyToDevice();
        
        // Setup ROI
        NppiSize roi = {width, height};
        
        // Run reference implementation
        NppStatus status = nppiAddC_8u_C1RSfs_Ctx_reference(
            src.hostData(), width, constant,
            dst_reference.hostData(), width, roi, scaleFactor
        );
        
        if (status != NPP_NO_ERROR) {
            printf("  ERROR: Reference implementation failed with status %d\n", status);
            return false;
        }
        
        // Run CPU implementation
        status = nppiAddC_8u_C1RSfs_Ctx_host(
            src.hostData(), width, constant,
            dst_cpu.hostData(), width, roi, scaleFactor, stream_ctx_
        );
        
        if (status != NPP_NO_ERROR) {
            printf("  ERROR: CPU implementation failed with status %d\n", status);
            return false;
        }
        
        // Run CUDA implementation
        status = nppiAddC_8u_C1RSfs_Ctx_cuda(
            src.deviceData(), width, constant,
            dst_cuda.deviceData(), width, roi, scaleFactor, stream_ctx_
        );
        
        if (status != NPP_NO_ERROR) {
            printf("  ERROR: CUDA implementation failed with status %d\n", status);
            return false;
        }
        
        cudaDeviceSynchronize();
        dst_cuda.copyFromDevice();
        
        // Validate results
        bool cpu_ok = dst_reference.compare(dst_cpu, 0);
        bool cuda_ok = dst_reference.compare(dst_cuda, 0);
        
        printf("  CPU vs Reference: %s\n", cpu_ok ? "PASS" : "FAIL");
        printf("  CUDA vs Reference: %s\n", cuda_ok ? "PASS" : "FAIL");
        
        if (!cpu_ok || !cuda_ok) {
            printf("  Detailed comparison:\n");
            src.printInfo("    Source");
            dst_reference.printInfo("    Reference");
            dst_cpu.printInfo("    CPU");
            dst_cuda.printInfo("    CUDA");
            return false;
        }
        
        printf("  Test PASSED\n\n");
        return true;
    }
    
    void runAllTests() {
        printf("=== NPP AddC 8u C1RSfs Test Suite ===\n\n");
        
        int passed = 0;
        int total = 0;
        
        // Test cases with different parameters
        struct TestCase {
            const char* name;
            int width;
            int height;
            Npp8u constant;
            int scaleFactor;
        };
        
        TestCase testCases[] = {
            {"Small Image", 32, 32, 50, 0},
            {"Medium Image", 256, 256, 100, 1},
            {"Large Image", 1024, 768, 128, 2},
            {"Edge Case - Zero Constant", 64, 64, 0, 1},
            {"Edge Case - Max Constant", 64, 64, 255, 1},
            {"Edge Case - Zero Scale", 64, 64, 50, 0},
            {"Edge Case - High Scale", 64, 64, 200, 4},
            {"Small Dimensions", 1, 1, 100, 1},
            {"Wide Image", 2048, 32, 75, 2},
            {"Tall Image", 32, 2048, 75, 2}
        };
        
        for (const auto& testCase : testCases) {
            total++;
            if (runTest(testCase.name, testCase.width, testCase.height, 
                       testCase.constant, testCase.scaleFactor)) {
                passed++;
            }
        }
        
        printf("=== Test Summary ===\n");
        printf("Passed: %d/%d\n", passed, total);
        printf("Overall: %s\n", (passed == total) ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    }
    
private:
    cudaStream_t stream_;
    NppStreamContext stream_ctx_;
};

/**
 * Main test function
 */
int main() {
    TestRunner runner;
    runner.runAllTests();
    return 0;
}
