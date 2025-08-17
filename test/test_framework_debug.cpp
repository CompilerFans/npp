#include <iostream>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>
#include "npp.h"

// 简化版的测试框架来定位问题
namespace NPPTest {

class SimpleTestResult {
public:
    bool passed = false;
    std::string error_message;
    
    void print(const std::string& test_name) const {
        printf("Test: %s - %s\n", test_name.c_str(), passed ? "PASS" : "FAIL");
        if (!passed) {
            printf("  Error: %s\n", error_message.c_str());
        }
    }
};

template<typename T>
class SimpleImageBuffer {
private:
    T* host_data_;
    T* device_data_;
    int width_;
    int height_;
    size_t total_elements_;
    
public:
    SimpleImageBuffer(int width, int height) : width_(width), height_(height) {
        printf("Creating SimpleImageBuffer %dx%d for type size %zu\n", width, height, sizeof(T));
        
        total_elements_ = width_ * height_;
        
        // 分配主机内存
        host_data_ = new T[total_elements_];
        memset(host_data_, 0, total_elements_ * sizeof(T));
        
        // 分配设备内存
        cudaError_t result = cudaMalloc(&device_data_, total_elements_ * sizeof(T));
        if (result != cudaSuccess) {
            printf("CUDA malloc failed: %s\n", cudaGetErrorString(result));
            device_data_ = nullptr;
            throw std::runtime_error("CUDA malloc failed");
        }
        cudaMemset(device_data_, 0, total_elements_ * sizeof(T));
        
        printf("SimpleImageBuffer created successfully\n");
    }
    
    ~SimpleImageBuffer() {
        printf("Destroying SimpleImageBuffer\n");
        delete[] host_data_;
        if (device_data_) {
            cudaFree(device_data_);
        }
        printf("SimpleImageBuffer destroyed\n");
    }
    
    void fillSimple() {
        printf("Filling buffer with simple pattern\n");
        for (size_t i = 0; i < total_elements_; i++) {
            host_data_[i] = static_cast<T>(i % 100);
        }
        printf("Buffer filled\n");
    }
    
    void copyToDevice() {
        printf("Copying to device\n");
        cudaMemcpy(device_data_, host_data_, total_elements_ * sizeof(T), cudaMemcpyHostToDevice);
        printf("Copy to device completed\n");
    }
    
    void copyFromDevice() {
        printf("Copying from device\n");
        cudaMemcpy(host_data_, device_data_, total_elements_ * sizeof(T), cudaMemcpyDeviceToHost);
        printf("Copy from device completed\n");
    }
    
    T* hostData() { return host_data_; }
    T* deviceData() { return device_data_; }
    int width() const { return width_; }
    int height() const { return height_; }
    int stepBytes() const { return width_ * sizeof(T); }
    NppiSize size() const { return {width_, height_}; }
    
    bool compareWith(const SimpleImageBuffer<T>& other) const {
        printf("Comparing buffers\n");
        for (size_t i = 0; i < total_elements_; i++) {
            if (host_data_[i] != other.host_data_[i]) {
                printf("Difference at %zu: %f vs %f\n", i, 
                       static_cast<double>(host_data_[i]), 
                       static_cast<double>(other.host_data_[i]));
                return false;
            }
        }
        return true;
    }
};

class SimpleAddCTest {
public:
    SimpleTestResult testAddC8u() {
        printf("Starting AddC 8u test\n");
        SimpleTestResult result;
        
        try {
            const int width = 32;
            const int height = 32;
            const Npp8u constant = 50;
            const int scale_factor = 1;
            
            printf("Creating buffers\n");
            SimpleImageBuffer<Npp8u> src(width, height);
            SimpleImageBuffer<Npp8u> dst(width, height);
            SimpleImageBuffer<Npp8u> ref(width, height);
            
            printf("Filling source buffer\n");
            src.fillSimple();
            src.copyToDevice();
            
            // 计算参考结果
            printf("Computing reference\n");
            for (int i = 0; i < width * height; i++) {
                int val = static_cast<int>(src.hostData()[i]) + static_cast<int>(constant);
                val = val >> scale_factor;
                val = std::max(0, std::min(255, val));
                ref.hostData()[i] = static_cast<Npp8u>(val);
            }
            
            printf("Running GPU implementation\n");
            NppStatus status = nppiAddC_8u_C1RSfs(
                src.deviceData(), src.stepBytes(), constant,
                dst.deviceData(), dst.stepBytes(), src.size(), scale_factor
            );
            
            if (status != NPP_NO_ERROR) {
                result.error_message = "GPU implementation failed: " + std::to_string(status);
                return result;
            }
            
            printf("Copying results back\n");
            dst.copyFromDevice();
            
            printf("Comparing results\n");
            if (dst.compareWith(ref)) {
                result.passed = true;
                printf("Test passed!\n");
            } else {
                result.error_message = "Results don't match";
                printf("Test failed - results don't match\n");
            }
            
        } catch (const std::exception& e) {
            result.error_message = std::string("Exception: ") + e.what();
            printf("Exception caught: %s\n", e.what());
        }
        
        printf("AddC 8u test completed\n");
        return result;
    }
};

} // namespace NPPTest

int main() {
    printf("Debug Test Framework\n");
    printf("====================\n");
    
    // 检查CUDA设备
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        printf("Error: No CUDA devices found!\n");
        return 1;
    }
    
    cudaSetDevice(0);
    printf("CUDA device set\n");
    
    try {
        printf("Creating test instance\n");
        NPPTest::SimpleAddCTest test;
        
        printf("Running test\n");
        auto result = test.testAddC8u();
        result.print("AddC_8u");
        
        printf("Test completed successfully\n");
        return result.passed ? 0 : 1;
        
    } catch (const std::exception& e) {
        printf("Main exception: %s\n", e.what());
        return 1;
    }
}