#ifndef NPP_VALIDATION_TEST_H
#define NPP_VALIDATION_TEST_H

#include "npp.h"
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>

/**
 * Comprehensive NPP validation test framework
 * Compares three implementations: OpenNPP vs NVIDIA NPP vs CPU Reference
 */

namespace npp_validation {

/**
 * Test result structure
 */
struct TestResult {
    std::string testName;
    bool openNppSuccess;
    bool nvidiaNppSuccess;
    bool cpuRefSuccess;
    bool resultsMatch;
    double maxDifference;
    std::string errorMessage;
    
    TestResult(const std::string& name) 
        : testName(name), openNppSuccess(false), nvidiaNppSuccess(false), 
          cpuRefSuccess(false), resultsMatch(false), maxDifference(0.0) {}
};

/**
 * Image buffer template for testing
 */
template<typename T>
class ValidationImageBuffer {
private:
    std::vector<T> hostData_;
    T* deviceData_;
    int width_;
    int height_;
    int step_;
    int channels_;
    
    // Helper to allocate device memory using appropriate NPPI function
    T* allocateNppiMemory(int width, int height, int channels, int* pStep) {
        if constexpr (std::is_same_v<T, Npp8u>) {
            if (channels == 1) return reinterpret_cast<T*>(nppiMalloc_8u_C1(width, height, pStep));
            else if (channels == 3) return reinterpret_cast<T*>(nppiMalloc_8u_C3(width, height, pStep));
            else if (channels == 4) return reinterpret_cast<T*>(nppiMalloc_8u_C4(width, height, pStep));
        } else if constexpr (std::is_same_v<T, Npp16u>) {
            if (channels == 1) return reinterpret_cast<T*>(nppiMalloc_16u_C1(width, height, pStep));
            else if (channels == 3) return reinterpret_cast<T*>(nppiMalloc_16u_C3(width, height, pStep));
            else if (channels == 4) return reinterpret_cast<T*>(nppiMalloc_16u_C4(width, height, pStep));
        } else if constexpr (std::is_same_v<T, Npp16s>) {
            if (channels == 1) return reinterpret_cast<T*>(nppiMalloc_16s_C1(width, height, pStep));
            else if (channels == 4) return reinterpret_cast<T*>(nppiMalloc_16s_C4(width, height, pStep));
        } else if constexpr (std::is_same_v<T, Npp32f>) {
            if (channels == 1) return reinterpret_cast<T*>(nppiMalloc_32f_C1(width, height, pStep));
            else if (channels == 3) return reinterpret_cast<T*>(nppiMalloc_32f_C3(width, height, pStep));
            else if (channels == 4) return reinterpret_cast<T*>(nppiMalloc_32f_C4(width, height, pStep));
        }
        // Fallback to cudaMallocPitch for unsupported types
        size_t pitch;
        void* devPtr = nullptr;
        cudaError_t result = cudaMallocPitch(&devPtr, &pitch, width * channels * sizeof(T), height);
        if (result == cudaSuccess) {
            *pStep = static_cast<int>(pitch);
            return static_cast<T*>(devPtr);
        }
        return nullptr;
    }
    
public:
    ValidationImageBuffer(int width, int height, int channels = 1) 
        : width_(width), height_(height), channels_(channels), deviceData_(nullptr) {
        
        // Allocate host memory
        hostData_.resize(width * height * channels);
        
        // Allocate device memory using NPPI functions
        deviceData_ = allocateNppiMemory(width, height, channels, &step_);
        
        if (!deviceData_) {
            throw std::runtime_error("NPPI memory allocation failed");
        }
    }
    
    ~ValidationImageBuffer() {
        if (deviceData_) {
            nppiFree(deviceData_);
            deviceData_ = nullptr;
        }
    }
    
    // Fill with random data
    void fillRandom(unsigned int seed = 12345) {
        std::mt19937 gen(seed);
        
        if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<int> dis(0, std::numeric_limits<T>::max());
            for (auto& pixel : hostData_) {
                pixel = static_cast<T>(dis(gen));
            }
        } else {
            std::uniform_real_distribution<float> dis(0.0f, 100.0f);
            for (auto& pixel : hostData_) {
                pixel = static_cast<T>(dis(gen));
            }
        }
        
        copyToDevice();
    }
    
    // Fill with specific pattern
    void fillPattern(T baseValue = T(50)) {
        for (int i = 0; i < width_ * height_ * channels_; ++i) {
            hostData_[i] = baseValue + static_cast<T>(i % 100);
        }
        copyToDevice();
    }
    
    void copyToDevice() {
        cudaError_t result = cudaMemcpy2D(
            deviceData_, step_,
            hostData_.data(), width_ * channels_ * sizeof(T),
            width_ * channels_ * sizeof(T), height_,
            cudaMemcpyHostToDevice
        );
        
        if (result != cudaSuccess) {
            throw std::runtime_error("Host to device copy failed");
        }
    }
    
    void copyFromDevice() {
        cudaError_t result = cudaMemcpy2D(
            hostData_.data(), width_ * channels_ * sizeof(T),
            deviceData_, step_,
            width_ * channels_ * sizeof(T), height_,
            cudaMemcpyDeviceToHost
        );
        
        if (result != cudaSuccess) {
            throw std::runtime_error("Device to host copy failed");
        }
    }
    
    T* devicePtr() { return deviceData_; }
    const T* devicePtr() const { return deviceData_; }
    T* hostPtr() { return hostData_.data(); }
    const T* hostPtr() const { return hostData_.data(); }
    
    int width() const { return width_; }
    int height() const { return height_; }
    int step() const { return step_; }
    int channels() const { return channels_; }
    NppiSize size() const { return {width_, height_}; }
    
    // Compare two buffers
    static double compareBuffers(const ValidationImageBuffer<T>& buf1, 
                               const ValidationImageBuffer<T>& buf2,
                               double tolerance = 1e-6) {
        if (buf1.width() != buf2.width() || buf1.height() != buf2.height()) {
            return std::numeric_limits<double>::max();
        }
        
        double maxDiff = 0.0;
        int totalPixels = buf1.width() * buf1.height() * buf1.channels_;
        for (int i = 0; i < totalPixels; ++i) {
            double diff = std::abs(static_cast<double>(buf1.hostData_[i]) - 
                                 static_cast<double>(buf2.hostData_[i]));
            maxDiff = std::max(maxDiff, diff);
        }
        
        return maxDiff;
    }
};

/**
 * Base validation test class
 */
class NPPValidationTestBase {
protected:
    std::vector<TestResult> results_;
    bool verbose_;
    
public:
    NPPValidationTestBase(bool verbose = true) : verbose_(verbose) {}
    virtual ~NPPValidationTestBase() = default;
    
    virtual void runAllTests() = 0;
    
    void printResults() const {
        std::cout << "\n=== NPP Validation Test Results ===\n";
        std::cout << std::left << std::setw(30) << "Test Name"
                  << std::setw(12) << "OpenNPP"
                  << std::setw(12) << "NVIDIA NPP"
                  << std::setw(12) << "CPU Ref"
                  << std::setw(12) << "Match"
                  << std::setw(15) << "Max Diff"
                  << "Error" << std::endl;
        std::cout << std::string(100, '-') << std::endl;
        
        int passCount = 0;
        for (const auto& result : results_) {
            std::cout << std::left << std::setw(30) << result.testName
                      << std::setw(12) << (result.openNppSuccess ? "PASS" : "FAIL")
                      << std::setw(12) << (result.nvidiaNppSuccess ? "PASS" : "FAIL")
                      << std::setw(12) << (result.cpuRefSuccess ? "PASS" : "FAIL")
                      << std::setw(12) << (result.resultsMatch ? "YES" : "NO")
                      << std::setw(15) << std::scientific << std::setprecision(2) << result.maxDifference;
            
            if (!result.errorMessage.empty()) {
                std::cout << result.errorMessage;
            }
            std::cout << std::endl;
            
            if (result.resultsMatch && result.openNppSuccess && 
                result.nvidiaNppSuccess && result.cpuRefSuccess) {
                passCount++;
            }
        }
        
        std::cout << std::string(100, '-') << std::endl;
        std::cout << "Total tests: " << results_.size() << ", Passed: " << passCount 
                  << ", Failed: " << (results_.size() - passCount) << std::endl;
    }
    
    bool allTestsPassed() const {
        for (const auto& result : results_) {
            if (!result.resultsMatch || !result.openNppSuccess || 
                !result.nvidiaNppSuccess || !result.cpuRefSuccess) {
                return false;
            }
        }
        return true;
    }
};

} // namespace npp_validation

#endif // NPP_VALIDATION_TEST_H