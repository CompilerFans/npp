#ifndef NPP_TEST_FRAMEWORK_H
#define NPP_TEST_FRAMEWORK_H

#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <functional>
#include <map>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <cstring>

#include "nppdefs.h"
#include <cuda_runtime.h>

/**
 * @brief Test framework for comparing CPU, GPU and NVIDIA NPP implementations
 * 
 * This framework provides a structured way to test NPP functions against
 * multiple implementations (CPU, CUDA, NVIDIA NPP) and compare results.
 */

namespace NPPTest {

/**
 * @brief Test result structure
 */
struct TestResult {
    bool passed;
    double max_error;
    double avg_error;
    double cpu_time_ms;
    double gpu_time_ms;
    double nvidia_time_ms;
    std::string error_message;
    std::vector<double> errors;
    
    TestResult() : passed(false), max_error(0.0), avg_error(0.0), 
                   cpu_time_ms(0.0), gpu_time_ms(0.0), nvidia_time_ms(0.0) {}
};

/**
 * @brief Test parameters base class
 */
class TestParameters {
public:
    virtual ~TestParameters() = default;
    virtual std::string toString() const = 0;
    virtual bool isValid() const = 0;
};

/**
 * @brief Image buffer for testing
 */
template<typename T>
class ImageBuffer {
private:
    T* host_data_;
    T* device_data_;
    int width_, height_, step_, channels_;
    size_t size_;

public:
    ImageBuffer(int width, int height, int channels = 1) 
        : width_(width), height_(height), channels_(channels) {
        step_ = width_ * channels_ * sizeof(T);
        size_ = height_ * step_;
        
        host_data_ = new T[width_ * height_ * channels_];
        cudaMalloc(&device_data_, size_);
    }
    
    ~ImageBuffer() {
        delete[] host_data_;
        if (device_data_) cudaFree(device_data_);
    }
    
    // Copy operations
    void copyToDevice() {
        cudaMemcpy(device_data_, host_data_, size_, cudaMemcpyHostToDevice);
    }
    
    void copyFromDevice() {
        cudaMemcpy(host_data_, device_data_, size_, cudaMemcpyDeviceToHost);
    }
    
    void copyFrom(const ImageBuffer<T>& other) {
        assert(width_ == other.width_ && height_ == other.height_);
        memcpy(host_data_, other.host_data_, size_);
    }
    
    // Fill methods
    void fillRandom() {
        for (size_t i = 0; i < width_ * height_ * channels_; i++) {
            if constexpr (std::is_floating_point<T>::value) {
                host_data_[i] = static_cast<T>(rand()) / RAND_MAX;
            } else {
                host_data_[i] = static_cast<T>(rand() % (std::numeric_limits<T>::max() + 1));
            }
        }
    }
    
    void fillPattern(int pattern_type = 0) {
        switch (pattern_type) {
            case 0: // Gradient
                for (int y = 0; y < height_; y++) {
                    for (int x = 0; x < width_; x++) {
                        for (int c = 0; c < channels_; c++) {
                            host_data_[(y * width_ + x) * channels_ + c] = 
                                static_cast<T>((x + y) % (std::numeric_limits<T>::max() + 1));
                        }
                    }
                }
                break;
            case 1: // Checkerboard
                for (int y = 0; y < height_; y++) {
                    for (int x = 0; x < width_; x++) {
                        for (int c = 0; c < channels_; c++) {
                            bool black = ((x / 8) + (y / 8)) % 2;
                            host_data_[(y * width_ + x) * channels_ + c] = 
                                black ? std::numeric_limits<T>::min() : std::numeric_limits<T>::max();
                        }
                    }
                }
                break;
            case 2: // Solid colors
                for (int y = 0; y < height_; y++) {
                    for (int x = 0; x < width_; x++) {
                        for (int c = 0; c < channels_; c++) {
                            host_data_[(y * width_ + x) * channels_ + c] = 
                                static_cast<T>((x * y) % (std::numeric_limits<T>::max() + 1));
                        }
                    }
                }
                break;
        }
    }
    
    // Comparison
    double compare(const ImageBuffer<T>& other, double tolerance = 0.0) const {
        assert(width_ == other.width_ && height_ == other.height_);
        
        double max_error = 0.0;
        for (size_t i = 0; i < width_ * height_ * channels_; i++) {
            double error = std::abs(static_cast<double>(host_data_[i]) - 
                                   static_cast<double>(other.host_data_[i]));
            max_error = std::max(max_error, error);
            if (error > tolerance) {
                return error;
            }
        }
        return max_error;
    }
    
    // Statistics
    void getStats(T& min_val, T& max_val, double& avg_val) const {
        min_val = max_val = host_data_[0];
        double sum = 0.0;
        
        for (size_t i = 0; i < width_ * height_ * channels_; i++) {
            min_val = std::min(min_val, host_data_[i]);
            max_val = std::max(max_val, host_data_[i]);
            sum += static_cast<double>(host_data_[i]);
        }
        
        avg_val = sum / (width_ * height_ * channels_);
    }
    
    // Accessors
    T* host() { return host_data_; }
    const T* host() const { return host_data_; }
    T* device() { return device_data_; }
    const T* device() const { return device_data_; }
    int width() const { return width_; }
    int height() const { return height_; }
    int channels() const { return channels_; }
    int step() const { return step_; }
    size_t size() const { return size_; }
};

/**
 * @brief Timer for performance measurement
 */
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    cudaEvent_t cuda_start_, cuda_stop_;
    bool use_cuda_;

public:
    Timer(bool use_cuda = false) : use_cuda_(use_cuda) {
        if (use_cuda_) {
            cudaEventCreate(&cuda_start_);
            cudaEventCreate(&cuda_stop_);
        }
    }
    
    ~Timer() {
        if (use_cuda_) {
            cudaEventDestroy(cuda_start_);
            cudaEventDestroy(cuda_stop_);
        }
    }
    
    void start() {
        if (use_cuda_) {
            cudaEventRecord(cuda_start_);
        } else {
            start_time_ = std::chrono::high_resolution_clock::now();
        }
    }
    
    double stop() {
        if (use_cuda_) {
            cudaEventRecord(cuda_stop_);
            cudaEventSynchronize(cuda_stop_);
            float milliseconds = 0.0f;
            cudaEventElapsedTime(&milliseconds, cuda_start_, cuda_stop_);
            return static_cast<double>(milliseconds);
        } else {
            auto end_time = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(end_time - start_time_).count();
        }
    }
};

/**
 * @brief Base class for all NPP tests
 */
class NPPTestBase {
protected:
    std::string test_name_;
    std::string function_name_;
    NppStreamContext stream_ctx_;
    
public:
    NPPTestBase(const std::string& test_name, const std::string& function_name)
        : test_name_(test_name), function_name_(function_name) {
        
        // Initialize stream context
        memset(&stream_ctx_, 0, sizeof(NppStreamContext));
        cudaStreamCreate(&stream_ctx_.hStream);
        
        // Get device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        stream_ctx_.nCudaDeviceId = 0;
        stream_ctx_.nMultiProcessorCount = prop.multiProcessorCount;
        stream_ctx_.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
        stream_ctx_.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
        stream_ctx_.nSharedMemPerBlock = prop.sharedMemPerBlock;
        stream_ctx_.nCudaDevAttrComputeCapabilityMajor = prop.major;
        stream_ctx_.nCudaDevAttrComputeCapabilityMinor = prop.minor;
    }
    
    virtual ~NPPTestBase() {
        cudaStreamDestroy(stream_ctx_.hStream);
    }
    
    virtual std::string getName() const { return test_name_; }
    virtual std::string getFunctionName() const { return function_name_; }
    
    /**
     * @brief Run the test with given parameters
     */
    virtual TestResult run(const TestParameters& params) = 0;
    
    /**
     * @brief Get test cases for this function
     */
    virtual std::vector<std::unique_ptr<TestParameters>> getTestCases() const = 0;
    
    /**
     * @brief Check if NVIDIA NPP is available
     */
    virtual bool isNVIDIAAvailable() const = 0;
};

/**
 * @brief Test registry for managing all tests
 */
class TestRegistry {
private:
    static std::map<std::string, std::unique_ptr<NPPTestBase>> tests_;
    
public:
    static void registerTest(std::unique_ptr<NPPTestBase> test) {
        tests_[test->getFunctionName()] = std::move(test);
    }
    
    static NPPTestBase* getTest(const std::string& function_name) {
        auto it = tests_.find(function_name);
        return (it != tests_.end()) ? it->second.get() : nullptr;
    }
    
    static const std::map<std::string, std::unique_ptr<NPPTestBase>>& getAllTests() {
        return tests_;
    }
};

// Static member definition
std::map<std::string, std::unique_ptr<NPPTestBase>> TestRegistry::tests_;

/**
 * @brief Test runner for executing all tests
 */
class TestRunner {
private:
    std::string output_dir_;
    bool verbose_;
    
public:
    TestRunner(const std::string& output_dir = "test_results", bool verbose = false)
        : output_dir_(output_dir), verbose_(verbose) {
    }
    
    void runAllTests() {
        printf("=== OpenNPP Test Suite ===\n");
        printf("Output directory: %s\n", output_dir_.c_str());
        
        int total_tests = 0;
        int passed_tests = 0;
        
        for (const auto& [func_name, test] : TestRegistry::getAllTests()) {
            printf("\n--- Testing %s ---\n", func_name.c_str());
            
            auto test_cases = test->getTestCases();
            for (const auto& params : test_cases) {
                if (!params->isValid()) {
                    printf("  Skipping invalid test case: %s\n", params->toString().c_str());
                    continue;
                }
                
                total_tests++;
                auto result = test->run(*params);
                
                if (result.passed) {
                    printf("  ✓ PASS: %s\n", params->toString().c_str());
                    passed_tests++;
                } else {
                    printf("  ✗ FAIL: %s - %s\n", params->toString().c_str(), result.error_message.c_str());
                }
                
                if (verbose_) {
                    printf("    CPU: %.2f ms, GPU: %.2f ms, NVIDIA: %.2f ms\n",
                           result.cpu_time_ms, result.gpu_time_ms, result.nvidia_time_ms);
                    printf("    Max error: %.4f, Avg error: %.4f\n",
                           result.max_error, result.avg_error);
                }
            }
        }
        
        printf("\n=== Test Summary ===\n");
        printf("Total tests: %d\n", total_tests);
        printf("Passed: %d\n", passed_tests);
        printf("Failed: %d\n", total_tests - passed_tests);
        printf("Success rate: %.2f%%\n", 100.0 * passed_tests / total_tests);
    }
};

} // namespace NPPTest

#endif // NPP_TEST_FRAMEWORK_H
