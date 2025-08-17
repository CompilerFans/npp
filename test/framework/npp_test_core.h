#ifndef NPP_TEST_CORE_H
#define NPP_TEST_CORE_H

#include <memory>
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <functional>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>
#include <cuda_runtime.h>
#include "npp.h"

namespace NPPTest {

/**
 * NPP测试结果结构
 */
struct TestResult {
    bool passed = false;
    double cpu_time_ms = 0.0;
    double gpu_time_ms = 0.0;
    double nvidia_time_ms = 0.0;
    double max_error_vs_cpu = 0.0;
    double max_error_vs_nvidia = 0.0;
    std::string error_message;
    
    void print(const std::string& test_name) const {
        printf("Test: %s\n", test_name.c_str());
        printf("  Status: %s\n", passed ? "PASS" : "FAIL");
        if (!passed) {
            printf("  Error: %s\n", error_message.c_str());
        }
        printf("  CPU Time: %.3f ms\n", cpu_time_ms);
        printf("  GPU Time: %.3f ms\n", gpu_time_ms);
        if (nvidia_time_ms > 0) {
            printf("  NVIDIA Time: %.3f ms\n", nvidia_time_ms);
            printf("  Speedup vs NVIDIA: %.2fx\n", nvidia_time_ms / gpu_time_ms);
        }
        printf("  Max Error vs CPU: %.6f\n", max_error_vs_cpu);
        if (max_error_vs_nvidia >= 0) {
            printf("  Max Error vs NVIDIA: %.6f\n", max_error_vs_nvidia);
        }
        printf("\n");
    }
};

/**
 * 测试参数基类
 */
class TestParameters {
public:
    virtual ~TestParameters() = default;
    virtual std::string description() const = 0;
    virtual bool isValid() const = 0;
};

/**
 * 通用图像缓冲区模板
 */
template<typename T>
class ImageBuffer {
private:
    T* host_data_;
    T* device_data_;
    int width_;
    int height_;
    int channels_;
    size_t pitch_bytes_;
    size_t total_elements_;
    
public:
    ImageBuffer(int width, int height, int channels = 1) 
        : width_(width), height_(height), channels_(channels) {
        total_elements_ = width_ * height_ * channels_;
        
        // 分配主机内存
        host_data_ = new T[total_elements_];
        memset(host_data_, 0, total_elements_ * sizeof(T));
        
        // 分配设备内存（使用pitched内存以获得更好的性能）
        if (channels_ == 1) {
            cudaMallocPitch(&device_data_, &pitch_bytes_, width_ * sizeof(T), height_);
        } else {
            cudaMalloc(&device_data_, total_elements_ * sizeof(T));
            pitch_bytes_ = width_ * channels_ * sizeof(T);
        }
        cudaMemset(device_data_, 0, total_elements_ * sizeof(T));
    }
    
    ~ImageBuffer() {
        delete[] host_data_;
        if (device_data_) {
            cudaFree(device_data_);
        }
    }
    
    // 填充测试数据
    void fillRandom(T maxVal = T(255)) {
        for (size_t i = 0; i < total_elements_; i++) {
            if (std::is_floating_point<T>::value) {
                host_data_[i] = static_cast<T>((rand() / (double)RAND_MAX) * static_cast<double>(maxVal));
            } else {
                int max_int = static_cast<int>(maxVal);
                if (max_int <= 0) max_int = 255;
                host_data_[i] = static_cast<T>(rand() % max_int);
            }
        }
    }
    
    void fillPattern(int pattern_type = 0) {
        switch (pattern_type) {
            case 0: // 渐变
                for (int y = 0; y < height_; y++) {
                    for (int x = 0; x < width_; x++) {
                        for (int c = 0; c < channels_; c++) {
                            int idx = (y * width_ + x) * channels_ + c;
                            if (std::is_floating_point<T>::value) {
                                host_data_[idx] = static_cast<T>((x + y + c) % 100 / 100.0);
                            } else {
                                host_data_[idx] = static_cast<T>((x + y + c) % 256);
                            }
                        }
                    }
                }
                break;
            case 1: // 棋盘格
                for (int y = 0; y < height_; y++) {
                    for (int x = 0; x < width_; x++) {
                        for (int c = 0; c < channels_; c++) {
                            int idx = (y * width_ + x) * channels_ + c;
                            T val = ((x/16 + y/16) % 2) ? (std::is_floating_point<T>::value ? T(100.0) : T(255)) : T(0);
                            host_data_[idx] = val;
                        }
                    }
                }
                break;
            default: // 均匀值
                T val = std::is_floating_point<T>::value ? T(50.0) : T(128);
                for (size_t i = 0; i < total_elements_; i++) {
                    host_data_[i] = val;
                }
                break;
        }
    }
    
    void copyToDevice() {
        if (channels_ == 1 && pitch_bytes_ != width_ * sizeof(T)) {
            cudaMemcpy2D(device_data_, pitch_bytes_, host_data_, width_ * sizeof(T),
                        width_ * sizeof(T), height_, cudaMemcpyHostToDevice);
        } else {
            cudaMemcpy(device_data_, host_data_, total_elements_ * sizeof(T), cudaMemcpyHostToDevice);
        }
    }
    
    void copyFromDevice() {
        if (channels_ == 1 && pitch_bytes_ != width_ * sizeof(T)) {
            cudaMemcpy2D(host_data_, width_ * sizeof(T), device_data_, pitch_bytes_,
                        width_ * sizeof(T), height_, cudaMemcpyDeviceToHost);
        } else {
            cudaMemcpy(host_data_, device_data_, total_elements_ * sizeof(T), cudaMemcpyDeviceToHost);
        }
    }
    
    double compareWith(const ImageBuffer<T>& other, double tolerance = 1e-6) const {
        if (width_ != other.width_ || height_ != other.height_ || channels_ != other.channels_) {
            return -1.0; // 无效比较
        }
        
        double max_diff = 0.0;
        for (size_t i = 0; i < total_elements_; i++) {
            double diff = std::abs(static_cast<double>(host_data_[i]) - static_cast<double>(other.host_data_[i]));
            max_diff = std::max(max_diff, diff);
        }
        return max_diff;
    }
    
    void printFirstDifferences(const ImageBuffer<T>& other, int max_count = 10) const {
        int count = 0;
        for (size_t i = 0; i < total_elements_ && count < max_count; i++) {
            if (host_data_[i] != other.host_data_[i]) {
                printf("    [%zu] ours=%.6f vs other=%.6f\n", i, 
                       static_cast<double>(host_data_[i]), static_cast<double>(other.host_data_[i]));
                count++;
            }
        }
    }
    
    // 访问器
    T* hostData() { return host_data_; }
    const T* hostData() const { return host_data_; }
    T* deviceData() { return device_data_; }
    const T* deviceData() const { return device_data_; }
    int width() const { return width_; }
    int height() const { return height_; }
    int channels() const { return channels_; }
    size_t pitchBytes() const { return pitch_bytes_; }
    int stepBytes() const { return static_cast<int>(pitch_bytes_); }
    NppiSize size() const { return {width_, height_}; }
};

/**
 * 性能计时器
 */
class Timer {
private:
    cudaEvent_t start_event_, stop_event_;
    bool use_cuda_;
    
public:
    Timer(bool use_cuda = true) : use_cuda_(use_cuda) {
        if (use_cuda_) {
            cudaEventCreate(&start_event_);
            cudaEventCreate(&stop_event_);
        }
    }
    
    ~Timer() {
        if (use_cuda_) {
            cudaEventDestroy(start_event_);
            cudaEventDestroy(stop_event_);
        }
    }
    
    void start() {
        if (use_cuda_) {
            cudaEventRecord(start_event_);
        }
    }
    
    double stop() {
        if (use_cuda_) {
            cudaEventRecord(stop_event_);
            cudaEventSynchronize(stop_event_);
            float ms = 0;
            cudaEventElapsedTime(&ms, start_event_, stop_event_);
            return static_cast<double>(ms);
        }
        return 0.0;
    }
};

/**
 * NPP测试基类
 */
class NPPTestBase {
protected:
    std::string test_name_;
    std::string category_;
    NppStreamContext stream_ctx_;
    bool nvidia_npp_available_;
    
public:
    NPPTestBase(const std::string& test_name, const std::string& category) 
        : test_name_(test_name), category_(category) {
        
        // 初始化CUDA流上下文
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        stream_ctx_.hStream = stream;
        stream_ctx_.nCudaDeviceId = 0;
        
        // 获取设备属性
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        stream_ctx_.nMultiProcessorCount = prop.multiProcessorCount;
        stream_ctx_.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
        stream_ctx_.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
        stream_ctx_.nSharedMemPerBlock = prop.sharedMemPerBlock;
        stream_ctx_.nCudaDevAttrComputeCapabilityMajor = prop.major;
        stream_ctx_.nCudaDevAttrComputeCapabilityMinor = prop.minor;
        stream_ctx_.nStreamFlags = 0;
        
        // 检查NVIDIA NPP可用性
        nvidia_npp_available_ = checkNVIDIANPPAvailability();
    }
    
    virtual ~NPPTestBase() {
        cudaStreamDestroy(stream_ctx_.hStream);
    }
    
    virtual TestResult runTest(const TestParameters& params) = 0;
    virtual std::vector<std::unique_ptr<TestParameters>> generateTestCases() const = 0;
    
    const std::string& name() const { return test_name_; }
    const std::string& category() const { return category_; }
    bool isNVIDIAAvailable() const { return nvidia_npp_available_; }
    
protected:
    bool checkNVIDIANPPAvailability() {
        // 尝试调用一个简单的NVIDIA NPP函数来检查可用性
        try {
            const NppLibraryVersion* version = nppGetLibVersion();
            return (version != nullptr);
        } catch (...) {
            return false;
        }
    }
};

/**
 * 测试注册表
 */
class TestRegistry {
private:
    static std::map<std::string, std::vector<std::unique_ptr<NPPTestBase>>>* getRegistryPtr() {
        static std::map<std::string, std::vector<std::unique_ptr<NPPTestBase>>> registry;
        return &registry;
    }
    
public:
    static void registerTest(std::unique_ptr<NPPTestBase> test) {
        printf("Registering test: %s in category: %s\n", test->name().c_str(), test->category().c_str());
        auto* registry = getRegistryPtr();
        (*registry)[test->category()].push_back(std::move(test));
        printf("Test registered successfully\n");
    }
    
    static void runAllTests() {
        printf("Running all tests\n");
        auto* registry = getRegistryPtr();
        
        int total_tests = 0;
        int passed_tests = 0;
        
        for (const auto& [category, tests] : *registry) {
            printf("=== Testing Category: %s ===\n", category.c_str());
            
            for (const auto& test : tests) {
                printf("\n--- Test: %s ---\n", test->name().c_str());
                
                auto test_cases = test->generateTestCases();
                for (const auto& test_case : test_cases) {
                    printf("Parameters: %s\n", test_case->description().c_str());
                    
                    auto result = test->runTest(*test_case);
                    result.print(test->name());
                    
                    total_tests++;
                    if (result.passed) {
                        passed_tests++;
                    }
                }
            }
        }
        
        printf("=== Summary ===\n");
        printf("Passed: %d/%d (%.1f%%)\n", passed_tests, total_tests, 
               100.0 * passed_tests / total_tests);
        printf("Overall: %s\n", (passed_tests == total_tests) ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    }
    
    static void runCategoryTests(const std::string& category) {
        printf("Running category tests: %s\n", category.c_str());
        auto* registry = getRegistryPtr();
        
        if (registry->find(category) == registry->end()) {
            printf("Category '%s' not found!\n", category.c_str());
            return;
        }
        
        printf("=== Testing Category: %s ===\n", category.c_str());
        
        int total_tests = 0;
        int passed_tests = 0;
        
        for (const auto& test : (*registry)[category]) {
            printf("\n--- Test: %s ---\n", test->name().c_str());
            
            auto test_cases = test->generateTestCases();
            for (const auto& test_case : test_cases) {
                printf("Parameters: %s\n", test_case->description().c_str());
                
                auto result = test->runTest(*test_case);
                result.print(test->name());
                
                total_tests++;
                if (result.passed) {
                    passed_tests++;
                }
            }
        }
        
        printf("=== Category '%s' Summary ===\n", category.c_str());
        printf("Passed: %d/%d (%.1f%%)\n", passed_tests, total_tests, 
               100.0 * passed_tests / total_tests);
    }
};

} // namespace NPPTest

#endif // NPP_TEST_CORE_H