#pragma once

#include "npp.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <cstring>
#include <random>
#include <type_traits>

/**
 * 改进的图像缓冲区类，正确处理pitched内存
 * 主机内存也使用pitched布局以匹配设备内存
 */
template<typename T>
class ValidationImageBufferV2 {
private:
    T* hostData_;       // 主机内存（使用pitched布局）
    T* deviceData_;     // 设备内存（NPPI分配）
    int width_;         // 宽度（像素）
    int height_;        // 高度（像素）
    int channels_;      // 通道数
    int hostStep_;      // 主机内存步长（字节）
    int deviceStep_;    // 设备内存步长（字节）
    
    // NPPI内存分配辅助函数
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
        
        // 后备方案：使用cudaMallocPitch
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
    ValidationImageBufferV2(int width, int height, int channels = 1) 
        : width_(width), height_(height), channels_(channels), 
          hostData_(nullptr), deviceData_(nullptr) {
        
        // 分配设备内存
        deviceData_ = allocateNppiMemory(width, height, channels, &deviceStep_);
        if (!deviceData_) {
            throw std::runtime_error("NPPI device memory allocation failed");
        }
        
        // 为主机分配pitched内存（与设备步长匹配）
        hostStep_ = deviceStep_;  // 使用相同的步长
        size_t totalSize = hostStep_ * height;
        hostData_ = static_cast<T*>(std::aligned_alloc(128, totalSize));
        if (!hostData_) {
            nppiFree(deviceData_);
            throw std::runtime_error("Host memory allocation failed");
        }
        
        // 初始化主机内存为0
        std::memset(hostData_, 0, totalSize);
    }
    
    ~ValidationImageBufferV2() {
        if (deviceData_) {
            nppiFree(deviceData_);
            deviceData_ = nullptr;
        }
        if (hostData_) {
            std::free(hostData_);
            hostData_ = nullptr;
        }
    }
    
    // 禁用拷贝构造和赋值
    ValidationImageBufferV2(const ValidationImageBufferV2&) = delete;
    ValidationImageBufferV2& operator=(const ValidationImageBufferV2&) = delete;
    
    // 填充特定模式
    void fillPattern(T baseValue = T(50)) {
        for (int y = 0; y < height_; y++) {
            T* rowPtr = reinterpret_cast<T*>(
                reinterpret_cast<char*>(hostData_) + y * hostStep_);
            for (int x = 0; x < width_ * channels_; x++) {
                rowPtr[x] = baseValue + static_cast<T>((y * width_ * channels_ + x) % 100);
            }
        }
        copyToDevice();
    }
    
    // 填充随机数据
    void fillRandom(unsigned int seed = 12345) {
        std::mt19937 gen(seed);
        
        for (int y = 0; y < height_; y++) {
            T* rowPtr = reinterpret_cast<T*>(
                reinterpret_cast<char*>(hostData_) + y * hostStep_);
            
            if constexpr (std::is_integral_v<T>) {
                std::uniform_int_distribution<int> dis(0, std::numeric_limits<T>::max());
                for (int x = 0; x < width_ * channels_; x++) {
                    rowPtr[x] = static_cast<T>(dis(gen));
                }
            } else {
                std::uniform_real_distribution<float> dis(0.0f, 100.0f);
                for (int x = 0; x < width_ * channels_; x++) {
                    rowPtr[x] = static_cast<T>(dis(gen));
                }
            }
        }
        copyToDevice();
    }
    
    // 拷贝到设备（使用正确的步长）
    void copyToDevice() {
        cudaError_t result = cudaMemcpy2D(
            deviceData_, deviceStep_,
            hostData_, hostStep_,
            width_ * channels_ * sizeof(T), height_,
            cudaMemcpyHostToDevice
        );
        
        if (result != cudaSuccess) {
            throw std::runtime_error("Host to device copy failed");
        }
    }
    
    // 从设备拷贝（使用正确的步长）
    void copyFromDevice() {
        cudaError_t result = cudaMemcpy2D(
            hostData_, hostStep_,
            deviceData_, deviceStep_,
            width_ * channels_ * sizeof(T), height_,
            cudaMemcpyDeviceToHost
        );
        
        if (result != cudaSuccess) {
            throw std::runtime_error("Device to host copy failed");
        }
    }
    
    // 访问器
    T* devicePtr() { return deviceData_; }
    const T* devicePtr() const { return deviceData_; }
    T* hostPtr() { return hostData_; }
    const T* hostPtr() const { return hostData_; }
    
    int width() const { return width_; }
    int height() const { return height_; }
    int channels() const { return channels_; }
    int deviceStep() const { return deviceStep_; }
    int hostStep() const { return hostStep_; }
    NppiSize size() const { return {width_, height_}; }
    
    // 获取主机内存中特定行的指针
    T* getHostRow(int y) {
        return reinterpret_cast<T*>(
            reinterpret_cast<char*>(hostData_) + y * hostStep_);
    }
    
    const T* getHostRow(int y) const {
        return reinterpret_cast<const T*>(
            reinterpret_cast<const char*>(hostData_) + y * hostStep_);
    }
    
    // 比较两个缓冲区（考虑pitched布局）
    static double compareBuffers(const ValidationImageBufferV2<T>& buf1, 
                                const ValidationImageBufferV2<T>& buf2,
                                double tolerance = 1e-6) {
        if (buf1.width() != buf2.width() || 
            buf1.height() != buf2.height() || 
            buf1.channels() != buf2.channels()) {
            return std::numeric_limits<double>::max();
        }
        
        double maxDiff = 0.0;
        
        for (int y = 0; y < buf1.height(); y++) {
            const T* row1 = buf1.getHostRow(y);
            const T* row2 = buf2.getHostRow(y);
            
            for (int x = 0; x < buf1.width() * buf1.channels(); x++) {
                double diff = std::abs(static_cast<double>(row1[x]) - 
                                     static_cast<double>(row2[x]));
                maxDiff = std::max(maxDiff, diff);
            }
        }
        
        return maxDiff;
    }
};

/**
 * 测试结果结构
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
 * 基础验证测试类
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
                      << std::setw(15) << std::scientific << std::setprecision(2) << result.maxDifference
                      << result.errorMessage << std::endl;
            
            if (result.resultsMatch) passCount++;
        }
        
        std::cout << std::string(100, '-') << std::endl;
        std::cout << "Total tests: " << results_.size() 
                  << ", Passed: " << passCount 
                  << ", Failed: " << (results_.size() - passCount) << std::endl;
    }
};