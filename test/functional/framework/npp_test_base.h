/**
 * @file npp_test_base.h
 * @brief NPP功能测试基础框架
 * 
 * 提供纯功能单元测试的基础设施，专注于API功能验证，不依赖NVIDIA NPP库
 */

#pragma once

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <memory>
#include <type_traits>
#include "npp.h"

namespace npp_functional_test {

/**
 * @brief NPP功能测试基类
 * 
 * 提供CUDA设备管理、内存管理、数据生成等通用功能
 */
class NppTestBase : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化CUDA设备
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess) << "Failed to set CUDA device";
        
        // 检查设备可用性
        int deviceCount;
        err = cudaGetDeviceCount(&deviceCount);
        ASSERT_EQ(err, cudaSuccess) << "Failed to get device count";
        ASSERT_GT(deviceCount, 0) << "No CUDA devices available";
        
        // 获取设备属性
        err = cudaGetDeviceProperties(&deviceProp_, 0);
        ASSERT_EQ(err, cudaSuccess) << "Failed to get device properties";
        
        std::cout << "Using GPU: " << deviceProp_.name << std::endl;
    }
    
    void TearDown() override {
        // 同步设备并检查错误
        cudaError_t err = cudaDeviceSynchronize();
        EXPECT_EQ(err, cudaSuccess) << "CUDA error after test: " 
                                   << cudaGetErrorString(err);
    }
    
    // 设备属性访问
    const cudaDeviceProp& getDeviceProperties() const {
        return deviceProp_;
    }
    
    // 检查计算能力
    bool hasComputeCapability(int major, int minor) const {
        return deviceProp_.major > major || 
               (deviceProp_.major == major && deviceProp_.minor >= minor);
    }

private:
    cudaDeviceProp deviceProp_;
};

/**
 * @brief 内存管理辅助类模板
 */
template<typename T>
class DeviceMemory {
public:
    DeviceMemory() : ptr_(nullptr), size_(0) {}
    
    explicit DeviceMemory(size_t size) : size_(size) {
        allocate(size);
    }
    
    ~DeviceMemory() {
        free();
    }
    
    // 禁止拷贝
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    
    // 支持移动
    DeviceMemory(DeviceMemory&& other) noexcept 
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    DeviceMemory& operator=(DeviceMemory&& other) noexcept {
        if (this != &other) {
            free();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    void allocate(size_t size) {
        free();
        cudaError_t err = cudaMalloc(&ptr_, size * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        size_ = size;
    }
    
    void free() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }
    
    T* get() const { return ptr_; }
    size_t size() const { return size_; }
    
    // 数据传输
    void copyFromHost(const std::vector<T>& hostData) {
        ASSERT_EQ(hostData.size(), size_) << "Size mismatch in copyFromHost";
        cudaError_t err = cudaMemcpy(ptr_, hostData.data(), 
                                   size_ * sizeof(T), cudaMemcpyHostToDevice);
        ASSERT_EQ(err, cudaSuccess) << "Failed to copy from host";
    }
    
    void copyToHost(std::vector<T>& hostData) const {
        hostData.resize(size_);
        cudaError_t err = cudaMemcpy(hostData.data(), ptr_, 
                                   size_ * sizeof(T), cudaMemcpyDeviceToHost);
        ASSERT_EQ(err, cudaSuccess) << "Failed to copy to host";
    }

private:
    T* ptr_;
    size_t size_;
};

/**
 * @brief NPP图像内存管理辅助类
 */
template<typename T>
class NppImageMemory {
public:
    NppImageMemory() : ptr_(nullptr), step_(0), width_(0), height_(0) {}
    
    NppImageMemory(int width, int height) : width_(width), height_(height) {
        allocate(width, height);
    }
    
    ~NppImageMemory() {
        free();
    }
    
    // 禁止拷贝
    NppImageMemory(const NppImageMemory&) = delete;
    NppImageMemory& operator=(const NppImageMemory&) = delete;
    
    // 支持移动
    NppImageMemory(NppImageMemory&& other) noexcept 
        : ptr_(other.ptr_), step_(other.step_), 
          width_(other.width_), height_(other.height_) {
        other.ptr_ = nullptr;
        other.step_ = 0;
        other.width_ = 0;
        other.height_ = 0;
    }
    
    void allocate(int width, int height) {
        free();
        width_ = width;
        height_ = height;
        
        if constexpr (std::is_same_v<T, Npp8u>) {
            ptr_ = nppiMalloc_8u_C1(width, height, &step_);
        } else if constexpr (std::is_same_v<T, Npp16u>) {
            ptr_ = nppiMalloc_16u_C1(width, height, &step_);
        } else if constexpr (std::is_same_v<T, Npp16s>) {
            ptr_ = nppiMalloc_16s_C1(width, height, &step_);
        } else if constexpr (std::is_same_v<T, Npp32f>) {
            ptr_ = nppiMalloc_32f_C1(width, height, &step_);
        } else if constexpr (std::is_same_v<T, Npp32fc>) {
            ptr_ = nppiMalloc_32fc_C1(width, height, &step_);
        } else {
            static_assert(sizeof(T) == 0, "Unsupported NPP data type");
        }
        
        if (!ptr_) {
            throw std::runtime_error("Failed to allocate NPP image memory");
        }
    }
    
    void free() {
        if (ptr_) {
            nppiFree(ptr_);
            ptr_ = nullptr;
            step_ = 0;
            width_ = 0;
            height_ = 0;
        }
    }
    
    T* get() const { return ptr_; }
    int step() const { return step_; }
    int width() const { return width_; }
    int height() const { return height_; }
    NppiSize size() const { return {width_, height_}; }
    
    // 数据传输
    void copyFromHost(const std::vector<T>& hostData) {
        ASSERT_EQ(hostData.size(), width_ * height_) << "Size mismatch in copyFromHost";
        cudaError_t err = cudaMemcpy2D(ptr_, step_, hostData.data(), 
                                     width_ * sizeof(T), width_ * sizeof(T), 
                                     height_, cudaMemcpyHostToDevice);
        ASSERT_EQ(err, cudaSuccess) << "Failed to copy from host";
    }
    
    void copyToHost(std::vector<T>& hostData) const {
        hostData.resize(width_ * height_);
        cudaError_t err = cudaMemcpy2D(hostData.data(), width_ * sizeof(T),
                                     ptr_, step_, width_ * sizeof(T), 
                                     height_, cudaMemcpyDeviceToHost);
        ASSERT_EQ(err, cudaSuccess) << "Failed to copy to host";
    }
    
    void fill(T value) {
        std::vector<T> hostData(width_ * height_, value);
        copyFromHost(hostData);
    }

private:
    T* ptr_;
    int step_;
    int width_;
    int height_;
};

/**
 * @brief 数据生成工具类
 */
class TestDataGenerator {
public:
    // 生成随机数据
    template<typename T>
    static void generateRandom(std::vector<T>& data, T minVal, T maxVal, 
                              unsigned seed = std::random_device{}()) {
        std::mt19937 gen(seed);
        
        if constexpr (std::is_floating_point_v<T>) {
            std::uniform_real_distribution<T> dis(minVal, maxVal);
            for (auto& val : data) {
                val = dis(gen);
            }
        } else {
            std::uniform_int_distribution<int> dis(minVal, maxVal);
            for (auto& val : data) {
                val = static_cast<T>(dis(gen));
            }
        }
    }
    
    // 生成递增数据
    template<typename T>
    static void generateSequential(std::vector<T>& data, T startVal = T{}, T step = T{1}) {
        T current = startVal;
        for (auto& val : data) {
            val = current;
            current += step;
        }
    }
    
    // 生成常量数据
    template<typename T>
    static void generateConstant(std::vector<T>& data, T value) {
        std::fill(data.begin(), data.end(), value);
    }
    
    // 生成测试模式数据（棋盘、条纹等）
    template<typename T>
    static void generateCheckerboard(std::vector<T>& data, int width, int height, 
                                    T value1, T value2) {
        ASSERT_EQ(data.size(), width * height) << "Data size mismatch";
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                data[idx] = ((x + y) % 2 == 0) ? value1 : value2;
            }
        }
    }
};

/**
 * @brief 结果验证工具类
 */
class ResultValidator {
public:
    // 验证数组相等（整数类型）
    template<typename T>
    static bool arraysEqual(const std::vector<T>& a, const std::vector<T>& b, 
                           T tolerance = T{}) {
        if (a.size() != b.size()) return false;
        
        if constexpr (std::is_floating_point_v<T>) {
            for (size_t i = 0; i < a.size(); i++) {
                if (std::abs(a[i] - b[i]) > tolerance) {
                    return false;
                }
            }
        } else {
            for (size_t i = 0; i < a.size(); i++) {
                if (std::abs(static_cast<long long>(a[i]) - static_cast<long long>(b[i])) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }
    
    // 查找第一个不匹配的位置
    template<typename T>
    static std::pair<bool, size_t> findFirstMismatch(const std::vector<T>& a, 
                                                     const std::vector<T>& b,
                                                     T tolerance = T{}) {
        if (a.size() != b.size()) {
            return {false, 0};
        }
        
        for (size_t i = 0; i < a.size(); i++) {
            bool match;
            if constexpr (std::is_floating_point_v<T>) {
                match = (std::abs(a[i] - b[i]) <= tolerance);
            } else {
                match = (std::abs(static_cast<long long>(a[i]) - static_cast<long long>(b[i])) <= tolerance);
            }
            
            if (!match) {
                return {false, i};
            }
        }
        return {true, 0};
    }
    
    // 计算数组的统计信息
    template<typename T>
    static void computeStats(const std::vector<T>& data, T& minVal, T& maxVal, 
                           double& mean, double& stddev) {
        if (data.empty()) {
            minVal = maxVal = T{};
            mean = stddev = 0.0;
            return;
        }
        
        minVal = *std::min_element(data.begin(), data.end());
        maxVal = *std::max_element(data.begin(), data.end());
        
        double sum = 0.0;
        for (const auto& val : data) {
            sum += static_cast<double>(val);
        }
        mean = sum / data.size();
        
        double variance = 0.0;
        for (const auto& val : data) {
            double diff = static_cast<double>(val) - mean;
            variance += diff * diff;
        }
        stddev = std::sqrt(variance / data.size());
    }
};

} // namespace npp_functional_test