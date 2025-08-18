/**
 * OpenNPP测试框架 - 图像缓冲区类
 * 正确处理pitched内存的图像缓冲区
 */
#pragma once

#include "npp_test_common.h"

namespace opennpp {
namespace test {

/**
 * 图像缓冲区类 - 正确处理pitched内存
 */
template<typename T>
class ImageBuffer {
private:
    T* hostData_;       // 主机内存（pitched布局）
    T* deviceData_;     // 设备内存（NPPI分配）
    int width_;         // 宽度（像素）
    int height_;        // 高度（像素）  
    int channels_;      // 通道数
    int hostStep_;      // 主机步长（字节）
    int deviceStep_;    // 设备步长（字节）
    bool ownsMemory_;   // 是否拥有内存（用于包装外部内存）
    
public:
    // 默认构造函数 - 分配新内存
    ImageBuffer(int width, int height, int channels = 1) 
        : width_(width), height_(height), channels_(channels),
          hostData_(nullptr), deviceData_(nullptr), ownsMemory_(true) {
        
        // 分配设备内存
        deviceData_ = nppiMallocHelper<T>(width, height, channels, &deviceStep_);
        if (!deviceData_) {
            throw std::runtime_error("Failed to allocate device memory");
        }
        
        // 分配主机内存（与设备步长对齐）
        hostStep_ = deviceStep_;
        size_t totalSize = hostStep_ * height;
        
        // 使用对齐分配以获得更好的性能
        hostData_ = static_cast<T*>(std::aligned_alloc(128, totalSize));
        if (!hostData_) {
            nppiFree(deviceData_);
            throw std::runtime_error("Failed to allocate host memory");
        }
        
        // 初始化为0
        std::memset(hostData_, 0, totalSize);
    }
    
    // 包装已存在的设备内存
    ImageBuffer(T* devicePtr, int width, int height, int channels, int step)
        : width_(width), height_(height), channels_(channels),
          deviceData_(devicePtr), deviceStep_(step),
          hostData_(nullptr), hostStep_(0), ownsMemory_(false) {
    }
    
    // 析构函数
    ~ImageBuffer() {
        if (ownsMemory_) {
            if (deviceData_) {
                nppiFree(deviceData_);
            }
            if (hostData_) {
                std::free(hostData_);
            }
        }
    }
    
    // 禁用拷贝
    ImageBuffer(const ImageBuffer&) = delete;
    ImageBuffer& operator=(const ImageBuffer&) = delete;
    
    // 启用移动
    ImageBuffer(ImageBuffer&& other) noexcept
        : hostData_(other.hostData_), deviceData_(other.deviceData_),
          width_(other.width_), height_(other.height_), channels_(other.channels_),
          hostStep_(other.hostStep_), deviceStep_(other.deviceStep_),
          ownsMemory_(other.ownsMemory_) {
        other.hostData_ = nullptr;
        other.deviceData_ = nullptr;
        other.ownsMemory_ = false;
    }
    
    ImageBuffer& operator=(ImageBuffer&& other) noexcept {
        if (this != &other) {
            // 清理当前资源
            if (ownsMemory_) {
                if (deviceData_) nppiFree(deviceData_);
                if (hostData_) std::free(hostData_);
            }
            
            // 移动资源
            hostData_ = other.hostData_;
            deviceData_ = other.deviceData_;
            width_ = other.width_;
            height_ = other.height_;
            channels_ = other.channels_;
            hostStep_ = other.hostStep_;
            deviceStep_ = other.deviceStep_;
            ownsMemory_ = other.ownsMemory_;
            
            other.hostData_ = nullptr;
            other.deviceData_ = nullptr;
            other.ownsMemory_ = false;
        }
        return *this;
    }
    
    // 填充数据模式
    void fill(TestPattern pattern, T baseValue = T(50), unsigned int seed = 12345) {
        if (!hostData_) {
            throw std::runtime_error("No host memory allocated");
        }
        
        switch (pattern) {
            case TestPattern::CONSTANT:
                fillConstant(baseValue);
                break;
            case TestPattern::GRADIENT:
                fillGradient(baseValue);
                break;
            case TestPattern::RANDOM:
                fillRandom(seed);
                break;
            case TestPattern::CHECKERBOARD:
                fillCheckerboard(baseValue);
                break;
            case TestPattern::EDGE_CASE:
                fillEdgeCase();
                break;
        }
        
        copyToDevice();
    }
    
    // 拷贝到设备
    void copyToDevice() {
        if (!hostData_ || !deviceData_) return;
        
        cudaError_t result = cudaMemcpy2D(
            deviceData_, deviceStep_,
            hostData_, hostStep_,
            width_ * channels_ * sizeof(T), height_,
            cudaMemcpyHostToDevice
        );
        
        if (result != cudaSuccess) {
            throw std::runtime_error("Failed to copy data to device: " + 
                                   std::string(cudaGetErrorString(result)));
        }
    }
    
    // 从设备拷贝
    void copyFromDevice() {
        if (!hostData_ || !deviceData_) {
            // 如果没有主机内存，分配它
            if (!hostData_ && deviceData_) {
                hostStep_ = deviceStep_;
                size_t totalSize = hostStep_ * height_;
                hostData_ = static_cast<T*>(std::aligned_alloc(128, totalSize));
                if (!hostData_) {
                    throw std::runtime_error("Failed to allocate host memory for copy");
                }
            } else {
                return;
            }
        }
        
        cudaError_t result = cudaMemcpy2D(
            hostData_, hostStep_,
            deviceData_, deviceStep_,
            width_ * channels_ * sizeof(T), height_,
            cudaMemcpyDeviceToHost
        );
        
        if (result != cudaSuccess) {
            throw std::runtime_error("Failed to copy data from device: " + 
                                   std::string(cudaGetErrorString(result)));
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
    
    // 获取特定行
    T* getHostRow(int y) {
        if (!hostData_ || y < 0 || y >= height_) return nullptr;
        return reinterpret_cast<T*>(
            reinterpret_cast<char*>(hostData_) + y * hostStep_);
    }
    
    const T* getHostRow(int y) const {
        if (!hostData_ || y < 0 || y >= height_) return nullptr;
        return reinterpret_cast<const T*>(
            reinterpret_cast<const char*>(hostData_) + y * hostStep_);
    }
    
    // 比较两个缓冲区
    static double compare(const ImageBuffer<T>& buf1, 
                         const ImageBuffer<T>& buf2,
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
            
            if (!row1 || !row2) continue;
            
            for (int x = 0; x < buf1.width() * buf1.channels(); x++) {
                double diff = std::abs(static_cast<double>(row1[x]) - 
                                     static_cast<double>(row2[x]));
                maxDiff = std::max(maxDiff, diff);
            }
        }
        
        return maxDiff;
    }
    
    // 打印缓冲区内容（调试用）
    void print(int maxRows = 5, int maxCols = 10) const {
        if (!hostData_) {
            std::cout << "Buffer has no host data" << std::endl;
            return;
        }
        
        std::cout << "Buffer " << width_ << "x" << height_ 
                  << "x" << channels_ << " (step=" << hostStep_ << " bytes):" << std::endl;
        
        int rowsToPrint = std::min(maxRows, height_);
        int colsToPrint = std::min(maxCols, width_ * channels_);
        
        for (int y = 0; y < rowsToPrint; y++) {
            const T* row = getHostRow(y);
            std::cout << "Row " << y << ": ";
            for (int x = 0; x < colsToPrint; x++) {
                if constexpr (std::is_floating_point_v<T>) {
                    std::cout << std::fixed << std::setprecision(2) << row[x] << " ";
                } else {
                    std::cout << static_cast<int>(row[x]) << " ";
                }
            }
            if (colsToPrint < width_ * channels_) {
                std::cout << "...";
            }
            std::cout << std::endl;
        }
        if (rowsToPrint < height_) {
            std::cout << "..." << std::endl;
        }
    }
    
private:
    void fillConstant(T value) {
        for (int y = 0; y < height_; y++) {
            T* row = getHostRow(y);
            for (int x = 0; x < width_ * channels_; x++) {
                row[x] = value;
            }
        }
    }
    
    void fillGradient(T baseValue) {
        for (int y = 0; y < height_; y++) {
            T* row = getHostRow(y);
            for (int x = 0; x < width_ * channels_; x++) {
                row[x] = baseValue + static_cast<T>((y * width_ * channels_ + x) % 100);
            }
        }
    }
    
    void fillRandom(unsigned int seed) {
        std::mt19937 gen(seed);
        
        for (int y = 0; y < height_; y++) {
            T* row = getHostRow(y);
            
            if constexpr (std::is_integral_v<T>) {
                std::uniform_int_distribution<int> dis(0, std::numeric_limits<T>::max());
                for (int x = 0; x < width_ * channels_; x++) {
                    row[x] = static_cast<T>(dis(gen));
                }
            } else {
                std::uniform_real_distribution<float> dis(0.0f, 100.0f);
                for (int x = 0; x < width_ * channels_; x++) {
                    row[x] = static_cast<T>(dis(gen));
                }
            }
        }
    }
    
    void fillCheckerboard(T baseValue) {
        T altValue = baseValue + T(50);
        for (int y = 0; y < height_; y++) {
            T* row = getHostRow(y);
            for (int x = 0; x < width_; x++) {
                T value = ((x / 8 + y / 8) % 2 == 0) ? baseValue : altValue;
                for (int c = 0; c < channels_; c++) {
                    row[x * channels_ + c] = value;
                }
            }
        }
    }
    
    void fillEdgeCase() {
        for (int y = 0; y < height_; y++) {
            T* row = getHostRow(y);
            for (int x = 0; x < width_ * channels_; x++) {
                if constexpr (std::is_integral_v<T>) {
                    // 测试边界值
                    if (x % 3 == 0) row[x] = std::numeric_limits<T>::min();
                    else if (x % 3 == 1) row[x] = std::numeric_limits<T>::max();
                    else row[x] = T(0);
                } else {
                    // 浮点数边界测试
                    if (x % 4 == 0) row[x] = T(0);
                    else if (x % 4 == 1) row[x] = T(-1);
                    else if (x % 4 == 2) row[x] = T(1);
                    else row[x] = std::numeric_limits<T>::max() / T(2);
                }
            }
        }
    }
};

} // namespace test
} // namespace opennpp