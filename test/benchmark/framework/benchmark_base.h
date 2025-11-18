#pragma once

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <random>
#include <stdexcept>

namespace npp_benchmark {

// 性能测试基类
class NppBenchmarkBase {
public:
    // CUDA 设备初始化
    static void InitializeCuda() {
        cudaSetDevice(0);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
    }
    
    // 同步 GPU 并检查错误
    static void SyncAndCheckError() {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error: ") + 
                                   cudaGetErrorString(err));
        }
    }
    
    // GPU 内存分配辅助函数
    template<typename T>
    static T* AllocateDevice(size_t numElements) {
        T* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, numElements * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory");
        }
        return ptr;
    }
    
    // GPU 内存释放
    static void FreeDevice(void* ptr) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    
    // 生成随机测试数据
    template<typename T>
    static void GenerateRandomData(std::vector<T>& data, T minVal, T maxVal) {
        std::random_device rd;
        std::mt19937 gen(rd());
        
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
    
    // 计算理论带宽利用率
    static double ComputeBandwidthUtilization(
        size_t bytesProcessed, 
        double timeSeconds,
        double peakBandwidthGBps = 900.0)  // 默认 A100 带宽
    {
        double actualBandwidthGBps = (bytesProcessed / 1e9) / timeSeconds;
        return (actualBandwidthGBps / peakBandwidthGBps) * 100.0;
    }
};

// 图像操作基准测试基类
template<typename PixelType>
class ImageBenchmarkBase : public NppBenchmarkBase {
public:
    PixelType* d_src1_ = nullptr;
    PixelType* d_src2_ = nullptr;
    PixelType* d_dst_ = nullptr;
    int width_;
    int height_;
    int step_;
    
    void SetupImageMemory(int width, int height) {
        width_ = width;
        height_ = height;
        step_ = width * sizeof(PixelType);
        
        // 使用 nppiMalloc 保证对齐
        d_src1_ = AllocateDevice<PixelType>(width * height);
        d_src2_ = AllocateDevice<PixelType>(width * height);
        d_dst_ = AllocateDevice<PixelType>(width * height);
        
        // 初始化数据
        std::vector<PixelType> hostData(width * height);
        GenerateRandomData(hostData, 
                          static_cast<PixelType>(0), 
                          static_cast<PixelType>(255));
        
        cudaMemcpy(d_src1_, hostData.data(), 
                   width * height * sizeof(PixelType), 
                   cudaMemcpyHostToDevice);
        GenerateRandomData(hostData, 
                          static_cast<PixelType>(0), 
                          static_cast<PixelType>(255));
        cudaMemcpy(d_src2_, hostData.data(), 
                   width * height * sizeof(PixelType), 
                   cudaMemcpyHostToDevice);
    }
    
    void TeardownImageMemory() {
        FreeDevice(d_src1_);
        FreeDevice(d_src2_);
        FreeDevice(d_dst_);
        d_src1_ = d_src2_ = d_dst_ = nullptr;
    }
    
    // 计算图像操作的数据吞吐量
    size_t ComputeImageBytes(int numInputs, int numOutputs) const {
        size_t pixelSize = sizeof(PixelType);
        size_t imageSize = width_ * height_ * pixelSize;
        return imageSize * (numInputs + numOutputs);
    }
};

// 性能指标报告辅助宏
#define REPORT_THROUGHPUT(state, bytesProcessed) \
    do { \
        state.SetBytesProcessed( \
            static_cast<int64_t>(state.iterations() * (bytesProcessed))); \
    } while(0)

#define REPORT_ITEMS_PROCESSED(state, itemsProcessed) \
    do { \
        state.SetItemsProcessed( \
            static_cast<int64_t>(state.iterations() * (itemsProcessed))); \
    } while(0)

#define REPORT_CUSTOM_METRIC(state, key, value) \
    do { \
        state.counters[key] = value; \
    } while(0)

} // namespace npp_benchmark
