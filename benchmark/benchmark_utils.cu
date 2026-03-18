#include "benchmark_base.h"

namespace npp_benchmark {

// Initialize random data on device
__global__ void initRandomKernel(Npp8u* data, int size, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Simple LCG random
        unsigned int val = (seed + idx) * 1103515245 + 12345;
        data[idx] = static_cast<Npp8u>(val % 256);
    }
}

__global__ void initRandomKernel16u(Npp16u* data, int size, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned int val = (seed + idx) * 1103515245 + 12345;
        data[idx] = static_cast<Npp16u>(val % 65536);
    }
}

__global__ void initRandomKernel16s(Npp16s* data, int size, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned int val = (seed + idx) * 1103515245 + 12345;
        data[idx] = static_cast<Npp16s>((val % 65536) - 32768);
    }
}

__global__ void initRandomKernel32f(Npp32f* data, int size, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned int val = (seed + idx) * 1103515245 + 12345;
        data[idx] = static_cast<Npp32f>(val % 65536) / 65536.0f;
    }
}

void initDeviceRandom8u(Npp8u* data, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    initRandomKernel<<<numBlocks, blockSize>>>(data, size, kBenchmarkRandomSeed + 0x08u);
    cudaDeviceSynchronize();
}

void initDeviceRandom16u(Npp16u* data, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    initRandomKernel16u<<<numBlocks, blockSize>>>(data, size, kBenchmarkRandomSeed + 0x10u);
    cudaDeviceSynchronize();
}

void initDeviceRandom16s(Npp16s* data, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    initRandomKernel16s<<<numBlocks, blockSize>>>(data, size, kBenchmarkRandomSeed + 0x11u);
    cudaDeviceSynchronize();
}

void initDeviceRandom32f(Npp32f* data, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    initRandomKernel32f<<<numBlocks, blockSize>>>(data, size, kBenchmarkRandomSeed + 0x20u);
    cudaDeviceSynchronize();
}

// Get device info
DeviceInfo getDeviceInfo() {
    DeviceInfo info;
    int deviceCount = 0;
    cudaError_t st = cudaGetDeviceCount(&deviceCount);
    if (st != cudaSuccess || deviceCount <= 0) {
        info.name = std::string("CUDA unavailable: ") + cudaGetErrorString(st);
        return info;
    }

    int deviceId = 0;
    st = cudaGetDevice(&deviceId);
    if (st != cudaSuccess) {
        info.name = std::string("CUDA error(cudaGetDevice): ") + cudaGetErrorString(st);
        return info;
    }

    cudaDeviceProp prop{};
    st = cudaGetDeviceProperties(&prop, deviceId);
    if (st != cudaSuccess) {
        info.name = std::string("CUDA error(cudaGetDeviceProperties): ") + cudaGetErrorString(st);
        return info;
    }

    info.name = prop.name;
    info.major = prop.major;
    info.minor = prop.minor;
    info.globalMemBytes = prop.totalGlobalMem;
    info.smCount = prop.multiProcessorCount;
    info.clockRateKHz = prop.clockRate;
    info.memoryClockRateKHz = prop.memoryClockRate;
    info.memoryBusWidthBits = prop.memoryBusWidth;
    return info;
}

void printDeviceInfo() {
    const DeviceInfo info = getDeviceInfo();
    std::cout << "========================================" << std::endl;
    std::cout << "Device: " << info.name << std::endl;
    if (info.globalMemBytes == 0) {
        std::cout << "Compute Capability: N/A" << std::endl;
        std::cout << "Global Memory: N/A" << std::endl;
        std::cout << "SM Count: N/A" << std::endl;
        std::cout << "Clock Rate: N/A" << std::endl;
        std::cout << "Memory Clock Rate: N/A" << std::endl;
        std::cout << "Memory Bus Width: N/A" << std::endl;
    } else {
        std::cout << "Compute Capability: " << info.major << "." << info.minor << std::endl;
        std::cout << "Global Memory: " << info.globalMemBytes / (1024 * 1024) << " MB" << std::endl;
        std::cout << "SM Count: " << info.smCount << std::endl;
        std::cout << "Clock Rate: " << info.clockRateKHz / 1000 << " MHz" << std::endl;
        std::cout << "Memory Clock Rate: " << info.memoryClockRateKHz / 1000 << " MHz" << std::endl;
        std::cout << "Memory Bus Width: " << info.memoryBusWidthBits << " bits" << std::endl;
    }
    std::cout << "========================================" << std::endl;
}

} // namespace npp_benchmark
