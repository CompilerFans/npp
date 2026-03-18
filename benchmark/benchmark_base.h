#ifndef NPP_BENCHMARK_BASE_H
#define NPP_BENCHMARK_BASE_H

#include <string>
#include <vector>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdint>
#include <type_traits>

#if defined(__linux__)
#include <dlfcn.h>
#endif

#include <cuda_runtime.h>
#include <npp.h>

namespace npp_benchmark {

inline constexpr int kBenchmarkWarmupDefault = 5;
inline constexpr int kBenchmarkMeasureDefault = 100;
inline constexpr uint32_t kBenchmarkRandomSeed = 0x4D505042u;

#if defined(__linux__)
inline void* openSharedLibraryHandle(const char* soname) {
    void* handle = dlopen(soname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
    if (handle != nullptr) return handle;
    return dlopen(soname, RTLD_NOW | RTLD_GLOBAL);
}

inline const std::vector<void*>& preferredSymbolHandles() {
    static const std::vector<void*> handles = [] {
        std::vector<void*> out;
#if defined(USE_NVIDIA_NPP)
        const char* libs[] = {
            "libnppial.so.12", "libnppicc.so.12", "libnppidei.so.12", "libnppif.so.12", "libnppig.so.12",
            "libnppim.so.12",  "libnppist.so.12", "libnppisu.so.12",  "libnppitc.so.12", "libnpps.so.12",
            "libnppc.so.12",
        };
        for (const char* lib : libs) {
            if (void* h = openSharedLibraryHandle(lib)) out.push_back(h);
        }
#endif
        return out;
    }();
    return handles;
}
#endif

struct DeviceInfo {
    std::string name;
    int major = 0;
    int minor = 0;
    size_t globalMemBytes = 0;
    int smCount = 0;
    int clockRateKHz = 0;
    int memoryClockRateKHz = 0;
    int memoryBusWidthBits = 0;
};

DeviceInfo getDeviceInfo();
void printDeviceInfo();

inline NppStreamContext makeNppStreamContext(cudaStream_t stream = 0) {
    NppStreamContext ctx{};
    ctx.hStream = stream;

    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        return ctx;
    }
    ctx.nCudaDeviceId = dev;

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        return ctx;
    }
    ctx.nMultiProcessorCount = prop.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;

    (void)cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, dev);
    (void)cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, dev);

    unsigned int flags = 0;
    (void)cudaStreamGetFlags(stream, &flags);
    ctx.nStreamFlags = flags;
    return ctx;
}

inline uint32_t mixDeterministicSeed(uint32_t seed, uint32_t index) {
    uint32_t value = seed + 0x9E3779B9u + index;
    value ^= value >> 16;
    value *= 0x85EBCA6Bu;
    value ^= value >> 13;
    value *= 0xC2B2AE35u;
    value ^= value >> 16;
    return value;
}

template<typename T>
inline T deterministicRandomValue(size_t index, uint32_t seed = kBenchmarkRandomSeed) {
    const uint32_t value = mixDeterministicSeed(seed, static_cast<uint32_t>(index));
    if constexpr (std::is_same_v<T, Npp8u>) {
        return static_cast<Npp8u>(value & 0xFFu);
    } else if constexpr (std::is_same_v<T, Npp16u>) {
        return static_cast<Npp16u>(value & 0xFFFFu);
    } else if constexpr (std::is_same_v<T, Npp16s>) {
        return static_cast<Npp16s>(static_cast<int32_t>(value & 0xFFFFu) - 32768);
    } else if constexpr (std::is_same_v<T, Npp32f>) {
        return static_cast<Npp32f>(value & 0xFFFFu) / 65535.0f;
    } else if constexpr (std::is_floating_point_v<T>) {
        return static_cast<T>(value & 0xFFFFu) / static_cast<T>(65535.0);
    } else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
        return static_cast<T>(value);
    } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        return static_cast<T>(static_cast<int32_t>(value));
    } else {
        return T{};
    }
}

template<typename T>
inline void fillDeterministicHostBuffer(T* data, size_t count, uint32_t seed = kBenchmarkRandomSeed) {
    if (data == nullptr) return;
    for (size_t i = 0; i < count; ++i) {
        data[i] = deterministicRandomValue<T>(i, seed);
    }
}

template<typename T>
inline void copyDeterministicHostDataToDevice(T* data, size_t count, uint32_t seed = kBenchmarkRandomSeed) {
    if (data == nullptr || count == 0) return;
    std::vector<T> host(count);
    fillDeterministicHostBuffer(host.data(), count, seed);
    cudaMemcpy(data, host.data(), count * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename Fn>
class DynamicSymbol {
public:
    explicit DynamicSymbol(const char* name) : name_(name) {}

    Fn get(std::string* error = nullptr) const {
#if defined(__linux__)
        for (void* handle : preferredSymbolHandles()) {
            dlerror();
            void* sym = dlsym(handle, name_);
            const char* err = dlerror();
            if (err == nullptr && sym != nullptr) {
                if (error) error->clear();
                return reinterpret_cast<Fn>(sym);
            }
        }

        dlerror();
        void* sym = dlsym(RTLD_DEFAULT, name_);
        const char* err = dlerror();
        if (err != nullptr || sym == nullptr) {
            if (error) *error = err != nullptr ? err : "symbol not found";
            return nullptr;
        }
        if (error) error->clear();
        return reinterpret_cast<Fn>(sym);
#else
        if (error) *error = "Dynamic symbol lookup is only implemented for Linux";
        return nullptr;
#endif
    }

    const char* name() const { return name_; }

private:
    const char* name_;
};

// Image size configuration for benchmarks
struct ImageSize {
    int width;
    int height;

    int pixels() const { return width * height; }
    std::string toString() const {
        return std::to_string(width) + "x" + std::to_string(height);
    }
};

// Standard benchmark sizes
inline std::vector<ImageSize> getStandardImageSizes() {
    return {
        {640, 480},      // VGA
        {1280, 720},     // HD
        {1280, 1024},    // SXGA
        {1920, 1080},    // Full HD
        {3840, 2160},    // 4K
    };
}

// Signal length configuration
struct SignalSize {
    int length;

    std::string toString() const {
        if (length >= 1000000) {
            return std::to_string(length / 1000000) + "M";
        } else if (length >= 1000) {
            return std::to_string(length / 1000) + "K";
        }
        return std::to_string(length);
    }
};

inline std::vector<SignalSize> getStandardSignalSizes() {
    return {
        {1024},
        {4096},
        {16384},
        {65536},
        {262144},
        {1048576},
    };
}

// Benchmark result for a single run
struct BenchmarkResult {
    std::string functionName;
    std::string size;
    std::string variantTags;
    std::string dataType;
    int channels = 0;

    double minTimeMs = 0.0;
    double maxTimeMs = 0.0;
    double avgTimeMs = 0.0;
    double stdDevMs = 0.0;
    double throughputGBps = 0.0;

    int iterations = 0;
    bool success = false;
    std::string errorMessage;

    void print() const {
        if (!success) {
            std::cout << std::left << std::setw(40) << functionName
                      << std::setw(18) << size
                      << std::setw(16) << (variantTags.empty() ? "-" : variantTags)
                      << std::setw(8) << dataType
                      << std::setw(4) << channels
                      << " FAILED: " << errorMessage << std::endl;
            return;
        }

        std::cout << std::left << std::setw(40) << functionName
                  << std::setw(18) << size
                  << std::setw(16) << (variantTags.empty() ? "-" : variantTags)
                  << std::setw(8) << dataType
                  << std::setw(4) << channels
                  << std::fixed << std::setprecision(3)
                  << std::setw(12) << avgTimeMs << " ms"
                  << std::setw(12) << throughputGBps << " GB/s"
                  << std::endl;
    }
};

// Timer using CUDA events for accurate GPU timing
class CudaTimer {
public:
    CudaTimer() {
        start_ = nullptr;
        stop_ = nullptr;
        valid_ = (cudaEventCreate(&start_) == cudaSuccess) && (cudaEventCreate(&stop_) == cudaSuccess);
        if (!valid_) {
            if (start_) cudaEventDestroy(start_);
            if (stop_) cudaEventDestroy(stop_);
            start_ = nullptr;
            stop_ = nullptr;
        }
    }

    ~CudaTimer() {
        if (start_) cudaEventDestroy(start_);
        if (stop_) cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        if (!valid_) return;
        cudaEventRecord(start_, stream);
    }

    void stop(cudaStream_t stream = 0) {
        if (!valid_) return;
        cudaEventRecord(stop_, stream);
        cudaEventSynchronize(stop_);
    }

    float elapsedMs() const {
        if (!valid_) return 0.0f;
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
    bool valid_ = false;
};

// Base class for all benchmarks
class BenchmarkBase {
public:
    BenchmarkBase(const std::string& name) : name_(name) {}
    virtual ~BenchmarkBase() = default;

    const std::string& name() const { return name_; }

    // Run benchmark with specified warmup count and timed-batch call count
    virtual std::vector<BenchmarkResult> run(int warmupIterations = kBenchmarkWarmupDefault,
                                              int measureIterations = kBenchmarkMeasureDefault) = 0;

protected:
    std::string name_;

    // Helper to compute statistics from timing samples
    static void computeStatistics(const std::vector<double>& samples,
                                  double& minVal, double& maxVal,
                                  double& avgVal, double& stdDev) {
        if (samples.empty()) {
            minVal = maxVal = avgVal = stdDev = 0.0;
            return;
        }

        minVal = *std::min_element(samples.begin(), samples.end());
        maxVal = *std::max_element(samples.begin(), samples.end());
        avgVal = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();

        double sqSum = 0.0;
        for (double v : samples) {
            sqSum += (v - avgVal) * (v - avgVal);
        }
        stdDev = std::sqrt(sqSum / samples.size());
    }

    // Compute throughput in GB/s
    static double computeThroughput(size_t bytes, double timeMs) {
        if (timeMs <= 0) return 0.0;
        return (bytes / 1e9) / (timeMs / 1000.0);
    }
};

inline void setBatchedMeasurementStats(BenchmarkResult& result, double totalTimeMs, int batchIterations) {
    const int effectiveBatchIterations = std::max(1, batchIterations);
    const double avgTimeMs = totalTimeMs / static_cast<double>(effectiveBatchIterations);
    result.minTimeMs = avgTimeMs;
    result.maxTimeMs = avgTimeMs;
    result.avgTimeMs = avgTimeMs;
    result.stdDevMs = 0.0;
    result.success = true;
}

inline void resetCudaIfFatal(cudaError_t err) {
    if (err == cudaErrorIllegalAddress || err == cudaErrorLaunchFailure || err == cudaErrorLaunchTimeout ||
        err == cudaErrorAssert || err == cudaErrorDevicesUnavailable || err == cudaErrorDeviceUninitialized) {
        (void)cudaDeviceReset();
        (void)cudaGetLastError();
    }
}

template<typename Call>
inline bool runBatchedMeasurement(BenchmarkResult& result, int batchIterations, Call&& call,
                                  bool checkCudaError = false,
                                  const char* nppErrorPrefix = "NPP error",
                                  const char* cudaErrorPrefix = "CUDA error during measure") {
    const int effectiveBatchIterations = std::max(1, batchIterations);
    CudaTimer timer;

    timer.start();
    for (int i = 0; i < effectiveBatchIterations; ++i) {
        const NppStatus status = call();
        if (status != NPP_SUCCESS) {
            result.success = false;
            result.errorMessage = std::string(nppErrorPrefix) + ": " + std::to_string(status);
            return false;
        }
    }
    timer.stop();

    if (checkCudaError) {
        const cudaError_t iterErr = cudaGetLastError();
        if (iterErr != cudaSuccess) {
            result.success = false;
            result.errorMessage = std::string(cudaErrorPrefix) + ": " + cudaGetErrorString(iterErr);
            resetCudaIfFatal(iterErr);
            return false;
        }
    }

    setBatchedMeasurementStats(result, static_cast<double>(timer.elapsedMs()), effectiveBatchIterations);
    return true;
}

// Device memory RAII wrapper
template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : ptr_(nullptr), size_(0), lastError_(cudaSuccess) {}

    explicit DeviceBuffer(size_t count) : ptr_(nullptr), size_(count), lastError_(cudaSuccess) {
        if (count == 0) return;
        lastError_ = cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T));
        if (lastError_ != cudaSuccess) {
            ptr_ = nullptr;
        }
    }

    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_), lastError_(other.lastError_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
        other.lastError_ = cudaSuccess;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            lastError_ = other.lastError_;
            other.ptr_ = nullptr;
            other.size_ = 0;
            other.lastError_ = cudaSuccess;
        }
        return *this;
    }

    // Non-copyable
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }
    size_t bytes() const { return size_ * sizeof(T); }
    cudaError_t lastError() const { return lastError_; }
    bool ok() const { return ptr_ != nullptr || size_ == 0; }

    void resize(size_t count) {
        if (ptr_) cudaFree(ptr_);
        ptr_ = nullptr;
        size_ = count;
        lastError_ = cudaSuccess;
        if (count == 0) return;
        lastError_ = cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T));
        if (lastError_ != cudaSuccess) {
            ptr_ = nullptr;
        }
    }

    void fillRandom() {
        if (!ptr_) return;
        copyDeterministicHostDataToDevice(ptr_, size_);
    }

    void fillZero() {
        if (!ptr_) return;
        cudaMemset(ptr_, 0, bytes());
    }

private:
    T* ptr_;
    size_t size_;
    cudaError_t lastError_;
};

template<typename T>
inline bool validateDeviceBuffer(const DeviceBuffer<T>& buf, BenchmarkResult& result, const char* bufName) {
    if (buf.ok()) return true;
    result.success = false;
    result.errorMessage =
        std::string("cudaMalloc failed for ") + bufName + ": " + cudaGetErrorString(buf.lastError());
    // Some CUDA errors (e.g. illegal address) put the device into a sticky failure state where subsequent allocations
    // will fail until the context is reset. Reset here to avoid cascading benchmark failures.
    const cudaError_t st = buf.lastError();
    resetCudaIfFatal(st);
    return false;
}

// Benchmark registry
class BenchmarkRegistry {
public:
    static BenchmarkRegistry& instance() {
        static BenchmarkRegistry registry;
        return registry;
    }

    void registerBenchmark(std::unique_ptr<BenchmarkBase> benchmark) {
        benchmarks_.push_back(std::move(benchmark));
    }

    const std::vector<std::unique_ptr<BenchmarkBase>>& benchmarks() const {
        return benchmarks_;
    }

    std::vector<BenchmarkResult> runAll(int warmup = kBenchmarkWarmupDefault,
                                        int iterations = kBenchmarkMeasureDefault,
                                        bool verbose = false) {
        std::vector<BenchmarkResult> allResults;

        for (const auto& bench : benchmarks_) {
            if (verbose) {
                std::cout << "Running benchmark: " << bench->name() << std::endl;
            }
            auto results = bench->run(warmup, iterations);
            allResults.insert(allResults.end(), results.begin(), results.end());
            // If a benchmark triggered a fatal CUDA error, subsequent benchmarks can become meaningless noise. Reset
            // the device here as a best-effort containment strategy.
            cudaError_t st = cudaGetLastError();
            resetCudaIfFatal(st);
        }

        return allResults;
    }

    std::vector<BenchmarkResult> runFiltered(const std::string& filter,
                                             int warmup = kBenchmarkWarmupDefault,
                                             int iterations = kBenchmarkMeasureDefault,
                                             bool verbose = false) {
        std::vector<BenchmarkResult> allResults;

        for (const auto& bench : benchmarks_) {
            if (bench->name().find(filter) != std::string::npos) {
                if (verbose) {
                    std::cout << "Running benchmark: " << bench->name() << std::endl;
                }
                auto results = bench->run(warmup, iterations);
                allResults.insert(allResults.end(), results.begin(), results.end());
                cudaError_t st = cudaGetLastError();
                resetCudaIfFatal(st);
            }
        }

        return allResults;
    }

private:
    BenchmarkRegistry() = default;
    std::vector<std::unique_ptr<BenchmarkBase>> benchmarks_;
};

// Macro to register a benchmark
#define REGISTER_BENCHMARK(BenchClass) \
    namespace { \
        struct BenchClass##Registrar { \
            BenchClass##Registrar() { \
                npp_benchmark::BenchmarkRegistry::instance().registerBenchmark( \
                    std::make_unique<BenchClass>()); \
            } \
        } g_##BenchClass##Registrar; \
    }

} // namespace npp_benchmark

#endif // NPP_BENCHMARK_BASE_H
