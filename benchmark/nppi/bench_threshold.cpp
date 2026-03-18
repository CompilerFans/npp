#include "../benchmark_base.h"
#include <nppi_threshold_and_compare_operations.h>

namespace npp_benchmark {

extern void initDeviceRandom8u(Npp8u* data, int size);
extern void initDeviceRandom16u(Npp16u* data, int size);
extern void initDeviceRandom16s(Npp16s* data, int size);
extern void initDeviceRandom32f(Npp32f* data, int size);

namespace {

std::vector<ImageSize> getThresholdImageSizes() {
    return {
        {1280, 720},
        {1280, 1024},
        {1920, 1080},
    };
}

template<typename T>
void initThresholdInput(T* data, int count) {
    copyDeterministicHostDataToDevice(data, static_cast<size_t>(count));
}

template<>
void initThresholdInput<Npp8u>(Npp8u* data, int count) {
    initDeviceRandom8u(data, count);
}

template<>
void initThresholdInput<Npp16u>(Npp16u* data, int count) {
    initDeviceRandom16u(data, count);
}

template<>
void initThresholdInput<Npp16s>(Npp16s* data, int count) {
    initDeviceRandom16s(data, count);
}

template<>
void initThresholdInput<Npp32f>(Npp32f* data, int count) {
    initDeviceRandom32f(data, count);
}

template<typename T, int Channels, typename Fn>
class NppiThresholdScalarBenchmark : public BenchmarkBase {
public:
    NppiThresholdScalarBenchmark(const std::string& benchName, const char* symbolName, const char* dataType, T threshold)
        : BenchmarkBase(benchName), symbolName_(symbolName), dataType_(dataType), threshold_(threshold) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string symErr;
        const Fn fn = DynamicSymbol<Fn>(symbolName_).get(&symErr);

        std::vector<BenchmarkResult> results;
        for (const auto& imgSize : getThresholdImageSizes()) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = dataType_;
            result.size = imgSize.toString();
            result.channels = Channels;
            result.iterations = measureIterations;

            if (fn == nullptr) {
                result.success = false;
                result.errorMessage = std::string("Missing symbol ") + symbolName_ + ": " + symErr;
                results.push_back(result);
                continue;
            }

            const int pixelCount = imgSize.width * imgSize.height * Channels;
            const int step = imgSize.width * Channels * static_cast<int>(sizeof(T));
            DeviceBuffer<T> src(pixelCount);
            DeviceBuffer<T> dst(pixelCount);
            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }

            initThresholdInput<T>(src.get(), pixelCount);
            dst.fillZero();
            const NppiSize roi{imgSize.width, imgSize.height};

            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st = fn(src.get(), step, dst.get(), step, roi, threshold_, NPP_CMP_GREATER);
                if (st != NPP_SUCCESS) {
                    result.success = false;
                    result.errorMessage = "NPP error during warmup: " + std::to_string(st);
                    break;
                }
            }
            if (!result.errorMessage.empty()) {
                results.push_back(result);
                continue;
            }
            (void)cudaDeviceSynchronize();

            if (runBatchedMeasurement(result, measureIterations,
                                      [&]() { return fn(src.get(), step, dst.get(), step, roi, threshold_, NPP_CMP_GREATER); })) {
                result.throughputGBps =
                    computeThroughput(2ULL * static_cast<size_t>(pixelCount) * sizeof(T), result.avgTimeMs);
            }

            results.push_back(result);
        }
        return results;
    }

private:
    const char* symbolName_;
    const char* dataType_;
    T threshold_;
};

template<typename T, int Channels, typename Fn>
class NppiThresholdVectorBenchmark : public BenchmarkBase {
public:
    NppiThresholdVectorBenchmark(const std::string& benchName, const char* symbolName, const char* dataType)
        : BenchmarkBase(benchName), symbolName_(symbolName), dataType_(dataType) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string symErr;
        const Fn fn = DynamicSymbol<Fn>(symbolName_).get(&symErr);

        std::vector<BenchmarkResult> results;
        for (const auto& imgSize : getThresholdImageSizes()) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = dataType_;
            result.size = imgSize.toString();
            result.channels = Channels;
            result.iterations = measureIterations;

            if (fn == nullptr) {
                result.success = false;
                result.errorMessage = std::string("Missing symbol ") + symbolName_ + ": " + symErr;
                results.push_back(result);
                continue;
            }

            const int pixelCount = imgSize.width * imgSize.height * Channels;
            const int step = imgSize.width * Channels * static_cast<int>(sizeof(T));
            DeviceBuffer<T> src(pixelCount);
            DeviceBuffer<T> dst(pixelCount);
            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }

            initThresholdInput<T>(src.get(), pixelCount);
            dst.fillZero();
            const NppiSize roi{imgSize.width, imgSize.height};

            T thresholds[3] = {static_cast<T>(32), static_cast<T>(64), static_cast<T>(96)};
            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st = fn(src.get(), step, dst.get(), step, roi, thresholds, NPP_CMP_GREATER);
                if (st != NPP_SUCCESS) {
                    result.success = false;
                    result.errorMessage = "NPP error during warmup: " + std::to_string(st);
                    break;
                }
            }
            if (!result.errorMessage.empty()) {
                results.push_back(result);
                continue;
            }
            (void)cudaDeviceSynchronize();

            if (runBatchedMeasurement(
                    result, measureIterations,
                    [&]() { return fn(src.get(), step, dst.get(), step, roi, thresholds, NPP_CMP_GREATER); })) {
                result.throughputGBps =
                    computeThroughput(2ULL * static_cast<size_t>(pixelCount) * sizeof(T), result.avgTimeMs);
            }

            results.push_back(result);
        }
        return results;
    }

private:
    const char* symbolName_;
    const char* dataType_;
};

#define REGISTER_THRESHOLD_SCALAR_BENCH(BenchClass, Symbol, Type, Channels, DType, Threshold)                        \
    class BenchClass final : public NppiThresholdScalarBenchmark<Type, Channels, decltype(&Symbol)> {               \
    public:                                                                                                          \
        BenchClass() : NppiThresholdScalarBenchmark<Type, Channels, decltype(&Symbol)>(#Symbol, #Symbol, DType, Threshold) {} \
    };                                                                                                               \
    REGISTER_BENCHMARK(BenchClass)

#define REGISTER_THRESHOLD_VECTOR_BENCH(BenchClass, Symbol, Type, Channels, DType)                                  \
    class BenchClass final : public NppiThresholdVectorBenchmark<Type, Channels, decltype(&Symbol)> {              \
    public:                                                                                                          \
        BenchClass() : NppiThresholdVectorBenchmark<Type, Channels, decltype(&Symbol)>(#Symbol, #Symbol, DType) {} \
    };                                                                                                               \
    REGISTER_BENCHMARK(BenchClass)

REGISTER_THRESHOLD_SCALAR_BENCH(BenchThreshold_8u_C1R, nppiThreshold_8u_C1R, Npp8u, 1, "8u", static_cast<Npp8u>(127));
REGISTER_THRESHOLD_SCALAR_BENCH(BenchThreshold_16u_C1R, nppiThreshold_16u_C1R, Npp16u, 1, "16u", static_cast<Npp16u>(1024));
REGISTER_THRESHOLD_SCALAR_BENCH(BenchThreshold_16s_C1R, nppiThreshold_16s_C1R, Npp16s, 1, "16s", static_cast<Npp16s>(64));
REGISTER_THRESHOLD_SCALAR_BENCH(BenchThreshold_32f_C1R, nppiThreshold_32f_C1R, Npp32f, 1, "32f", static_cast<Npp32f>(0.5f));

REGISTER_THRESHOLD_VECTOR_BENCH(BenchThreshold_8u_C3R, nppiThreshold_8u_C3R, Npp8u, 3, "8u");
REGISTER_THRESHOLD_VECTOR_BENCH(BenchThreshold_16u_C3R, nppiThreshold_16u_C3R, Npp16u, 3, "16u");
REGISTER_THRESHOLD_VECTOR_BENCH(BenchThreshold_16s_C3R, nppiThreshold_16s_C3R, Npp16s, 3, "16s");
REGISTER_THRESHOLD_VECTOR_BENCH(BenchThreshold_32f_C3R, nppiThreshold_32f_C3R, Npp32f, 3, "32f");

REGISTER_THRESHOLD_VECTOR_BENCH(BenchThreshold_8u_AC4R, nppiThreshold_8u_AC4R, Npp8u, 4, "8u");
REGISTER_THRESHOLD_VECTOR_BENCH(BenchThreshold_16u_AC4R, nppiThreshold_16u_AC4R, Npp16u, 4, "16u");
REGISTER_THRESHOLD_VECTOR_BENCH(BenchThreshold_16s_AC4R, nppiThreshold_16s_AC4R, Npp16s, 4, "16s");
REGISTER_THRESHOLD_VECTOR_BENCH(BenchThreshold_32f_AC4R, nppiThreshold_32f_AC4R, Npp32f, 4, "32f");

#undef REGISTER_THRESHOLD_SCALAR_BENCH
#undef REGISTER_THRESHOLD_VECTOR_BENCH

} // namespace

} // namespace npp_benchmark
