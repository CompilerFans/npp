#include "../benchmark_base.h"
#include <nppi_threshold_and_compare_operations.h>

namespace npp_benchmark {

extern void initDeviceRandom8u(Npp8u* data, int size);
extern void initDeviceRandom16u(Npp16u* data, int size);
extern void initDeviceRandom16s(Npp16s* data, int size);
extern void initDeviceRandom32f(Npp32f* data, int size);

namespace {

std::vector<ImageSize> getCompareImageSizes() {
    return {
        {1280, 720},
        {1280, 1024},
        {1920, 1080},
    };
}

template<typename T>
void initCompareInput(T* data, int count) {
    copyDeterministicHostDataToDevice(data, static_cast<size_t>(count));
}

template<>
void initCompareInput<Npp8u>(Npp8u* data, int count) { initDeviceRandom8u(data, count); }
template<>
void initCompareInput<Npp16u>(Npp16u* data, int count) { initDeviceRandom16u(data, count); }
template<>
void initCompareInput<Npp16s>(Npp16s* data, int count) { initDeviceRandom16s(data, count); }
template<>
void initCompareInput<Npp32f>(Npp32f* data, int count) { initDeviceRandom32f(data, count); }

template<typename SrcT, int Channels, typename Fn>
class NppiCompareBenchmark : public BenchmarkBase {
public:
    NppiCompareBenchmark(const std::string& benchName, const char* symbolName, const char* dataType)
        : BenchmarkBase(benchName), symbolName_(symbolName), dataType_(dataType) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string symErr;
        const Fn fn = DynamicSymbol<Fn>(symbolName_).get(&symErr);
        std::vector<BenchmarkResult> results;

        for (const auto& imgSize : getCompareImageSizes()) {
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

            const int srcPixels = imgSize.width * imgSize.height * Channels;
            const int srcStep = imgSize.width * Channels * static_cast<int>(sizeof(SrcT));
            const int dstStep = imgSize.width * static_cast<int>(sizeof(Npp8u));

            DeviceBuffer<SrcT> src1(srcPixels);
            DeviceBuffer<SrcT> src2(srcPixels);
            DeviceBuffer<Npp8u> dst(imgSize.width * imgSize.height);
            if (!validateDeviceBuffer(src1, result, "src1") || !validateDeviceBuffer(src2, result, "src2") ||
                !validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }

            initCompareInput<SrcT>(src1.get(), srcPixels);
            initCompareInput<SrcT>(src2.get(), srcPixels);
            dst.fillZero();
            const NppiSize roi{imgSize.width, imgSize.height};

            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st =
                    fn(src1.get(), srcStep, src2.get(), srcStep, dst.get(), dstStep, roi, NPP_CMP_GREATER);
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
                                      [&]() {
                                          return fn(src1.get(), srcStep, src2.get(), srcStep, dst.get(), dstStep, roi,
                                                    NPP_CMP_GREATER);
                                      })) {
                result.throughputGBps =
                    computeThroughput(2ULL * srcPixels * sizeof(SrcT) + static_cast<size_t>(imgSize.width) *
                                                                     static_cast<size_t>(imgSize.height),
                                      result.avgTimeMs);
            }
            results.push_back(result);
        }
        return results;
    }

private:
    const char* symbolName_;
    const char* dataType_;
};

template<typename SrcT, int Channels, typename Fn>
class NppiCompareCBenchmark : public BenchmarkBase {
public:
    NppiCompareCBenchmark(const std::string& benchName, const char* symbolName, const char* dataType)
        : BenchmarkBase(benchName), symbolName_(symbolName), dataType_(dataType) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string symErr;
        const Fn fn = DynamicSymbol<Fn>(symbolName_).get(&symErr);
        std::vector<BenchmarkResult> results;

        for (const auto& imgSize : getCompareImageSizes()) {
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

            const int srcPixels = imgSize.width * imgSize.height * Channels;
            const int srcStep = imgSize.width * Channels * static_cast<int>(sizeof(SrcT));
            const int dstStep = imgSize.width * static_cast<int>(sizeof(Npp8u));

            DeviceBuffer<SrcT> src(srcPixels);
            DeviceBuffer<Npp8u> dst(imgSize.width * imgSize.height);
            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }

            initCompareInput<SrcT>(src.get(), srcPixels);
            dst.fillZero();
            const NppiSize roi{imgSize.width, imgSize.height};

            if constexpr (Channels == 1) {
                const SrcT constantValue = static_cast<SrcT>(32);
                for (int i = 0; i < warmupIterations; ++i) {
                    const NppStatus st = fn(src.get(), srcStep, constantValue, dst.get(), dstStep, roi, NPP_CMP_GREATER);
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
                                          [&]() {
                                              return fn(src.get(), srcStep, constantValue, dst.get(), dstStep, roi,
                                                        NPP_CMP_GREATER);
                                          })) {
                    result.throughputGBps =
                        computeThroughput(static_cast<size_t>(srcPixels) * sizeof(SrcT) +
                                              static_cast<size_t>(imgSize.width) * static_cast<size_t>(imgSize.height),
                                          result.avgTimeMs);
                }
            } else {
                SrcT constants[3] = {static_cast<SrcT>(16), static_cast<SrcT>(32), static_cast<SrcT>(64)};
                for (int i = 0; i < warmupIterations; ++i) {
                    const NppStatus st = fn(src.get(), srcStep, constants, dst.get(), dstStep, roi, NPP_CMP_GREATER);
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
                                          [&]() {
                                              return fn(src.get(), srcStep, constants, dst.get(), dstStep, roi,
                                                        NPP_CMP_GREATER);
                                          })) {
                    result.throughputGBps =
                        computeThroughput(static_cast<size_t>(srcPixels) * sizeof(SrcT) +
                                              static_cast<size_t>(imgSize.width) * static_cast<size_t>(imgSize.height),
                                          result.avgTimeMs);
                }
            }

            results.push_back(result);
        }
        return results;
    }

private:
    const char* symbolName_;
    const char* dataType_;
};

template<int Channels, typename Fn>
class NppiCompareEqualEpsBenchmark : public BenchmarkBase {
public:
    explicit NppiCompareEqualEpsBenchmark(const std::string& benchName, const char* symbolName)
        : BenchmarkBase(benchName), symbolName_(symbolName) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string symErr;
        const Fn fn = DynamicSymbol<Fn>(symbolName_).get(&symErr);
        std::vector<BenchmarkResult> results;
        for (const auto& imgSize : getCompareImageSizes()) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = "32f";
            result.size = imgSize.toString();
            result.channels = Channels;
            result.iterations = measureIterations;
            if (fn == nullptr) {
                result.success = false;
                result.errorMessage = std::string("Missing symbol ") + symbolName_ + ": " + symErr;
                results.push_back(result);
                continue;
            }

            const int srcPixels = imgSize.width * imgSize.height * Channels;
            const int srcStep = imgSize.width * Channels * static_cast<int>(sizeof(Npp32f));
            const int dstStep = imgSize.width * static_cast<int>(sizeof(Npp8u));
            DeviceBuffer<Npp32f> src1(srcPixels), src2(srcPixels);
            DeviceBuffer<Npp8u> dst(imgSize.width * imgSize.height);
            if (!validateDeviceBuffer(src1, result, "src1") || !validateDeviceBuffer(src2, result, "src2") ||
                !validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }
            initCompareInput<Npp32f>(src1.get(), srcPixels);
            initCompareInput<Npp32f>(src2.get(), srcPixels);
            dst.fillZero();
            const NppiSize roi{imgSize.width, imgSize.height};
            constexpr Npp32f kEps = 0.01f;
            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st = fn(src1.get(), srcStep, src2.get(), srcStep, dst.get(), dstStep, roi, kEps);
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
                                      [&]() { return fn(src1.get(), srcStep, src2.get(), srcStep, dst.get(), dstStep, roi, kEps); })) {
                result.throughputGBps =
                    computeThroughput(2ULL * srcPixels * sizeof(Npp32f) +
                                          static_cast<size_t>(imgSize.width) * static_cast<size_t>(imgSize.height),
                                      result.avgTimeMs);
            }
            results.push_back(result);
        }
        return results;
    }

private:
    const char* symbolName_;
};

template<int Channels, typename Fn>
class NppiCompareEqualEpsCBenchmark : public BenchmarkBase {
public:
    explicit NppiCompareEqualEpsCBenchmark(const std::string& benchName, const char* symbolName)
        : BenchmarkBase(benchName), symbolName_(symbolName) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string symErr;
        const Fn fn = DynamicSymbol<Fn>(symbolName_).get(&symErr);
        std::vector<BenchmarkResult> results;
        for (const auto& imgSize : getCompareImageSizes()) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = "32f";
            result.size = imgSize.toString();
            result.channels = Channels;
            result.iterations = measureIterations;
            if (fn == nullptr) {
                result.success = false;
                result.errorMessage = std::string("Missing symbol ") + symbolName_ + ": " + symErr;
                results.push_back(result);
                continue;
            }

            const int srcPixels = imgSize.width * imgSize.height * Channels;
            const int srcStep = imgSize.width * Channels * static_cast<int>(sizeof(Npp32f));
            const int dstStep = imgSize.width * static_cast<int>(sizeof(Npp8u));
            DeviceBuffer<Npp32f> src(srcPixels);
            DeviceBuffer<Npp8u> dst(imgSize.width * imgSize.height);
            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }
            initCompareInput<Npp32f>(src.get(), srcPixels);
            dst.fillZero();
            const NppiSize roi{imgSize.width, imgSize.height};
            constexpr Npp32f kEps = 0.01f;

            if constexpr (Channels == 1) {
                constexpr Npp32f kValue = 0.5f;
                for (int i = 0; i < warmupIterations; ++i) {
                    const NppStatus st = fn(src.get(), srcStep, kValue, dst.get(), dstStep, roi, kEps);
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
                                          [&]() { return fn(src.get(), srcStep, kValue, dst.get(), dstStep, roi, kEps); })) {
                    result.throughputGBps =
                        computeThroughput(static_cast<size_t>(srcPixels) * sizeof(Npp32f) +
                                              static_cast<size_t>(imgSize.width) * static_cast<size_t>(imgSize.height),
                                          result.avgTimeMs);
                }
            } else {
                Npp32f constants[3] = {0.25f, 0.5f, 0.75f};
                for (int i = 0; i < warmupIterations; ++i) {
                    const NppStatus st = fn(src.get(), srcStep, constants, dst.get(), dstStep, roi, kEps);
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
                                          [&]() { return fn(src.get(), srcStep, constants, dst.get(), dstStep, roi, kEps); })) {
                    result.throughputGBps =
                        computeThroughput(static_cast<size_t>(srcPixels) * sizeof(Npp32f) +
                                              static_cast<size_t>(imgSize.width) * static_cast<size_t>(imgSize.height),
                                          result.avgTimeMs);
                }
            }

            results.push_back(result);
        }
        return results;
    }

private:
    const char* symbolName_;
};

#define REGISTER_COMPARE_BENCH(BenchClass, Symbol, Type, Channels, DType)                                 \
    class BenchClass final : public NppiCompareBenchmark<Type, Channels, decltype(&Symbol)> {            \
    public:                                                                                                \
        BenchClass() : NppiCompareBenchmark<Type, Channels, decltype(&Symbol)>(#Symbol, #Symbol, DType) {} \
    };                                                                                                     \
    REGISTER_BENCHMARK(BenchClass)

#define REGISTER_COMPAREC_BENCH(BenchClass, Symbol, Type, Channels, DType)                                \
    class BenchClass final : public NppiCompareCBenchmark<Type, Channels, decltype(&Symbol)> {           \
    public:                                                                                                \
        BenchClass() : NppiCompareCBenchmark<Type, Channels, decltype(&Symbol)>(#Symbol, #Symbol, DType) {} \
    };                                                                                                     \
    REGISTER_BENCHMARK(BenchClass)

#define REGISTER_COMPARE_EPS_BENCH(BenchClass, Symbol, Channels)                                            \
    class BenchClass final : public NppiCompareEqualEpsBenchmark<Channels, decltype(&Symbol)> {           \
    public:                                                                                                \
        BenchClass() : NppiCompareEqualEpsBenchmark<Channels, decltype(&Symbol)>(#Symbol, #Symbol) {}    \
    };                                                                                                     \
    REGISTER_BENCHMARK(BenchClass)

#define REGISTER_COMPARE_EPSC_BENCH(BenchClass, Symbol, Channels)                                           \
    class BenchClass final : public NppiCompareEqualEpsCBenchmark<Channels, decltype(&Symbol)> {         \
    public:                                                                                                \
        BenchClass() : NppiCompareEqualEpsCBenchmark<Channels, decltype(&Symbol)>(#Symbol, #Symbol) {}   \
    };                                                                                                     \
    REGISTER_BENCHMARK(BenchClass)

REGISTER_COMPARE_BENCH(BenchCompare_8u_C1R, nppiCompare_8u_C1R, Npp8u, 1, "8u");
REGISTER_COMPARE_BENCH(BenchCompare_8u_C3R, nppiCompare_8u_C3R, Npp8u, 3, "8u");
REGISTER_COMPARE_BENCH(BenchCompare_8u_C4R, nppiCompare_8u_C4R, Npp8u, 4, "8u");
REGISTER_COMPARE_BENCH(BenchCompare_8u_AC4R, nppiCompare_8u_AC4R, Npp8u, 4, "8u");
REGISTER_COMPARE_BENCH(BenchCompare_16u_C1R, nppiCompare_16u_C1R, Npp16u, 1, "16u");
REGISTER_COMPARE_BENCH(BenchCompare_16u_C3R, nppiCompare_16u_C3R, Npp16u, 3, "16u");
REGISTER_COMPARE_BENCH(BenchCompare_16u_C4R, nppiCompare_16u_C4R, Npp16u, 4, "16u");
REGISTER_COMPARE_BENCH(BenchCompare_16u_AC4R, nppiCompare_16u_AC4R, Npp16u, 4, "16u");
REGISTER_COMPARE_BENCH(BenchCompare_16s_C1R, nppiCompare_16s_C1R, Npp16s, 1, "16s");
REGISTER_COMPARE_BENCH(BenchCompare_16s_C3R, nppiCompare_16s_C3R, Npp16s, 3, "16s");
REGISTER_COMPARE_BENCH(BenchCompare_16s_C4R, nppiCompare_16s_C4R, Npp16s, 4, "16s");
REGISTER_COMPARE_BENCH(BenchCompare_16s_AC4R, nppiCompare_16s_AC4R, Npp16s, 4, "16s");
REGISTER_COMPARE_BENCH(BenchCompare_32f_C1R, nppiCompare_32f_C1R, Npp32f, 1, "32f");
REGISTER_COMPARE_BENCH(BenchCompare_32f_C3R, nppiCompare_32f_C3R, Npp32f, 3, "32f");
REGISTER_COMPARE_BENCH(BenchCompare_32f_C4R, nppiCompare_32f_C4R, Npp32f, 4, "32f");
REGISTER_COMPARE_BENCH(BenchCompare_32f_AC4R, nppiCompare_32f_AC4R, Npp32f, 4, "32f");

REGISTER_COMPAREC_BENCH(BenchCompareC_8u_C1R, nppiCompareC_8u_C1R, Npp8u, 1, "8u");
REGISTER_COMPAREC_BENCH(BenchCompareC_8u_C3R, nppiCompareC_8u_C3R, Npp8u, 3, "8u");
REGISTER_COMPAREC_BENCH(BenchCompareC_8u_C4R, nppiCompareC_8u_C4R, Npp8u, 4, "8u");
REGISTER_COMPAREC_BENCH(BenchCompareC_8u_AC4R, nppiCompareC_8u_AC4R, Npp8u, 4, "8u");
REGISTER_COMPAREC_BENCH(BenchCompareC_16u_C1R, nppiCompareC_16u_C1R, Npp16u, 1, "16u");
REGISTER_COMPAREC_BENCH(BenchCompareC_16u_C3R, nppiCompareC_16u_C3R, Npp16u, 3, "16u");
REGISTER_COMPAREC_BENCH(BenchCompareC_16u_C4R, nppiCompareC_16u_C4R, Npp16u, 4, "16u");
REGISTER_COMPAREC_BENCH(BenchCompareC_16u_AC4R, nppiCompareC_16u_AC4R, Npp16u, 4, "16u");
REGISTER_COMPAREC_BENCH(BenchCompareC_16s_C1R, nppiCompareC_16s_C1R, Npp16s, 1, "16s");
REGISTER_COMPAREC_BENCH(BenchCompareC_16s_C3R, nppiCompareC_16s_C3R, Npp16s, 3, "16s");
REGISTER_COMPAREC_BENCH(BenchCompareC_16s_C4R, nppiCompareC_16s_C4R, Npp16s, 4, "16s");
REGISTER_COMPAREC_BENCH(BenchCompareC_16s_AC4R, nppiCompareC_16s_AC4R, Npp16s, 4, "16s");
REGISTER_COMPAREC_BENCH(BenchCompareC_32f_C1R, nppiCompareC_32f_C1R, Npp32f, 1, "32f");
REGISTER_COMPAREC_BENCH(BenchCompareC_32f_C3R, nppiCompareC_32f_C3R, Npp32f, 3, "32f");
REGISTER_COMPAREC_BENCH(BenchCompareC_32f_C4R, nppiCompareC_32f_C4R, Npp32f, 4, "32f");
REGISTER_COMPAREC_BENCH(BenchCompareC_32f_AC4R, nppiCompareC_32f_AC4R, Npp32f, 4, "32f");

REGISTER_COMPARE_EPS_BENCH(BenchCompareEqualEps_32f_C1R, nppiCompareEqualEps_32f_C1R, 1);
REGISTER_COMPARE_EPS_BENCH(BenchCompareEqualEps_32f_C3R, nppiCompareEqualEps_32f_C3R, 3);
REGISTER_COMPARE_EPS_BENCH(BenchCompareEqualEps_32f_C4R, nppiCompareEqualEps_32f_C4R, 4);
REGISTER_COMPARE_EPS_BENCH(BenchCompareEqualEps_32f_AC4R, nppiCompareEqualEps_32f_AC4R, 4);

REGISTER_COMPARE_EPSC_BENCH(BenchCompareEqualEpsC_32f_C1R, nppiCompareEqualEpsC_32f_C1R, 1);
REGISTER_COMPARE_EPSC_BENCH(BenchCompareEqualEpsC_32f_C3R, nppiCompareEqualEpsC_32f_C3R, 3);
REGISTER_COMPARE_EPSC_BENCH(BenchCompareEqualEpsC_32f_C4R, nppiCompareEqualEpsC_32f_C4R, 4);
REGISTER_COMPARE_EPSC_BENCH(BenchCompareEqualEpsC_32f_AC4R, nppiCompareEqualEpsC_32f_AC4R, 4);

#undef REGISTER_COMPARE_BENCH
#undef REGISTER_COMPAREC_BENCH
#undef REGISTER_COMPARE_EPS_BENCH
#undef REGISTER_COMPARE_EPSC_BENCH

} // namespace

} // namespace npp_benchmark
