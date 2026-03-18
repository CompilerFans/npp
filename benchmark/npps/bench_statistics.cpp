#include "../benchmark_base.h"
#include <npps.h>

namespace npp_benchmark {

extern void initDeviceRandom32f(Npp32f* data, int size);

namespace {

#ifdef USE_NVIDIA_NPP_TESTS
using HostSizeType = size_t;
#else
using HostSizeType = int;
#endif

template<typename T>
void initStatsSignalInput(T* data, int count) {
    std::vector<T> host(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        host[static_cast<size_t>(i)] = static_cast<T>((i % 251) - 125);
    }
    (void)cudaMemcpy(data, host.data(), static_cast<size_t>(count) * sizeof(T), cudaMemcpyHostToDevice);
}

template<>
void initStatsSignalInput<Npp32f>(Npp32f* data, int count) { initDeviceRandom32f(data, count); }

template<typename SrcT, typename DstT, typename FnGetBufferSize, typename FnOp>
class NppsStatisticsBenchmark : public BenchmarkBase {
public:
    NppsStatisticsBenchmark(const std::string& benchName, const char* bufferSymbol, const char* opSymbol,
                            const char* dataType)
        : BenchmarkBase(benchName), bufferSymbol_(bufferSymbol), opSymbol_(opSymbol), dataType_(dataType) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string bufferErr;
        std::string opErr;
        const FnGetBufferSize fnGetBufferSize = DynamicSymbol<FnGetBufferSize>(bufferSymbol_).get(&bufferErr);
        const FnOp fnOp = DynamicSymbol<FnOp>(opSymbol_).get(&opErr);
        std::vector<BenchmarkResult> results;

        for (const auto& sigSize : getStandardSignalSizes()) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = dataType_;
            result.size = sigSize.toString();
            result.channels = 1;
            result.iterations = measureIterations;

            if (fnGetBufferSize == nullptr) {
                result.success = false;
                result.errorMessage = std::string("Missing symbol ") + bufferSymbol_ + ": " + bufferErr;
                results.push_back(result);
                continue;
            }
            if (fnOp == nullptr) {
                result.success = false;
                result.errorMessage = std::string("Missing symbol ") + opSymbol_ + ": " + opErr;
                results.push_back(result);
                continue;
            }

            DeviceBuffer<SrcT> src(sigSize.length);
            DeviceBuffer<DstT> dst(1);
            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }

            HostSizeType bufferSize = 0;
            const NppStatus bufferStatus = fnGetBufferSize(static_cast<HostSizeType>(sigSize.length), &bufferSize);
            if (bufferStatus != NPP_SUCCESS) {
                result.success = false;
                result.errorMessage = "NPP error during buffer-size query: " + std::to_string(bufferStatus);
                results.push_back(result);
                continue;
            }

            DeviceBuffer<Npp8u> scratch(static_cast<size_t>(bufferSize));
            if (!validateDeviceBuffer(scratch, result, "scratch")) {
                results.push_back(result);
                continue;
            }

            initStatsSignalInput<SrcT>(src.get(), sigSize.length);
            dst.fillZero();
            scratch.fillZero();

            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st =
                    fnOp(src.get(), static_cast<HostSizeType>(sigSize.length), dst.get(), scratch.get());
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
                                          return fnOp(src.get(), static_cast<HostSizeType>(sigSize.length), dst.get(),
                                                      scratch.get());
                                      })) {
                result.throughputGBps =
                    computeThroughput(static_cast<size_t>(sigSize.length) * sizeof(SrcT), result.avgTimeMs);
            }
            results.push_back(result);
        }

        return results;
    }

private:
    const char* bufferSymbol_;
    const char* opSymbol_;
    const char* dataType_;
};

template<typename SrcT, typename DstT, typename FnGetBufferSize, typename FnOp>
class NppsExtremaBenchmark : public BenchmarkBase {
public:
    NppsExtremaBenchmark(const std::string& benchName, const char* bufferSymbol, const char* opSymbol,
                         const char* dataType)
        : BenchmarkBase(benchName), bufferSymbol_(bufferSymbol), opSymbol_(opSymbol), dataType_(dataType) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string bufferErr;
        std::string opErr;
        const FnGetBufferSize fnGetBufferSize = DynamicSymbol<FnGetBufferSize>(bufferSymbol_).get(&bufferErr);
        const FnOp fnOp = DynamicSymbol<FnOp>(opSymbol_).get(&opErr);
        std::vector<BenchmarkResult> results;

        for (const auto& sigSize : getStandardSignalSizes()) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = dataType_;
            result.size = sigSize.toString();
            result.channels = 1;
            result.iterations = measureIterations;

            if (fnGetBufferSize == nullptr) {
                result.success = false;
                result.errorMessage = std::string("Missing symbol ") + bufferSymbol_ + ": " + bufferErr;
                results.push_back(result);
                continue;
            }
            if (fnOp == nullptr) {
                result.success = false;
                result.errorMessage = std::string("Missing symbol ") + opSymbol_ + ": " + opErr;
                results.push_back(result);
                continue;
            }

            DeviceBuffer<SrcT> src(sigSize.length);
            DeviceBuffer<DstT> dst(1);
            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }

            HostSizeType bufferSize = 0;
            const NppStatus bufferStatus = fnGetBufferSize(static_cast<HostSizeType>(sigSize.length), &bufferSize);
            if (bufferStatus != NPP_SUCCESS) {
                result.success = false;
                result.errorMessage = "NPP error during buffer-size query: " + std::to_string(bufferStatus);
                results.push_back(result);
                continue;
            }

            DeviceBuffer<Npp8u> scratch(static_cast<size_t>(bufferSize));
            if (!validateDeviceBuffer(scratch, result, "scratch")) {
                results.push_back(result);
                continue;
            }

            initStatsSignalInput<SrcT>(src.get(), sigSize.length);
            dst.fillZero();
            scratch.fillZero();

            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st =
                    fnOp(src.get(), static_cast<HostSizeType>(sigSize.length), dst.get(), scratch.get());
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
                                          return fnOp(src.get(), static_cast<HostSizeType>(sigSize.length), dst.get(),
                                                      scratch.get());
                                      })) {
                result.throughputGBps =
                    computeThroughput(static_cast<size_t>(sigSize.length) * sizeof(SrcT), result.avgTimeMs);
            }
            results.push_back(result);
        }

        return results;
    }

private:
    const char* bufferSymbol_;
    const char* opSymbol_;
    const char* dataType_;
};

template<typename SrcT, typename DstT, typename FnGetBufferSize, typename FnOp>
class NppsMeanStdDevBenchmark : public BenchmarkBase {
public:
    NppsMeanStdDevBenchmark(const std::string& benchName, const char* bufferSymbol, const char* opSymbol,
                            const char* dataType)
        : BenchmarkBase(benchName), bufferSymbol_(bufferSymbol), opSymbol_(opSymbol), dataType_(dataType) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string bufferErr;
        std::string opErr;
        const FnGetBufferSize fnGetBufferSize = DynamicSymbol<FnGetBufferSize>(bufferSymbol_).get(&bufferErr);
        const FnOp fnOp = DynamicSymbol<FnOp>(opSymbol_).get(&opErr);
        std::vector<BenchmarkResult> results;

        for (const auto& sigSize : getStandardSignalSizes()) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = dataType_;
            result.size = sigSize.toString();
            result.channels = 1;
            result.iterations = measureIterations;

            if (fnGetBufferSize == nullptr) {
                result.success = false;
                result.errorMessage = std::string("Missing symbol ") + bufferSymbol_ + ": " + bufferErr;
                results.push_back(result);
                continue;
            }
            if (fnOp == nullptr) {
                result.success = false;
                result.errorMessage = std::string("Missing symbol ") + opSymbol_ + ": " + opErr;
                results.push_back(result);
                continue;
            }

            DeviceBuffer<SrcT> src(sigSize.length);
            DeviceBuffer<DstT> mean(1);
            DeviceBuffer<DstT> stddev(1);
            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(mean, result, "mean") ||
                !validateDeviceBuffer(stddev, result, "stddev")) {
                results.push_back(result);
                continue;
            }

            HostSizeType bufferSize = 0;
            const NppStatus bufferStatus = fnGetBufferSize(static_cast<HostSizeType>(sigSize.length), &bufferSize);
            if (bufferStatus != NPP_SUCCESS) {
                result.success = false;
                result.errorMessage = "NPP error during buffer-size query: " + std::to_string(bufferStatus);
                results.push_back(result);
                continue;
            }

            DeviceBuffer<Npp8u> scratch(static_cast<size_t>(bufferSize));
            if (!validateDeviceBuffer(scratch, result, "scratch")) {
                results.push_back(result);
                continue;
            }

            initStatsSignalInput<SrcT>(src.get(), sigSize.length);
            mean.fillZero();
            stddev.fillZero();
            scratch.fillZero();

            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st = fnOp(src.get(), static_cast<HostSizeType>(sigSize.length), mean.get(),
                                          stddev.get(), scratch.get());
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
                                          return fnOp(src.get(), static_cast<HostSizeType>(sigSize.length),
                                                      mean.get(), stddev.get(), scratch.get());
                                      })) {
                result.throughputGBps =
                    computeThroughput(static_cast<size_t>(sigSize.length) * sizeof(SrcT), result.avgTimeMs);
            }
            results.push_back(result);
        }

        return results;
    }

private:
    const char* bufferSymbol_;
    const char* opSymbol_;
    const char* dataType_;
};

#define REGISTER_NPPS_STATS_BENCH(BenchClass, BufferFn, OpFn, SrcType, DstType, DType)                            \
    class BenchClass final                                                                                          \
        : public NppsStatisticsBenchmark<SrcType, DstType, decltype(&BufferFn), decltype(&OpFn)> {                \
    public:                                                                                                         \
        BenchClass()                                                                                                \
            : NppsStatisticsBenchmark<SrcType, DstType, decltype(&BufferFn), decltype(&OpFn)>(                    \
                  #OpFn, #BufferFn, #OpFn, DType) {}                                                                \
    };                                                                                                              \
    REGISTER_BENCHMARK(BenchClass)

#define REGISTER_NPPS_EXTREMA_BENCH(BenchClass, BufferFn, OpFn, SrcType, DstType, DType)                          \
    class BenchClass final                                                                                          \
        : public NppsExtremaBenchmark<SrcType, DstType, decltype(&BufferFn), decltype(&OpFn)> {                   \
    public:                                                                                                         \
        BenchClass()                                                                                                \
            : NppsExtremaBenchmark<SrcType, DstType, decltype(&BufferFn), decltype(&OpFn)>(                       \
                  #OpFn, #BufferFn, #OpFn, DType) {}                                                                \
    };                                                                                                              \
    REGISTER_BENCHMARK(BenchClass)

#define REGISTER_NPPS_MEANSTDDEV_BENCH(BenchClass, BufferFn, OpFn, SrcType, DstType, DType)                       \
    class BenchClass final                                                                                          \
        : public NppsMeanStdDevBenchmark<SrcType, DstType, decltype(&BufferFn), decltype(&OpFn)> {                \
    public:                                                                                                         \
        BenchClass()                                                                                                \
            : NppsMeanStdDevBenchmark<SrcType, DstType, decltype(&BufferFn), decltype(&OpFn)>(                    \
                  #OpFn, #BufferFn, #OpFn, DType) {}                                                                \
    };                                                                                                              \
    REGISTER_BENCHMARK(BenchClass)

REGISTER_NPPS_STATS_BENCH(BenchSum_64f, nppsSumGetBufferSize_64f, nppsSum_64f, Npp64f, Npp64f, "64f");
REGISTER_NPPS_STATS_BENCH(BenchMean_32f, nppsMeanGetBufferSize_32f, nppsMean_32f, Npp32f, Npp32f, "32f");
REGISTER_NPPS_STATS_BENCH(BenchMean_64f, nppsMeanGetBufferSize_64f, nppsMean_64f, Npp64f, Npp64f, "64f");
REGISTER_NPPS_EXTREMA_BENCH(BenchMin_16s, nppsMinGetBufferSize_16s, nppsMin_16s, Npp16s, Npp16s, "16s");
REGISTER_NPPS_EXTREMA_BENCH(BenchMin_32s, nppsMinGetBufferSize_32s, nppsMin_32s, Npp32s, Npp32s, "32s");
REGISTER_NPPS_EXTREMA_BENCH(BenchMin_32f, nppsMinGetBufferSize_32f, nppsMin_32f, Npp32f, Npp32f, "32f");
REGISTER_NPPS_EXTREMA_BENCH(BenchMin_64f, nppsMinGetBufferSize_64f, nppsMin_64f, Npp64f, Npp64f, "64f");
REGISTER_NPPS_EXTREMA_BENCH(BenchMax_16s, nppsMaxGetBufferSize_16s, nppsMax_16s, Npp16s, Npp16s, "16s");
REGISTER_NPPS_EXTREMA_BENCH(BenchMax_32s, nppsMaxGetBufferSize_32s, nppsMax_32s, Npp32s, Npp32s, "32s");
REGISTER_NPPS_EXTREMA_BENCH(BenchMax_32f, nppsMaxGetBufferSize_32f, nppsMax_32f, Npp32f, Npp32f, "32f");
REGISTER_NPPS_EXTREMA_BENCH(BenchMax_64f, nppsMaxGetBufferSize_64f, nppsMax_64f, Npp64f, Npp64f, "64f");
REGISTER_NPPS_STATS_BENCH(BenchStdDev_32f, nppsStdDevGetBufferSize_32f, nppsStdDev_32f, Npp32f, Npp32f, "32f");
REGISTER_NPPS_STATS_BENCH(BenchStdDev_64f, nppsStdDevGetBufferSize_64f, nppsStdDev_64f, Npp64f, Npp64f, "64f");
REGISTER_NPPS_MEANSTDDEV_BENCH(BenchMeanStdDev_32f, nppsMeanStdDevGetBufferSize_32f, nppsMeanStdDev_32f, Npp32f,
                               Npp32f, "32f");
REGISTER_NPPS_MEANSTDDEV_BENCH(BenchMeanStdDev_64f, nppsMeanStdDevGetBufferSize_64f, nppsMeanStdDev_64f, Npp64f,
                               Npp64f, "64f");

#undef REGISTER_NPPS_STATS_BENCH
#undef REGISTER_NPPS_EXTREMA_BENCH
#undef REGISTER_NPPS_MEANSTDDEV_BENCH

} // namespace

} // namespace npp_benchmark
