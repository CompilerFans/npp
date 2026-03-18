#include "../benchmark_base.h"
#include <nppi_statistics_functions.h>

namespace npp_benchmark {

extern void initDeviceRandom8u(Npp8u* data, int size);
extern void initDeviceRandom16u(Npp16u* data, int size);
extern void initDeviceRandom16s(Npp16s* data, int size);
extern void initDeviceRandom32f(Npp32f* data, int size);

namespace {

template<typename T>
void initStatisticsInput(T* data, int count) {
    std::vector<T> host(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        host[static_cast<size_t>(i)] = static_cast<T>((i % 251) - 125);
    }
    (void)cudaMemcpy(data, host.data(), static_cast<size_t>(count) * sizeof(T), cudaMemcpyHostToDevice);
}

template<>
void initStatisticsInput<Npp8u>(Npp8u* data, int count) { initDeviceRandom8u(data, count); }
template<>
void initStatisticsInput<Npp16u>(Npp16u* data, int count) { initDeviceRandom16u(data, count); }
template<>
void initStatisticsInput<Npp16s>(Npp16s* data, int count) { initDeviceRandom16s(data, count); }
template<>
void initStatisticsInput<Npp32f>(Npp32f* data, int count) { initDeviceRandom32f(data, count); }

#ifdef USE_NVIDIA_NPP_TESTS
using HostSizeType = size_t;
#else
using HostSizeType = int;
#endif

template<typename SrcT, int SrcChannels, int OutputCount, typename FnGetBufferSize, typename FnOp>
class NppiStatisticsBenchmark : public BenchmarkBase {
public:
    NppiStatisticsBenchmark(const std::string& benchName, const char* bufferSymbol, const char* opSymbol,
                            const char* dataType)
        : BenchmarkBase(benchName), bufferSymbol_(bufferSymbol), opSymbol_(opSymbol), dataType_(dataType) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string bufferErr;
        std::string opErr;
        const FnGetBufferSize fnGetBufferSize = DynamicSymbol<FnGetBufferSize>(bufferSymbol_).get(&bufferErr);
        const FnOp fnOp = DynamicSymbol<FnOp>(opSymbol_).get(&opErr);

        std::vector<BenchmarkResult> results;
        for (const auto& imgSize : getStandardImageSizes()) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = dataType_;
            result.size = imgSize.toString();
            result.channels = SrcChannels;
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

            const int pixelCount = imgSize.width * imgSize.height;
            const int elemCount = pixelCount * SrcChannels;
            const int srcStep = imgSize.width * SrcChannels * static_cast<int>(sizeof(SrcT));
            const NppiSize roi{imgSize.width, imgSize.height};

            DeviceBuffer<SrcT> src(elemCount);
            DeviceBuffer<Npp64f> dst(OutputCount);
            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }

            HostSizeType bufferSize = 0;
            const NppStatus bufferStatus = fnGetBufferSize(roi, &bufferSize);
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

            initStatisticsInput<SrcT>(src.get(), elemCount);
            dst.fillZero();
            scratch.fillZero();

            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st = fnOp(src.get(), srcStep, roi, scratch.get(), dst.get());
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
                    [&]() { return fnOp(src.get(), srcStep, roi, scratch.get(), dst.get()); })) {
                result.throughputGBps =
                    computeThroughput(static_cast<size_t>(elemCount) * sizeof(SrcT), result.avgTimeMs);
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

template<typename SrcT, typename DstT, int SrcChannels, int OutputCount, typename FnGetBufferSize, typename FnOp>
class NppiExtremaBenchmark : public BenchmarkBase {
public:
    NppiExtremaBenchmark(const std::string& benchName, const char* bufferSymbol, const char* opSymbol,
                         const char* dataType)
        : BenchmarkBase(benchName), bufferSymbol_(bufferSymbol), opSymbol_(opSymbol), dataType_(dataType) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string bufferErr;
        std::string opErr;
        const FnGetBufferSize fnGetBufferSize = DynamicSymbol<FnGetBufferSize>(bufferSymbol_).get(&bufferErr);
        const FnOp fnOp = DynamicSymbol<FnOp>(opSymbol_).get(&opErr);

        std::vector<BenchmarkResult> results;
        for (const auto& imgSize : getStandardImageSizes()) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = dataType_;
            result.size = imgSize.toString();
            result.channels = SrcChannels;
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

            const int pixelCount = imgSize.width * imgSize.height;
            const int elemCount = pixelCount * SrcChannels;
            const int srcStep = imgSize.width * SrcChannels * static_cast<int>(sizeof(SrcT));
            const NppiSize roi{imgSize.width, imgSize.height};

            DeviceBuffer<SrcT> src(elemCount);
            DeviceBuffer<DstT> dst(OutputCount);
            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }

            HostSizeType bufferSize = 0;
            const NppStatus bufferStatus = fnGetBufferSize(roi, &bufferSize);
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

            initStatisticsInput<SrcT>(src.get(), elemCount);
            dst.fillZero();
            scratch.fillZero();

            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st = fnOp(src.get(), srcStep, roi, scratch.get(), dst.get());
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
                    [&]() { return fnOp(src.get(), srcStep, roi, scratch.get(), dst.get()); })) {
                result.throughputGBps =
                    computeThroughput(static_cast<size_t>(elemCount) * sizeof(SrcT), result.avgTimeMs);
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

template<typename SrcT, int SrcChannels, int OutputCount, typename FnGetBufferSize, typename FnOp>
class NppiNormBenchmark : public BenchmarkBase {
public:
    NppiNormBenchmark(const std::string& benchName, const char* bufferSymbol, const char* opSymbol,
                      const char* dataType)
        : BenchmarkBase(benchName), bufferSymbol_(bufferSymbol), opSymbol_(opSymbol), dataType_(dataType) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string bufferErr;
        std::string opErr;
        const FnGetBufferSize fnGetBufferSize = DynamicSymbol<FnGetBufferSize>(bufferSymbol_).get(&bufferErr);
        const FnOp fnOp = DynamicSymbol<FnOp>(opSymbol_).get(&opErr);

        std::vector<BenchmarkResult> results;
        for (const auto& imgSize : getStandardImageSizes()) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = dataType_;
            result.size = imgSize.toString();
            result.channels = SrcChannels;
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

            const int pixelCount = imgSize.width * imgSize.height;
            const int elemCount = pixelCount * SrcChannels;
            const int srcStep = imgSize.width * SrcChannels * static_cast<int>(sizeof(SrcT));
            const NppiSize roi{imgSize.width, imgSize.height};

            DeviceBuffer<SrcT> src(elemCount);
            DeviceBuffer<Npp64f> dst(OutputCount);
            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }

            HostSizeType bufferSize = 0;
            const NppStatus bufferStatus = fnGetBufferSize(roi, &bufferSize);
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

            initStatisticsInput<SrcT>(src.get(), elemCount);
            dst.fillZero();
            scratch.fillZero();

            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st = fnOp(src.get(), srcStep, roi, dst.get(), scratch.get());
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
                    [&]() { return fnOp(src.get(), srcStep, roi, dst.get(), scratch.get()); })) {
                result.throughputGBps =
                    computeThroughput(static_cast<size_t>(elemCount) * sizeof(SrcT), result.avgTimeMs);
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

template<typename SrcT, typename DstT, int SrcChannels, int OutputCount, typename FnGetBufferSize, typename FnOp>
class NppiMinMaxBenchmark : public BenchmarkBase {
public:
    NppiMinMaxBenchmark(const std::string& benchName, const char* bufferSymbol, const char* opSymbol,
                        const char* dataType)
        : BenchmarkBase(benchName), bufferSymbol_(bufferSymbol), opSymbol_(opSymbol), dataType_(dataType) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string bufferErr;
        std::string opErr;
        const FnGetBufferSize fnGetBufferSize = DynamicSymbol<FnGetBufferSize>(bufferSymbol_).get(&bufferErr);
        const FnOp fnOp = DynamicSymbol<FnOp>(opSymbol_).get(&opErr);

        std::vector<BenchmarkResult> results;
        for (const auto& imgSize : getStandardImageSizes()) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = dataType_;
            result.size = imgSize.toString();
            result.channels = SrcChannels;
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

            const int pixelCount = imgSize.width * imgSize.height;
            const int elemCount = pixelCount * SrcChannels;
            const int srcStep = imgSize.width * SrcChannels * static_cast<int>(sizeof(SrcT));
            const NppiSize roi{imgSize.width, imgSize.height};

            DeviceBuffer<SrcT> src(elemCount);
            DeviceBuffer<DstT> minDst(OutputCount);
            DeviceBuffer<DstT> maxDst(OutputCount);
            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(minDst, result, "min") ||
                !validateDeviceBuffer(maxDst, result, "max")) {
                results.push_back(result);
                continue;
            }

            HostSizeType bufferSize = 0;
            const NppStatus bufferStatus = fnGetBufferSize(roi, &bufferSize);
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

            initStatisticsInput<SrcT>(src.get(), elemCount);
            minDst.fillZero();
            maxDst.fillZero();
            scratch.fillZero();

            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st = fnOp(src.get(), srcStep, roi, minDst.get(), maxDst.get(), scratch.get());
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
                                          return fnOp(src.get(), srcStep, roi, minDst.get(), maxDst.get(),
                                                      scratch.get());
                                      })) {
                result.throughputGBps =
                    computeThroughput(static_cast<size_t>(elemCount) * sizeof(SrcT), result.avgTimeMs);
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

template<typename SrcT, typename DstT, int SrcChannels, int OutputCount, typename FnGetBufferSize, typename FnOp>
class NppiIndexExtremaBenchmark : public BenchmarkBase {
public:
    NppiIndexExtremaBenchmark(const std::string& benchName, const char* bufferSymbol, const char* opSymbol,
                              const char* dataType)
        : BenchmarkBase(benchName), bufferSymbol_(bufferSymbol), opSymbol_(opSymbol), dataType_(dataType) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string bufferErr;
        std::string opErr;
        const FnGetBufferSize fnGetBufferSize = DynamicSymbol<FnGetBufferSize>(bufferSymbol_).get(&bufferErr);
        const FnOp fnOp = DynamicSymbol<FnOp>(opSymbol_).get(&opErr);

        std::vector<BenchmarkResult> results;
        for (const auto& imgSize : getStandardImageSizes()) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = dataType_;
            result.size = imgSize.toString();
            result.channels = SrcChannels;
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

            const int pixelCount = imgSize.width * imgSize.height;
            const int elemCount = pixelCount * SrcChannels;
            const int srcStep = imgSize.width * SrcChannels * static_cast<int>(sizeof(SrcT));
            const NppiSize roi{imgSize.width, imgSize.height};

            DeviceBuffer<SrcT> src(elemCount);
            DeviceBuffer<DstT> extrema(OutputCount);
            DeviceBuffer<int> indexX(OutputCount);
            DeviceBuffer<int> indexY(OutputCount);
            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(extrema, result, "extrema") ||
                !validateDeviceBuffer(indexX, result, "indexX") || !validateDeviceBuffer(indexY, result, "indexY")) {
                results.push_back(result);
                continue;
            }

            HostSizeType bufferSize = 0;
            const NppStatus bufferStatus = fnGetBufferSize(roi, &bufferSize);
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

            initStatisticsInput<SrcT>(src.get(), elemCount);
            extrema.fillZero();
            indexX.fillZero();
            indexY.fillZero();
            scratch.fillZero();

            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st =
                    fnOp(src.get(), srcStep, roi, scratch.get(), extrema.get(), indexX.get(), indexY.get());
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
                                          return fnOp(src.get(), srcStep, roi, scratch.get(), extrema.get(),
                                                      indexX.get(), indexY.get());
                                      })) {
                result.throughputGBps =
                    computeThroughput(static_cast<size_t>(elemCount) * sizeof(SrcT), result.avgTimeMs);
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

template<typename SrcT, typename FnGetBufferSize, typename FnOp>
class NppiMeanStdDevBenchmark : public BenchmarkBase {
public:
    NppiMeanStdDevBenchmark(const std::string& benchName, const char* bufferSymbol, const char* opSymbol,
                            const char* dataType)
        : BenchmarkBase(benchName), bufferSymbol_(bufferSymbol), opSymbol_(opSymbol), dataType_(dataType) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string bufferErr;
        std::string opErr;
        const FnGetBufferSize fnGetBufferSize = DynamicSymbol<FnGetBufferSize>(bufferSymbol_).get(&bufferErr);
        const FnOp fnOp = DynamicSymbol<FnOp>(opSymbol_).get(&opErr);

        std::vector<BenchmarkResult> results;
        for (const auto& imgSize : getStandardImageSizes()) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = dataType_;
            result.size = imgSize.toString();
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

            const int pixelCount = imgSize.width * imgSize.height;
            const int srcStep = imgSize.width * static_cast<int>(sizeof(SrcT));
            const NppiSize roi{imgSize.width, imgSize.height};

            DeviceBuffer<SrcT> src(pixelCount);
            DeviceBuffer<Npp64f> mean(1);
            DeviceBuffer<Npp64f> stddev(1);
            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(mean, result, "mean") ||
                !validateDeviceBuffer(stddev, result, "stddev")) {
                results.push_back(result);
                continue;
            }

            HostSizeType bufferSize = 0;
            const NppStatus bufferStatus = fnGetBufferSize(roi, &bufferSize);
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

            initStatisticsInput<SrcT>(src.get(), pixelCount);
            mean.fillZero();
            stddev.fillZero();
            scratch.fillZero();

            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st = fnOp(src.get(), srcStep, roi, scratch.get(), mean.get(), stddev.get());
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
                                          return fnOp(src.get(), srcStep, roi, scratch.get(), mean.get(),
                                                      stddev.get());
                                      })) {
                result.throughputGBps = computeThroughput(static_cast<size_t>(pixelCount) * sizeof(SrcT),
                                                          result.avgTimeMs);
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

#define REGISTER_NPPI_STATS_BENCH(BenchClass, BufferFn, OpFn, SrcType, SrcCh, OutCount, DType)                    \
    class BenchClass final                                                                                          \
        : public NppiStatisticsBenchmark<SrcType, SrcCh, OutCount, decltype(&BufferFn), decltype(&OpFn)> {        \
    public:                                                                                                         \
        BenchClass()                                                                                                \
            : NppiStatisticsBenchmark<SrcType, SrcCh, OutCount, decltype(&BufferFn), decltype(&OpFn)>(            \
                  #OpFn, #BufferFn, #OpFn, DType) {}                                                                \
    };                                                                                                              \
    REGISTER_BENCHMARK(BenchClass)

#define REGISTER_NPPI_EXTREMA_BENCH(BenchClass, BufferFn, OpFn, SrcType, DstType, SrcCh, OutCount, DType)         \
    class BenchClass final                                                                                          \
        : public NppiExtremaBenchmark<SrcType, DstType, SrcCh, OutCount, decltype(&BufferFn), decltype(&OpFn)> {  \
    public:                                                                                                         \
        BenchClass()                                                                                                \
            : NppiExtremaBenchmark<SrcType, DstType, SrcCh, OutCount, decltype(&BufferFn), decltype(&OpFn)>(      \
                  #OpFn, #BufferFn, #OpFn, DType) {}                                                                \
    };                                                                                                              \
    REGISTER_BENCHMARK(BenchClass)

#define REGISTER_NPPI_NORM_BENCH(BenchClass, BufferFn, OpFn, SrcType, SrcCh, OutCount, DType)                     \
    class BenchClass final                                                                                          \
        : public NppiNormBenchmark<SrcType, SrcCh, OutCount, decltype(&BufferFn), decltype(&OpFn)> {              \
    public:                                                                                                         \
        BenchClass()                                                                                                \
            : NppiNormBenchmark<SrcType, SrcCh, OutCount, decltype(&BufferFn), decltype(&OpFn)>(                  \
                  #OpFn, #BufferFn, #OpFn, DType) {}                                                                \
    };                                                                                                              \
    REGISTER_BENCHMARK(BenchClass)

#define REGISTER_NPPI_MINMAX_BENCH(BenchClass, BufferFn, OpFn, SrcType, DstType, SrcCh, OutCount, DType)         \
    class BenchClass final                                                                                          \
        : public NppiMinMaxBenchmark<SrcType, DstType, SrcCh, OutCount, decltype(&BufferFn), decltype(&OpFn)> {   \
    public:                                                                                                         \
        BenchClass()                                                                                                \
            : NppiMinMaxBenchmark<SrcType, DstType, SrcCh, OutCount, decltype(&BufferFn), decltype(&OpFn)>(       \
                  #OpFn, #BufferFn, #OpFn, DType) {}                                                                \
    };                                                                                                              \
    REGISTER_BENCHMARK(BenchClass)

#define REGISTER_NPPI_INDEX_BENCH(BenchClass, BufferFn, OpFn, SrcType, DstType, SrcCh, OutCount, DType)          \
    class BenchClass final                                                                                          \
        : public NppiIndexExtremaBenchmark<SrcType, DstType, SrcCh, OutCount, decltype(&BufferFn),               \
                                           decltype(&OpFn)> {                                                       \
    public:                                                                                                         \
        BenchClass()                                                                                                \
            : NppiIndexExtremaBenchmark<SrcType, DstType, SrcCh, OutCount, decltype(&BufferFn),                   \
                                        decltype(&OpFn)>(#OpFn, #BufferFn, #OpFn, DType) {}                        \
    };                                                                                                              \
    REGISTER_BENCHMARK(BenchClass)

#define REGISTER_NPPI_MEANSTDDEV_BENCH(BenchClass, BufferFn, OpFn, SrcType, DType)                                 \
    class BenchClass final                                                                                            \
        : public NppiMeanStdDevBenchmark<SrcType, decltype(&BufferFn), decltype(&OpFn)> {                           \
    public:                                                                                                           \
        BenchClass()                                                                                                  \
            : NppiMeanStdDevBenchmark<SrcType, decltype(&BufferFn), decltype(&OpFn)>(#OpFn, #BufferFn, #OpFn,      \
                                                                                     DType) {}                        \
    };                                                                                                                \
    REGISTER_BENCHMARK(BenchClass)

REGISTER_NPPI_STATS_BENCH(BenchSum_8u_C1R, nppiSumGetBufferHostSize_8u_C1R, nppiSum_8u_C1R, Npp8u, 1, 1, "8u");
REGISTER_NPPI_STATS_BENCH(BenchSum_16u_C1R, nppiSumGetBufferHostSize_16u_C1R, nppiSum_16u_C1R, Npp16u, 1, 1,
                          "16u");
REGISTER_NPPI_STATS_BENCH(BenchSum_16s_C1R, nppiSumGetBufferHostSize_16s_C1R, nppiSum_16s_C1R, Npp16s, 1, 1,
                          "16s");
REGISTER_NPPI_STATS_BENCH(BenchSum_32f_C1R, nppiSumGetBufferHostSize_32f_C1R, nppiSum_32f_C1R, Npp32f, 1, 1,
                          "32f");
REGISTER_NPPI_STATS_BENCH(BenchSum_8u_C3R, nppiSumGetBufferHostSize_8u_C3R, nppiSum_8u_C3R, Npp8u, 3, 3, "8u");
REGISTER_NPPI_STATS_BENCH(BenchSum_16u_C3R, nppiSumGetBufferHostSize_16u_C3R, nppiSum_16u_C3R, Npp16u, 3, 3,
                          "16u");
REGISTER_NPPI_STATS_BENCH(BenchSum_16s_C3R, nppiSumGetBufferHostSize_16s_C3R, nppiSum_16s_C3R, Npp16s, 3, 3,
                          "16s");
REGISTER_NPPI_STATS_BENCH(BenchSum_32f_C3R, nppiSumGetBufferHostSize_32f_C3R, nppiSum_32f_C3R, Npp32f, 3, 3,
                          "32f");
REGISTER_NPPI_STATS_BENCH(BenchSum_8u_C4R, nppiSumGetBufferHostSize_8u_C4R, nppiSum_8u_C4R, Npp8u, 4, 4, "8u");
REGISTER_NPPI_STATS_BENCH(BenchSum_16u_C4R, nppiSumGetBufferHostSize_16u_C4R, nppiSum_16u_C4R, Npp16u, 4, 4,
                          "16u");
REGISTER_NPPI_STATS_BENCH(BenchSum_16s_C4R, nppiSumGetBufferHostSize_16s_C4R, nppiSum_16s_C4R, Npp16s, 4, 4,
                          "16s");
REGISTER_NPPI_STATS_BENCH(BenchSum_32f_C4R, nppiSumGetBufferHostSize_32f_C4R, nppiSum_32f_C4R, Npp32f, 4, 4,
                          "32f");
REGISTER_NPPI_STATS_BENCH(BenchSum_8u_AC4R, nppiSumGetBufferHostSize_8u_AC4R, nppiSum_8u_AC4R, Npp8u, 4, 3,
                          "8u");
REGISTER_NPPI_STATS_BENCH(BenchSum_16u_AC4R, nppiSumGetBufferHostSize_16u_AC4R, nppiSum_16u_AC4R, Npp16u, 4, 3,
                          "16u");
REGISTER_NPPI_STATS_BENCH(BenchSum_16s_AC4R, nppiSumGetBufferHostSize_16s_AC4R, nppiSum_16s_AC4R, Npp16s, 4, 3,
                          "16s");
REGISTER_NPPI_STATS_BENCH(BenchSum_32f_AC4R, nppiSumGetBufferHostSize_32f_AC4R, nppiSum_32f_AC4R, Npp32f, 4, 3,
                          "32f");

REGISTER_NPPI_STATS_BENCH(BenchMean_8u_C1R, nppiMeanGetBufferHostSize_8u_C1R, nppiMean_8u_C1R, Npp8u, 1, 1, "8u");
REGISTER_NPPI_STATS_BENCH(BenchMean_16u_C1R, nppiMeanGetBufferHostSize_16u_C1R, nppiMean_16u_C1R, Npp16u, 1, 1,
                          "16u");
REGISTER_NPPI_STATS_BENCH(BenchMean_16s_C1R, nppiMeanGetBufferHostSize_16s_C1R, nppiMean_16s_C1R, Npp16s, 1, 1,
                          "16s");
REGISTER_NPPI_STATS_BENCH(BenchMean_32f_C1R, nppiMeanGetBufferHostSize_32f_C1R, nppiMean_32f_C1R, Npp32f, 1, 1,
                          "32f");
REGISTER_NPPI_STATS_BENCH(BenchMean_8u_C3R, nppiMeanGetBufferHostSize_8u_C3R, nppiMean_8u_C3R, Npp8u, 3, 3, "8u");
REGISTER_NPPI_STATS_BENCH(BenchMean_16u_C3R, nppiMeanGetBufferHostSize_16u_C3R, nppiMean_16u_C3R, Npp16u, 3, 3,
                          "16u");
REGISTER_NPPI_STATS_BENCH(BenchMean_16s_C3R, nppiMeanGetBufferHostSize_16s_C3R, nppiMean_16s_C3R, Npp16s, 3, 3,
                          "16s");
REGISTER_NPPI_STATS_BENCH(BenchMean_32f_C3R, nppiMeanGetBufferHostSize_32f_C3R, nppiMean_32f_C3R, Npp32f, 3, 3,
                          "32f");
REGISTER_NPPI_STATS_BENCH(BenchMean_8u_C4R, nppiMeanGetBufferHostSize_8u_C4R, nppiMean_8u_C4R, Npp8u, 4, 4, "8u");
REGISTER_NPPI_STATS_BENCH(BenchMean_16u_C4R, nppiMeanGetBufferHostSize_16u_C4R, nppiMean_16u_C4R, Npp16u, 4, 4,
                          "16u");
REGISTER_NPPI_STATS_BENCH(BenchMean_16s_C4R, nppiMeanGetBufferHostSize_16s_C4R, nppiMean_16s_C4R, Npp16s, 4, 4,
                          "16s");
REGISTER_NPPI_STATS_BENCH(BenchMean_32f_C4R, nppiMeanGetBufferHostSize_32f_C4R, nppiMean_32f_C4R, Npp32f, 4, 4,
                          "32f");
REGISTER_NPPI_STATS_BENCH(BenchMean_8u_AC4R, nppiMeanGetBufferHostSize_8u_AC4R, nppiMean_8u_AC4R, Npp8u, 4, 3,
                          "8u");
REGISTER_NPPI_STATS_BENCH(BenchMean_16u_AC4R, nppiMeanGetBufferHostSize_16u_AC4R, nppiMean_16u_AC4R, Npp16u, 4, 3,
                          "16u");
REGISTER_NPPI_STATS_BENCH(BenchMean_16s_AC4R, nppiMeanGetBufferHostSize_16s_AC4R, nppiMean_16s_AC4R, Npp16s, 4, 3,
                          "16s");
REGISTER_NPPI_STATS_BENCH(BenchMean_32f_AC4R, nppiMeanGetBufferHostSize_32f_AC4R, nppiMean_32f_AC4R, Npp32f, 4, 3,
                          "32f");

REGISTER_NPPI_EXTREMA_BENCH(BenchMin_8u_C1R, nppiMinGetBufferHostSize_8u_C1R, nppiMin_8u_C1R, Npp8u, Npp8u, 1, 1,
                            "8u");
REGISTER_NPPI_EXTREMA_BENCH(BenchMin_16u_C1R, nppiMinGetBufferHostSize_16u_C1R, nppiMin_16u_C1R, Npp16u, Npp16u, 1,
                            1, "16u");
REGISTER_NPPI_EXTREMA_BENCH(BenchMin_16s_C1R, nppiMinGetBufferHostSize_16s_C1R, nppiMin_16s_C1R, Npp16s, Npp16s, 1,
                            1, "16s");
REGISTER_NPPI_EXTREMA_BENCH(BenchMin_32f_C1R, nppiMinGetBufferHostSize_32f_C1R, nppiMin_32f_C1R, Npp32f, Npp32f, 1,
                            1, "32f");
REGISTER_NPPI_EXTREMA_BENCH(BenchMin_8u_C3R, nppiMinGetBufferHostSize_8u_C3R, nppiMin_8u_C3R, Npp8u, Npp8u, 3, 3,
                            "8u");
REGISTER_NPPI_EXTREMA_BENCH(BenchMin_16u_C3R, nppiMinGetBufferHostSize_16u_C3R, nppiMin_16u_C3R, Npp16u, Npp16u, 3,
                            3, "16u");
REGISTER_NPPI_EXTREMA_BENCH(BenchMin_16s_C3R, nppiMinGetBufferHostSize_16s_C3R, nppiMin_16s_C3R, Npp16s, Npp16s, 3,
                            3, "16s");
REGISTER_NPPI_EXTREMA_BENCH(BenchMin_32f_C3R, nppiMinGetBufferHostSize_32f_C3R, nppiMin_32f_C3R, Npp32f, Npp32f, 3,
                            3, "32f");
REGISTER_NPPI_EXTREMA_BENCH(BenchMin_8u_C4R, nppiMinGetBufferHostSize_8u_C4R, nppiMin_8u_C4R, Npp8u, Npp8u, 4, 4,
                            "8u");
REGISTER_NPPI_EXTREMA_BENCH(BenchMin_16u_C4R, nppiMinGetBufferHostSize_16u_C4R, nppiMin_16u_C4R, Npp16u, Npp16u, 4,
                            4, "16u");
REGISTER_NPPI_EXTREMA_BENCH(BenchMin_16s_C4R, nppiMinGetBufferHostSize_16s_C4R, nppiMin_16s_C4R, Npp16s, Npp16s, 4,
                            4, "16s");
REGISTER_NPPI_EXTREMA_BENCH(BenchMin_32f_C4R, nppiMinGetBufferHostSize_32f_C4R, nppiMin_32f_C4R, Npp32f, Npp32f, 4,
                            4, "32f");
REGISTER_NPPI_EXTREMA_BENCH(BenchMin_8u_AC4R, nppiMinGetBufferHostSize_8u_AC4R, nppiMin_8u_AC4R, Npp8u, Npp8u, 4, 3,
                            "8u");
REGISTER_NPPI_EXTREMA_BENCH(BenchMin_16u_AC4R, nppiMinGetBufferHostSize_16u_AC4R, nppiMin_16u_AC4R, Npp16u, Npp16u,
                            4, 3, "16u");
REGISTER_NPPI_EXTREMA_BENCH(BenchMin_16s_AC4R, nppiMinGetBufferHostSize_16s_AC4R, nppiMin_16s_AC4R, Npp16s, Npp16s,
                            4, 3, "16s");
REGISTER_NPPI_EXTREMA_BENCH(BenchMin_32f_AC4R, nppiMinGetBufferHostSize_32f_AC4R, nppiMin_32f_AC4R, Npp32f, Npp32f,
                            4, 3, "32f");

REGISTER_NPPI_EXTREMA_BENCH(BenchMax_8u_C1R, nppiMaxGetBufferHostSize_8u_C1R, nppiMax_8u_C1R, Npp8u, Npp8u, 1, 1,
                            "8u");
REGISTER_NPPI_EXTREMA_BENCH(BenchMax_16u_C1R, nppiMaxGetBufferHostSize_16u_C1R, nppiMax_16u_C1R, Npp16u, Npp16u, 1,
                            1, "16u");
REGISTER_NPPI_EXTREMA_BENCH(BenchMax_16s_C1R, nppiMaxGetBufferHostSize_16s_C1R, nppiMax_16s_C1R, Npp16s, Npp16s, 1,
                            1, "16s");
REGISTER_NPPI_EXTREMA_BENCH(BenchMax_32f_C1R, nppiMaxGetBufferHostSize_32f_C1R, nppiMax_32f_C1R, Npp32f, Npp32f, 1,
                            1, "32f");
REGISTER_NPPI_EXTREMA_BENCH(BenchMax_8u_C3R, nppiMaxGetBufferHostSize_8u_C3R, nppiMax_8u_C3R, Npp8u, Npp8u, 3, 3,
                            "8u");
REGISTER_NPPI_EXTREMA_BENCH(BenchMax_16u_C3R, nppiMaxGetBufferHostSize_16u_C3R, nppiMax_16u_C3R, Npp16u, Npp16u, 3,
                            3, "16u");
REGISTER_NPPI_EXTREMA_BENCH(BenchMax_16s_C3R, nppiMaxGetBufferHostSize_16s_C3R, nppiMax_16s_C3R, Npp16s, Npp16s, 3,
                            3, "16s");
REGISTER_NPPI_EXTREMA_BENCH(BenchMax_32f_C3R, nppiMaxGetBufferHostSize_32f_C3R, nppiMax_32f_C3R, Npp32f, Npp32f, 3,
                            3, "32f");
REGISTER_NPPI_EXTREMA_BENCH(BenchMax_8u_C4R, nppiMaxGetBufferHostSize_8u_C4R, nppiMax_8u_C4R, Npp8u, Npp8u, 4, 4,
                            "8u");
REGISTER_NPPI_EXTREMA_BENCH(BenchMax_16u_C4R, nppiMaxGetBufferHostSize_16u_C4R, nppiMax_16u_C4R, Npp16u, Npp16u, 4,
                            4, "16u");
REGISTER_NPPI_EXTREMA_BENCH(BenchMax_16s_C4R, nppiMaxGetBufferHostSize_16s_C4R, nppiMax_16s_C4R, Npp16s, Npp16s, 4,
                            4, "16s");
REGISTER_NPPI_EXTREMA_BENCH(BenchMax_32f_C4R, nppiMaxGetBufferHostSize_32f_C4R, nppiMax_32f_C4R, Npp32f, Npp32f, 4,
                            4, "32f");
REGISTER_NPPI_EXTREMA_BENCH(BenchMax_8u_AC4R, nppiMaxGetBufferHostSize_8u_AC4R, nppiMax_8u_AC4R, Npp8u, Npp8u, 4, 3,
                            "8u");
REGISTER_NPPI_EXTREMA_BENCH(BenchMax_16u_AC4R, nppiMaxGetBufferHostSize_16u_AC4R, nppiMax_16u_AC4R, Npp16u, Npp16u,
                            4, 3, "16u");
REGISTER_NPPI_EXTREMA_BENCH(BenchMax_16s_AC4R, nppiMaxGetBufferHostSize_16s_AC4R, nppiMax_16s_AC4R, Npp16s, Npp16s,
                            4, 3, "16s");
REGISTER_NPPI_EXTREMA_BENCH(BenchMax_32f_AC4R, nppiMaxGetBufferHostSize_32f_AC4R, nppiMax_32f_AC4R, Npp32f, Npp32f,
                            4, 3, "32f");

REGISTER_NPPI_NORM_BENCH(BenchNormInf_8u_C1R, nppiNormInfGetBufferHostSize_8u_C1R, nppiNorm_Inf_8u_C1R, Npp8u, 1, 1,
                         "8u");
REGISTER_NPPI_NORM_BENCH(BenchNormInf_16u_C1R, nppiNormInfGetBufferHostSize_16u_C1R, nppiNorm_Inf_16u_C1R, Npp16u, 1,
                         1, "16u");
REGISTER_NPPI_NORM_BENCH(BenchNormInf_16s_C1R, nppiNormInfGetBufferHostSize_16s_C1R, nppiNorm_Inf_16s_C1R, Npp16s, 1,
                         1, "16s");
REGISTER_NPPI_NORM_BENCH(BenchNormInf_32f_C1R, nppiNormInfGetBufferHostSize_32f_C1R, nppiNorm_Inf_32f_C1R, Npp32f, 1,
                         1, "32f");
REGISTER_NPPI_NORM_BENCH(BenchNormInf_8u_C3R, nppiNormInfGetBufferHostSize_8u_C3R, nppiNorm_Inf_8u_C3R, Npp8u, 3, 3,
                         "8u");
REGISTER_NPPI_NORM_BENCH(BenchNormInf_16u_C3R, nppiNormInfGetBufferHostSize_16u_C3R, nppiNorm_Inf_16u_C3R, Npp16u, 3,
                         3, "16u");
REGISTER_NPPI_NORM_BENCH(BenchNormInf_16s_C3R, nppiNormInfGetBufferHostSize_16s_C3R, nppiNorm_Inf_16s_C3R, Npp16s, 3,
                         3, "16s");
REGISTER_NPPI_NORM_BENCH(BenchNormInf_32f_C3R, nppiNormInfGetBufferHostSize_32f_C3R, nppiNorm_Inf_32f_C3R, Npp32f, 3,
                         3, "32f");
REGISTER_NPPI_NORM_BENCH(BenchNormInf_8u_C4R, nppiNormInfGetBufferHostSize_8u_C4R, nppiNorm_Inf_8u_C4R, Npp8u, 4, 4,
                         "8u");
REGISTER_NPPI_NORM_BENCH(BenchNormInf_16u_C4R, nppiNormInfGetBufferHostSize_16u_C4R, nppiNorm_Inf_16u_C4R, Npp16u, 4,
                         4, "16u");
REGISTER_NPPI_NORM_BENCH(BenchNormInf_16s_C4R, nppiNormInfGetBufferHostSize_16s_C4R, nppiNorm_Inf_16s_C4R, Npp16s, 4,
                         4, "16s");
REGISTER_NPPI_NORM_BENCH(BenchNormInf_32f_C4R, nppiNormInfGetBufferHostSize_32f_C4R, nppiNorm_Inf_32f_C4R, Npp32f, 4,
                         4, "32f");
REGISTER_NPPI_NORM_BENCH(BenchNormInf_8u_AC4R, nppiNormInfGetBufferHostSize_8u_AC4R, nppiNorm_Inf_8u_AC4R, Npp8u, 4,
                         3, "8u");
REGISTER_NPPI_NORM_BENCH(BenchNormInf_16u_AC4R, nppiNormInfGetBufferHostSize_16u_AC4R, nppiNorm_Inf_16u_AC4R, Npp16u,
                         4, 3, "16u");
REGISTER_NPPI_NORM_BENCH(BenchNormInf_16s_AC4R, nppiNormInfGetBufferHostSize_16s_AC4R, nppiNorm_Inf_16s_AC4R, Npp16s,
                         4, 3, "16s");
REGISTER_NPPI_NORM_BENCH(BenchNormInf_32f_AC4R, nppiNormInfGetBufferHostSize_32f_AC4R, nppiNorm_Inf_32f_AC4R, Npp32f,
                         4, 3, "32f");

REGISTER_NPPI_NORM_BENCH(BenchNormL1_8u_C1R, nppiNormL1GetBufferHostSize_8u_C1R, nppiNorm_L1_8u_C1R, Npp8u, 1, 1,
                         "8u");
REGISTER_NPPI_NORM_BENCH(BenchNormL1_16u_C1R, nppiNormL1GetBufferHostSize_16u_C1R, nppiNorm_L1_16u_C1R, Npp16u, 1, 1,
                         "16u");
REGISTER_NPPI_NORM_BENCH(BenchNormL1_16s_C1R, nppiNormL1GetBufferHostSize_16s_C1R, nppiNorm_L1_16s_C1R, Npp16s, 1, 1,
                         "16s");
REGISTER_NPPI_NORM_BENCH(BenchNormL1_32f_C1R, nppiNormL1GetBufferHostSize_32f_C1R, nppiNorm_L1_32f_C1R, Npp32f, 1, 1,
                         "32f");
REGISTER_NPPI_NORM_BENCH(BenchNormL1_8u_C3R, nppiNormL1GetBufferHostSize_8u_C3R, nppiNorm_L1_8u_C3R, Npp8u, 3, 3,
                         "8u");
REGISTER_NPPI_NORM_BENCH(BenchNormL1_16u_C3R, nppiNormL1GetBufferHostSize_16u_C3R, nppiNorm_L1_16u_C3R, Npp16u, 3, 3,
                         "16u");
REGISTER_NPPI_NORM_BENCH(BenchNormL1_16s_C3R, nppiNormL1GetBufferHostSize_16s_C3R, nppiNorm_L1_16s_C3R, Npp16s, 3, 3,
                         "16s");
REGISTER_NPPI_NORM_BENCH(BenchNormL1_32f_C3R, nppiNormL1GetBufferHostSize_32f_C3R, nppiNorm_L1_32f_C3R, Npp32f, 3, 3,
                         "32f");

REGISTER_NPPI_NORM_BENCH(BenchNormL2_8u_C1R, nppiNormL2GetBufferHostSize_8u_C1R, nppiNorm_L2_8u_C1R, Npp8u, 1, 1,
                         "8u");
REGISTER_NPPI_NORM_BENCH(BenchNormL2_16u_C1R, nppiNormL2GetBufferHostSize_16u_C1R, nppiNorm_L2_16u_C1R, Npp16u, 1, 1,
                         "16u");
REGISTER_NPPI_NORM_BENCH(BenchNormL2_16s_C1R, nppiNormL2GetBufferHostSize_16s_C1R, nppiNorm_L2_16s_C1R, Npp16s, 1, 1,
                         "16s");
REGISTER_NPPI_NORM_BENCH(BenchNormL2_32f_C1R, nppiNormL2GetBufferHostSize_32f_C1R, nppiNorm_L2_32f_C1R, Npp32f, 1, 1,
                         "32f");
REGISTER_NPPI_NORM_BENCH(BenchNormL2_8u_C3R, nppiNormL2GetBufferHostSize_8u_C3R, nppiNorm_L2_8u_C3R, Npp8u, 3, 3,
                         "8u");
REGISTER_NPPI_NORM_BENCH(BenchNormL2_16u_C3R, nppiNormL2GetBufferHostSize_16u_C3R, nppiNorm_L2_16u_C3R, Npp16u, 3, 3,
                         "16u");
REGISTER_NPPI_NORM_BENCH(BenchNormL2_16s_C3R, nppiNormL2GetBufferHostSize_16s_C3R, nppiNorm_L2_16s_C3R, Npp16s, 3, 3,
                         "16s");
REGISTER_NPPI_NORM_BENCH(BenchNormL2_32f_C3R, nppiNormL2GetBufferHostSize_32f_C3R, nppiNorm_L2_32f_C3R, Npp32f, 3, 3,
                         "32f");

REGISTER_NPPI_MINMAX_BENCH(BenchMinMax_8u_C1R, nppiMinMaxGetBufferHostSize_8u_C1R, nppiMinMax_8u_C1R, Npp8u, Npp8u,
                           1, 1, "8u");
REGISTER_NPPI_MINMAX_BENCH(BenchMinMax_16u_C1R, nppiMinMaxGetBufferHostSize_16u_C1R, nppiMinMax_16u_C1R, Npp16u,
                           Npp16u, 1, 1, "16u");
REGISTER_NPPI_MINMAX_BENCH(BenchMinMax_16s_C1R, nppiMinMaxGetBufferHostSize_16s_C1R, nppiMinMax_16s_C1R, Npp16s,
                           Npp16s, 1, 1, "16s");
REGISTER_NPPI_MINMAX_BENCH(BenchMinMax_32f_C1R, nppiMinMaxGetBufferHostSize_32f_C1R, nppiMinMax_32f_C1R, Npp32f,
                           Npp32f, 1, 1, "32f");
REGISTER_NPPI_MINMAX_BENCH(BenchMinMax_8u_C3R, nppiMinMaxGetBufferHostSize_8u_C3R, nppiMinMax_8u_C3R, Npp8u, Npp8u,
                           3, 3, "8u");
REGISTER_NPPI_MINMAX_BENCH(BenchMinMax_16u_C3R, nppiMinMaxGetBufferHostSize_16u_C3R, nppiMinMax_16u_C3R, Npp16u,
                           Npp16u, 3, 3, "16u");
REGISTER_NPPI_MINMAX_BENCH(BenchMinMax_16s_C3R, nppiMinMaxGetBufferHostSize_16s_C3R, nppiMinMax_16s_C3R, Npp16s,
                           Npp16s, 3, 3, "16s");
REGISTER_NPPI_MINMAX_BENCH(BenchMinMax_32f_C3R, nppiMinMaxGetBufferHostSize_32f_C3R, nppiMinMax_32f_C3R, Npp32f,
                           Npp32f, 3, 3, "32f");
REGISTER_NPPI_MINMAX_BENCH(BenchMinMax_8u_C4R, nppiMinMaxGetBufferHostSize_8u_C4R, nppiMinMax_8u_C4R, Npp8u, Npp8u,
                           4, 4, "8u");
REGISTER_NPPI_MINMAX_BENCH(BenchMinMax_16u_C4R, nppiMinMaxGetBufferHostSize_16u_C4R, nppiMinMax_16u_C4R, Npp16u,
                           Npp16u, 4, 4, "16u");
REGISTER_NPPI_MINMAX_BENCH(BenchMinMax_16s_C4R, nppiMinMaxGetBufferHostSize_16s_C4R, nppiMinMax_16s_C4R, Npp16s,
                           Npp16s, 4, 4, "16s");
REGISTER_NPPI_MINMAX_BENCH(BenchMinMax_32f_C4R, nppiMinMaxGetBufferHostSize_32f_C4R, nppiMinMax_32f_C4R, Npp32f,
                           Npp32f, 4, 4, "32f");
REGISTER_NPPI_MINMAX_BENCH(BenchMinMax_8u_AC4R, nppiMinMaxGetBufferHostSize_8u_AC4R, nppiMinMax_8u_AC4R, Npp8u,
                           Npp8u, 4, 3, "8u");
REGISTER_NPPI_MINMAX_BENCH(BenchMinMax_16u_AC4R, nppiMinMaxGetBufferHostSize_16u_AC4R, nppiMinMax_16u_AC4R, Npp16u,
                           Npp16u, 4, 3, "16u");
REGISTER_NPPI_MINMAX_BENCH(BenchMinMax_16s_AC4R, nppiMinMaxGetBufferHostSize_16s_AC4R, nppiMinMax_16s_AC4R, Npp16s,
                           Npp16s, 4, 3, "16s");
REGISTER_NPPI_MINMAX_BENCH(BenchMinMax_32f_AC4R, nppiMinMaxGetBufferHostSize_32f_AC4R, nppiMinMax_32f_AC4R, Npp32f,
                           Npp32f, 4, 3, "32f");

REGISTER_NPPI_INDEX_BENCH(BenchMinIndx_8u_C1R, nppiMinIndxGetBufferHostSize_8u_C1R, nppiMinIndx_8u_C1R, Npp8u,
                          Npp8u, 1, 1, "8u");
REGISTER_NPPI_INDEX_BENCH(BenchMinIndx_16u_C1R, nppiMinIndxGetBufferHostSize_16u_C1R, nppiMinIndx_16u_C1R, Npp16u,
                          Npp16u, 1, 1, "16u");
REGISTER_NPPI_INDEX_BENCH(BenchMinIndx_16s_C1R, nppiMinIndxGetBufferHostSize_16s_C1R, nppiMinIndx_16s_C1R, Npp16s,
                          Npp16s, 1, 1, "16s");
REGISTER_NPPI_INDEX_BENCH(BenchMinIndx_32f_C1R, nppiMinIndxGetBufferHostSize_32f_C1R, nppiMinIndx_32f_C1R, Npp32f,
                          Npp32f, 1, 1, "32f");
REGISTER_NPPI_INDEX_BENCH(BenchMinIndx_8u_C3R, nppiMinIndxGetBufferHostSize_8u_C3R, nppiMinIndx_8u_C3R, Npp8u,
                          Npp8u, 3, 3, "8u");
REGISTER_NPPI_INDEX_BENCH(BenchMinIndx_16u_C3R, nppiMinIndxGetBufferHostSize_16u_C3R, nppiMinIndx_16u_C3R, Npp16u,
                          Npp16u, 3, 3, "16u");
REGISTER_NPPI_INDEX_BENCH(BenchMinIndx_16s_C3R, nppiMinIndxGetBufferHostSize_16s_C3R, nppiMinIndx_16s_C3R, Npp16s,
                          Npp16s, 3, 3, "16s");
REGISTER_NPPI_INDEX_BENCH(BenchMinIndx_32f_C3R, nppiMinIndxGetBufferHostSize_32f_C3R, nppiMinIndx_32f_C3R, Npp32f,
                          Npp32f, 3, 3, "32f");

REGISTER_NPPI_INDEX_BENCH(BenchMaxIndx_8u_C1R, nppiMaxIndxGetBufferHostSize_8u_C1R, nppiMaxIndx_8u_C1R, Npp8u,
                          Npp8u, 1, 1, "8u");
REGISTER_NPPI_INDEX_BENCH(BenchMaxIndx_16u_C1R, nppiMaxIndxGetBufferHostSize_16u_C1R, nppiMaxIndx_16u_C1R, Npp16u,
                          Npp16u, 1, 1, "16u");
REGISTER_NPPI_INDEX_BENCH(BenchMaxIndx_16s_C1R, nppiMaxIndxGetBufferHostSize_16s_C1R, nppiMaxIndx_16s_C1R, Npp16s,
                          Npp16s, 1, 1, "16s");
REGISTER_NPPI_INDEX_BENCH(BenchMaxIndx_32f_C1R, nppiMaxIndxGetBufferHostSize_32f_C1R, nppiMaxIndx_32f_C1R, Npp32f,
                          Npp32f, 1, 1, "32f");
REGISTER_NPPI_INDEX_BENCH(BenchMaxIndx_8u_C3R, nppiMaxIndxGetBufferHostSize_8u_C3R, nppiMaxIndx_8u_C3R, Npp8u,
                          Npp8u, 3, 3, "8u");
REGISTER_NPPI_INDEX_BENCH(BenchMaxIndx_16u_C3R, nppiMaxIndxGetBufferHostSize_16u_C3R, nppiMaxIndx_16u_C3R, Npp16u,
                          Npp16u, 3, 3, "16u");
REGISTER_NPPI_INDEX_BENCH(BenchMaxIndx_16s_C3R, nppiMaxIndxGetBufferHostSize_16s_C3R, nppiMaxIndx_16s_C3R, Npp16s,
                          Npp16s, 3, 3, "16s");
REGISTER_NPPI_INDEX_BENCH(BenchMaxIndx_32f_C3R, nppiMaxIndxGetBufferHostSize_32f_C3R, nppiMaxIndx_32f_C3R, Npp32f,
                          Npp32f, 3, 3, "32f");

REGISTER_NPPI_MEANSTDDEV_BENCH(BenchMeanStdDev_8u_C1R, nppiMeanStdDevGetBufferHostSize_8u_C1R, nppiMean_StdDev_8u_C1R,
                               Npp8u, "8u");
REGISTER_NPPI_MEANSTDDEV_BENCH(BenchMeanStdDev_8s_C1R, nppiMeanStdDevGetBufferHostSize_8s_C1R, nppiMean_StdDev_8s_C1R,
                               Npp8s, "8s");
REGISTER_NPPI_MEANSTDDEV_BENCH(BenchMeanStdDev_16u_C1R, nppiMeanStdDevGetBufferHostSize_16u_C1R,
                               nppiMean_StdDev_16u_C1R, Npp16u, "16u");
REGISTER_NPPI_MEANSTDDEV_BENCH(BenchMeanStdDev_32f_C1R, nppiMeanStdDevGetBufferHostSize_32f_C1R,
                               nppiMean_StdDev_32f_C1R, Npp32f, "32f");

#undef REGISTER_NPPI_STATS_BENCH
#undef REGISTER_NPPI_EXTREMA_BENCH
#undef REGISTER_NPPI_NORM_BENCH
#undef REGISTER_NPPI_MINMAX_BENCH
#undef REGISTER_NPPI_INDEX_BENCH
#undef REGISTER_NPPI_MEANSTDDEV_BENCH

} // namespace

} // namespace npp_benchmark
