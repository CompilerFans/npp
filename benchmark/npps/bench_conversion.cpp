#include "../benchmark_base.h"
#include <npps.h>

namespace npp_benchmark {

namespace {

template<typename T>
void initConversionInput(T* data, int count) {
    std::vector<T> host(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        host[static_cast<size_t>(i)] = static_cast<T>((i % 251) - 125);
    }
    cudaMemcpy(data, host.data(), static_cast<size_t>(count) * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename SrcT, typename DstT, typename Fn>
class NppsConvertBenchmark : public BenchmarkBase {
public:
    NppsConvertBenchmark(const std::string& benchName, const char* symbolName, const char* dataType)
        : BenchmarkBase(benchName), symbolName_(symbolName), dataType_(dataType) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string symErr;
        const Fn fn = DynamicSymbol<Fn>(symbolName_).get(&symErr);
        std::vector<BenchmarkResult> results;

        for (const auto& sigSize : getStandardSignalSizes()) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = dataType_;
            result.size = sigSize.toString();
            result.channels = 1;
            result.iterations = measureIterations;
            if (fn == nullptr) {
                result.success = false;
                result.errorMessage = std::string("Missing symbol ") + symbolName_ + ": " + symErr;
                results.push_back(result);
                continue;
            }

            DeviceBuffer<SrcT> src(sigSize.length);
            DeviceBuffer<DstT> dst(sigSize.length);
            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }
            initConversionInput<SrcT>(src.get(), sigSize.length);
            dst.fillZero();

            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st = fn(src.get(), dst.get(), sigSize.length);
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
                                      [&]() { return fn(src.get(), dst.get(), sigSize.length); })) {
                result.throughputGBps =
                    computeThroughput(static_cast<size_t>(sigSize.length) * (sizeof(SrcT) + sizeof(DstT)),
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

#define REGISTER_SIGNAL_CONVERT_BENCH(BenchClass, Symbol, SrcType, DstType, DType)                                \
    class BenchClass final : public NppsConvertBenchmark<SrcType, DstType, decltype(&Symbol)> {                  \
    public:                                                                                                        \
        BenchClass() : NppsConvertBenchmark<SrcType, DstType, decltype(&Symbol)>(#Symbol, #Symbol, DType) {}     \
    };                                                                                                             \
    REGISTER_BENCHMARK(BenchClass)

REGISTER_SIGNAL_CONVERT_BENCH(BenchConvert_8s16s, nppsConvert_8s16s, Npp8s, Npp16s, "8s16s");
REGISTER_SIGNAL_CONVERT_BENCH(BenchConvert_8s32f, nppsConvert_8s32f, Npp8s, Npp32f, "8s32f");
REGISTER_SIGNAL_CONVERT_BENCH(BenchConvert_8u32f, nppsConvert_8u32f, Npp8u, Npp32f, "8u32f");
REGISTER_SIGNAL_CONVERT_BENCH(BenchConvert_16s32s, nppsConvert_16s32s, Npp16s, Npp32s, "16s32s");
REGISTER_SIGNAL_CONVERT_BENCH(BenchConvert_16s32f, nppsConvert_16s32f, Npp16s, Npp32f, "16s32f");
REGISTER_SIGNAL_CONVERT_BENCH(BenchConvert_16u32f, nppsConvert_16u32f, Npp16u, Npp32f, "16u32f");
REGISTER_SIGNAL_CONVERT_BENCH(BenchConvert_32s16s, nppsConvert_32s16s, Npp32s, Npp16s, "32s16s");
REGISTER_SIGNAL_CONVERT_BENCH(BenchConvert_32s32f, nppsConvert_32s32f, Npp32s, Npp32f, "32s32f");
REGISTER_SIGNAL_CONVERT_BENCH(BenchConvert_32s64f, nppsConvert_32s64f, Npp32s, Npp64f, "32s64f");
REGISTER_SIGNAL_CONVERT_BENCH(BenchConvert_32f64f, nppsConvert_32f64f, Npp32f, Npp64f, "32f64f");
REGISTER_SIGNAL_CONVERT_BENCH(BenchConvert_64s64f, nppsConvert_64s64f, Npp64s, Npp64f, "64s64f");
REGISTER_SIGNAL_CONVERT_BENCH(BenchConvert_64f32f, nppsConvert_64f32f, Npp64f, Npp32f, "64f32f");

#undef REGISTER_SIGNAL_CONVERT_BENCH

} // namespace

} // namespace npp_benchmark
