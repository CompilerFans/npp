#include "../benchmark_base.h"
#include <npps_filtering_functions.h>

namespace npp_benchmark {

namespace {

#ifdef USE_NVIDIA_NPP_TESTS
using HostSizeType = size_t;
#else
using HostSizeType = int;
#endif

void initIntegralInput(Npp32s* data, int count) {
    std::vector<Npp32s> host(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        host[static_cast<size_t>(i)] = static_cast<Npp32s>((i % 17) - 8);
    }
    (void)cudaMemcpy(data, host.data(), static_cast<size_t>(count) * sizeof(Npp32s), cudaMemcpyHostToDevice);
}

class NppsIntegral32sBenchmark final : public BenchmarkBase {
public:
    NppsIntegral32sBenchmark() : BenchmarkBase("nppsIntegral_32s") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string bufferErr;
        std::string opErr;
        const auto fnGetBufferSize =
            DynamicSymbol<decltype(&nppsIntegralGetBufferSize_32s)>("nppsIntegralGetBufferSize_32s").get(&bufferErr);
        const auto fnOp = DynamicSymbol<decltype(&nppsIntegral_32s)>("nppsIntegral_32s").get(&opErr);

        std::vector<BenchmarkResult> results;
        for (const auto& sigSize : getStandardSignalSizes()) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = "32s";
            result.size = sigSize.toString();
            result.channels = 1;
            result.iterations = measureIterations;

            if (fnGetBufferSize == nullptr) {
                result.success = false;
                result.errorMessage = std::string("Missing symbol nppsIntegralGetBufferSize_32s: ") + bufferErr;
                results.push_back(result);
                continue;
            }
            if (fnOp == nullptr) {
                result.success = false;
                result.errorMessage = std::string("Missing symbol nppsIntegral_32s: ") + opErr;
                results.push_back(result);
                continue;
            }

            DeviceBuffer<Npp32s> src(sigSize.length);
            DeviceBuffer<Npp32s> dst(sigSize.length);
            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }

            HostSizeType bufferSize = 0;
            const NppStatus bufferStatus = fnGetBufferSize(sigSize.length, &bufferSize);
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

            initIntegralInput(src.get(), sigSize.length);
            dst.fillZero();
            scratch.fillZero();

            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st = fnOp(src.get(), dst.get(), sigSize.length, scratch.get());
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
                                      [&]() { return fnOp(src.get(), dst.get(), sigSize.length, scratch.get()); })) {
                result.throughputGBps =
                    computeThroughput(static_cast<size_t>(sigSize.length) * sizeof(Npp32s) * 2, result.avgTimeMs);
            }
            results.push_back(result);
        }

        return results;
    }
};

REGISTER_BENCHMARK(NppsIntegral32sBenchmark)

} // namespace

} // namespace npp_benchmark
