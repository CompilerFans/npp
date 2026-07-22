#include "../benchmark_base.h"
#include <nppcore.h>

namespace npp_benchmark {

namespace {

template<typename Fn>
class NppcoreHostBenchmark : public BenchmarkBase {
public:
    using ValidateFn = bool (*)(Fn);

    NppcoreHostBenchmark(const std::string& benchName, const char* symbolName, ValidateFn validate)
        : BenchmarkBase(benchName), symbolName_(symbolName), validate_(validate) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string symErr;
        const Fn fn = DynamicSymbol<Fn>(symbolName_).get(&symErr);

        BenchmarkResult result;
        result.functionName = name_;
        result.dataType = "host";
        result.size = "host";
        result.channels = 0;
        result.iterations = measureIterations;

        if (fn == nullptr) {
            result.success = false;
            result.errorMessage = std::string("Missing symbol ") + symbolName_ + ": " + symErr;
            return {result};
        }

        for (int i = 0; i < warmupIterations; ++i) {
            if (!validate_(fn)) {
                result.success = false;
                result.errorMessage = "Validation failed during warmup";
                return {result};
            }
        }

        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < measureIterations; ++i) {
            if (!validate_(fn)) {
                result.success = false;
                result.errorMessage = "Validation failed during timed batch";
                return {result};
            }
        }
        const auto stop = std::chrono::steady_clock::now();

        const double totalMs = std::chrono::duration<double, std::milli>(stop - start).count();
        result.success = true;
        result.minTimeMs = totalMs / measureIterations;
        result.maxTimeMs = result.minTimeMs;
        result.avgTimeMs = result.minTimeMs;
        result.stdDevMs = 0.0;
        result.throughputGBps = 0.0;
        return {result};
    }

private:
    const char* symbolName_;
    ValidateFn validate_;
};

#define REGISTER_NPPCORE_QUERY_BENCH(BenchClass, Symbol, FnType, ValidateFn)                                     \
    class BenchClass final : public NppcoreHostBenchmark<FnType> {                                               \
    public:                                                                                                      \
        BenchClass()                                                                                             \
            : NppcoreHostBenchmark<FnType>(#Symbol, #Symbol, ValidateFn) {}                                      \
    };                                                                                                           \
    REGISTER_BENCHMARK(BenchClass)

using FnGetLibVersion = const NppLibraryVersion* (*)();
using FnGetGpuNumSMs = int (*)();
using FnGetMaxThreadsPerBlock = int (*)();
using FnGetMaxThreadsPerSM = int (*)();
using FnGetGpuDeviceProperties = int (*)(int*, int*, int*);
using FnGetGpuName = const char* (*)();
using FnGetStream = cudaStream_t (*)();
using FnGetStreamContext = NppStatus (*)(NppStreamContext*);
using FnGetStreamNumSMs = int (*)();
using FnGetStreamMaxThreadsPerSM = int (*)();

bool validateGetLibVersion(FnGetLibVersion fn) { return fn() != nullptr; }
bool validateGetGpuNumSMs(FnGetGpuNumSMs fn) { return fn() >= 0; }
bool validateGetMaxThreadsPerBlock(FnGetMaxThreadsPerBlock fn) { return fn() > 0; }
bool validateGetMaxThreadsPerSM(FnGetMaxThreadsPerSM fn) { return fn() > 0; }
bool validateGetGpuDeviceProperties(FnGetGpuDeviceProperties fn) {
    int maxThreadsPerSM = 0;
    int maxThreadsPerBlock = 0;
    int numberOfSMs = 0;
    return fn(&maxThreadsPerSM, &maxThreadsPerBlock, &numberOfSMs) == 0 && maxThreadsPerSM > 0 &&
           maxThreadsPerBlock > 0 && numberOfSMs > 0;
}
bool validateGetGpuName(FnGetGpuName fn) {
    const char* name = fn();
    return name != nullptr && name[0] != '\0';
}
bool validateGetStream(FnGetStream fn) {
    (void)fn();
    return true;
}
bool validateGetStreamContext(FnGetStreamContext fn) {
    NppStreamContext ctx{};
    return fn(&ctx) == NPP_SUCCESS;
}
bool validateGetStreamNumSMs(FnGetStreamNumSMs fn) { return fn() > 0; }
bool validateGetStreamMaxThreadsPerSM(FnGetStreamMaxThreadsPerSM fn) { return fn() > 0; }

REGISTER_NPPCORE_QUERY_BENCH(BenchGetLibVersion, nppGetLibVersion, FnGetLibVersion, validateGetLibVersion);
REGISTER_NPPCORE_QUERY_BENCH(BenchGetGpuNumSMs, nppGetGpuNumSMs, FnGetGpuNumSMs, validateGetGpuNumSMs);
REGISTER_NPPCORE_QUERY_BENCH(
    BenchGetMaxThreadsPerBlock, nppGetMaxThreadsPerBlock, FnGetMaxThreadsPerBlock, validateGetMaxThreadsPerBlock);
REGISTER_NPPCORE_QUERY_BENCH(
    BenchGetMaxThreadsPerSM, nppGetMaxThreadsPerSM, FnGetMaxThreadsPerSM, validateGetMaxThreadsPerSM);
REGISTER_NPPCORE_QUERY_BENCH(
    BenchGetGpuDeviceProperties, nppGetGpuDeviceProperties, FnGetGpuDeviceProperties, validateGetGpuDeviceProperties);
REGISTER_NPPCORE_QUERY_BENCH(BenchGetGpuName, nppGetGpuName, FnGetGpuName, validateGetGpuName);
REGISTER_NPPCORE_QUERY_BENCH(BenchGetStream, nppGetStream, FnGetStream, validateGetStream);
REGISTER_NPPCORE_QUERY_BENCH(
    BenchGetStreamContext, nppGetStreamContext, FnGetStreamContext, validateGetStreamContext);
REGISTER_NPPCORE_QUERY_BENCH(
    BenchGetStreamNumSMs, nppGetStreamNumSMs, FnGetStreamNumSMs, validateGetStreamNumSMs);
REGISTER_NPPCORE_QUERY_BENCH(
    BenchGetStreamMaxThreadsPerSM, nppGetStreamMaxThreadsPerSM, FnGetStreamMaxThreadsPerSM,
    validateGetStreamMaxThreadsPerSM);

#undef REGISTER_NPPCORE_QUERY_BENCH

} // namespace

} // namespace npp_benchmark
