#include "../benchmark_base.h"
#include <npps.h>

#include <type_traits>

namespace npp_benchmark {

extern void initDeviceRandom8u(Npp8u* data, int size);
extern void initDeviceRandom16u(Npp16u* data, int size);
extern void initDeviceRandom16s(Npp16s* data, int size);
extern void initDeviceRandom32f(Npp32f* data, int size);

namespace {

template<typename T>
void initSignalInput(T* data, int count) {
    copyDeterministicHostDataToDevice(data, static_cast<size_t>(count));
}

template<>
void initSignalInput<Npp8u>(Npp8u* data, int count) {
    initDeviceRandom8u(data, count);
}

template<>
void initSignalInput<Npp16u>(Npp16u* data, int count) {
    initDeviceRandom16u(data, count);
}

template<>
void initSignalInput<Npp16s>(Npp16s* data, int count) {
    initDeviceRandom16s(data, count);
}

template<>
void initSignalInput<Npp32f>(Npp32f* data, int count) {
    initDeviceRandom32f(data, count);
}

std::vector<BenchmarkResult> makeMissingSignalResults(const std::string& functionName, const std::string& dataType,
                                                      int measureIterations, const std::string& error) {
    std::vector<BenchmarkResult> results;
    for (const auto& sigSize : getStandardSignalSizes()) {
        BenchmarkResult result;
        result.functionName = functionName;
        result.dataType = dataType;
        result.size = sigSize.toString();
        result.channels = 1;
        result.iterations = measureIterations;
        result.success = false;
        result.errorMessage = error;
        results.push_back(result);
    }
    return results;
}

template<typename T, typename Fn>
class NppsBinarySignalBenchmark : public BenchmarkBase {
public:
    NppsBinarySignalBenchmark(const std::string& benchName, const char* symbolName, const char* dataType)
        : BenchmarkBase(benchName), symbolName_(symbolName), dataType_(dataType) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string symErr;
        const Fn fn = DynamicSymbol<Fn>(symbolName_).get(&symErr);
        if (fn == nullptr) {
            return makeMissingSignalResults(name_, dataType_, measureIterations,
                                            std::string("Missing symbol ") + symbolName_ + ": " + symErr);
        }

        std::vector<BenchmarkResult> results;
        for (const auto& sigSize : getStandardSignalSizes()) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = dataType_;
            result.size = sigSize.toString();
            result.channels = 1;
            result.iterations = measureIterations;

            DeviceBuffer<T> src1(sigSize.length);
            DeviceBuffer<T> src2(sigSize.length);
            DeviceBuffer<T> dst(sigSize.length);
            if (!validateDeviceBuffer(src1, result, "src1") || !validateDeviceBuffer(src2, result, "src2") ||
                !validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }

            initSignalInput<T>(src1.get(), sigSize.length);
            initSignalInput<T>(src2.get(), sigSize.length);
            dst.fillZero();

            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st = fn(src1.get(), src2.get(), dst.get(), sigSize.length);
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
                                      [&]() { return fn(src1.get(), src2.get(), dst.get(), sigSize.length); })) {
                result.throughputGBps = computeThroughput(3ULL * sigSize.length * sizeof(T), result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }

private:
    const char* symbolName_;
    const char* dataType_;
};

template<typename T, typename Fn>
class NppsBinarySignalSfsBenchmark : public BenchmarkBase {
public:
    NppsBinarySignalSfsBenchmark(const std::string& benchName, const char* symbolName, const char* dataType)
        : BenchmarkBase(benchName), symbolName_(symbolName), dataType_(dataType) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string symErr;
        const Fn fn = DynamicSymbol<Fn>(symbolName_).get(&symErr);
        if (fn == nullptr) {
            return makeMissingSignalResults(name_, dataType_, measureIterations,
                                            std::string("Missing symbol ") + symbolName_ + ": " + symErr);
        }

        std::vector<BenchmarkResult> results;
        constexpr int kScaleFactor = 0;
        for (const auto& sigSize : getStandardSignalSizes()) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = dataType_;
            result.size = sigSize.toString();
            result.channels = 1;
            result.iterations = measureIterations;

            DeviceBuffer<T> src1(sigSize.length);
            DeviceBuffer<T> src2(sigSize.length);
            DeviceBuffer<T> dst(sigSize.length);
            if (!validateDeviceBuffer(src1, result, "src1") || !validateDeviceBuffer(src2, result, "src2") ||
                !validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }

            initSignalInput<T>(src1.get(), sigSize.length);
            initSignalInput<T>(src2.get(), sigSize.length);
            dst.fillZero();

            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st = fn(src1.get(), src2.get(), dst.get(), sigSize.length, kScaleFactor);
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
                                      [&]() { return fn(src1.get(), src2.get(), dst.get(), sigSize.length, kScaleFactor); })) {
                result.throughputGBps = computeThroughput(3ULL * sigSize.length * sizeof(T), result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }

private:
    const char* symbolName_;
    const char* dataType_;
};

template<typename T, typename Fn>
class NppsBinarySignalInPlaceBenchmark : public BenchmarkBase {
public:
    NppsBinarySignalInPlaceBenchmark(const std::string& benchName, const char* symbolName, const char* dataType)
        : BenchmarkBase(benchName), symbolName_(symbolName), dataType_(dataType) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string symErr;
        const Fn fn = DynamicSymbol<Fn>(symbolName_).get(&symErr);
        if (fn == nullptr) {
            return makeMissingSignalResults(name_, dataType_, measureIterations,
                                            std::string("Missing symbol ") + symbolName_ + ": " + symErr);
        }

        std::vector<BenchmarkResult> results;
        for (const auto& sigSize : getStandardSignalSizes()) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = dataType_;
            result.size = sigSize.toString();
            result.channels = 1;
            result.iterations = measureIterations;

            DeviceBuffer<T> src(sigSize.length);
            DeviceBuffer<T> srcDst(sigSize.length);
            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(srcDst, result, "srcDst")) {
                results.push_back(result);
                continue;
            }

            initSignalInput<T>(src.get(), sigSize.length);
            initSignalInput<T>(srcDst.get(), sigSize.length);

            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st = fn(src.get(), srcDst.get(), sigSize.length);
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
                                      [&]() { return fn(src.get(), srcDst.get(), sigSize.length); })) {
                result.throughputGBps = computeThroughput(2ULL * sigSize.length * sizeof(T), result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }

private:
    const char* symbolName_;
    const char* dataType_;
};

template<typename T, typename Fn>
class NppsBinarySignalInPlaceSfsBenchmark : public BenchmarkBase {
public:
    NppsBinarySignalInPlaceSfsBenchmark(const std::string& benchName, const char* symbolName, const char* dataType)
        : BenchmarkBase(benchName), symbolName_(symbolName), dataType_(dataType) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string symErr;
        const Fn fn = DynamicSymbol<Fn>(symbolName_).get(&symErr);
        if (fn == nullptr) {
            return makeMissingSignalResults(name_, dataType_, measureIterations,
                                            std::string("Missing symbol ") + symbolName_ + ": " + symErr);
        }

        std::vector<BenchmarkResult> results;
        constexpr int kScaleFactor = 0;
        for (const auto& sigSize : getStandardSignalSizes()) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = dataType_;
            result.size = sigSize.toString();
            result.channels = 1;
            result.iterations = measureIterations;

            DeviceBuffer<T> src(sigSize.length);
            DeviceBuffer<T> srcDst(sigSize.length);
            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(srcDst, result, "srcDst")) {
                results.push_back(result);
                continue;
            }

            initSignalInput<T>(src.get(), sigSize.length);
            initSignalInput<T>(srcDst.get(), sigSize.length);

            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st = fn(src.get(), srcDst.get(), sigSize.length, kScaleFactor);
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
                                      [&]() { return fn(src.get(), srcDst.get(), sigSize.length, kScaleFactor); })) {
                result.throughputGBps = computeThroughput(2ULL * sigSize.length * sizeof(T), result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }

private:
    const char* symbolName_;
    const char* dataType_;
};

#define REGISTER_SIGNAL_BINARY_BENCH(BenchClass, Symbol, Type, DType)                                         \
    class BenchClass final : public NppsBinarySignalBenchmark<Type, decltype(&Symbol)> {                      \
    public:                                                                                                    \
        BenchClass() : NppsBinarySignalBenchmark<Type, decltype(&Symbol)>(#Symbol, #Symbol, DType) {}        \
    };                                                                                                         \
    REGISTER_BENCHMARK(BenchClass)

#define REGISTER_SIGNAL_BINARY_SFS_BENCH(BenchClass, Symbol, Type, DType)                                     \
    class BenchClass final : public NppsBinarySignalSfsBenchmark<Type, decltype(&Symbol)> {                   \
    public:                                                                                                    \
        BenchClass() : NppsBinarySignalSfsBenchmark<Type, decltype(&Symbol)>(#Symbol, #Symbol, DType) {}     \
    };                                                                                                         \
    REGISTER_BENCHMARK(BenchClass)

#define REGISTER_SIGNAL_BINARY_INPLACE_BENCH(BenchClass, Symbol, Type, DType)                                 \
    class BenchClass final : public NppsBinarySignalInPlaceBenchmark<Type, decltype(&Symbol)> {               \
    public:                                                                                                    \
        BenchClass() : NppsBinarySignalInPlaceBenchmark<Type, decltype(&Symbol)>(#Symbol, #Symbol, DType) {} \
    };                                                                                                         \
    REGISTER_BENCHMARK(BenchClass)

#define REGISTER_SIGNAL_BINARY_INPLACE_SFS_BENCH(BenchClass, Symbol, Type, DType)                                 \
    class BenchClass final : public NppsBinarySignalInPlaceSfsBenchmark<Type, decltype(&Symbol)> {                \
    public:                                                                                                        \
        BenchClass() : NppsBinarySignalInPlaceSfsBenchmark<Type, decltype(&Symbol)>(#Symbol, #Symbol, DType) {}  \
    };                                                                                                             \
    REGISTER_BENCHMARK(BenchClass)

REGISTER_SIGNAL_BINARY_BENCH(BenchAdd_16s, nppsAdd_16s, Npp16s, "16s");
REGISTER_SIGNAL_BINARY_BENCH(BenchAdd_16u, nppsAdd_16u, Npp16u, "16u");
REGISTER_SIGNAL_BINARY_SFS_BENCH(BenchAdd_8u_Sfs, nppsAdd_8u_Sfs, Npp8u, "8u");
REGISTER_SIGNAL_BINARY_SFS_BENCH(BenchAdd_16u_Sfs, nppsAdd_16u_Sfs, Npp16u, "16u");
REGISTER_SIGNAL_BINARY_SFS_BENCH(BenchAdd_16s_Sfs, nppsAdd_16s_Sfs, Npp16s, "16s");
REGISTER_SIGNAL_BINARY_INPLACE_BENCH(BenchAdd_16s_I, nppsAdd_16s_I, Npp16s, "16s");
REGISTER_SIGNAL_BINARY_INPLACE_SFS_BENCH(BenchAdd_8u_ISfs, nppsAdd_8u_ISfs, Npp8u, "8u");
REGISTER_SIGNAL_BINARY_INPLACE_SFS_BENCH(BenchAdd_16u_ISfs, nppsAdd_16u_ISfs, Npp16u, "16u");
REGISTER_SIGNAL_BINARY_INPLACE_SFS_BENCH(BenchAdd_16s_ISfs, nppsAdd_16s_ISfs, Npp16s, "16s");

REGISTER_SIGNAL_BINARY_BENCH(BenchSub_16s, nppsSub_16s, Npp16s, "16s");
REGISTER_SIGNAL_BINARY_BENCH(BenchSub_32f, nppsSub_32f, Npp32f, "32f");
REGISTER_SIGNAL_BINARY_SFS_BENCH(BenchSub_8u_Sfs, nppsSub_8u_Sfs, Npp8u, "8u");
REGISTER_SIGNAL_BINARY_SFS_BENCH(BenchSub_16u_Sfs, nppsSub_16u_Sfs, Npp16u, "16u");
REGISTER_SIGNAL_BINARY_SFS_BENCH(BenchSub_16s_Sfs, nppsSub_16s_Sfs, Npp16s, "16s");
REGISTER_SIGNAL_BINARY_INPLACE_BENCH(BenchSub_16s_I, nppsSub_16s_I, Npp16s, "16s");
REGISTER_SIGNAL_BINARY_INPLACE_BENCH(BenchSub_32f_I, nppsSub_32f_I, Npp32f, "32f");
REGISTER_SIGNAL_BINARY_INPLACE_SFS_BENCH(BenchSub_8u_ISfs, nppsSub_8u_ISfs, Npp8u, "8u");
REGISTER_SIGNAL_BINARY_INPLACE_SFS_BENCH(BenchSub_16u_ISfs, nppsSub_16u_ISfs, Npp16u, "16u");
REGISTER_SIGNAL_BINARY_INPLACE_SFS_BENCH(BenchSub_16s_ISfs, nppsSub_16s_ISfs, Npp16s, "16s");

REGISTER_SIGNAL_BINARY_BENCH(BenchMul_16s, nppsMul_16s, Npp16s, "16s");
REGISTER_SIGNAL_BINARY_BENCH(BenchMul_32f, nppsMul_32f, Npp32f, "32f");
REGISTER_SIGNAL_BINARY_SFS_BENCH(BenchMul_8u_Sfs, nppsMul_8u_Sfs, Npp8u, "8u");
REGISTER_SIGNAL_BINARY_SFS_BENCH(BenchMul_16u_Sfs, nppsMul_16u_Sfs, Npp16u, "16u");
REGISTER_SIGNAL_BINARY_SFS_BENCH(BenchMul_16s_Sfs, nppsMul_16s_Sfs, Npp16s, "16s");
REGISTER_SIGNAL_BINARY_INPLACE_BENCH(BenchMul_16s_I, nppsMul_16s_I, Npp16s, "16s");
REGISTER_SIGNAL_BINARY_INPLACE_BENCH(BenchMul_32f_I, nppsMul_32f_I, Npp32f, "32f");
REGISTER_SIGNAL_BINARY_INPLACE_SFS_BENCH(BenchMul_8u_ISfs, nppsMul_8u_ISfs, Npp8u, "8u");
REGISTER_SIGNAL_BINARY_INPLACE_SFS_BENCH(BenchMul_16u_ISfs, nppsMul_16u_ISfs, Npp16u, "16u");
REGISTER_SIGNAL_BINARY_INPLACE_SFS_BENCH(BenchMul_16s_ISfs, nppsMul_16s_ISfs, Npp16s, "16s");

REGISTER_SIGNAL_BINARY_BENCH(BenchDiv_32f, nppsDiv_32f, Npp32f, "32f");
REGISTER_SIGNAL_BINARY_SFS_BENCH(BenchDiv_8u_Sfs, nppsDiv_8u_Sfs, Npp8u, "8u");
REGISTER_SIGNAL_BINARY_SFS_BENCH(BenchDiv_16u_Sfs, nppsDiv_16u_Sfs, Npp16u, "16u");
REGISTER_SIGNAL_BINARY_SFS_BENCH(BenchDiv_16s_Sfs, nppsDiv_16s_Sfs, Npp16s, "16s");
REGISTER_SIGNAL_BINARY_INPLACE_BENCH(BenchDiv_32f_I, nppsDiv_32f_I, Npp32f, "32f");
REGISTER_SIGNAL_BINARY_INPLACE_SFS_BENCH(BenchDiv_8u_ISfs, nppsDiv_8u_ISfs, Npp8u, "8u");
REGISTER_SIGNAL_BINARY_INPLACE_SFS_BENCH(BenchDiv_16u_ISfs, nppsDiv_16u_ISfs, Npp16u, "16u");
REGISTER_SIGNAL_BINARY_INPLACE_SFS_BENCH(BenchDiv_16s_ISfs, nppsDiv_16s_ISfs, Npp16s, "16s");

REGISTER_SIGNAL_BINARY_BENCH(BenchAnd_8u, nppsAnd_8u, Npp8u, "8u");
REGISTER_SIGNAL_BINARY_BENCH(BenchAnd_16u, nppsAnd_16u, Npp16u, "16u");
REGISTER_SIGNAL_BINARY_INPLACE_BENCH(BenchAnd_8u_I, nppsAnd_8u_I, Npp8u, "8u");
REGISTER_SIGNAL_BINARY_INPLACE_BENCH(BenchAnd_16u_I, nppsAnd_16u_I, Npp16u, "16u");

REGISTER_SIGNAL_BINARY_BENCH(BenchOr_8u, nppsOr_8u, Npp8u, "8u");
REGISTER_SIGNAL_BINARY_BENCH(BenchOr_16u, nppsOr_16u, Npp16u, "16u");
REGISTER_SIGNAL_BINARY_INPLACE_BENCH(BenchOr_8u_I, nppsOr_8u_I, Npp8u, "8u");
REGISTER_SIGNAL_BINARY_INPLACE_BENCH(BenchOr_16u_I, nppsOr_16u_I, Npp16u, "16u");

REGISTER_SIGNAL_BINARY_BENCH(BenchXor_8u, nppsXor_8u, Npp8u, "8u");
REGISTER_SIGNAL_BINARY_BENCH(BenchXor_16u, nppsXor_16u, Npp16u, "16u");
REGISTER_SIGNAL_BINARY_INPLACE_BENCH(BenchXor_8u_I, nppsXor_8u_I, Npp8u, "8u");
REGISTER_SIGNAL_BINARY_INPLACE_BENCH(BenchXor_16u_I, nppsXor_16u_I, Npp16u, "16u");

#undef REGISTER_SIGNAL_BINARY_BENCH
#undef REGISTER_SIGNAL_BINARY_SFS_BENCH
#undef REGISTER_SIGNAL_BINARY_INPLACE_BENCH
#undef REGISTER_SIGNAL_BINARY_INPLACE_SFS_BENCH

} // namespace

} // namespace npp_benchmark
