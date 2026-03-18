#include "../benchmark_base.h"
#include <npps.h>

namespace npp_benchmark {

namespace {

inline Npp16sc make16scValue() { return Npp16sc{2, -2}; }
inline Npp32sc make32scValue() { return Npp32sc{3, -3}; }
inline Npp32fc make32fcValue() { return Npp32fc{1.5f, -1.5f}; }
inline Npp64sc make64scValue() { return Npp64sc{4, -4}; }
inline Npp64fc make64fcValue() { return Npp64fc{2.5, -2.5}; }

template<typename T>
void initInitInput(T* data, int count) {
    std::vector<T> host(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        host[static_cast<size_t>(i)] = static_cast<T>((i % 29) - 14);
    }
    (void)cudaMemcpy(data, host.data(), static_cast<size_t>(count) * sizeof(T), cudaMemcpyHostToDevice);
}

template<>
void initInitInput<Npp8u>(Npp8u* data, int count) {
    std::vector<Npp8u> host(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        host[static_cast<size_t>(i)] = static_cast<Npp8u>(i % 251);
    }
    (void)cudaMemcpy(data, host.data(), static_cast<size_t>(count) * sizeof(Npp8u), cudaMemcpyHostToDevice);
}

template<>
void initInitInput<Npp16sc>(Npp16sc* data, int count) {
    std::vector<Npp16sc> host(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        host[static_cast<size_t>(i)] = Npp16sc{static_cast<Npp16s>(i % 19), static_cast<Npp16s>(-(i % 19))};
    }
    (void)cudaMemcpy(data, host.data(), static_cast<size_t>(count) * sizeof(Npp16sc), cudaMemcpyHostToDevice);
}

template<>
void initInitInput<Npp32sc>(Npp32sc* data, int count) {
    std::vector<Npp32sc> host(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        host[static_cast<size_t>(i)] = Npp32sc{i % 23, -(i % 23)};
    }
    (void)cudaMemcpy(data, host.data(), static_cast<size_t>(count) * sizeof(Npp32sc), cudaMemcpyHostToDevice);
}

template<>
void initInitInput<Npp32fc>(Npp32fc* data, int count) {
    std::vector<Npp32fc> host(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        host[static_cast<size_t>(i)] = Npp32fc{static_cast<float>(i % 17), static_cast<float>(-(i % 17))};
    }
    (void)cudaMemcpy(data, host.data(), static_cast<size_t>(count) * sizeof(Npp32fc), cudaMemcpyHostToDevice);
}

template<>
void initInitInput<Npp64sc>(Npp64sc* data, int count) {
    std::vector<Npp64sc> host(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        host[static_cast<size_t>(i)] = Npp64sc{static_cast<Npp64s>(i % 31), static_cast<Npp64s>(-(i % 31))};
    }
    (void)cudaMemcpy(data, host.data(), static_cast<size_t>(count) * sizeof(Npp64sc), cudaMemcpyHostToDevice);
}

template<>
void initInitInput<Npp64fc>(Npp64fc* data, int count) {
    std::vector<Npp64fc> host(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        host[static_cast<size_t>(i)] = Npp64fc{static_cast<double>(i % 13), static_cast<double>(-(i % 13))};
    }
    (void)cudaMemcpy(data, host.data(), static_cast<size_t>(count) * sizeof(Npp64fc), cudaMemcpyHostToDevice);
}

template<typename T, typename Fn>
class NppsSetBenchmark : public BenchmarkBase {
public:
    NppsSetBenchmark(const std::string& benchName, const char* symbolName, const char* dataType, T value)
        : BenchmarkBase(benchName), symbolName_(symbolName), dataType_(dataType), value_(value) {}

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

            DeviceBuffer<T> dst(sigSize.length);
            if (!validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }

            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st = fn(value_, dst.get(), sigSize.length);
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
                                      [&]() { return fn(value_, dst.get(), sigSize.length); })) {
                result.throughputGBps = computeThroughput(static_cast<size_t>(sigSize.length) * sizeof(T), result.avgTimeMs);
            }
            results.push_back(result);
        }
        return results;
    }

private:
    const char* symbolName_;
    const char* dataType_;
    T value_;
};

template<typename T, typename Fn>
class NppsZeroBenchmark : public BenchmarkBase {
public:
    NppsZeroBenchmark(const std::string& benchName, const char* symbolName, const char* dataType)
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

            DeviceBuffer<T> dst(sigSize.length);
            if (!validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }

            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st = fn(dst.get(), sigSize.length);
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

            if (runBatchedMeasurement(result, measureIterations, [&]() { return fn(dst.get(), sigSize.length); })) {
                result.throughputGBps =
                    computeThroughput(static_cast<size_t>(sigSize.length) * sizeof(T), result.avgTimeMs);
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
class NppsCopyBenchmark : public BenchmarkBase {
public:
    NppsCopyBenchmark(const std::string& benchName, const char* symbolName, const char* dataType)
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

            DeviceBuffer<T> src(sigSize.length);
            DeviceBuffer<T> dst(sigSize.length);
            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }

            initInitInput<T>(src.get(), sigSize.length);
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
                    computeThroughput(static_cast<size_t>(sigSize.length) * sizeof(T) * 2, result.avgTimeMs);
            }
            results.push_back(result);
        }
        return results;
    }

private:
    const char* symbolName_;
    const char* dataType_;
};

#define REGISTER_SIGNAL_SET_BENCH(BenchClass, Symbol, Type, DType, Value)                                      \
    class BenchClass final : public NppsSetBenchmark<Type, decltype(&Symbol)> {                                \
    public:                                                                                                      \
        BenchClass() : NppsSetBenchmark<Type, decltype(&Symbol)>(#Symbol, #Symbol, DType, Value) {}           \
    };                                                                                                           \
    REGISTER_BENCHMARK(BenchClass)

#define REGISTER_SIGNAL_ZERO_BENCH(BenchClass, Symbol, Type, DType)                                             \
    class BenchClass final : public NppsZeroBenchmark<Type, decltype(&Symbol)> {                                \
    public:                                                                                                      \
        BenchClass() : NppsZeroBenchmark<Type, decltype(&Symbol)>(#Symbol, #Symbol, DType) {}                  \
    };                                                                                                           \
    REGISTER_BENCHMARK(BenchClass)

#define REGISTER_SIGNAL_COPY_BENCH(BenchClass, Symbol, Type, DType)                                             \
    class BenchClass final : public NppsCopyBenchmark<Type, decltype(&Symbol)> {                                \
    public:                                                                                                      \
        BenchClass() : NppsCopyBenchmark<Type, decltype(&Symbol)>(#Symbol, #Symbol, DType) {}                  \
    };                                                                                                           \
    REGISTER_BENCHMARK(BenchClass)

REGISTER_SIGNAL_SET_BENCH(BenchSet_8u, nppsSet_8u, Npp8u, "8u", static_cast<Npp8u>(7));
REGISTER_SIGNAL_SET_BENCH(BenchSet_8s, nppsSet_8s, Npp8s, "8s", static_cast<Npp8s>(-7));
REGISTER_SIGNAL_SET_BENCH(BenchSet_16u, nppsSet_16u, Npp16u, "16u", static_cast<Npp16u>(17));
REGISTER_SIGNAL_SET_BENCH(BenchSet_16s, nppsSet_16s, Npp16s, "16s", static_cast<Npp16s>(-3));
REGISTER_SIGNAL_SET_BENCH(BenchSet_16sc, nppsSet_16sc, Npp16sc, "16sc", make16scValue());
REGISTER_SIGNAL_SET_BENCH(BenchSet_32u, nppsSet_32u, Npp32u, "32u", static_cast<Npp32u>(23));
REGISTER_SIGNAL_SET_BENCH(BenchSet_32s, nppsSet_32s, Npp32s, "32s", static_cast<Npp32s>(-23));
REGISTER_SIGNAL_SET_BENCH(BenchSet_32sc, nppsSet_32sc, Npp32sc, "32sc", make32scValue());
REGISTER_SIGNAL_SET_BENCH(BenchSet_32f, nppsSet_32f, Npp32f, "32f", static_cast<Npp32f>(1.0f));
REGISTER_SIGNAL_SET_BENCH(BenchSet_32fc, nppsSet_32fc, Npp32fc, "32fc", make32fcValue());
REGISTER_SIGNAL_SET_BENCH(BenchSet_64s, nppsSet_64s, Npp64s, "64s", static_cast<Npp64s>(-31));
REGISTER_SIGNAL_SET_BENCH(BenchSet_64sc, nppsSet_64sc, Npp64sc, "64sc", make64scValue());
REGISTER_SIGNAL_SET_BENCH(BenchSet_64f, nppsSet_64f, Npp64f, "64f", static_cast<Npp64f>(1.0));
REGISTER_SIGNAL_SET_BENCH(BenchSet_64fc, nppsSet_64fc, Npp64fc, "64fc", make64fcValue());

REGISTER_SIGNAL_ZERO_BENCH(BenchZero_8u, nppsZero_8u, Npp8u, "8u");
REGISTER_SIGNAL_ZERO_BENCH(BenchZero_16s, nppsZero_16s, Npp16s, "16s");
REGISTER_SIGNAL_ZERO_BENCH(BenchZero_16sc, nppsZero_16sc, Npp16sc, "16sc");
REGISTER_SIGNAL_ZERO_BENCH(BenchZero_32s, nppsZero_32s, Npp32s, "32s");
REGISTER_SIGNAL_ZERO_BENCH(BenchZero_32sc, nppsZero_32sc, Npp32sc, "32sc");
REGISTER_SIGNAL_ZERO_BENCH(BenchZero_32f, nppsZero_32f, Npp32f, "32f");
REGISTER_SIGNAL_ZERO_BENCH(BenchZero_32fc, nppsZero_32fc, Npp32fc, "32fc");
REGISTER_SIGNAL_ZERO_BENCH(BenchZero_64s, nppsZero_64s, Npp64s, "64s");
REGISTER_SIGNAL_ZERO_BENCH(BenchZero_64sc, nppsZero_64sc, Npp64sc, "64sc");
REGISTER_SIGNAL_ZERO_BENCH(BenchZero_64f, nppsZero_64f, Npp64f, "64f");
REGISTER_SIGNAL_ZERO_BENCH(BenchZero_64fc, nppsZero_64fc, Npp64fc, "64fc");

REGISTER_SIGNAL_COPY_BENCH(BenchCopy_8u, nppsCopy_8u, Npp8u, "8u");
REGISTER_SIGNAL_COPY_BENCH(BenchCopy_16s, nppsCopy_16s, Npp16s, "16s");
REGISTER_SIGNAL_COPY_BENCH(BenchCopy_16sc, nppsCopy_16sc, Npp16sc, "16sc");
REGISTER_SIGNAL_COPY_BENCH(BenchCopy_32s, nppsCopy_32s, Npp32s, "32s");
REGISTER_SIGNAL_COPY_BENCH(BenchCopy_32sc, nppsCopy_32sc, Npp32sc, "32sc");
REGISTER_SIGNAL_COPY_BENCH(BenchCopy_32f, nppsCopy_32f, Npp32f, "32f");
REGISTER_SIGNAL_COPY_BENCH(BenchCopy_32fc, nppsCopy_32fc, Npp32fc, "32fc");
REGISTER_SIGNAL_COPY_BENCH(BenchCopy_64s, nppsCopy_64s, Npp64s, "64s");
REGISTER_SIGNAL_COPY_BENCH(BenchCopy_64sc, nppsCopy_64sc, Npp64sc, "64sc");
REGISTER_SIGNAL_COPY_BENCH(BenchCopy_64fc, nppsCopy_64fc, Npp64fc, "64fc");

#undef REGISTER_SIGNAL_SET_BENCH
#undef REGISTER_SIGNAL_ZERO_BENCH
#undef REGISTER_SIGNAL_COPY_BENCH

} // namespace

} // namespace npp_benchmark
