#include "../benchmark_base.h"
#include <nppi_arithmetic_and_logical_operations.h>

namespace npp_benchmark {

extern void initDeviceRandom8u(Npp8u* data, int size);
extern void initDeviceRandom16u(Npp16u* data, int size);
extern void initDeviceRandom16s(Npp16s* data, int size);
extern void initDeviceRandom32f(Npp32f* data, int size);

// Generic image benchmark template
template<typename T>
class NppiArithmeticBenchmark : public BenchmarkBase {
public:
    using BenchFunc = std::function<NppStatus(const T*, int, T*, int, NppiSize)>;

    NppiArithmeticBenchmark(const std::string& name, const std::string& dataType,
                           int channels, BenchFunc func)
        : BenchmarkBase(name), dataType_(dataType), channels_(channels), func_(func) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::vector<BenchmarkResult> results;
        auto sizes = getStandardImageSizes();

        for (const auto& size : sizes) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = dataType_;
            result.size = size.toString();
            result.channels = channels_;
            result.iterations = measureIterations;

            int pixelCount = size.width * size.height * channels_;
            size_t bytes = pixelCount * sizeof(T);
            int step = size.width * channels_ * sizeof(T);

            DeviceBuffer<T> src(pixelCount);
            DeviceBuffer<T> dst(pixelCount);

            // Initialize source with random data
            initData(src.get(), pixelCount);

            NppiSize roiSize = {size.width, size.height};

            // Warmup
            for (int i = 0; i < warmupIterations; ++i) {
                func_(src.get(), step, dst.get(), step, roiSize);
            }
            cudaDeviceSynchronize();

            if (runBatchedMeasurement(result, measureIterations,
                                      [&]() { return func_(src.get(), step, dst.get(), step, roiSize); })) {
                // Throughput: read src + write dst
                result.throughputGBps = computeThroughput(2 * bytes, result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }

private:
    std::string dataType_;
    int channels_;
    BenchFunc func_;

    void initData(T* data, int count);
};

template<>
void NppiArithmeticBenchmark<Npp8u>::initData(Npp8u* data, int count) {
    initDeviceRandom8u(data, count);
}

template<>
void NppiArithmeticBenchmark<Npp16u>::initData(Npp16u* data, int count) {
    initDeviceRandom16u(data, count);
}

template<>
void NppiArithmeticBenchmark<Npp16s>::initData(Npp16s* data, int count) {
    initDeviceRandom16s(data, count);
}

template<>
void NppiArithmeticBenchmark<Npp32f>::initData(Npp32f* data, int count) {
    initDeviceRandom32f(data, count);
}

// Benchmark for Add operation with two source images
template<typename T>
class NppiAddBenchmark : public BenchmarkBase {
public:
    using BenchFunc = std::function<NppStatus(const T*, int, const T*, int, T*, int, NppiSize)>;

    NppiAddBenchmark(const std::string& name, const std::string& dataType,
                     int channels, BenchFunc func)
        : BenchmarkBase(name), dataType_(dataType), channels_(channels), func_(func) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::vector<BenchmarkResult> results;
        auto sizes = getStandardImageSizes();

        for (const auto& size : sizes) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = dataType_;
            result.size = size.toString();
            result.channels = channels_;
            result.iterations = measureIterations;

            int pixelCount = size.width * size.height * channels_;
            size_t bytes = pixelCount * sizeof(T);
            int step = size.width * channels_ * sizeof(T);

            DeviceBuffer<T> src1(pixelCount);
            DeviceBuffer<T> src2(pixelCount);
            DeviceBuffer<T> dst(pixelCount);

            initData(src1.get(), pixelCount);
            initData(src2.get(), pixelCount);

            NppiSize roiSize = {size.width, size.height};

            // Warmup
            for (int i = 0; i < warmupIterations; ++i) {
                func_(src1.get(), step, src2.get(), step, dst.get(), step, roiSize);
            }
            cudaDeviceSynchronize();

            if (runBatchedMeasurement(
                    result, measureIterations,
                    [&]() { return func_(src1.get(), step, src2.get(), step, dst.get(), step, roiSize); })) {
                // Throughput: read src1 + src2 + write dst
                result.throughputGBps = computeThroughput(3 * bytes, result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }

private:
    std::string dataType_;
    int channels_;
    BenchFunc func_;

    void initData(T* data, int count);
};

template<>
void NppiAddBenchmark<Npp8u>::initData(Npp8u* data, int count) {
    initDeviceRandom8u(data, count);
}

template<>
void NppiAddBenchmark<Npp16u>::initData(Npp16u* data, int count) {
    initDeviceRandom16u(data, count);
}

template<>
void NppiAddBenchmark<Npp32f>::initData(Npp32f* data, int count) {
    initDeviceRandom32f(data, count);
}

// Register arithmetic benchmarks
class BenchAdd_8u_C1R : public NppiAddBenchmark<Npp8u> {
public:
    BenchAdd_8u_C1R() : NppiAddBenchmark("nppiAdd_8u_C1RSfs", "8u", 1,
        [](const Npp8u* s1, int step1, const Npp8u* s2, int step2,
           Npp8u* d, int dstep, NppiSize sz) {
            using Fn = decltype(&nppiAdd_8u_C1RSfs);
            static Fn fn = DynamicSymbol<Fn>("nppiAdd_8u_C1RSfs").get();
            if (fn == nullptr) return NPP_NOT_SUPPORTED_MODE_ERROR;
            return fn(s1, step1, s2, step2, d, dstep, sz, 0);
        }) {}
};
REGISTER_BENCHMARK(BenchAdd_8u_C1R)

class BenchAdd_8u_C3R : public NppiAddBenchmark<Npp8u> {
public:
    BenchAdd_8u_C3R() : NppiAddBenchmark("nppiAdd_8u_C3RSfs", "8u", 3,
        [](const Npp8u* s1, int step1, const Npp8u* s2, int step2,
           Npp8u* d, int dstep, NppiSize sz) {
            using Fn = decltype(&nppiAdd_8u_C3RSfs);
            static Fn fn = DynamicSymbol<Fn>("nppiAdd_8u_C3RSfs").get();
            if (fn == nullptr) return NPP_NOT_SUPPORTED_MODE_ERROR;
            return fn(s1, step1, s2, step2, d, dstep, sz, 0);
        }) {}
};
REGISTER_BENCHMARK(BenchAdd_8u_C3R)

class BenchAdd_8u_C4R : public NppiAddBenchmark<Npp8u> {
public:
    BenchAdd_8u_C4R() : NppiAddBenchmark("nppiAdd_8u_C4RSfs", "8u", 4,
        [](const Npp8u* s1, int step1, const Npp8u* s2, int step2,
           Npp8u* d, int dstep, NppiSize sz) {
            using Fn = decltype(&nppiAdd_8u_C4RSfs);
            static Fn fn = DynamicSymbol<Fn>("nppiAdd_8u_C4RSfs").get();
            if (fn == nullptr) return NPP_NOT_SUPPORTED_MODE_ERROR;
            return fn(s1, step1, s2, step2, d, dstep, sz, 0);
        }) {}
};
REGISTER_BENCHMARK(BenchAdd_8u_C4R)

class BenchAdd_16u_C1R : public NppiAddBenchmark<Npp16u> {
public:
    BenchAdd_16u_C1R() : NppiAddBenchmark("nppiAdd_16u_C1RSfs", "16u", 1,
        [](const Npp16u* s1, int step1, const Npp16u* s2, int step2,
           Npp16u* d, int dstep, NppiSize sz) {
            using Fn = decltype(&nppiAdd_16u_C1RSfs);
            static Fn fn = DynamicSymbol<Fn>("nppiAdd_16u_C1RSfs").get();
            if (fn == nullptr) return NPP_NOT_SUPPORTED_MODE_ERROR;
            return fn(s1, step1, s2, step2, d, dstep, sz, 0);
        }) {}
};
REGISTER_BENCHMARK(BenchAdd_16u_C1R)

class BenchAdd_32f_C1R : public NppiAddBenchmark<Npp32f> {
public:
    BenchAdd_32f_C1R() : NppiAddBenchmark("nppiAdd_32f_C1R", "32f", 1,
        [](const Npp32f* s1, int step1, const Npp32f* s2, int step2,
           Npp32f* d, int dstep, NppiSize sz) {
            using Fn = decltype(&nppiAdd_32f_C1R);
            static Fn fn = DynamicSymbol<Fn>("nppiAdd_32f_C1R").get();
            if (fn == nullptr) return NPP_NOT_SUPPORTED_MODE_ERROR;
            return fn(s1, step1, s2, step2, d, dstep, sz);
        }) {}
};
REGISTER_BENCHMARK(BenchAdd_32f_C1R)

// Abs benchmarks
class BenchAbs_16s_C1R : public NppiArithmeticBenchmark<Npp16s> {
public:
    BenchAbs_16s_C1R() : NppiArithmeticBenchmark("nppiAbs_16s_C1R", "16s", 1,
        [](const Npp16s* src, int srcStep, Npp16s* dst, int dstStep, NppiSize sz) {
            using Fn = decltype(&nppiAbs_16s_C1R);
            static Fn fn = DynamicSymbol<Fn>("nppiAbs_16s_C1R").get();
            if (fn == nullptr) return NPP_NOT_SUPPORTED_MODE_ERROR;
            return fn(src, srcStep, dst, dstStep, sz);
        }) {}
};
REGISTER_BENCHMARK(BenchAbs_16s_C1R)

class BenchAbs_32f_C1R : public NppiArithmeticBenchmark<Npp32f> {
public:
    BenchAbs_32f_C1R() : NppiArithmeticBenchmark("nppiAbs_32f_C1R", "32f", 1,
        [](const Npp32f* src, int srcStep, Npp32f* dst, int dstStep, NppiSize sz) {
            using Fn = decltype(&nppiAbs_32f_C1R);
            static Fn fn = DynamicSymbol<Fn>("nppiAbs_32f_C1R").get();
            if (fn == nullptr) return NPP_NOT_SUPPORTED_MODE_ERROR;
            return fn(src, srcStep, dst, dstStep, sz);
        }) {}
};
REGISTER_BENCHMARK(BenchAbs_32f_C1R)

// Sub benchmarks
class BenchSub_8u_C1R : public NppiAddBenchmark<Npp8u> {
public:
    BenchSub_8u_C1R() : NppiAddBenchmark("nppiSub_8u_C1RSfs", "8u", 1,
        [](const Npp8u* s1, int step1, const Npp8u* s2, int step2,
           Npp8u* d, int dstep, NppiSize sz) {
            using Fn = decltype(&nppiSub_8u_C1RSfs);
            static Fn fn = DynamicSymbol<Fn>("nppiSub_8u_C1RSfs").get();
            if (fn == nullptr) return NPP_NOT_SUPPORTED_MODE_ERROR;
            return fn(s1, step1, s2, step2, d, dstep, sz, 0);
        }) {}
};
REGISTER_BENCHMARK(BenchSub_8u_C1R)

class BenchSub_32f_C1R : public NppiAddBenchmark<Npp32f> {
public:
    BenchSub_32f_C1R() : NppiAddBenchmark("nppiSub_32f_C1R", "32f", 1,
        [](const Npp32f* s1, int step1, const Npp32f* s2, int step2,
           Npp32f* d, int dstep, NppiSize sz) {
            using Fn = decltype(&nppiSub_32f_C1R);
            static Fn fn = DynamicSymbol<Fn>("nppiSub_32f_C1R").get();
            if (fn == nullptr) return NPP_NOT_SUPPORTED_MODE_ERROR;
            return fn(s1, step1, s2, step2, d, dstep, sz);
        }) {}
};
REGISTER_BENCHMARK(BenchSub_32f_C1R)

// Mul benchmarks
class BenchMul_8u_C1R : public NppiAddBenchmark<Npp8u> {
public:
    BenchMul_8u_C1R() : NppiAddBenchmark("nppiMul_8u_C1RSfs", "8u", 1,
        [](const Npp8u* s1, int step1, const Npp8u* s2, int step2,
           Npp8u* d, int dstep, NppiSize sz) {
            using Fn = decltype(&nppiMul_8u_C1RSfs);
            static Fn fn = DynamicSymbol<Fn>("nppiMul_8u_C1RSfs").get();
            if (fn == nullptr) return NPP_NOT_SUPPORTED_MODE_ERROR;
            return fn(s1, step1, s2, step2, d, dstep, sz, 0);
        }) {}
};
REGISTER_BENCHMARK(BenchMul_8u_C1R)

class BenchMul_32f_C1R : public NppiAddBenchmark<Npp32f> {
public:
    BenchMul_32f_C1R() : NppiAddBenchmark("nppiMul_32f_C1R", "32f", 1,
        [](const Npp32f* s1, int step1, const Npp32f* s2, int step2,
           Npp32f* d, int dstep, NppiSize sz) {
            using Fn = decltype(&nppiMul_32f_C1R);
            static Fn fn = DynamicSymbol<Fn>("nppiMul_32f_C1R").get();
            if (fn == nullptr) return NPP_NOT_SUPPORTED_MODE_ERROR;
            return fn(s1, step1, s2, step2, d, dstep, sz);
        }) {}
};
REGISTER_BENCHMARK(BenchMul_32f_C1R)

// Div benchmarks
class BenchDiv_8u_C1R : public NppiAddBenchmark<Npp8u> {
public:
    BenchDiv_8u_C1R() : NppiAddBenchmark("nppiDiv_8u_C1RSfs", "8u", 1,
        [](const Npp8u* s1, int step1, const Npp8u* s2, int step2,
           Npp8u* d, int dstep, NppiSize sz) {
            using Fn = decltype(&nppiDiv_8u_C1RSfs);
            static Fn fn = DynamicSymbol<Fn>("nppiDiv_8u_C1RSfs").get();
            if (fn == nullptr) return NPP_NOT_SUPPORTED_MODE_ERROR;
            return fn(s1, step1, s2, step2, d, dstep, sz, 0);
        }) {}
};
REGISTER_BENCHMARK(BenchDiv_8u_C1R)

class BenchDiv_32f_C1R : public NppiAddBenchmark<Npp32f> {
public:
    BenchDiv_32f_C1R() : NppiAddBenchmark("nppiDiv_32f_C1R", "32f", 1,
        [](const Npp32f* s1, int step1, const Npp32f* s2, int step2,
           Npp32f* d, int dstep, NppiSize sz) {
            using Fn = decltype(&nppiDiv_32f_C1R);
            static Fn fn = DynamicSymbol<Fn>("nppiDiv_32f_C1R").get();
            if (fn == nullptr) return NPP_NOT_SUPPORTED_MODE_ERROR;
            return fn(s1, step1, s2, step2, d, dstep, sz);
        }) {}
};
REGISTER_BENCHMARK(BenchDiv_32f_C1R)

} // namespace npp_benchmark
