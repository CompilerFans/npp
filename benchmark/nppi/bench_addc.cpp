/**
 * @file bench_addc.cpp
 * @brief Performance benchmarks for nppi AddC APIs
 *
 * Generated via:
 *   ./benchmark/scripts/new_benchmark.sh nppi addc nppi_arithmetic_and_logical_operations.h
 */

#include "../benchmark_base.h"
#include <nppi_arithmetic_and_logical_operations.h>

namespace npp_benchmark {

extern void initDeviceRandom8u(Npp8u* data, int size);
extern void initDeviceRandom16u(Npp16u* data, int size);
extern void initDeviceRandom32f(Npp32f* data, int size);

namespace {

template<typename T>
void initData(T* data, int count);

template<>
void initData<Npp8u>(Npp8u* data, int count) {
    initDeviceRandom8u(data, count);
}

template<>
void initData<Npp16u>(Npp16u* data, int count) {
    initDeviceRandom16u(data, count);
}

template<>
void initData<Npp32f>(Npp32f* data, int count) {
    initDeviceRandom32f(data, count);
}

template<typename Fn>
std::vector<BenchmarkResult> missingSymbolResults(const std::string& benchName, const std::string& dataType,
                                                  int channels, int iterations, const std::string& symbolName,
                                                  const std::string& symErr) {
    std::vector<BenchmarkResult> results;
    for (const auto& imgSize : getStandardImageSizes()) {
        BenchmarkResult r;
        r.functionName = benchName;
        r.dataType = dataType;
        r.size = imgSize.toString();
        r.channels = channels;
        r.iterations = iterations;
        r.success = false;
        r.errorMessage = "Missing symbol " + symbolName + ": " + symErr;
        results.push_back(r);
    }
    return results;
}

template<typename T, int Channels>
int elemCount(const ImageSize& img) {
    return img.width * img.height * Channels;
}

template<typename T, int Channels>
int stepBytes(const ImageSize& img) {
    return img.width * Channels * static_cast<int>(sizeof(T));
}

template<typename T, int Channels>
size_t totalBytes(const ImageSize& img) {
    return static_cast<size_t>(img.width) * static_cast<size_t>(img.height) * static_cast<size_t>(Channels) *
           sizeof(T);
}

template<typename T, int Channels, typename Call>
std::vector<BenchmarkResult> runPerSize(const std::string& benchName, const std::string& dataType, int warmupIterations,
                                        int measureIterations, Call&& call) {
    auto computeThroughput = [](size_t bytes, double timeMs) -> double {
        if (timeMs <= 0.0) return 0.0;
        return (bytes / 1e9) / (timeMs / 1000.0);
    };

    std::vector<BenchmarkResult> results;
    for (const auto& imgSize : getStandardImageSizes()) {
        BenchmarkResult result;
        result.functionName = benchName;
        result.dataType = dataType;
        result.size = imgSize.toString();
        result.channels = Channels;
        result.iterations = measureIterations;

        const int n = elemCount<T, Channels>(imgSize);
        const int step = stepBytes<T, Channels>(imgSize);
        const size_t bytes = totalBytes<T, Channels>(imgSize);
        const NppiSize roiSize = {imgSize.width, imgSize.height};

        DeviceBuffer<T> src(n);
        DeviceBuffer<T> dst(n);
        initData<T>(src.get(), n);

        for (int i = 0; i < warmupIterations; ++i) {
            (void)call(src.get(), step, dst.get(), step, roiSize);
        }
        cudaDeviceSynchronize();

        if (runBatchedMeasurement(result, measureIterations,
                                  [&]() { return call(src.get(), step, dst.get(), step, roiSize); })) {
            result.throughputGBps = computeThroughput(2 * bytes, result.avgTimeMs);
        }

        results.push_back(result);
    }
    return results;
}

} // namespace

class BenchAddC_8u_C1RSfs : public BenchmarkBase {
public:
    BenchAddC_8u_C1RSfs() : BenchmarkBase("nppiAddC_8u_C1RSfs") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiAddC_8u_C1RSfs);
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>("nppiAddC_8u_C1RSfs").get(&symErr);
        if (fn == nullptr) {
            return missingSymbolResults<Fn>(name_, "8u", 1, measureIterations, "nppiAddC_8u_C1RSfs", symErr);
        }

        const Npp8u constant = 7;
        const int scaleFactor = 0;
        return runPerSize<Npp8u, 1>(
            name_, "8u", warmupIterations, measureIterations,
            [&](const Npp8u* src, int srcStep, Npp8u* dst, int dstStep, NppiSize roi) {
                return fn(src, srcStep, constant, dst, dstStep, roi, scaleFactor);
            });
    }
};
REGISTER_BENCHMARK(BenchAddC_8u_C1RSfs)

class BenchAddC_8u_C3RSfs : public BenchmarkBase {
public:
    BenchAddC_8u_C3RSfs() : BenchmarkBase("nppiAddC_8u_C3RSfs") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiAddC_8u_C3RSfs);
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>("nppiAddC_8u_C3RSfs").get(&symErr);
        if (fn == nullptr) {
            return missingSymbolResults<Fn>(name_, "8u", 3, measureIterations, "nppiAddC_8u_C3RSfs", symErr);
        }

        const Npp8u constants[3] = {7, 3, 1};
        const int scaleFactor = 0;
        return runPerSize<Npp8u, 3>(
            name_, "8u", warmupIterations, measureIterations,
            [&](const Npp8u* src, int srcStep, Npp8u* dst, int dstStep, NppiSize roi) {
                return fn(src, srcStep, constants, dst, dstStep, roi, scaleFactor);
            });
    }
};
REGISTER_BENCHMARK(BenchAddC_8u_C3RSfs)

class BenchAddC_16u_C1RSfs : public BenchmarkBase {
public:
    BenchAddC_16u_C1RSfs() : BenchmarkBase("nppiAddC_16u_C1RSfs") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiAddC_16u_C1RSfs);
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>("nppiAddC_16u_C1RSfs").get(&symErr);
        if (fn == nullptr) {
            return missingSymbolResults<Fn>(name_, "16u", 1, measureIterations, "nppiAddC_16u_C1RSfs", symErr);
        }

        const Npp16u constant = 17;
        const int scaleFactor = 0;
        return runPerSize<Npp16u, 1>(
            name_, "16u", warmupIterations, measureIterations,
            [&](const Npp16u* src, int srcStep, Npp16u* dst, int dstStep, NppiSize roi) {
                return fn(src, srcStep, constant, dst, dstStep, roi, scaleFactor);
            });
    }
};
REGISTER_BENCHMARK(BenchAddC_16u_C1RSfs)

class BenchAddC_32f_C1R : public BenchmarkBase {
public:
    BenchAddC_32f_C1R() : BenchmarkBase("nppiAddC_32f_C1R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiAddC_32f_C1R);
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>("nppiAddC_32f_C1R").get(&symErr);
        if (fn == nullptr) {
            return missingSymbolResults<Fn>(name_, "32f", 1, measureIterations, "nppiAddC_32f_C1R", symErr);
        }

        const Npp32f constant = 0.25f;
        return runPerSize<Npp32f, 1>(
            name_, "32f", warmupIterations, measureIterations,
            [&](const Npp32f* src, int srcStep, Npp32f* dst, int dstStep, NppiSize roi) {
                return fn(src, srcStep, constant, dst, dstStep, roi);
            });
    }
};
REGISTER_BENCHMARK(BenchAddC_32f_C1R)

class BenchAddC_32f_C3R : public BenchmarkBase {
public:
    BenchAddC_32f_C3R() : BenchmarkBase("nppiAddC_32f_C3R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiAddC_32f_C3R);
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>("nppiAddC_32f_C3R").get(&symErr);
        if (fn == nullptr) {
            return missingSymbolResults<Fn>(name_, "32f", 3, measureIterations, "nppiAddC_32f_C3R", symErr);
        }

        const Npp32f constants[3] = {0.25f, 0.5f, 0.75f};
        return runPerSize<Npp32f, 3>(
            name_, "32f", warmupIterations, measureIterations,
            [&](const Npp32f* src, int srcStep, Npp32f* dst, int dstStep, NppiSize roi) {
                return fn(src, srcStep, constants, dst, dstStep, roi);
            });
    }
};
REGISTER_BENCHMARK(BenchAddC_32f_C3R)

} // namespace npp_benchmark
