#include "../benchmark_base.h"
#include <npps.h>

namespace npp_benchmark {

extern void initDeviceRandom32f(Npp32f* data, int size);

namespace {

std::vector<BenchmarkResult> makeMissingSymbolResults(const std::string& functionName, const std::string& dataType,
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

} // namespace

// Signal Add benchmark
class BenchAdd_32f : public BenchmarkBase {
public:
    BenchAdd_32f() : BenchmarkBase("nppsAdd_32f") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppsAdd_32f);
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>("nppsAdd_32f").get(&symErr);
        if (fn == nullptr) {
            return makeMissingSymbolResults(name_, "32f", measureIterations, "Missing symbol nppsAdd_32f: " + symErr);
        }

        std::vector<BenchmarkResult> results;
        auto sizes = getStandardSignalSizes();

        for (const auto& sigSize : sizes) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = "32f";
            result.size = sigSize.toString();
            result.channels = 1;
            result.iterations = measureIterations;

            size_t bytes = sigSize.length * sizeof(Npp32f);

            DeviceBuffer<Npp32f> src1(sigSize.length);
            DeviceBuffer<Npp32f> src2(sigSize.length);
            DeviceBuffer<Npp32f> dst(sigSize.length);

            initDeviceRandom32f(src1.get(), sigSize.length);
            initDeviceRandom32f(src2.get(), sigSize.length);

            // Warmup
            for (int i = 0; i < warmupIterations; ++i) {
                fn(src1.get(), src2.get(), dst.get(), sigSize.length);
            }
            cudaDeviceSynchronize();

            if (runBatchedMeasurement(
                    result, measureIterations,
                    [&]() { return fn(src1.get(), src2.get(), dst.get(), sigSize.length); })) {
                result.throughputGBps = computeThroughput(3 * bytes, result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }
};
REGISTER_BENCHMARK(BenchAdd_32f)

// Signal Add in-place benchmark
class BenchAdd_32f_I : public BenchmarkBase {
public:
    BenchAdd_32f_I() : BenchmarkBase("nppsAdd_32f_I") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppsAdd_32f_I);
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>("nppsAdd_32f_I").get(&symErr);
        if (fn == nullptr) {
            return makeMissingSymbolResults(name_, "32f", measureIterations, "Missing symbol nppsAdd_32f_I: " + symErr);
        }

        std::vector<BenchmarkResult> results;
        auto sizes = getStandardSignalSizes();

        for (const auto& sigSize : sizes) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = "32f";
            result.size = sigSize.toString();
            result.channels = 1;
            result.iterations = measureIterations;

            size_t bytes = sigSize.length * sizeof(Npp32f);

            DeviceBuffer<Npp32f> src(sigSize.length);
            DeviceBuffer<Npp32f> srcDst(sigSize.length);

            initDeviceRandom32f(src.get(), sigSize.length);
            initDeviceRandom32f(srcDst.get(), sigSize.length);

            // Warmup
            for (int i = 0; i < warmupIterations; ++i) {
                fn(src.get(), srcDst.get(), sigSize.length);
            }
            cudaDeviceSynchronize();

            if (runBatchedMeasurement(
                    result, measureIterations,
                    [&]() { return fn(src.get(), srcDst.get(), sigSize.length); })) {
                result.throughputGBps = computeThroughput(2 * bytes, result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }
};
REGISTER_BENCHMARK(BenchAdd_32f_I)

// Signal Sum benchmark
class BenchSum_32f : public BenchmarkBase {
public:
    BenchSum_32f() : BenchmarkBase("nppsSum_32f") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using FnGetBuffer = decltype(&nppsSumGetBufferSize_32f);
        using Fn = decltype(&nppsSum_32f);
        std::string symErr;
        FnGetBuffer fnGetBuffer = DynamicSymbol<FnGetBuffer>("nppsSumGetBufferSize_32f").get(&symErr);
        if (fnGetBuffer == nullptr) {
            return makeMissingSymbolResults(name_, "32f", measureIterations,
                                            "Missing symbol nppsSumGetBufferSize_32f: " + symErr);
        }
        Fn fn = DynamicSymbol<Fn>("nppsSum_32f").get(&symErr);
        if (fn == nullptr) {
            return makeMissingSymbolResults(name_, "32f", measureIterations, "Missing symbol nppsSum_32f: " + symErr);
        }

        std::vector<BenchmarkResult> results;
        auto sizes = getStandardSignalSizes();

        for (const auto& sigSize : sizes) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = "32f";
            result.size = sigSize.toString();
            result.channels = 1;
            result.iterations = measureIterations;

            size_t bytes = sigSize.length * sizeof(Npp32f);

            DeviceBuffer<Npp32f> src(sigSize.length);
            DeviceBuffer<Npp32f> sum(1);

            initDeviceRandom32f(src.get(), sigSize.length);

            // Get scratch buffer size
            // int bufferSize = 0;
	  #ifdef USE_NVIDIA_NPP_TESTS
	    size_t bufferSize = 0;
	  #else
	    int bufferSize = 0;
	  #endif
            fnGetBuffer(sigSize.length, &bufferSize);
            DeviceBuffer<Npp8u> scratch(bufferSize);

            // Warmup
            for (int i = 0; i < warmupIterations; ++i) {
                fn(src.get(), sigSize.length, sum.get(), scratch.get());
            }
            cudaDeviceSynchronize();

            if (runBatchedMeasurement(
                    result, measureIterations,
                    [&]() { return fn(src.get(), sigSize.length, sum.get(), scratch.get()); })) {
                result.throughputGBps = computeThroughput(bytes, result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }
};
REGISTER_BENCHMARK(BenchSum_32f)

// Signal Set benchmark
class BenchSet_32f : public BenchmarkBase {
public:
    BenchSet_32f() : BenchmarkBase("nppsSet_32f") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppsSet_32f);
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>("nppsSet_32f").get(&symErr);
        if (fn == nullptr) {
            return makeMissingSymbolResults(name_, "32f", measureIterations, "Missing symbol nppsSet_32f: " + symErr);
        }

        std::vector<BenchmarkResult> results;
        auto sizes = getStandardSignalSizes();

        for (const auto& sigSize : sizes) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = "32f";
            result.size = sigSize.toString();
            result.channels = 1;
            result.iterations = measureIterations;

            size_t bytes = sigSize.length * sizeof(Npp32f);

            DeviceBuffer<Npp32f> dst(sigSize.length);
            Npp32f value = 1.0f;

            // Warmup
            for (int i = 0; i < warmupIterations; ++i) {
                fn(value, dst.get(), sigSize.length);
            }
            cudaDeviceSynchronize();

            if (runBatchedMeasurement(
                    result, measureIterations,
                    [&]() { return fn(value, dst.get(), sigSize.length); })) {
                result.throughputGBps = computeThroughput(bytes, result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }
};
REGISTER_BENCHMARK(BenchSet_32f)

} // namespace npp_benchmark
