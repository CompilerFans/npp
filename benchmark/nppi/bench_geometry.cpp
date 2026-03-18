#include "../benchmark_base.h"
#include <nppi_geometry_transforms.h>
#include <type_traits>

namespace npp_benchmark {

extern void initDeviceRandom8u(Npp8u* data, int size);
extern void initDeviceRandom16u(Npp16u* data, int size);
extern void initDeviceRandom16s(Npp16s* data, int size);
extern void initDeviceRandom32f(Npp32f* data, int size);

namespace {

std::vector<ImageSize> getRequestedImageSizes() {
    return {
        {1280, 720},
        {1280, 1024},
        {1920, 1080},
    };
}

struct ResizeScenario {
    ImageSize src;
    ImageSize dst;
    std::string desc;
};

std::vector<ResizeScenario> makeResizeScenarios() {
    std::vector<ResizeScenario> out;
    out.reserve(getRequestedImageSizes().size() * 2);
    for (const auto& s : getRequestedImageSizes()) {
        out.push_back({s, {s.width / 2, s.height / 2}, s.toString() + "->" + std::to_string(s.width / 2) + "x" +
                                                           std::to_string(s.height / 2)});
        out.push_back({s, {s.width * 2, s.height * 2}, s.toString() + "->" + std::to_string(s.width * 2) + "x" +
                                                           std::to_string(s.height * 2)});
    }
    return out;
}

template<typename T>
void initRandom(T* data, int count) {
    copyDeterministicHostDataToDevice(data, static_cast<size_t>(count));
}

template<>
void initRandom<Npp8u>(Npp8u* data, int count) {
    initDeviceRandom8u(data, count);
}

template<>
void initRandom<Npp16u>(Npp16u* data, int count) {
    initDeviceRandom16u(data, count);
}

template<>
void initRandom<Npp16s>(Npp16s* data, int count) {
    initDeviceRandom16s(data, count);
}

template<>
void initRandom<Npp32f>(Npp32f* data, int count) {
    initDeviceRandom32f(data, count);
}

} // namespace

// Resize benchmark (packed pixel variants)
template<typename T, int Channels, typename Fn>
class NppiResizeBenchmark : public BenchmarkBase {
public:
    NppiResizeBenchmark(const std::string& benchName, const char* symbolName, const char* dataType)
        : BenchmarkBase(benchName), symbolName_(symbolName), dataType_(dataType) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::vector<BenchmarkResult> results;
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>(symbolName_).get(&symErr);
        if (fn == nullptr) {
            for (const auto& scenario : makeResizeScenarios()) {
                BenchmarkResult result;
                result.functionName = name_;
                result.dataType = dataType_;
                result.size = scenario.desc;
                result.channels = Channels;
                result.iterations = measureIterations;
                result.success = false;
                result.errorMessage = std::string("Missing symbol ") + symbolName_ + ": " + symErr;
                results.push_back(result);
            }
            return results;
        }

        auto scenarios = makeResizeScenarios();
        const int interpolation = NPPI_INTER_LINEAR;

        auto runScenario = [&](const ResizeScenario& scenario, auto&& call) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = dataType_;
            result.size = scenario.desc;
            result.channels = Channels;
            result.iterations = measureIterations;

            int srcPixels = scenario.src.width * scenario.src.height * Channels;
            int dstPixels = scenario.dst.width * scenario.dst.height * Channels;
            int srcStep = scenario.src.width * Channels * sizeof(T);
            int dstStep = scenario.dst.width * Channels * sizeof(T);

            DeviceBuffer<T> src(srcPixels);
            DeviceBuffer<T> dst(dstPixels);

            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                return;
            }

            initRandom<T>(src.get(), srcPixels);

            NppiSize srcSize = {scenario.src.width, scenario.src.height};
            NppiRect srcROI = {0, 0, scenario.src.width, scenario.src.height};
            NppiSize dstSize = {scenario.dst.width, scenario.dst.height};
            NppiRect dstROI = {0, 0, scenario.dst.width, scenario.dst.height};

            for (int i = 0; i < warmupIterations; ++i) {
                call(src.get(), srcStep, srcSize, srcROI, dst.get(), dstStep, dstSize, dstROI, interpolation);
            }
            cudaDeviceSynchronize();

            if (runBatchedMeasurement(
                    result, measureIterations,
                    [&]() {
                        return call(src.get(), srcStep, srcSize, srcROI, dst.get(), dstStep, dstSize, dstROI,
                                    interpolation);
                    })) {
                size_t totalBytes = srcPixels * sizeof(T) + dstPixels * sizeof(T);
                result.throughputGBps = computeThroughput(totalBytes, result.avgTimeMs);
            }

            results.push_back(result);
        };

        for (const auto& scenario : scenarios) {
            runScenario(scenario, [&](const T* src, int srcStep, NppiSize srcSize, NppiRect srcROI, T* dst, int dstStep,
                                      NppiSize dstSize, NppiRect dstROI, int interp) {
                return fn(src, srcStep, srcSize, srcROI, dst, dstStep, dstSize, dstROI, interp);
            });
        }

        return results;
    }

protected:
    const char* symbolName_;
    const char* dataType_;
};

// Concrete resize benchmarks (implemented variants from coverage.csv)
class BenchResize_8u_C1R : public NppiResizeBenchmark<Npp8u, 1, decltype(&nppiResize_8u_C1R)> {
public:
    BenchResize_8u_C1R() : NppiResizeBenchmark("nppiResize_8u_C1R", "nppiResize_8u_C1R", "8u") {}
};
REGISTER_BENCHMARK(BenchResize_8u_C1R)

class BenchResize_8u_C3R : public NppiResizeBenchmark<Npp8u, 3, decltype(&nppiResize_8u_C3R)> {
public:
    BenchResize_8u_C3R() : NppiResizeBenchmark("nppiResize_8u_C3R", "nppiResize_8u_C3R", "8u") {}
};
REGISTER_BENCHMARK(BenchResize_8u_C3R)

class BenchResize_8u_C4R : public NppiResizeBenchmark<Npp8u, 4, decltype(&nppiResize_8u_C4R)> {
public:
    BenchResize_8u_C4R() : NppiResizeBenchmark("nppiResize_8u_C4R", "nppiResize_8u_C4R", "8u") {}
};
REGISTER_BENCHMARK(BenchResize_8u_C4R)

class BenchResize_8u_AC4R : public NppiResizeBenchmark<Npp8u, 4, decltype(&nppiResize_8u_AC4R)> {
public:
    BenchResize_8u_AC4R() : NppiResizeBenchmark("nppiResize_8u_AC4R", "nppiResize_8u_AC4R", "8u") {}
};
REGISTER_BENCHMARK(BenchResize_8u_AC4R)

class BenchResize_16u_C1R : public NppiResizeBenchmark<Npp16u, 1, decltype(&nppiResize_16u_C1R)> {
public:
    BenchResize_16u_C1R() : NppiResizeBenchmark("nppiResize_16u_C1R", "nppiResize_16u_C1R", "16u") {}
};
REGISTER_BENCHMARK(BenchResize_16u_C1R)

class BenchResize_16u_C3R : public NppiResizeBenchmark<Npp16u, 3, decltype(&nppiResize_16u_C3R)> {
public:
    BenchResize_16u_C3R() : NppiResizeBenchmark("nppiResize_16u_C3R", "nppiResize_16u_C3R", "16u") {}
};
REGISTER_BENCHMARK(BenchResize_16u_C3R)

class BenchResize_16u_C4R : public NppiResizeBenchmark<Npp16u, 4, decltype(&nppiResize_16u_C4R)> {
public:
    BenchResize_16u_C4R() : NppiResizeBenchmark("nppiResize_16u_C4R", "nppiResize_16u_C4R", "16u") {}
};
REGISTER_BENCHMARK(BenchResize_16u_C4R)

class BenchResize_16u_AC4R : public NppiResizeBenchmark<Npp16u, 4, decltype(&nppiResize_16u_AC4R)> {
public:
    BenchResize_16u_AC4R() : NppiResizeBenchmark("nppiResize_16u_AC4R", "nppiResize_16u_AC4R", "16u") {}
};
REGISTER_BENCHMARK(BenchResize_16u_AC4R)

class BenchResize_16s_C1R : public NppiResizeBenchmark<Npp16s, 1, decltype(&nppiResize_16s_C1R)> {
public:
    BenchResize_16s_C1R() : NppiResizeBenchmark("nppiResize_16s_C1R", "nppiResize_16s_C1R", "16s") {}
};
REGISTER_BENCHMARK(BenchResize_16s_C1R)

class BenchResize_16s_C3R : public NppiResizeBenchmark<Npp16s, 3, decltype(&nppiResize_16s_C3R)> {
public:
    BenchResize_16s_C3R() : NppiResizeBenchmark("nppiResize_16s_C3R", "nppiResize_16s_C3R", "16s") {}
};
REGISTER_BENCHMARK(BenchResize_16s_C3R)

class BenchResize_16s_C4R : public NppiResizeBenchmark<Npp16s, 4, decltype(&nppiResize_16s_C4R)> {
public:
    BenchResize_16s_C4R() : NppiResizeBenchmark("nppiResize_16s_C4R", "nppiResize_16s_C4R", "16s") {}
};
REGISTER_BENCHMARK(BenchResize_16s_C4R)

class BenchResize_16s_AC4R : public NppiResizeBenchmark<Npp16s, 4, decltype(&nppiResize_16s_AC4R)> {
public:
    BenchResize_16s_AC4R() : NppiResizeBenchmark("nppiResize_16s_AC4R", "nppiResize_16s_AC4R", "16s") {}
};
REGISTER_BENCHMARK(BenchResize_16s_AC4R)

class BenchResize_32f_C1R : public NppiResizeBenchmark<Npp32f, 1, decltype(&nppiResize_32f_C1R)> {
public:
    BenchResize_32f_C1R() : NppiResizeBenchmark("nppiResize_32f_C1R", "nppiResize_32f_C1R", "32f") {}
};
REGISTER_BENCHMARK(BenchResize_32f_C1R)

class BenchResize_32f_C3R : public NppiResizeBenchmark<Npp32f, 3, decltype(&nppiResize_32f_C3R)> {
public:
    BenchResize_32f_C3R() : NppiResizeBenchmark("nppiResize_32f_C3R", "nppiResize_32f_C3R", "32f") {}
};
REGISTER_BENCHMARK(BenchResize_32f_C3R)

class BenchResize_32f_C4R : public NppiResizeBenchmark<Npp32f, 4, decltype(&nppiResize_32f_C4R)> {
public:
    BenchResize_32f_C4R() : NppiResizeBenchmark("nppiResize_32f_C4R", "nppiResize_32f_C4R", "32f") {}
};
REGISTER_BENCHMARK(BenchResize_32f_C4R)

class BenchResize_32f_AC4R : public NppiResizeBenchmark<Npp32f, 4, decltype(&nppiResize_32f_AC4R)> {
public:
    BenchResize_32f_AC4R() : NppiResizeBenchmark("nppiResize_32f_AC4R", "nppiResize_32f_AC4R", "32f") {}
};
REGISTER_BENCHMARK(BenchResize_32f_AC4R)

// Mirror benchmark
class BenchMirror_8u_C1R : public BenchmarkBase {
public:
    BenchMirror_8u_C1R() : BenchmarkBase("nppiMirror_8u_C1R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiMirror_8u_C1R);
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>("nppiMirror_8u_C1R").get(&symErr);
        if (fn == nullptr) {
            std::vector<BenchmarkResult> results;
            for (const auto& size : getStandardImageSizes()) {
                BenchmarkResult result;
                result.functionName = name_;
                result.dataType = "8u";
                result.size = size.toString();
                result.channels = 1;
                result.iterations = measureIterations;
                result.success = false;
                result.errorMessage = "Missing symbol nppiMirror_8u_C1R: " + symErr;
                results.push_back(result);
            }
            return results;
        }

        std::vector<BenchmarkResult> results;
        auto sizes = getStandardImageSizes();

        for (const auto& size : sizes) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = "8u";
            result.size = size.toString();
            result.channels = 1;
            result.iterations = measureIterations;

            int pixelCount = size.width * size.height;
            int step = size.width * sizeof(Npp8u);

            DeviceBuffer<Npp8u> src(pixelCount);
            DeviceBuffer<Npp8u> dst(pixelCount);

            initDeviceRandom8u(src.get(), pixelCount);

            NppiSize roiSize = {size.width, size.height};

            // Warmup
            for (int i = 0; i < warmupIterations; ++i) {
                fn(src.get(), step, dst.get(), step, roiSize, NPP_HORIZONTAL_AXIS);
            }
            cudaDeviceSynchronize();

            if (runBatchedMeasurement(
                    result, measureIterations,
                    [&]() { return fn(src.get(), step, dst.get(), step, roiSize, NPP_HORIZONTAL_AXIS); })) {
                result.throughputGBps = computeThroughput(2 * pixelCount * sizeof(Npp8u), result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }
};
REGISTER_BENCHMARK(BenchMirror_8u_C1R)

} // namespace npp_benchmark
