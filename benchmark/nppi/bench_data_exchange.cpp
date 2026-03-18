#include "../benchmark_base.h"
#include <nppi_data_exchange_and_initialization.h>
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

// Copy benchmark (packed pixel CxR)
template<typename T, int Channels, typename Fn>
class NppiCopyBenchmark : public BenchmarkBase {
public:
    NppiCopyBenchmark(const std::string& benchName, const char* symbolName, const char* dataType)
        : BenchmarkBase(benchName), dataType_(dataType), symbolName_(symbolName) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>(symbolName_).get(&symErr);
        if (fn == nullptr) {
            std::vector<BenchmarkResult> results;
            for (const auto& imgSize : getRequestedImageSizes()) {
                BenchmarkResult result;
                result.functionName = name_;
                result.dataType = dataType_;
                result.size = imgSize.toString();
                result.channels = Channels;
                result.iterations = measureIterations;
                result.success = false;
                result.errorMessage = std::string("Missing symbol ") + symbolName_ + ": " + symErr;
                results.push_back(result);
            }
            return results;
        }

        std::vector<BenchmarkResult> results;
        const auto sizes = getRequestedImageSizes();
        auto runLoops = [&](auto&& call) {
            for (const auto& imgSize : sizes) {
                BenchmarkResult result;
                result.functionName = name_;
                result.dataType = dataType_;
                result.size = imgSize.toString();
                result.channels = Channels;
                result.iterations = measureIterations;

                int pixelCount = imgSize.width * imgSize.height * Channels;
                size_t bytes = pixelCount * sizeof(T);
                int step = imgSize.width * Channels * sizeof(T);

                DeviceBuffer<T> src(pixelCount);
                DeviceBuffer<T> dst(pixelCount);

                if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(dst, result, "dst")) {
                    results.push_back(result);
                    continue;
                }

                initRandom<T>(src.get(), pixelCount);

                NppiSize roiSize = {imgSize.width, imgSize.height};

                // Warmup
                for (int i = 0; i < warmupIterations; ++i) {
                    call(src.get(), step, dst.get(), step, roiSize);
                }
                cudaDeviceSynchronize();

                if (runBatchedMeasurement(result, measureIterations,
                                          [&]() { return call(src.get(), step, dst.get(), step, roiSize); })) {
                    result.throughputGBps = computeThroughput(2 * bytes, result.avgTimeMs);
                }

                results.push_back(result);
            }
        };

        runLoops([&](const T* src, int srcStep, T* dst, int dstStep, NppiSize roiSize) {
            return fn(src, srcStep, dst, dstStep, roiSize);
        });

        return results;
    }

protected:
    std::string dataType_;
    const char* symbolName_;
};

// Concrete copy benchmarks
class BenchCopy_8u_C1R : public NppiCopyBenchmark<Npp8u, 1, decltype(&nppiCopy_8u_C1R)> {
public:
    BenchCopy_8u_C1R() : NppiCopyBenchmark("nppiCopy_8u_C1R", "nppiCopy_8u_C1R", "8u") {}
};
REGISTER_BENCHMARK(BenchCopy_8u_C1R)

class BenchCopy_8u_C3R : public NppiCopyBenchmark<Npp8u, 3, decltype(&nppiCopy_8u_C3R)> {
public:
    BenchCopy_8u_C3R() : NppiCopyBenchmark("nppiCopy_8u_C3R", "nppiCopy_8u_C3R", "8u") {}
};
REGISTER_BENCHMARK(BenchCopy_8u_C3R)

class BenchCopy_8u_C4R : public NppiCopyBenchmark<Npp8u, 4, decltype(&nppiCopy_8u_C4R)> {
public:
    BenchCopy_8u_C4R() : NppiCopyBenchmark("nppiCopy_8u_C4R", "nppiCopy_8u_C4R", "8u") {}
};
REGISTER_BENCHMARK(BenchCopy_8u_C4R)

class BenchCopy_16u_C1R : public NppiCopyBenchmark<Npp16u, 1, decltype(&nppiCopy_16u_C1R)> {
public:
    BenchCopy_16u_C1R() : NppiCopyBenchmark("nppiCopy_16u_C1R", "nppiCopy_16u_C1R", "16u") {}
};
REGISTER_BENCHMARK(BenchCopy_16u_C1R)

class BenchCopy_16u_C3R : public NppiCopyBenchmark<Npp16u, 3, decltype(&nppiCopy_16u_C3R)> {
public:
    BenchCopy_16u_C3R() : NppiCopyBenchmark("nppiCopy_16u_C3R", "nppiCopy_16u_C3R", "16u") {}
};
REGISTER_BENCHMARK(BenchCopy_16u_C3R)

class BenchCopy_16u_C4R : public NppiCopyBenchmark<Npp16u, 4, decltype(&nppiCopy_16u_C4R)> {
public:
    BenchCopy_16u_C4R() : NppiCopyBenchmark("nppiCopy_16u_C4R", "nppiCopy_16u_C4R", "16u") {}
};
REGISTER_BENCHMARK(BenchCopy_16u_C4R)

class BenchCopy_16s_C1R : public NppiCopyBenchmark<Npp16s, 1, decltype(&nppiCopy_16s_C1R)> {
public:
    BenchCopy_16s_C1R() : NppiCopyBenchmark("nppiCopy_16s_C1R", "nppiCopy_16s_C1R", "16s") {}
};
REGISTER_BENCHMARK(BenchCopy_16s_C1R)

class BenchCopy_16s_C3R : public NppiCopyBenchmark<Npp16s, 3, decltype(&nppiCopy_16s_C3R)> {
public:
    BenchCopy_16s_C3R() : NppiCopyBenchmark("nppiCopy_16s_C3R", "nppiCopy_16s_C3R", "16s") {}
};
REGISTER_BENCHMARK(BenchCopy_16s_C3R)

class BenchCopy_16s_C4R : public NppiCopyBenchmark<Npp16s, 4, decltype(&nppiCopy_16s_C4R)> {
public:
    BenchCopy_16s_C4R() : NppiCopyBenchmark("nppiCopy_16s_C4R", "nppiCopy_16s_C4R", "16s") {}
};
REGISTER_BENCHMARK(BenchCopy_16s_C4R)

class BenchCopy_32s_C1R : public NppiCopyBenchmark<Npp32s, 1, decltype(&nppiCopy_32s_C1R)> {
public:
    BenchCopy_32s_C1R() : NppiCopyBenchmark("nppiCopy_32s_C1R", "nppiCopy_32s_C1R", "32s") {}
};
REGISTER_BENCHMARK(BenchCopy_32s_C1R)

class BenchCopy_32s_C3R : public NppiCopyBenchmark<Npp32s, 3, decltype(&nppiCopy_32s_C3R)> {
public:
    BenchCopy_32s_C3R() : NppiCopyBenchmark("nppiCopy_32s_C3R", "nppiCopy_32s_C3R", "32s") {}
};
REGISTER_BENCHMARK(BenchCopy_32s_C3R)

class BenchCopy_32s_C4R : public NppiCopyBenchmark<Npp32s, 4, decltype(&nppiCopy_32s_C4R)> {
public:
    BenchCopy_32s_C4R() : NppiCopyBenchmark("nppiCopy_32s_C4R", "nppiCopy_32s_C4R", "32s") {}
};
REGISTER_BENCHMARK(BenchCopy_32s_C4R)

class BenchCopy_32f_C1R : public NppiCopyBenchmark<Npp32f, 1, decltype(&nppiCopy_32f_C1R)> {
public:
    BenchCopy_32f_C1R() : NppiCopyBenchmark("nppiCopy_32f_C1R", "nppiCopy_32f_C1R", "32f") {}
};
REGISTER_BENCHMARK(BenchCopy_32f_C1R)

class BenchCopy_32f_C3R : public NppiCopyBenchmark<Npp32f, 3, decltype(&nppiCopy_32f_C3R)> {
public:
    BenchCopy_32f_C3R() : NppiCopyBenchmark("nppiCopy_32f_C3R", "nppiCopy_32f_C3R", "32f") {}
};
REGISTER_BENCHMARK(BenchCopy_32f_C3R)

class BenchCopy_32f_C4R : public NppiCopyBenchmark<Npp32f, 4, decltype(&nppiCopy_32f_C4R)> {
public:
    BenchCopy_32f_C4R() : NppiCopyBenchmark("nppiCopy_32f_C4R", "nppiCopy_32f_C4R", "32f") {}
};
REGISTER_BENCHMARK(BenchCopy_32f_C4R)

// Copy benchmark (packed <-> planar)
template<typename T, int Channels, typename Fn>
class NppiCopyCXPXRBenchmark : public BenchmarkBase {
public:
    NppiCopyCXPXRBenchmark(const std::string& benchName, const char* symbolName, const char* dataType)
        : BenchmarkBase(benchName), dataType_(dataType), symbolName_(symbolName) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>(symbolName_).get(&symErr);

        std::vector<BenchmarkResult> results;
        for (const auto& imgSize : getRequestedImageSizes()) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = dataType_;
            result.size = imgSize.toString();
            result.channels = Channels;
            result.iterations = measureIterations;

            if (fn == nullptr) {
                result.success = false;
                result.errorMessage = std::string("Missing symbol ") + symbolName_ + ": " + symErr;
                results.push_back(result);
                continue;
            }

            const int width = imgSize.width;
            const int height = imgSize.height;
            const int pixels = width * height;
            const int packedStep = width * Channels * static_cast<int>(sizeof(T));
            const int planarStep = width * static_cast<int>(sizeof(T));

            DeviceBuffer<T> packed(static_cast<size_t>(pixels) * static_cast<size_t>(Channels));
            DeviceBuffer<T> planar0(pixels);
            DeviceBuffer<T> planar1(pixels);
            DeviceBuffer<T> planar2(pixels);
            DeviceBuffer<T> planar3(pixels);

            if (!validateDeviceBuffer(packed, result, "packed") || !validateDeviceBuffer(planar0, result, "planar0") ||
                !validateDeviceBuffer(planar1, result, "planar1") || !validateDeviceBuffer(planar2, result, "planar2") ||
                !validateDeviceBuffer(planar3, result, "planar3")) {
                results.push_back(result);
                continue;
            }

            initRandom<T>(packed.get(), pixels * Channels);
            planar0.fillZero();
            planar1.fillZero();
            planar2.fillZero();
            planar3.fillZero();

            T* planes[4] = {planar0.get(), planar1.get(), planar2.get(), planar3.get()};

            const NppiSize roi{width, height};

            auto runTimed = [&](auto&& call) {
                for (int i = 0; i < warmupIterations; ++i) {
                    call();
                }
                cudaError_t warmupSync = cudaDeviceSynchronize();
                if (warmupSync != cudaSuccess) {
                    result.success = false;
                    result.errorMessage = std::string("CUDA error after warmup: ") + cudaGetErrorString(warmupSync);
                    results.push_back(result);
                    return;
                }

                if (runBatchedMeasurement(result, measureIterations, call)) {
                    const size_t bytes = static_cast<size_t>(pixels) * sizeof(T) * (Channels + 1); // packed + planar
                    result.throughputGBps = computeThroughput(bytes, result.avgTimeMs);
                }
                results.push_back(result);
            };

            runTimed([&]() { return fn(packed.get(), packedStep, planes, planarStep, roi); });
        }

        return results;
    }

private:
    const char* dataType_;
    const char* symbolName_;
};

template<typename T, int Channels, typename Fn>
class NppiCopyPXCXRBenchmark : public BenchmarkBase {
public:
    NppiCopyPXCXRBenchmark(const std::string& benchName, const char* symbolName, const char* dataType)
        : BenchmarkBase(benchName), dataType_(dataType), symbolName_(symbolName) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>(symbolName_).get(&symErr);

        std::vector<BenchmarkResult> results;
        for (const auto& imgSize : getRequestedImageSizes()) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = dataType_;
            result.size = imgSize.toString();
            result.channels = Channels;
            result.iterations = measureIterations;

            if (fn == nullptr) {
                result.success = false;
                result.errorMessage = std::string("Missing symbol ") + symbolName_ + ": " + symErr;
                results.push_back(result);
                continue;
            }

            const int width = imgSize.width;
            const int height = imgSize.height;
            const int pixels = width * height;
            const int packedStep = width * Channels * static_cast<int>(sizeof(T));
            const int planarStep = width * static_cast<int>(sizeof(T));

            DeviceBuffer<T> packed(static_cast<size_t>(pixels) * static_cast<size_t>(Channels));
            DeviceBuffer<T> planar0(pixels);
            DeviceBuffer<T> planar1(pixels);
            DeviceBuffer<T> planar2(pixels);
            DeviceBuffer<T> planar3(pixels);

            if (!validateDeviceBuffer(packed, result, "packed") || !validateDeviceBuffer(planar0, result, "planar0") ||
                !validateDeviceBuffer(planar1, result, "planar1") || !validateDeviceBuffer(planar2, result, "planar2") ||
                !validateDeviceBuffer(planar3, result, "planar3")) {
                results.push_back(result);
                continue;
            }

            initRandom<T>(planar0.get(), pixels);
            initRandom<T>(planar1.get(), pixels);
            initRandom<T>(planar2.get(), pixels);
            initRandom<T>(planar3.get(), pixels);
            packed.fillZero();

            const T* planes[4] = {planar0.get(), planar1.get(), planar2.get(), planar3.get()};

            const NppiSize roi{width, height};

            auto runTimed = [&](auto&& call) {
                for (int i = 0; i < warmupIterations; ++i) {
                    call();
                }
                cudaError_t warmupSync = cudaDeviceSynchronize();
                if (warmupSync != cudaSuccess) {
                    result.success = false;
                    result.errorMessage = std::string("CUDA error after warmup: ") + cudaGetErrorString(warmupSync);
                    results.push_back(result);
                    return;
                }

                if (runBatchedMeasurement(result, measureIterations, call)) {
                    const size_t bytes = static_cast<size_t>(pixels) * sizeof(T) * (Channels + 1);
                    result.throughputGBps = computeThroughput(bytes, result.avgTimeMs);
                }
                results.push_back(result);
            };

            runTimed([&]() { return fn(planes, planarStep, packed.get(), packedStep, roi); });
        }

        return results;
    }

private:
    const char* dataType_;
    const char* symbolName_;
};

// Concrete packed<->planar copy benchmarks (most commonly used in pipelines)
class BenchCopy_8u_C3P3R : public NppiCopyCXPXRBenchmark<Npp8u, 3, decltype(&nppiCopy_8u_C3P3R)> {
public:
    BenchCopy_8u_C3P3R() : NppiCopyCXPXRBenchmark("nppiCopy_8u_C3P3R", "nppiCopy_8u_C3P3R", "8u") {}
};
REGISTER_BENCHMARK(BenchCopy_8u_C3P3R)

class BenchCopy_8u_P3C3R : public NppiCopyPXCXRBenchmark<Npp8u, 3, decltype(&nppiCopy_8u_P3C3R)> {
public:
    BenchCopy_8u_P3C3R() : NppiCopyPXCXRBenchmark("nppiCopy_8u_P3C3R", "nppiCopy_8u_P3C3R", "8u") {}
};
REGISTER_BENCHMARK(BenchCopy_8u_P3C3R)

class BenchCopy_8u_C4P4R : public NppiCopyCXPXRBenchmark<Npp8u, 4, decltype(&nppiCopy_8u_C4P4R)> {
public:
    BenchCopy_8u_C4P4R() : NppiCopyCXPXRBenchmark("nppiCopy_8u_C4P4R", "nppiCopy_8u_C4P4R", "8u") {}
};
REGISTER_BENCHMARK(BenchCopy_8u_C4P4R)

class BenchCopy_8u_P4C4R : public NppiCopyPXCXRBenchmark<Npp8u, 4, decltype(&nppiCopy_8u_P4C4R)> {
public:
    BenchCopy_8u_P4C4R() : NppiCopyPXCXRBenchmark("nppiCopy_8u_P4C4R", "nppiCopy_8u_P4C4R", "8u") {}
};
REGISTER_BENCHMARK(BenchCopy_8u_P4C4R)

class BenchCopy_16u_C3P3R : public NppiCopyCXPXRBenchmark<Npp16u, 3, decltype(&nppiCopy_16u_C3P3R)> {
public:
    BenchCopy_16u_C3P3R() : NppiCopyCXPXRBenchmark("nppiCopy_16u_C3P3R", "nppiCopy_16u_C3P3R", "16u") {}
};
REGISTER_BENCHMARK(BenchCopy_16u_C3P3R)

class BenchCopy_16u_P3C3R : public NppiCopyPXCXRBenchmark<Npp16u, 3, decltype(&nppiCopy_16u_P3C3R)> {
public:
    BenchCopy_16u_P3C3R() : NppiCopyPXCXRBenchmark("nppiCopy_16u_P3C3R", "nppiCopy_16u_P3C3R", "16u") {}
};
REGISTER_BENCHMARK(BenchCopy_16u_P3C3R)

class BenchCopy_16u_C4P4R : public NppiCopyCXPXRBenchmark<Npp16u, 4, decltype(&nppiCopy_16u_C4P4R)> {
public:
    BenchCopy_16u_C4P4R() : NppiCopyCXPXRBenchmark("nppiCopy_16u_C4P4R", "nppiCopy_16u_C4P4R", "16u") {}
};
REGISTER_BENCHMARK(BenchCopy_16u_C4P4R)

class BenchCopy_16u_P4C4R : public NppiCopyPXCXRBenchmark<Npp16u, 4, decltype(&nppiCopy_16u_P4C4R)> {
public:
    BenchCopy_16u_P4C4R() : NppiCopyPXCXRBenchmark("nppiCopy_16u_P4C4R", "nppiCopy_16u_P4C4R", "16u") {}
};
REGISTER_BENCHMARK(BenchCopy_16u_P4C4R)

class BenchCopy_32f_C3P3R : public NppiCopyCXPXRBenchmark<Npp32f, 3, decltype(&nppiCopy_32f_C3P3R)> {
public:
    BenchCopy_32f_C3P3R() : NppiCopyCXPXRBenchmark("nppiCopy_32f_C3P3R", "nppiCopy_32f_C3P3R", "32f") {}
};
REGISTER_BENCHMARK(BenchCopy_32f_C3P3R)

class BenchCopy_32f_P3C3R : public NppiCopyPXCXRBenchmark<Npp32f, 3, decltype(&nppiCopy_32f_P3C3R)> {
public:
    BenchCopy_32f_P3C3R() : NppiCopyPXCXRBenchmark("nppiCopy_32f_P3C3R", "nppiCopy_32f_P3C3R", "32f") {}
};
REGISTER_BENCHMARK(BenchCopy_32f_P3C3R)

class BenchCopy_32f_C4P4R : public NppiCopyCXPXRBenchmark<Npp32f, 4, decltype(&nppiCopy_32f_C4P4R)> {
public:
    BenchCopy_32f_C4P4R() : NppiCopyCXPXRBenchmark("nppiCopy_32f_C4P4R", "nppiCopy_32f_C4P4R", "32f") {}
};
REGISTER_BENCHMARK(BenchCopy_32f_C4P4R)

class BenchCopy_32f_P4C4R : public NppiCopyPXCXRBenchmark<Npp32f, 4, decltype(&nppiCopy_32f_P4C4R)> {
public:
    BenchCopy_32f_P4C4R() : NppiCopyPXCXRBenchmark("nppiCopy_32f_P4C4R", "nppiCopy_32f_P4C4R", "32f") {}
};
REGISTER_BENCHMARK(BenchCopy_32f_P4C4R)

// 16s/32s: only the 3-channel packed<->planar variants are marked implemented in coverage.csv
class BenchCopy_16s_C3P3R : public NppiCopyCXPXRBenchmark<Npp16s, 3, decltype(&nppiCopy_16s_C3P3R)> {
public:
    BenchCopy_16s_C3P3R() : NppiCopyCXPXRBenchmark("nppiCopy_16s_C3P3R", "nppiCopy_16s_C3P3R", "16s") {}
};
REGISTER_BENCHMARK(BenchCopy_16s_C3P3R)

class BenchCopy_16s_P3C3R : public NppiCopyPXCXRBenchmark<Npp16s, 3, decltype(&nppiCopy_16s_P3C3R)> {
public:
    BenchCopy_16s_P3C3R() : NppiCopyPXCXRBenchmark("nppiCopy_16s_P3C3R", "nppiCopy_16s_P3C3R", "16s") {}
};
REGISTER_BENCHMARK(BenchCopy_16s_P3C3R)

class BenchCopy_32s_C3P3R : public NppiCopyCXPXRBenchmark<Npp32s, 3, decltype(&nppiCopy_32s_C3P3R)> {
public:
    BenchCopy_32s_C3P3R() : NppiCopyCXPXRBenchmark("nppiCopy_32s_C3P3R", "nppiCopy_32s_C3P3R", "32s") {}
};
REGISTER_BENCHMARK(BenchCopy_32s_C3P3R)

class BenchCopy_32s_P3C3R : public NppiCopyPXCXRBenchmark<Npp32s, 3, decltype(&nppiCopy_32s_P3C3R)> {
public:
    BenchCopy_32s_P3C3R() : NppiCopyPXCXRBenchmark("nppiCopy_32s_P3C3R", "nppiCopy_32s_P3C3R", "32s") {}
};
REGISTER_BENCHMARK(BenchCopy_32s_P3C3R)

// Set benchmark
class BenchSet_8u_C1R : public BenchmarkBase {
public:
    BenchSet_8u_C1R() : BenchmarkBase("nppiSet_8u_C1R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiSet_8u_C1R);
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>("nppiSet_8u_C1R").get(&symErr);
        if (fn == nullptr) {
            std::vector<BenchmarkResult> results;
            for (const auto& imgSize : getStandardImageSizes()) {
                BenchmarkResult result;
                result.functionName = name_;
                result.dataType = "8u";
                result.size = imgSize.toString();
                result.channels = 1;
                result.iterations = measureIterations;
                result.success = false;
                result.errorMessage = "Missing symbol nppiSet_8u_C1R: " + symErr;
                results.push_back(result);
            }
            return results;
        }

        std::vector<BenchmarkResult> results;
        auto sizes = getStandardImageSizes();

        for (const auto& imgSize : sizes) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = "8u";
            result.size = imgSize.toString();
            result.channels = 1;
            result.iterations = measureIterations;

            int pixelCount = imgSize.width * imgSize.height;
            int step = imgSize.width * sizeof(Npp8u);

            DeviceBuffer<Npp8u> dst(pixelCount);

            NppiSize roiSize = {imgSize.width, imgSize.height};
            Npp8u value = 128;

            // Warmup
            for (int i = 0; i < warmupIterations; ++i) {
                fn(value, dst.get(), step, roiSize);
            }
            cudaDeviceSynchronize();

            if (runBatchedMeasurement(result, measureIterations,
                                      [&]() { return fn(value, dst.get(), step, roiSize); })) {
                result.throughputGBps = computeThroughput(pixelCount * sizeof(Npp8u), result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }
};
REGISTER_BENCHMARK(BenchSet_8u_C1R)

class BenchSet_16u_C1R : public BenchmarkBase {
public:
    BenchSet_16u_C1R() : BenchmarkBase("nppiSet_16u_C1R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiSet_16u_C1R);
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>("nppiSet_16u_C1R").get(&symErr);
        if (fn == nullptr) {
            std::vector<BenchmarkResult> results;
            for (const auto& imgSize : getStandardImageSizes()) {
                BenchmarkResult result;
                result.functionName = name_;
                result.dataType = "16u";
                result.size = imgSize.toString();
                result.channels = 1;
                result.iterations = measureIterations;
                result.success = false;
                result.errorMessage = "Missing symbol nppiSet_16u_C1R: " + symErr;
                results.push_back(result);
            }
            return results;
        }

        std::vector<BenchmarkResult> results;
        auto sizes = getStandardImageSizes();

        for (const auto& imgSize : sizes) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = "16u";
            result.size = imgSize.toString();
            result.channels = 1;
            result.iterations = measureIterations;

            int pixelCount = imgSize.width * imgSize.height;
            int step = imgSize.width * static_cast<int>(sizeof(Npp16u));

            DeviceBuffer<Npp16u> dst(pixelCount);

            NppiSize roiSize = {imgSize.width, imgSize.height};
            Npp16u value = 17;

            for (int i = 0; i < warmupIterations; ++i) {
                fn(value, dst.get(), step, roiSize);
            }
            cudaDeviceSynchronize();

            if (runBatchedMeasurement(result, measureIterations,
                                      [&]() { return fn(value, dst.get(), step, roiSize); })) {
                result.throughputGBps = computeThroughput(pixelCount * sizeof(Npp16u), result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }
};
REGISTER_BENCHMARK(BenchSet_16u_C1R)

// Transpose benchmark
class BenchTranspose_8u_C1R : public BenchmarkBase {
public:
    BenchTranspose_8u_C1R() : BenchmarkBase("nppiTranspose_8u_C1R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiTranspose_8u_C1R);
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>("nppiTranspose_8u_C1R").get(&symErr);
        if (fn == nullptr) {
            std::vector<BenchmarkResult> results;
            for (const auto& imgSize : getStandardImageSizes()) {
                BenchmarkResult result;
                result.functionName = name_;
                result.dataType = "8u";
                result.size = imgSize.toString();
                result.channels = 1;
                result.iterations = measureIterations;
                result.success = false;
                result.errorMessage = "Missing symbol nppiTranspose_8u_C1R: " + symErr;
                results.push_back(result);
            }
            return results;
        }

        std::vector<BenchmarkResult> results;
        auto sizes = getStandardImageSizes();

        for (const auto& imgSize : sizes) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = "8u";
            result.size = imgSize.toString();
            result.channels = 1;
            result.iterations = measureIterations;

            int srcPixels = imgSize.width * imgSize.height;
            int srcStep = imgSize.width * sizeof(Npp8u);
            int dstStep = imgSize.height * sizeof(Npp8u);  // Transposed

            DeviceBuffer<Npp8u> src(srcPixels);
            DeviceBuffer<Npp8u> dst(srcPixels);

            initDeviceRandom8u(src.get(), srcPixels);

            NppiSize srcSize = {imgSize.width, imgSize.height};

            // Warmup
            for (int i = 0; i < warmupIterations; ++i) {
                fn(src.get(), srcStep, dst.get(), dstStep, srcSize);
            }
            cudaDeviceSynchronize();

            if (runBatchedMeasurement(
                    result, measureIterations,
                    [&]() { return fn(src.get(), srcStep, dst.get(), dstStep, srcSize); })) {
                result.throughputGBps = computeThroughput(2 * srcPixels * sizeof(Npp8u), result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }
};
REGISTER_BENCHMARK(BenchTranspose_8u_C1R)

// Convert benchmark (8u to 32f)
class BenchConvert_8u32f_C1R : public BenchmarkBase {
public:
    BenchConvert_8u32f_C1R() : BenchmarkBase("nppiConvert_8u32f_C1R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiConvert_8u32f_C1R);
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>("nppiConvert_8u32f_C1R").get(&symErr);
        if (fn == nullptr) {
            std::vector<BenchmarkResult> results;
            for (const auto& imgSize : getStandardImageSizes()) {
                BenchmarkResult result;
                result.functionName = name_;
                result.dataType = "8u->32f";
                result.size = imgSize.toString();
                result.channels = 1;
                result.iterations = measureIterations;
                result.success = false;
                result.errorMessage = "Missing symbol nppiConvert_8u32f_C1R: " + symErr;
                results.push_back(result);
            }
            return results;
        }

        std::vector<BenchmarkResult> results;
        auto sizes = getStandardImageSizes();

        for (const auto& imgSize : sizes) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = "8u->32f";
            result.size = imgSize.toString();
            result.channels = 1;
            result.iterations = measureIterations;

            int pixelCount = imgSize.width * imgSize.height;
            int srcStep = imgSize.width * sizeof(Npp8u);
            int dstStep = imgSize.width * sizeof(Npp32f);

            DeviceBuffer<Npp8u> src(pixelCount);
            DeviceBuffer<Npp32f> dst(pixelCount);

            initDeviceRandom8u(src.get(), pixelCount);

            NppiSize roiSize = {imgSize.width, imgSize.height};

            // Warmup
            for (int i = 0; i < warmupIterations; ++i) {
                fn(src.get(), srcStep, dst.get(), dstStep, roiSize);
            }
            cudaDeviceSynchronize();

            if (runBatchedMeasurement(
                    result, measureIterations,
                    [&]() { return fn(src.get(), srcStep, dst.get(), dstStep, roiSize); })) {
                size_t totalBytes = pixelCount * sizeof(Npp8u) + pixelCount * sizeof(Npp32f);
                result.throughputGBps = computeThroughput(totalBytes, result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }
};
REGISTER_BENCHMARK(BenchConvert_8u32f_C1R)

} // namespace npp_benchmark
