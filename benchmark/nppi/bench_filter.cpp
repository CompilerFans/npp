#include "../benchmark_base.h"
#include <nppi_filtering_functions.h>

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

std::string kernelToString(const NppiSize& k) {
    return std::to_string(k.width) + "x" + std::to_string(k.height);
}

template<typename T>
void initRandomT(T* data, int count) {
    copyDeterministicHostDataToDevice(data, static_cast<size_t>(count));
}

template<>
void initRandomT<Npp8u>(Npp8u* data, int count) {
    initDeviceRandom8u(data, count);
}

template<>
void initRandomT<Npp16u>(Npp16u* data, int count) {
    initDeviceRandom16u(data, count);
}

template<>
void initRandomT<Npp16s>(Npp16s* data, int count) {
    initDeviceRandom16s(data, count);
}

template<>
void initRandomT<Npp32f>(Npp32f* data, int count) {
    initDeviceRandom32f(data, count);
}

template<typename T, int Channels, typename Fn>
class NppiFilterBoxBenchmark : public BenchmarkBase {
public:
    NppiFilterBoxBenchmark(const std::string& benchName, const char* symbolName, const char* dataType)
        : BenchmarkBase(benchName), symbolName_(symbolName), dataType_(dataType) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string symErr;
        auto fn = DynamicSymbol<Fn>(symbolName_).get(&symErr);

        std::vector<BenchmarkResult> results;
        const auto sizes = getRequestedImageSizes();

        // Test different kernel sizes
        std::vector<NppiSize> kernelSizes = {{3, 3}, {5, 5}, {7, 7}};

        auto runLoops = [&](auto&& call) {
            for (const auto& imgSize : sizes) {
                for (const auto& kernelSize : kernelSizes) {
                    BenchmarkResult result;
                    result.functionName = name_ + "_" + kernelToString(kernelSize);
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

                    const int pixelCount = imgSize.width * imgSize.height * Channels;
                    const int step = imgSize.width * Channels * static_cast<int>(sizeof(T));

                    DeviceBuffer<T> src(pixelCount);
                    DeviceBuffer<T> dst(pixelCount);

                    if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(dst, result, "dst")) {
                        results.push_back(result);
                        continue;
                    }

                    initRandomT<T>(src.get(), pixelCount);

                    NppiPoint anchor = {kernelSize.width / 2, kernelSize.height / 2};
                    const int roiWidth = imgSize.width - (kernelSize.width - 1);
                    const int roiHeight = imgSize.height - (kernelSize.height - 1);
                    if (roiWidth <= 0 || roiHeight <= 0) {
                        result.success = false;
                        result.errorMessage = "Invalid ROI for kernel " + kernelToString(kernelSize);
                        results.push_back(result);
                        continue;
                    }
                    NppiSize roiSize = {roiWidth, roiHeight};

                    // NPP expects pSrc/pDst to point to the top-left of the ROI. For neighborhood filters, the ROI must
                    // exclude borders unless the caller provides padding. We benchmark the inner region by offsetting
                    // pointers by the anchor and shrinking the ROI.
                    const int elemOffset = (anchor.y * imgSize.width + anchor.x) * Channels;
                    const T* srcRoi = src.get() + elemOffset;
                    T* dstRoi = dst.get() + elemOffset;

                    // Warmup
                    (void)cudaGetLastError();
                    for (int i = 0; i < warmupIterations; ++i) {
                        NppStatus st = call(srcRoi, step, dstRoi, step, roiSize, kernelSize, anchor);
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
                    cudaError_t warmupSync = cudaDeviceSynchronize();
                    if (warmupSync != cudaSuccess) {
                        result.success = false;
                        result.errorMessage =
                            std::string("CUDA error after warmup: ") + cudaGetErrorString(warmupSync);
                        results.push_back(result);
                        continue;
                    }

                    if (runBatchedMeasurement(
                            result, measureIterations,
                            [&]() { return call(srcRoi, step, dstRoi, step, roiSize, kernelSize, anchor); }, true)) {
                        const int roiPixels = roiSize.width * roiSize.height * Channels;
                        result.throughputGBps =
                            computeThroughput(2ULL * static_cast<size_t>(roiPixels) * sizeof(T), result.avgTimeMs);
                    }

                    results.push_back(result);
                }
            }
        };

        runLoops([&](const T* src, int srcStep, T* dst, int dstStep, NppiSize roiSize, NppiSize kernelSize,
                     NppiPoint anchor) { return fn(src, srcStep, dst, dstStep, roiSize, kernelSize, anchor); });

        return results;
    }

private:
    const char* symbolName_;
    const char* dataType_;
};

} // namespace

class BenchFilterBox_8u_C1R final : public NppiFilterBoxBenchmark<Npp8u, 1, decltype(&nppiFilterBox_8u_C1R)> {
public:
    BenchFilterBox_8u_C1R() : NppiFilterBoxBenchmark("nppiFilterBox_8u_C1R", "nppiFilterBox_8u_C1R", "8u") {}
};
REGISTER_BENCHMARK(BenchFilterBox_8u_C1R)

class BenchFilterBox_8u_C3R final : public NppiFilterBoxBenchmark<Npp8u, 3, decltype(&nppiFilterBox_8u_C3R)> {
public:
    BenchFilterBox_8u_C3R() : NppiFilterBoxBenchmark("nppiFilterBox_8u_C3R", "nppiFilterBox_8u_C3R", "8u") {}
};
REGISTER_BENCHMARK(BenchFilterBox_8u_C3R)

class BenchFilterBox_8u_C4R final : public NppiFilterBoxBenchmark<Npp8u, 4, decltype(&nppiFilterBox_8u_C4R)> {
public:
    BenchFilterBox_8u_C4R() : NppiFilterBoxBenchmark("nppiFilterBox_8u_C4R", "nppiFilterBox_8u_C4R", "8u") {}
};
REGISTER_BENCHMARK(BenchFilterBox_8u_C4R)

class BenchFilterBox_8u_AC4R final : public NppiFilterBoxBenchmark<Npp8u, 4, decltype(&nppiFilterBox_8u_AC4R)> {
public:
    BenchFilterBox_8u_AC4R() : NppiFilterBoxBenchmark("nppiFilterBox_8u_AC4R", "nppiFilterBox_8u_AC4R", "8u") {}
};
REGISTER_BENCHMARK(BenchFilterBox_8u_AC4R)

class BenchFilterBox_16u_C1R final : public NppiFilterBoxBenchmark<Npp16u, 1, decltype(&nppiFilterBox_16u_C1R)> {
public:
    BenchFilterBox_16u_C1R() : NppiFilterBoxBenchmark("nppiFilterBox_16u_C1R", "nppiFilterBox_16u_C1R", "16u") {}
};
REGISTER_BENCHMARK(BenchFilterBox_16u_C1R)

class BenchFilterBox_16u_C3R final : public NppiFilterBoxBenchmark<Npp16u, 3, decltype(&nppiFilterBox_16u_C3R)> {
public:
    BenchFilterBox_16u_C3R() : NppiFilterBoxBenchmark("nppiFilterBox_16u_C3R", "nppiFilterBox_16u_C3R", "16u") {}
};
REGISTER_BENCHMARK(BenchFilterBox_16u_C3R)

class BenchFilterBox_16u_C4R final : public NppiFilterBoxBenchmark<Npp16u, 4, decltype(&nppiFilterBox_16u_C4R)> {
public:
    BenchFilterBox_16u_C4R() : NppiFilterBoxBenchmark("nppiFilterBox_16u_C4R", "nppiFilterBox_16u_C4R", "16u") {}
};
REGISTER_BENCHMARK(BenchFilterBox_16u_C4R)

class BenchFilterBox_16u_AC4R final : public NppiFilterBoxBenchmark<Npp16u, 4, decltype(&nppiFilterBox_16u_AC4R)> {
public:
    BenchFilterBox_16u_AC4R() : NppiFilterBoxBenchmark("nppiFilterBox_16u_AC4R", "nppiFilterBox_16u_AC4R", "16u") {}
};
REGISTER_BENCHMARK(BenchFilterBox_16u_AC4R)

class BenchFilterBox_16s_C1R final : public NppiFilterBoxBenchmark<Npp16s, 1, decltype(&nppiFilterBox_16s_C1R)> {
public:
    BenchFilterBox_16s_C1R() : NppiFilterBoxBenchmark("nppiFilterBox_16s_C1R", "nppiFilterBox_16s_C1R", "16s") {}
};
REGISTER_BENCHMARK(BenchFilterBox_16s_C1R)

class BenchFilterBox_16s_C3R final : public NppiFilterBoxBenchmark<Npp16s, 3, decltype(&nppiFilterBox_16s_C3R)> {
public:
    BenchFilterBox_16s_C3R() : NppiFilterBoxBenchmark("nppiFilterBox_16s_C3R", "nppiFilterBox_16s_C3R", "16s") {}
};
REGISTER_BENCHMARK(BenchFilterBox_16s_C3R)

class BenchFilterBox_16s_C4R final : public NppiFilterBoxBenchmark<Npp16s, 4, decltype(&nppiFilterBox_16s_C4R)> {
public:
    BenchFilterBox_16s_C4R() : NppiFilterBoxBenchmark("nppiFilterBox_16s_C4R", "nppiFilterBox_16s_C4R", "16s") {}
};
REGISTER_BENCHMARK(BenchFilterBox_16s_C4R)

class BenchFilterBox_16s_AC4R final : public NppiFilterBoxBenchmark<Npp16s, 4, decltype(&nppiFilterBox_16s_AC4R)> {
public:
    BenchFilterBox_16s_AC4R() : NppiFilterBoxBenchmark("nppiFilterBox_16s_AC4R", "nppiFilterBox_16s_AC4R", "16s") {}
};
REGISTER_BENCHMARK(BenchFilterBox_16s_AC4R)

class BenchFilterBox_32f_C1R final : public NppiFilterBoxBenchmark<Npp32f, 1, decltype(&nppiFilterBox_32f_C1R)> {
public:
    BenchFilterBox_32f_C1R() : NppiFilterBoxBenchmark("nppiFilterBox_32f_C1R", "nppiFilterBox_32f_C1R", "32f") {}
};
REGISTER_BENCHMARK(BenchFilterBox_32f_C1R)

class BenchFilterBox_32f_C3R final : public NppiFilterBoxBenchmark<Npp32f, 3, decltype(&nppiFilterBox_32f_C3R)> {
public:
    BenchFilterBox_32f_C3R() : NppiFilterBoxBenchmark("nppiFilterBox_32f_C3R", "nppiFilterBox_32f_C3R", "32f") {}
};
REGISTER_BENCHMARK(BenchFilterBox_32f_C3R)

class BenchFilterBox_32f_C4R final : public NppiFilterBoxBenchmark<Npp32f, 4, decltype(&nppiFilterBox_32f_C4R)> {
public:
    BenchFilterBox_32f_C4R() : NppiFilterBoxBenchmark("nppiFilterBox_32f_C4R", "nppiFilterBox_32f_C4R", "32f") {}
};
REGISTER_BENCHMARK(BenchFilterBox_32f_C4R)

class BenchFilterBox_32f_AC4R final : public NppiFilterBoxBenchmark<Npp32f, 4, decltype(&nppiFilterBox_32f_AC4R)> {
public:
    BenchFilterBox_32f_AC4R() : NppiFilterBoxBenchmark("nppiFilterBox_32f_AC4R", "nppiFilterBox_32f_AC4R", "32f") {}
};
REGISTER_BENCHMARK(BenchFilterBox_32f_AC4R)

// FilterGauss benchmark
class BenchFilterGauss_8u_C1R : public BenchmarkBase {
public:
    BenchFilterGauss_8u_C1R() : BenchmarkBase("nppiFilterGauss_8u_C1R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiFilterGauss_8u_C1R);
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>("nppiFilterGauss_8u_C1R").get(&symErr);

        std::vector<BenchmarkResult> results;
        auto sizes = getStandardImageSizes();

        // Test different mask sizes
        std::vector<NppiMaskSize> maskSizes = {NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_5_X_5};
        std::vector<std::string> maskNames = {"3x3", "5x5"};
        std::vector<int> maskWidths = {3, 5};

        for (const auto& imgSize : sizes) {
            for (size_t m = 0; m < maskSizes.size(); ++m) {
                BenchmarkResult result;
                result.functionName = name_ + "_" + maskNames[m];
                result.dataType = "8u";
                result.size = imgSize.toString();
                result.channels = 1;
                result.iterations = measureIterations;

                if (fn == nullptr) {
                    result.success = false;
                    result.errorMessage = "Missing symbol nppiFilterGauss_8u_C1R: " + symErr;
                    results.push_back(result);
                    continue;
                }

                int pixelCount = imgSize.width * imgSize.height;
                int step = imgSize.width * sizeof(Npp8u);

                DeviceBuffer<Npp8u> src(pixelCount);
                DeviceBuffer<Npp8u> dst(pixelCount);

                if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(dst, result, "dst")) {
                    results.push_back(result);
                    continue;
                }

                initDeviceRandom8u(src.get(), pixelCount);

                const int k = maskWidths[m];
                const int roiWidth = imgSize.width - (k - 1);
                const int roiHeight = imgSize.height - (k - 1);
                if (roiWidth <= 0 || roiHeight <= 0) {
                    result.success = false;
                    result.errorMessage = "Invalid ROI for mask " + maskNames[m];
                    results.push_back(result);
                    continue;
                }
                const int anchor = k / 2;
                const int elemOffset = anchor * imgSize.width + anchor;
                const Npp8u* srcRoi = src.get() + elemOffset;
                Npp8u* dstRoi = dst.get() + elemOffset;
                NppiSize roiSize = {roiWidth, roiHeight};

                // Warmup
                (void)cudaGetLastError();
                for (int i = 0; i < warmupIterations; ++i) {
                    NppStatus st = fn(srcRoi, step, dstRoi, step, roiSize, maskSizes[m]);
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
                cudaError_t warmupSync = cudaDeviceSynchronize();
                if (warmupSync != cudaSuccess) {
                    result.success = false;
                    result.errorMessage =
                        std::string("CUDA error after warmup: ") + cudaGetErrorString(warmupSync);
                    results.push_back(result);
                    continue;
                }

                if (runBatchedMeasurement(
                        result, measureIterations,
                        [&]() { return fn(srcRoi, step, dstRoi, step, roiSize, maskSizes[m]); }, true)) {
                    const int roiPixels = roiSize.width * roiSize.height;
                    result.throughputGBps =
                        computeThroughput(2ULL * static_cast<size_t>(roiPixels) * sizeof(Npp8u), result.avgTimeMs);
                }

                results.push_back(result);
            }
        }

        return results;
    }
};
REGISTER_BENCHMARK(BenchFilterGauss_8u_C1R)

// FilterGauss 32f benchmark
class BenchFilterGauss_32f_C1R : public BenchmarkBase {
public:
    BenchFilterGauss_32f_C1R() : BenchmarkBase("nppiFilterGauss_32f_C1R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiFilterGauss_32f_C1R);
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>("nppiFilterGauss_32f_C1R").get(&symErr);
        if (fn == nullptr) {
            std::vector<BenchmarkResult> results;
            for (const auto& imgSize : getStandardImageSizes()) {
                BenchmarkResult result;
                result.functionName = name_;
                result.dataType = "32f";
                result.size = imgSize.toString();
                result.channels = 1;
                result.iterations = measureIterations;
                result.success = false;
                result.errorMessage = "Missing symbol nppiFilterGauss_32f_C1R: " + symErr;
                results.push_back(result);
            }
            return results;
        }

        std::vector<BenchmarkResult> results;
        auto sizes = getStandardImageSizes();

        for (const auto& imgSize : sizes) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = "32f";
            result.size = imgSize.toString();
            result.channels = 1;
            result.iterations = measureIterations;

            int pixelCount = imgSize.width * imgSize.height;
            int step = imgSize.width * sizeof(Npp32f);

            DeviceBuffer<Npp32f> src(pixelCount);
            DeviceBuffer<Npp32f> dst(pixelCount);

            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }

            initDeviceRandom32f(src.get(), pixelCount);

            const int k = 3;
            const int roiWidth = imgSize.width - (k - 1);
            const int roiHeight = imgSize.height - (k - 1);
            if (roiWidth <= 0 || roiHeight <= 0) {
                result.success = false;
                result.errorMessage = "Invalid ROI for mask 3x3";
                results.push_back(result);
                continue;
            }
            const int anchor = k / 2;
            const int elemOffset = anchor * imgSize.width + anchor;
            const Npp32f* srcRoi = src.get() + elemOffset;
            Npp32f* dstRoi = dst.get() + elemOffset;
            NppiSize roiSize = {roiWidth, roiHeight};

            // Warmup
            (void)cudaGetLastError();
            for (int i = 0; i < warmupIterations; ++i) {
                NppStatus st = fn(srcRoi, step, dstRoi, step, roiSize, NPP_MASK_SIZE_3_X_3);
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
            cudaError_t warmupSync = cudaDeviceSynchronize();
            if (warmupSync != cudaSuccess) {
                result.success = false;
                result.errorMessage =
                    std::string("CUDA error after warmup: ") + cudaGetErrorString(warmupSync);
                results.push_back(result);
                continue;
            }

            if (runBatchedMeasurement(
                    result, measureIterations,
                    [&]() { return fn(srcRoi, step, dstRoi, step, roiSize, NPP_MASK_SIZE_3_X_3); }, true)) {
                const int roiPixels = roiSize.width * roiSize.height;
                result.throughputGBps =
                    computeThroughput(2ULL * static_cast<size_t>(roiPixels) * sizeof(Npp32f), result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }
};
REGISTER_BENCHMARK(BenchFilterGauss_32f_C1R)

// FilterMin benchmark
class BenchFilterMin_8u_C1R : public BenchmarkBase {
public:
    BenchFilterMin_8u_C1R() : BenchmarkBase("nppiFilterMin_8u_C1R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiFilterMin_8u_C1R);
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>("nppiFilterMin_8u_C1R").get(&symErr);
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
                result.errorMessage = "Missing symbol nppiFilterMin_8u_C1R: " + symErr;
                results.push_back(result);
            }
            return results;
        }

        std::vector<BenchmarkResult> results;
        auto sizes = getStandardImageSizes();

        NppiSize maskSize = {3, 3};
        NppiPoint anchor = {1, 1};

        for (const auto& imgSize : sizes) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = "8u";
            result.size = imgSize.toString();
            result.channels = 1;
            result.iterations = measureIterations;

            int pixelCount = imgSize.width * imgSize.height;
            int step = imgSize.width * sizeof(Npp8u);

            DeviceBuffer<Npp8u> src(pixelCount);
            DeviceBuffer<Npp8u> dst(pixelCount);

            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }

            initDeviceRandom8u(src.get(), pixelCount);

            const int roiWidth = imgSize.width - (maskSize.width - 1);
            const int roiHeight = imgSize.height - (maskSize.height - 1);
            if (roiWidth <= 0 || roiHeight <= 0) {
                result.success = false;
                result.errorMessage = "Invalid ROI for 3x3 mask";
                results.push_back(result);
                continue;
            }
            const int elemOffset = anchor.y * imgSize.width + anchor.x;
            const Npp8u* srcRoi = src.get() + elemOffset;
            Npp8u* dstRoi = dst.get() + elemOffset;
            NppiSize roiSize = {roiWidth, roiHeight};

            // Warmup
            (void)cudaGetLastError();
            for (int i = 0; i < warmupIterations; ++i) {
                NppStatus st = fn(srcRoi, step, dstRoi, step, roiSize, maskSize, anchor);
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
            cudaError_t warmupSync = cudaDeviceSynchronize();
            if (warmupSync != cudaSuccess) {
                result.success = false;
                result.errorMessage =
                    std::string("CUDA error after warmup: ") + cudaGetErrorString(warmupSync);
                results.push_back(result);
                continue;
            }

            if (runBatchedMeasurement(
                    result, measureIterations,
                    [&]() { return fn(srcRoi, step, dstRoi, step, roiSize, maskSize, anchor); }, true)) {
                const int roiPixels = roiSize.width * roiSize.height;
                result.throughputGBps =
                    computeThroughput(2ULL * static_cast<size_t>(roiPixels) * sizeof(Npp8u), result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }
};
REGISTER_BENCHMARK(BenchFilterMin_8u_C1R)

// FilterMax benchmark
class BenchFilterMax_8u_C1R : public BenchmarkBase {
public:
    BenchFilterMax_8u_C1R() : BenchmarkBase("nppiFilterMax_8u_C1R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiFilterMax_8u_C1R);
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>("nppiFilterMax_8u_C1R").get(&symErr);
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
                result.errorMessage = "Missing symbol nppiFilterMax_8u_C1R: " + symErr;
                results.push_back(result);
            }
            return results;
        }

        std::vector<BenchmarkResult> results;
        auto sizes = getStandardImageSizes();

        NppiSize maskSize = {3, 3};
        NppiPoint anchor = {1, 1};

        for (const auto& imgSize : sizes) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = "8u";
            result.size = imgSize.toString();
            result.channels = 1;
            result.iterations = measureIterations;

            int pixelCount = imgSize.width * imgSize.height;
            int step = imgSize.width * sizeof(Npp8u);

            DeviceBuffer<Npp8u> src(pixelCount);
            DeviceBuffer<Npp8u> dst(pixelCount);

            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }

            initDeviceRandom8u(src.get(), pixelCount);

            const int roiWidth = imgSize.width - (maskSize.width - 1);
            const int roiHeight = imgSize.height - (maskSize.height - 1);
            if (roiWidth <= 0 || roiHeight <= 0) {
                result.success = false;
                result.errorMessage = "Invalid ROI for 3x3 mask";
                results.push_back(result);
                continue;
            }
            const int elemOffset = anchor.y * imgSize.width + anchor.x;
            const Npp8u* srcRoi = src.get() + elemOffset;
            Npp8u* dstRoi = dst.get() + elemOffset;
            NppiSize roiSize = {roiWidth, roiHeight};

            // Warmup
            (void)cudaGetLastError();
            for (int i = 0; i < warmupIterations; ++i) {
                NppStatus st = fn(srcRoi, step, dstRoi, step, roiSize, maskSize, anchor);
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
            cudaError_t warmupSync = cudaDeviceSynchronize();
            if (warmupSync != cudaSuccess) {
                result.success = false;
                result.errorMessage =
                    std::string("CUDA error after warmup: ") + cudaGetErrorString(warmupSync);
                results.push_back(result);
                continue;
            }

            if (runBatchedMeasurement(
                    result, measureIterations,
                    [&]() { return fn(srcRoi, step, dstRoi, step, roiSize, maskSize, anchor); }, true)) {
                const int roiPixels = roiSize.width * roiSize.height;
                result.throughputGBps =
                    computeThroughput(2ULL * static_cast<size_t>(roiPixels) * sizeof(Npp8u), result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }
};
REGISTER_BENCHMARK(BenchFilterMax_8u_C1R)

} // namespace npp_benchmark
