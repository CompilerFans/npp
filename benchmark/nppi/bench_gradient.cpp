#include "../benchmark_base.h"
#include <nppi_filtering_functions.h>

namespace npp_benchmark {

extern void initDeviceRandom8u(Npp8u* data, int size);
extern void initDeviceRandom16s(Npp16s* data, int size);
extern void initDeviceRandom32f(Npp32f* data, int size);

namespace {

std::vector<ImageSize> getGradientImageSizes() {
    return {
        {1280, 720},
        {1280, 1024},
        {1920, 1080},
    };
}

template<typename T>
void initGradientInput(T* data, int count) {
    copyDeterministicHostDataToDevice(data, static_cast<size_t>(count));
}

template<>
void initGradientInput<Npp8u>(Npp8u* data, int count) {
    initDeviceRandom8u(data, count);
}

template<>
void initGradientInput<Npp16s>(Npp16s* data, int count) {
    initDeviceRandom16s(data, count);
}

template<>
void initGradientInput<Npp32f>(Npp32f* data, int count) {
    initDeviceRandom32f(data, count);
}

template<typename T, int Channels, typename Fn>
class NppiGradientBenchmark : public BenchmarkBase {
public:
    NppiGradientBenchmark(const std::string& benchName, const char* symbolName, const char* dataType)
        : BenchmarkBase(benchName), symbolName_(symbolName), dataType_(dataType) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string symErr;
        const Fn fn = DynamicSymbol<Fn>(symbolName_).get(&symErr);

        std::vector<BenchmarkResult> results;
        for (const auto& imgSize : getGradientImageSizes()) {
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

            const int pixelCount = imgSize.width * imgSize.height * Channels;
            const int step = imgSize.width * Channels * static_cast<int>(sizeof(T));

            DeviceBuffer<T> src(pixelCount);
            DeviceBuffer<T> dst(pixelCount);
            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }

            initGradientInput<T>(src.get(), pixelCount);
            dst.fillZero();
            // Sobel/Prewitt are 3x3 neighborhood operators. Use the inner ROI and offset the source/destination
            // pointers to the anchor to avoid border over-read in NVIDIA NPP.
            const NppiPoint anchor{1, 1};
            const int roiWidth = imgSize.width - 2;
            const int roiHeight = imgSize.height - 2;
            if (roiWidth <= 0 || roiHeight <= 0) {
                result.success = false;
                result.errorMessage = "Invalid ROI for 3x3 gradient kernel";
                results.push_back(result);
                continue;
            }
            const NppiSize roi{roiWidth, roiHeight};
            const int elemOffset = (anchor.y * imgSize.width + anchor.x) * Channels;
            const T* srcRoi = src.get() + elemOffset;
            T* dstRoi = dst.get() + elemOffset;

            (void)cudaGetLastError();
            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st = fn(srcRoi, step, dstRoi, step, roi);
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

            const cudaError_t warmupSync = cudaDeviceSynchronize();
            if (warmupSync != cudaSuccess) {
                result.success = false;
                result.errorMessage = std::string("CUDA error after warmup: ") + cudaGetErrorString(warmupSync);
                resetCudaIfFatal(warmupSync);
                results.push_back(result);
                continue;
            }

            if (runBatchedMeasurement(
                    result, measureIterations,
                    [&]() { return fn(srcRoi, step, dstRoi, step, roi); }, true,
                    "NPP call failed in measurement", "CUDA error during measure")) {
                const int roiPixels = roi.width * roi.height * Channels;
                result.throughputGBps =
                    computeThroughput(2ULL * static_cast<size_t>(roiPixels) * sizeof(T), result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }

private:
    const char* symbolName_;
    const char* dataType_;
};

template<typename SrcT, typename DstT, int Channels, typename Fn>
class NppiMixedGradientBenchmark : public BenchmarkBase {
public:
    NppiMixedGradientBenchmark(const std::string& benchName, const char* symbolName, const char* dataType)
        : BenchmarkBase(benchName), symbolName_(symbolName), dataType_(dataType) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string symErr;
        const Fn fn = DynamicSymbol<Fn>(symbolName_).get(&symErr);

        std::vector<BenchmarkResult> results;
        for (const auto& imgSize : getGradientImageSizes()) {
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

            const int pixelCount = imgSize.width * imgSize.height * Channels;
            const int srcStep = imgSize.width * Channels * static_cast<int>(sizeof(SrcT));
            const int dstStep = imgSize.width * Channels * static_cast<int>(sizeof(DstT));

            DeviceBuffer<SrcT> src(pixelCount);
            DeviceBuffer<DstT> dst(pixelCount);
            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }

            initGradientInput<SrcT>(src.get(), pixelCount);
            dst.fillZero();

            const NppiPoint anchor{1, 1};
            const int roiWidth = imgSize.width - 2;
            const int roiHeight = imgSize.height - 2;
            if (roiWidth <= 0 || roiHeight <= 0) {
                result.success = false;
                result.errorMessage = "Invalid ROI for 3x3 gradient kernel";
                results.push_back(result);
                continue;
            }
            const NppiSize roi{roiWidth, roiHeight};
            const int elemOffset = (anchor.y * imgSize.width + anchor.x) * Channels;
            const SrcT* srcRoi = src.get() + elemOffset;
            DstT* dstRoi = dst.get() + elemOffset;

            (void)cudaGetLastError();
            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st = fn(srcRoi, srcStep, dstRoi, dstStep, roi);
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

            const cudaError_t warmupSync = cudaDeviceSynchronize();
            if (warmupSync != cudaSuccess) {
                result.success = false;
                result.errorMessage = std::string("CUDA error after warmup: ") + cudaGetErrorString(warmupSync);
                resetCudaIfFatal(warmupSync);
                results.push_back(result);
                continue;
            }

            if (runBatchedMeasurement(
                    result, measureIterations,
                    [&]() { return fn(srcRoi, srcStep, dstRoi, dstStep, roi); }, true,
                    "NPP call failed in measurement", "CUDA error during measure")) {
                const int roiPixels = roi.width * roi.height * Channels;
                result.throughputGBps = computeThroughput(
                    static_cast<size_t>(roiPixels) * (sizeof(SrcT) + sizeof(DstT)), result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }

private:
    const char* symbolName_;
    const char* dataType_;
};

#define REGISTER_GRADIENT_BENCH(BenchClass, Symbol, Type, Channels, DType)                                  \
    class BenchClass final : public NppiGradientBenchmark<Type, Channels, decltype(&Symbol)> {              \
    public:                                                                                                  \
        BenchClass() : NppiGradientBenchmark<Type, Channels, decltype(&Symbol)>(#Symbol, #Symbol, DType) {} \
    };                                                                                                       \
    REGISTER_BENCHMARK(BenchClass)

#define REGISTER_MIXED_GRADIENT_BENCH(BenchClass, Symbol, SrcType, DstType, Channels, DType)                        \
    class BenchClass final : public NppiMixedGradientBenchmark<SrcType, DstType, Channels, decltype(&Symbol)> {     \
    public:                                                                                                           \
        BenchClass()                                                                                                  \
            : NppiMixedGradientBenchmark<SrcType, DstType, Channels, decltype(&Symbol)>(#Symbol, #Symbol, DType) {} \
    };                                                                                                                \
    REGISTER_BENCHMARK(BenchClass)

REGISTER_GRADIENT_BENCH(BenchFilterSobelHoriz_8u_C1R, nppiFilterSobelHoriz_8u_C1R, Npp8u, 1, "8u");
REGISTER_GRADIENT_BENCH(BenchFilterSobelHoriz_8u_C3R, nppiFilterSobelHoriz_8u_C3R, Npp8u, 3, "8u");
REGISTER_GRADIENT_BENCH(BenchFilterSobelHoriz_8u_C4R, nppiFilterSobelHoriz_8u_C4R, Npp8u, 4, "8u");
REGISTER_GRADIENT_BENCH(BenchFilterSobelHoriz_16s_C1R, nppiFilterSobelHoriz_16s_C1R, Npp16s, 1, "16s");
REGISTER_GRADIENT_BENCH(BenchFilterSobelHoriz_16s_C3R, nppiFilterSobelHoriz_16s_C3R, Npp16s, 3, "16s");
REGISTER_GRADIENT_BENCH(BenchFilterSobelHoriz_16s_C4R, nppiFilterSobelHoriz_16s_C4R, Npp16s, 4, "16s");
REGISTER_GRADIENT_BENCH(BenchFilterSobelHoriz_32f_C1R, nppiFilterSobelHoriz_32f_C1R, Npp32f, 1, "32f");
REGISTER_GRADIENT_BENCH(BenchFilterSobelHoriz_32f_C3R, nppiFilterSobelHoriz_32f_C3R, Npp32f, 3, "32f");
REGISTER_GRADIENT_BENCH(BenchFilterSobelHoriz_32f_C4R, nppiFilterSobelHoriz_32f_C4R, Npp32f, 4, "32f");

REGISTER_GRADIENT_BENCH(BenchFilterSobelVert_8u_C1R, nppiFilterSobelVert_8u_C1R, Npp8u, 1, "8u");
REGISTER_GRADIENT_BENCH(BenchFilterSobelVert_8u_C3R, nppiFilterSobelVert_8u_C3R, Npp8u, 3, "8u");
REGISTER_GRADIENT_BENCH(BenchFilterSobelVert_8u_C4R, nppiFilterSobelVert_8u_C4R, Npp8u, 4, "8u");
REGISTER_GRADIENT_BENCH(BenchFilterSobelVert_16s_C1R, nppiFilterSobelVert_16s_C1R, Npp16s, 1, "16s");
REGISTER_GRADIENT_BENCH(BenchFilterSobelVert_16s_C3R, nppiFilterSobelVert_16s_C3R, Npp16s, 3, "16s");
REGISTER_GRADIENT_BENCH(BenchFilterSobelVert_16s_C4R, nppiFilterSobelVert_16s_C4R, Npp16s, 4, "16s");
REGISTER_GRADIENT_BENCH(BenchFilterSobelVert_32f_C1R, nppiFilterSobelVert_32f_C1R, Npp32f, 1, "32f");
REGISTER_GRADIENT_BENCH(BenchFilterSobelVert_32f_C3R, nppiFilterSobelVert_32f_C3R, Npp32f, 3, "32f");
REGISTER_GRADIENT_BENCH(BenchFilterSobelVert_32f_C4R, nppiFilterSobelVert_32f_C4R, Npp32f, 4, "32f");

REGISTER_GRADIENT_BENCH(BenchFilterPrewittHoriz_8u_C1R, nppiFilterPrewittHoriz_8u_C1R, Npp8u, 1, "8u");
REGISTER_GRADIENT_BENCH(BenchFilterPrewittHoriz_8u_C3R, nppiFilterPrewittHoriz_8u_C3R, Npp8u, 3, "8u");
REGISTER_GRADIENT_BENCH(BenchFilterPrewittHoriz_8u_C4R, nppiFilterPrewittHoriz_8u_C4R, Npp8u, 4, "8u");
REGISTER_GRADIENT_BENCH(BenchFilterPrewittHoriz_16s_C1R, nppiFilterPrewittHoriz_16s_C1R, Npp16s, 1, "16s");
REGISTER_GRADIENT_BENCH(BenchFilterPrewittHoriz_16s_C3R, nppiFilterPrewittHoriz_16s_C3R, Npp16s, 3, "16s");
REGISTER_GRADIENT_BENCH(BenchFilterPrewittHoriz_16s_C4R, nppiFilterPrewittHoriz_16s_C4R, Npp16s, 4, "16s");
REGISTER_GRADIENT_BENCH(BenchFilterPrewittHoriz_32f_C1R, nppiFilterPrewittHoriz_32f_C1R, Npp32f, 1, "32f");
REGISTER_GRADIENT_BENCH(BenchFilterPrewittHoriz_32f_C3R, nppiFilterPrewittHoriz_32f_C3R, Npp32f, 3, "32f");
REGISTER_GRADIENT_BENCH(BenchFilterPrewittHoriz_32f_C4R, nppiFilterPrewittHoriz_32f_C4R, Npp32f, 4, "32f");

REGISTER_GRADIENT_BENCH(BenchFilterPrewittVert_8u_C1R, nppiFilterPrewittVert_8u_C1R, Npp8u, 1, "8u");
REGISTER_GRADIENT_BENCH(BenchFilterPrewittVert_8u_C3R, nppiFilterPrewittVert_8u_C3R, Npp8u, 3, "8u");
REGISTER_GRADIENT_BENCH(BenchFilterPrewittVert_8u_C4R, nppiFilterPrewittVert_8u_C4R, Npp8u, 4, "8u");
REGISTER_GRADIENT_BENCH(BenchFilterPrewittVert_16s_C1R, nppiFilterPrewittVert_16s_C1R, Npp16s, 1, "16s");
REGISTER_GRADIENT_BENCH(BenchFilterPrewittVert_16s_C3R, nppiFilterPrewittVert_16s_C3R, Npp16s, 3, "16s");
REGISTER_GRADIENT_BENCH(BenchFilterPrewittVert_16s_C4R, nppiFilterPrewittVert_16s_C4R, Npp16s, 4, "16s");
REGISTER_GRADIENT_BENCH(BenchFilterPrewittVert_32f_C1R, nppiFilterPrewittVert_32f_C1R, Npp32f, 1, "32f");
REGISTER_GRADIENT_BENCH(BenchFilterPrewittVert_32f_C3R, nppiFilterPrewittVert_32f_C3R, Npp32f, 3, "32f");
REGISTER_GRADIENT_BENCH(BenchFilterPrewittVert_32f_C4R, nppiFilterPrewittVert_32f_C4R, Npp32f, 4, "32f");

REGISTER_MIXED_GRADIENT_BENCH(
    BenchFilterScharrHoriz_8u16s_C1R, nppiFilterScharrHoriz_8u16s_C1R, Npp8u, Npp16s, 1, "8u16s");
REGISTER_MIXED_GRADIENT_BENCH(
    BenchFilterScharrHoriz_8s16s_C1R, nppiFilterScharrHoriz_8s16s_C1R, Npp8s, Npp16s, 1, "8s16s");
REGISTER_MIXED_GRADIENT_BENCH(
    BenchFilterScharrHoriz_32f_C1R, nppiFilterScharrHoriz_32f_C1R, Npp32f, Npp32f, 1, "32f");
REGISTER_MIXED_GRADIENT_BENCH(
    BenchFilterScharrVert_8u16s_C1R, nppiFilterScharrVert_8u16s_C1R, Npp8u, Npp16s, 1, "8u16s");
REGISTER_MIXED_GRADIENT_BENCH(
    BenchFilterScharrVert_8s16s_C1R, nppiFilterScharrVert_8s16s_C1R, Npp8s, Npp16s, 1, "8s16s");
REGISTER_MIXED_GRADIENT_BENCH(
    BenchFilterScharrVert_32f_C1R, nppiFilterScharrVert_32f_C1R, Npp32f, Npp32f, 1, "32f");

#undef REGISTER_GRADIENT_BENCH
#undef REGISTER_MIXED_GRADIENT_BENCH

} // namespace

} // namespace npp_benchmark
