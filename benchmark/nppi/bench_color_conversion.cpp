#include "../benchmark_base.h"
#include <nppi_color_conversion.h>

namespace npp_benchmark {

extern void initDeviceRandom8u(Npp8u* data, int size);
extern void initDeviceRandom16u(Npp16u* data, int size);
extern void initDeviceRandom16s(Npp16s* data, int size);
extern void initDeviceRandom32f(Npp32f* data, int size);

namespace {

std::vector<ImageSize> getColorImageSizes() {
    return {
        {1280, 720},
        {1280, 1024},
        {1920, 1080},
    };
}

template<typename T>
void initColorInput(T* data, int count) {
    copyDeterministicHostDataToDevice(data, static_cast<size_t>(count));
}

template<>
void initColorInput<Npp8u>(Npp8u* data, int count) { initDeviceRandom8u(data, count); }
template<>
void initColorInput<Npp16u>(Npp16u* data, int count) { initDeviceRandom16u(data, count); }
template<>
void initColorInput<Npp16s>(Npp16s* data, int count) { initDeviceRandom16s(data, count); }
template<>
void initColorInput<Npp32f>(Npp32f* data, int count) { initDeviceRandom32f(data, count); }

template<typename SrcT, typename DstT, int SrcChannels, int DstChannels, typename Fn>
class NppiColorConvertBenchmark : public BenchmarkBase {
public:
    NppiColorConvertBenchmark(const std::string& benchName, const char* symbolName, const char* dataType)
        : BenchmarkBase(benchName), symbolName_(symbolName), dataType_(dataType) {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        std::string symErr;
        const Fn fn = DynamicSymbol<Fn>(symbolName_).get(&symErr);
        std::vector<BenchmarkResult> results;
        for (const auto& imgSize : getColorImageSizes()) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = dataType_;
            result.size = imgSize.toString();
            result.channels = DstChannels;
            result.iterations = measureIterations;
            if (fn == nullptr) {
                result.success = false;
                result.errorMessage = std::string("Missing symbol ") + symbolName_ + ": " + symErr;
                results.push_back(result);
                continue;
            }

            const int srcElems = imgSize.width * imgSize.height * SrcChannels;
            const int dstElems = imgSize.width * imgSize.height * DstChannels;
            const int srcStep = imgSize.width * SrcChannels * static_cast<int>(sizeof(SrcT));
            const int dstStep = imgSize.width * DstChannels * static_cast<int>(sizeof(DstT));

            DeviceBuffer<SrcT> src(srcElems);
            DeviceBuffer<DstT> dst(dstElems);
            if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(dst, result, "dst")) {
                results.push_back(result);
                continue;
            }
            initColorInput<SrcT>(src.get(), srcElems);
            dst.fillZero();
            const NppiSize roi{imgSize.width, imgSize.height};

            for (int i = 0; i < warmupIterations; ++i) {
                const NppStatus st = fn(src.get(), srcStep, dst.get(), dstStep, roi);
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
                                      [&]() { return fn(src.get(), srcStep, dst.get(), dstStep, roi); })) {
                result.throughputGBps = computeThroughput(static_cast<size_t>(srcElems) * sizeof(SrcT) +
                                                              static_cast<size_t>(dstElems) * sizeof(DstT),
                                                          result.avgTimeMs);
            }
            results.push_back(result);
        }
        return results;
    }

private:
    const char* symbolName_;
    const char* dataType_;
};

#define REGISTER_COLOR_BENCH(BenchClass, Symbol, SrcType, DstType, SrcCh, DstCh, DType)                            \
    class BenchClass final : public NppiColorConvertBenchmark<SrcType, DstType, SrcCh, DstCh, decltype(&Symbol)> { \
    public:                                                                                                           \
        BenchClass()                                                                                                  \
            : NppiColorConvertBenchmark<SrcType, DstType, SrcCh, DstCh, decltype(&Symbol)>(#Symbol, #Symbol, DType) {} \
    };                                                                                                                \
    REGISTER_BENCHMARK(BenchClass)

REGISTER_COLOR_BENCH(BenchRGBToYUV_8u_C3R, nppiRGBToYUV_8u_C3R, Npp8u, Npp8u, 3, 3, "8u");
REGISTER_COLOR_BENCH(BenchRGBToYUV_8u_AC4R, nppiRGBToYUV_8u_AC4R, Npp8u, Npp8u, 4, 4, "8u");
REGISTER_COLOR_BENCH(BenchBGRToYUV_8u_C3R, nppiBGRToYUV_8u_C3R, Npp8u, Npp8u, 3, 3, "8u");
REGISTER_COLOR_BENCH(BenchBGRToYUV_8u_AC4R, nppiBGRToYUV_8u_AC4R, Npp8u, Npp8u, 4, 4, "8u");
REGISTER_COLOR_BENCH(BenchYUVToRGB_8u_C3R, nppiYUVToRGB_8u_C3R, Npp8u, Npp8u, 3, 3, "8u");
REGISTER_COLOR_BENCH(BenchYUVToRGB_8u_AC4R, nppiYUVToRGB_8u_AC4R, Npp8u, Npp8u, 4, 4, "8u");
REGISTER_COLOR_BENCH(BenchYUVToBGR_8u_C3R, nppiYUVToBGR_8u_C3R, Npp8u, Npp8u, 3, 3, "8u");
REGISTER_COLOR_BENCH(BenchYUVToBGR_8u_AC4R, nppiYUVToBGR_8u_AC4R, Npp8u, Npp8u, 4, 4, "8u");
REGISTER_COLOR_BENCH(BenchRGBToXYZ_8u_C3R, nppiRGBToXYZ_8u_C3R, Npp8u, Npp8u, 3, 3, "8u");
REGISTER_COLOR_BENCH(BenchRGBToXYZ_8u_AC4R, nppiRGBToXYZ_8u_AC4R, Npp8u, Npp8u, 4, 4, "8u");
REGISTER_COLOR_BENCH(BenchXYZToRGB_8u_C3R, nppiXYZToRGB_8u_C3R, Npp8u, Npp8u, 3, 3, "8u");
REGISTER_COLOR_BENCH(BenchXYZToRGB_8u_AC4R, nppiXYZToRGB_8u_AC4R, Npp8u, Npp8u, 4, 4, "8u");
REGISTER_COLOR_BENCH(BenchRGBToLUV_8u_C3R, nppiRGBToLUV_8u_C3R, Npp8u, Npp8u, 3, 3, "8u");
REGISTER_COLOR_BENCH(BenchRGBToLUV_8u_AC4R, nppiRGBToLUV_8u_AC4R, Npp8u, Npp8u, 4, 4, "8u");
REGISTER_COLOR_BENCH(BenchLUVToRGB_8u_C3R, nppiLUVToRGB_8u_C3R, Npp8u, Npp8u, 3, 3, "8u");
REGISTER_COLOR_BENCH(BenchLUVToRGB_8u_AC4R, nppiLUVToRGB_8u_AC4R, Npp8u, Npp8u, 4, 4, "8u");
REGISTER_COLOR_BENCH(BenchRGBToYCC_8u_C3R, nppiRGBToYCC_8u_C3R, Npp8u, Npp8u, 3, 3, "8u");
REGISTER_COLOR_BENCH(BenchRGBToYCC_8u_AC4R, nppiRGBToYCC_8u_AC4R, Npp8u, Npp8u, 4, 4, "8u");
REGISTER_COLOR_BENCH(BenchYCCToRGB_8u_C3R, nppiYCCToRGB_8u_C3R, Npp8u, Npp8u, 3, 3, "8u");
REGISTER_COLOR_BENCH(BenchYCCToRGB_8u_AC4R, nppiYCCToRGB_8u_AC4R, Npp8u, Npp8u, 4, 4, "8u");
REGISTER_COLOR_BENCH(BenchRGBToHSV_8u_C3R, nppiRGBToHSV_8u_C3R, Npp8u, Npp8u, 3, 3, "8u");
REGISTER_COLOR_BENCH(BenchRGBToHSV_8u_AC4R, nppiRGBToHSV_8u_AC4R, Npp8u, Npp8u, 4, 4, "8u");
REGISTER_COLOR_BENCH(BenchHSVToRGB_8u_C3R, nppiHSVToRGB_8u_C3R, Npp8u, Npp8u, 3, 3, "8u");
REGISTER_COLOR_BENCH(BenchHSVToRGB_8u_AC4R, nppiHSVToRGB_8u_AC4R, Npp8u, Npp8u, 4, 4, "8u");
REGISTER_COLOR_BENCH(BenchRGBToGray_8u_C3C1R, nppiRGBToGray_8u_C3C1R, Npp8u, Npp8u, 3, 1, "8u");
REGISTER_COLOR_BENCH(BenchRGBToGray_8u_AC4C1R, nppiRGBToGray_8u_AC4C1R, Npp8u, Npp8u, 4, 1, "8u");
REGISTER_COLOR_BENCH(BenchRGBToGray_16u_C3C1R, nppiRGBToGray_16u_C3C1R, Npp16u, Npp16u, 3, 1, "16u");
REGISTER_COLOR_BENCH(BenchRGBToGray_16u_AC4C1R, nppiRGBToGray_16u_AC4C1R, Npp16u, Npp16u, 4, 1, "16u");
REGISTER_COLOR_BENCH(BenchRGBToGray_16s_C3C1R, nppiRGBToGray_16s_C3C1R, Npp16s, Npp16s, 3, 1, "16s");
REGISTER_COLOR_BENCH(BenchRGBToGray_16s_AC4C1R, nppiRGBToGray_16s_AC4C1R, Npp16s, Npp16s, 4, 1, "16s");
REGISTER_COLOR_BENCH(BenchRGBToGray_32f_C3C1R, nppiRGBToGray_32f_C3C1R, Npp32f, Npp32f, 3, 1, "32f");
REGISTER_COLOR_BENCH(BenchRGBToGray_32f_AC4C1R, nppiRGBToGray_32f_AC4C1R, Npp32f, Npp32f, 4, 1, "32f");

#undef REGISTER_COLOR_BENCH

} // namespace

} // namespace npp_benchmark
