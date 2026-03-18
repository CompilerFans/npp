#include "../benchmark_base.h"
#include <nppi_morphological_operations.h>
#include <type_traits>

namespace npp_benchmark {

extern void initDeviceRandom8u(Npp8u* data, int size);
extern void initDeviceRandom32f(Npp32f* data, int size);

namespace {

std::vector<ImageSize> getRequestedImageSizes() {
    return {
        {1280, 720},
        {1280, 1024},
        {1920, 1080},
    };
}

std::vector<NppiSize> getMaskSizes() {
    return {
        {3, 3},
        {5, 5},
        {7, 7},
    };
}

std::string maskToString(const NppiSize& mask) {
    return std::to_string(mask.width) + "x" + std::to_string(mask.height);
}

std::vector<Npp8u> makeOnesMaskHost(const NppiSize& mask) {
    std::vector<Npp8u> m(static_cast<size_t>(mask.width) * static_cast<size_t>(mask.height));
    std::fill(m.begin(), m.end(), static_cast<Npp8u>(1));
    return m;
}

template<typename Fn>
std::vector<BenchmarkResult> missingSymbolResults(const std::string& benchName, const std::string& dataType, int channels,
                                                  int iterations, const std::string& symbolName,
                                                  const std::string& symErr) {
    std::vector<BenchmarkResult> results;
    for (const auto& imgSize : getRequestedImageSizes()) {
        for (const auto& maskSize : getMaskSizes()) {
            BenchmarkResult r;
            r.functionName = benchName + "_" + maskToString(maskSize);
            r.dataType = dataType;
            r.size = imgSize.toString();
            r.channels = channels;
            r.iterations = iterations;
            r.success = false;
            r.errorMessage = "Missing symbol " + symbolName + ": " + symErr;
            results.push_back(r);
        }
    }
    return results;
}

template<typename T, int Channels, bool UseCtx, typename Fn>
std::vector<BenchmarkResult> runMorphology(const std::string& benchName, const std::string& dataType,
                                          const char* symbolName, int warmupIterations, int measureIterations) {
    auto computeThroughputLocal = [](size_t bytes, double timeMs) -> double {
        if (timeMs <= 0.0) return 0.0;
        return (bytes / 1e9) / (timeMs / 1000.0);
    };

    std::string symErr;
    Fn fn = DynamicSymbol<Fn>(symbolName).get(&symErr);
    if (fn == nullptr) {
        return missingSymbolResults<Fn>(benchName, dataType, Channels, measureIterations, symbolName, symErr);
    }

    std::vector<BenchmarkResult> results;
    const auto imageSizes = getRequestedImageSizes();
    const auto maskSizes = getMaskSizes();

    auto runLoops = [&](auto&& call) {
        for (const auto& imgSize : imageSizes) {
            for (const auto& maskSize : maskSizes) {
                BenchmarkResult result;
                result.functionName = benchName + "_" + maskToString(maskSize);
                result.dataType = dataType;
                result.size = imgSize.toString();
                result.channels = Channels;
                result.iterations = measureIterations;

                const int pixelCount = imgSize.width * imgSize.height * Channels;
                const int step = imgSize.width * Channels * static_cast<int>(sizeof(T));

                DeviceBuffer<T> src(pixelCount);
                DeviceBuffer<T> dst(pixelCount);

                if (!validateDeviceBuffer(src, result, "src") || !validateDeviceBuffer(dst, result, "dst")) {
                    results.push_back(result);
                    continue;
                }

                if constexpr (std::is_same<T, Npp8u>::value) {
                    initDeviceRandom8u(reinterpret_cast<Npp8u*>(src.get()), pixelCount);
                } else if constexpr (std::is_same<T, Npp32f>::value) {
                    initDeviceRandom32f(reinterpret_cast<Npp32f*>(src.get()), pixelCount);
                } else {
                    src.fillRandom();
                }

                const NppiPoint anchor = {maskSize.width / 2, maskSize.height / 2};
                const int roiWidth = imgSize.width - (maskSize.width - 1);
                const int roiHeight = imgSize.height - (maskSize.height - 1);
                if (roiWidth <= 0 || roiHeight <= 0) {
                    result.success = false;
                    result.errorMessage = "Invalid ROI for mask " + maskToString(maskSize);
                    results.push_back(result);
                    continue;
                }
                const NppiSize roiSize = {roiWidth, roiHeight};

                const int elemOffset = (anchor.y * imgSize.width + anchor.x) * Channels;
                const T* srcRoi = src.get() + elemOffset;
                T* dstRoi = dst.get() + elemOffset;

                const auto hostMask = makeOnesMaskHost(maskSize);
                DeviceBuffer<Npp8u> mask(hostMask.size());
                if (!validateDeviceBuffer(mask, result, "mask")) {
                    results.push_back(result);
                    continue;
                }
                cudaError_t maskCopy = cudaMemcpy(mask.get(), hostMask.data(), hostMask.size() * sizeof(Npp8u),
                                                  cudaMemcpyHostToDevice);
                if (maskCopy != cudaSuccess) {
                    result.success = false;
                    result.errorMessage = std::string("CUDA error copying mask: ") + cudaGetErrorString(maskCopy);
                    results.push_back(result);
                    continue;
                }

                (void)cudaGetLastError();
                for (int i = 0; i < warmupIterations; ++i) {
                    NppStatus st = call(srcRoi, step, dstRoi, step, roiSize, mask.get(), maskSize, anchor);
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
                    result.errorMessage = std::string("CUDA error after warmup: ") + cudaGetErrorString(warmupSync);
                    resetCudaIfFatal(warmupSync);
                    results.push_back(result);
                    continue;
                }

                if (runBatchedMeasurement(
                        result, measureIterations,
                        [&]() { return call(srcRoi, step, dstRoi, step, roiSize, mask.get(), maskSize, anchor); },
                        true)) {
                    const int roiPixels = roiSize.width * roiSize.height * Channels;
                    const size_t roiBytes = static_cast<size_t>(roiPixels) * sizeof(T);
                    result.throughputGBps = computeThroughputLocal(2 * roiBytes, result.avgTimeMs);
                }

                results.push_back(result);
            }
        }
    };

    if constexpr (UseCtx) {
        const NppStreamContext ctx = makeNppStreamContext(0);
        runLoops([&](const T* src, int srcStep, T* dst, int dstStep, NppiSize roiSize, const Npp8u* mask,
                     NppiSize maskSize, NppiPoint anchor) {
            return fn(src, srcStep, dst, dstStep, roiSize, mask, maskSize, anchor, ctx);
        });
    } else {
        runLoops([&](const T* src, int srcStep, T* dst, int dstStep, NppiSize roiSize, const Npp8u* mask,
                     NppiSize maskSize, NppiPoint anchor) { return fn(src, srcStep, dst, dstStep, roiSize, mask, maskSize, anchor); });
    }

    return results;
}

} // namespace

class BenchErode_8u_C1R : public BenchmarkBase {
public:
    BenchErode_8u_C1R() : BenchmarkBase("nppiErode_8u_C1R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiErode_8u_C1R);
        return runMorphology<Npp8u, 1, false, Fn>(name_, "8u", "nppiErode_8u_C1R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchErode_8u_C1R)

class BenchErode_8u_C4R : public BenchmarkBase {
public:
    BenchErode_8u_C4R() : BenchmarkBase("nppiErode_8u_C4R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiErode_8u_C4R);
        return runMorphology<Npp8u, 4, false, Fn>(name_, "8u", "nppiErode_8u_C4R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchErode_8u_C4R)

class BenchErode_8u_C3R : public BenchmarkBase {
public:
    BenchErode_8u_C3R() : BenchmarkBase("nppiErode_8u_C3R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiErode_8u_C3R);
        return runMorphology<Npp8u, 3, false, Fn>(name_, "8u", "nppiErode_8u_C3R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchErode_8u_C3R)

class BenchErode_8u_AC4R : public BenchmarkBase {
public:
    BenchErode_8u_AC4R() : BenchmarkBase("nppiErode_8u_AC4R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiErode_8u_AC4R);
        return runMorphology<Npp8u, 4, false, Fn>(name_, "8u", "nppiErode_8u_AC4R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchErode_8u_AC4R)

class BenchErode_16u_C1R : public BenchmarkBase {
public:
    BenchErode_16u_C1R() : BenchmarkBase("nppiErode_16u_C1R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiErode_16u_C1R);
        return runMorphology<Npp16u, 1, false, Fn>(name_, "16u", "nppiErode_16u_C1R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchErode_16u_C1R)

class BenchErode_16u_C3R : public BenchmarkBase {
public:
    BenchErode_16u_C3R() : BenchmarkBase("nppiErode_16u_C3R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiErode_16u_C3R);
        return runMorphology<Npp16u, 3, false, Fn>(name_, "16u", "nppiErode_16u_C3R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchErode_16u_C3R)

class BenchErode_16u_C4R : public BenchmarkBase {
public:
    BenchErode_16u_C4R() : BenchmarkBase("nppiErode_16u_C4R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiErode_16u_C4R);
        return runMorphology<Npp16u, 4, false, Fn>(name_, "16u", "nppiErode_16u_C4R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchErode_16u_C4R)

class BenchErode_16u_AC4R : public BenchmarkBase {
public:
    BenchErode_16u_AC4R() : BenchmarkBase("nppiErode_16u_AC4R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiErode_16u_AC4R);
        return runMorphology<Npp16u, 4, false, Fn>(name_, "16u", "nppiErode_16u_AC4R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchErode_16u_AC4R)

class BenchErode_32f_C1R : public BenchmarkBase {
public:
    BenchErode_32f_C1R() : BenchmarkBase("nppiErode_32f_C1R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiErode_32f_C1R);
        return runMorphology<Npp32f, 1, false, Fn>(name_, "32f", "nppiErode_32f_C1R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchErode_32f_C1R)

class BenchErode_32f_C3R : public BenchmarkBase {
public:
    BenchErode_32f_C3R() : BenchmarkBase("nppiErode_32f_C3R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiErode_32f_C3R);
        return runMorphology<Npp32f, 3, false, Fn>(name_, "32f", "nppiErode_32f_C3R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchErode_32f_C3R)

class BenchErode_32f_C4R : public BenchmarkBase {
public:
    BenchErode_32f_C4R() : BenchmarkBase("nppiErode_32f_C4R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiErode_32f_C4R);
        return runMorphology<Npp32f, 4, false, Fn>(name_, "32f", "nppiErode_32f_C4R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchErode_32f_C4R)

class BenchErode_32f_AC4R : public BenchmarkBase {
public:
    BenchErode_32f_AC4R() : BenchmarkBase("nppiErode_32f_AC4R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiErode_32f_AC4R);
        return runMorphology<Npp32f, 4, false, Fn>(name_, "32f", "nppiErode_32f_AC4R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchErode_32f_AC4R)

class BenchDilate_8u_C1R : public BenchmarkBase {
public:
    BenchDilate_8u_C1R() : BenchmarkBase("nppiDilate_8u_C1R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiDilate_8u_C1R);
        return runMorphology<Npp8u, 1, false, Fn>(name_, "8u", "nppiDilate_8u_C1R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchDilate_8u_C1R)

class BenchDilate_8u_C3R : public BenchmarkBase {
public:
    BenchDilate_8u_C3R() : BenchmarkBase("nppiDilate_8u_C3R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiDilate_8u_C3R);
        return runMorphology<Npp8u, 3, false, Fn>(name_, "8u", "nppiDilate_8u_C3R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchDilate_8u_C3R)

class BenchDilate_8u_C4R : public BenchmarkBase {
public:
    BenchDilate_8u_C4R() : BenchmarkBase("nppiDilate_8u_C4R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiDilate_8u_C4R);
        return runMorphology<Npp8u, 4, false, Fn>(name_, "8u", "nppiDilate_8u_C4R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchDilate_8u_C4R)

class BenchDilate_8u_AC4R : public BenchmarkBase {
public:
    BenchDilate_8u_AC4R() : BenchmarkBase("nppiDilate_8u_AC4R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiDilate_8u_AC4R);
        return runMorphology<Npp8u, 4, false, Fn>(name_, "8u", "nppiDilate_8u_AC4R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchDilate_8u_AC4R)

class BenchDilate_16u_C1R : public BenchmarkBase {
public:
    BenchDilate_16u_C1R() : BenchmarkBase("nppiDilate_16u_C1R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiDilate_16u_C1R);
        return runMorphology<Npp16u, 1, false, Fn>(name_, "16u", "nppiDilate_16u_C1R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchDilate_16u_C1R)

class BenchDilate_16u_C3R : public BenchmarkBase {
public:
    BenchDilate_16u_C3R() : BenchmarkBase("nppiDilate_16u_C3R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiDilate_16u_C3R);
        return runMorphology<Npp16u, 3, false, Fn>(name_, "16u", "nppiDilate_16u_C3R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchDilate_16u_C3R)

class BenchDilate_16u_C4R : public BenchmarkBase {
public:
    BenchDilate_16u_C4R() : BenchmarkBase("nppiDilate_16u_C4R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiDilate_16u_C4R);
        return runMorphology<Npp16u, 4, false, Fn>(name_, "16u", "nppiDilate_16u_C4R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchDilate_16u_C4R)

class BenchDilate_16u_AC4R : public BenchmarkBase {
public:
    BenchDilate_16u_AC4R() : BenchmarkBase("nppiDilate_16u_AC4R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiDilate_16u_AC4R);
        return runMorphology<Npp16u, 4, false, Fn>(name_, "16u", "nppiDilate_16u_AC4R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchDilate_16u_AC4R)

class BenchDilate_32f_C1R : public BenchmarkBase {
public:
    BenchDilate_32f_C1R() : BenchmarkBase("nppiDilate_32f_C1R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiDilate_32f_C1R);
        return runMorphology<Npp32f, 1, false, Fn>(name_, "32f", "nppiDilate_32f_C1R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchDilate_32f_C1R)

class BenchDilate_32f_C3R : public BenchmarkBase {
public:
    BenchDilate_32f_C3R() : BenchmarkBase("nppiDilate_32f_C3R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiDilate_32f_C3R);
        return runMorphology<Npp32f, 3, false, Fn>(name_, "32f", "nppiDilate_32f_C3R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchDilate_32f_C3R)

class BenchDilate_32f_C4R : public BenchmarkBase {
public:
    BenchDilate_32f_C4R() : BenchmarkBase("nppiDilate_32f_C4R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiDilate_32f_C4R);
        return runMorphology<Npp32f, 4, false, Fn>(name_, "32f", "nppiDilate_32f_C4R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchDilate_32f_C4R)

class BenchDilate_32f_AC4R : public BenchmarkBase {
public:
    BenchDilate_32f_AC4R() : BenchmarkBase("nppiDilate_32f_AC4R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiDilate_32f_AC4R);
        return runMorphology<Npp32f, 4, false, Fn>(name_, "32f", "nppiDilate_32f_AC4R", warmupIterations, measureIterations);
    }
};
REGISTER_BENCHMARK(BenchDilate_32f_AC4R)

} // namespace npp_benchmark
