#include "../benchmark_base.h"
#include <nppi_color_conversion.h>

namespace npp_benchmark {

extern void initDeviceRandom8u(Npp8u* data, int size);
extern void initDeviceRandom32f(Npp32f* data, int size);

// RGB to Gray benchmark
class BenchRGBToGray_8u_C3C1R : public BenchmarkBase {
public:
    BenchRGBToGray_8u_C3C1R() : BenchmarkBase("nppiRGBToGray_8u_C3C1R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiRGBToGray_8u_C3C1R);
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>("nppiRGBToGray_8u_C3C1R").get(&symErr);
        if (fn == nullptr) {
            std::vector<BenchmarkResult> results;
            for (const auto& imgSize : getStandardImageSizes()) {
                BenchmarkResult result;
                result.functionName = name_;
                result.dataType = "8u";
                result.size = imgSize.toString();
                result.channels = 3;
                result.iterations = measureIterations;
                result.success = false;
                result.errorMessage = "Missing symbol nppiRGBToGray_8u_C3C1R: " + symErr;
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
            result.channels = 3;
            result.iterations = measureIterations;

            int srcPixels = imgSize.width * imgSize.height * 3;
            int dstPixels = imgSize.width * imgSize.height;
            int srcStep = imgSize.width * 3 * sizeof(Npp8u);
            int dstStep = imgSize.width * sizeof(Npp8u);

            DeviceBuffer<Npp8u> src(srcPixels);
            DeviceBuffer<Npp8u> dst(dstPixels);

            initDeviceRandom8u(src.get(), srcPixels);

            NppiSize roiSize = {imgSize.width, imgSize.height};

            // Warmup
            for (int i = 0; i < warmupIterations; ++i) {
                fn(src.get(), srcStep, dst.get(), dstStep, roiSize);
            }
            cudaDeviceSynchronize();

            if (runBatchedMeasurement(
                    result, measureIterations,
                    [&]() { return fn(src.get(), srcStep, dst.get(), dstStep, roiSize); })) {
                result.throughputGBps =
                    computeThroughput((srcPixels + dstPixels) * sizeof(Npp8u), result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }
};
REGISTER_BENCHMARK(BenchRGBToGray_8u_C3C1R)

// RGB to YUV benchmark
class BenchRGBToYUV_8u_C3R : public BenchmarkBase {
public:
    BenchRGBToYUV_8u_C3R() : BenchmarkBase("nppiRGBToYUV_8u_C3R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiRGBToYUV_8u_C3R);
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>("nppiRGBToYUV_8u_C3R").get(&symErr);
        if (fn == nullptr) {
            std::vector<BenchmarkResult> results;
            for (const auto& imgSize : getStandardImageSizes()) {
                BenchmarkResult result;
                result.functionName = name_;
                result.dataType = "8u";
                result.size = imgSize.toString();
                result.channels = 3;
                result.iterations = measureIterations;
                result.success = false;
                result.errorMessage = "Missing symbol nppiRGBToYUV_8u_C3R: " + symErr;
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
            result.channels = 3;
            result.iterations = measureIterations;

            int pixelCount = imgSize.width * imgSize.height * 3;
            int step = imgSize.width * 3 * sizeof(Npp8u);

            DeviceBuffer<Npp8u> src(pixelCount);
            DeviceBuffer<Npp8u> dst(pixelCount);

            initDeviceRandom8u(src.get(), pixelCount);

            NppiSize roiSize = {imgSize.width, imgSize.height};

            // Warmup
            for (int i = 0; i < warmupIterations; ++i) {
                fn(src.get(), step, dst.get(), step, roiSize);
            }
            cudaDeviceSynchronize();

            if (runBatchedMeasurement(
                    result, measureIterations,
                    [&]() { return fn(src.get(), step, dst.get(), step, roiSize); })) {
                result.throughputGBps = computeThroughput(2 * pixelCount * sizeof(Npp8u), result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }
};
REGISTER_BENCHMARK(BenchRGBToYUV_8u_C3R)

// YUV to RGB benchmark
class BenchYUVToRGB_8u_C3R : public BenchmarkBase {
public:
    BenchYUVToRGB_8u_C3R() : BenchmarkBase("nppiYUVToRGB_8u_C3R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiYUVToRGB_8u_C3R);
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>("nppiYUVToRGB_8u_C3R").get(&symErr);
        if (fn == nullptr) {
            std::vector<BenchmarkResult> results;
            for (const auto& imgSize : getStandardImageSizes()) {
                BenchmarkResult result;
                result.functionName = name_;
                result.dataType = "8u";
                result.size = imgSize.toString();
                result.channels = 3;
                result.iterations = measureIterations;
                result.success = false;
                result.errorMessage = "Missing symbol nppiYUVToRGB_8u_C3R: " + symErr;
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
            result.channels = 3;
            result.iterations = measureIterations;

            int pixelCount = imgSize.width * imgSize.height * 3;
            int step = imgSize.width * 3 * sizeof(Npp8u);

            DeviceBuffer<Npp8u> src(pixelCount);
            DeviceBuffer<Npp8u> dst(pixelCount);

            initDeviceRandom8u(src.get(), pixelCount);

            NppiSize roiSize = {imgSize.width, imgSize.height};

            // Warmup
            for (int i = 0; i < warmupIterations; ++i) {
                fn(src.get(), step, dst.get(), step, roiSize);
            }
            cudaDeviceSynchronize();

            if (runBatchedMeasurement(
                    result, measureIterations,
                    [&]() { return fn(src.get(), step, dst.get(), step, roiSize); })) {
                result.throughputGBps = computeThroughput(2 * pixelCount * sizeof(Npp8u), result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }
};
REGISTER_BENCHMARK(BenchYUVToRGB_8u_C3R)

// Color Twist benchmark
class BenchColorTwist32f_8u_C3R : public BenchmarkBase {
public:
    BenchColorTwist32f_8u_C3R() : BenchmarkBase("nppiColorTwist32f_8u_C3R") {}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {
        using Fn = decltype(&nppiColorTwist32f_8u_C3R);
        std::string symErr;
        Fn fn = DynamicSymbol<Fn>("nppiColorTwist32f_8u_C3R").get(&symErr);
        if (fn == nullptr) {
            std::vector<BenchmarkResult> results;
            for (const auto& imgSize : getStandardImageSizes()) {
                BenchmarkResult result;
                result.functionName = name_;
                result.dataType = "8u";
                result.size = imgSize.toString();
                result.channels = 3;
                result.iterations = measureIterations;
                result.success = false;
                result.errorMessage = "Missing symbol nppiColorTwist32f_8u_C3R: " + symErr;
                results.push_back(result);
            }
            return results;
        }

        std::vector<BenchmarkResult> results;
        auto sizes = getStandardImageSizes();

        // Identity matrix for benchmark (actual values don't affect performance)
        Npp32f twist[3][4] = {
            {1.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 1.0f, 0.0f}
        };

        for (const auto& imgSize : sizes) {
            BenchmarkResult result;
            result.functionName = name_;
            result.dataType = "8u";
            result.size = imgSize.toString();
            result.channels = 3;
            result.iterations = measureIterations;

            int pixelCount = imgSize.width * imgSize.height * 3;
            int step = imgSize.width * 3 * sizeof(Npp8u);

            DeviceBuffer<Npp8u> src(pixelCount);
            DeviceBuffer<Npp8u> dst(pixelCount);

            initDeviceRandom8u(src.get(), pixelCount);

            NppiSize roiSize = {imgSize.width, imgSize.height};

            // Warmup
            for (int i = 0; i < warmupIterations; ++i) {
                fn(src.get(), step, dst.get(), step, roiSize, twist);
            }
            cudaDeviceSynchronize();

            if (runBatchedMeasurement(
                    result, measureIterations,
                    [&]() { return fn(src.get(), step, dst.get(), step, roiSize, twist); })) {
                result.throughputGBps = computeThroughput(2 * pixelCount * sizeof(Npp8u), result.avgTimeMs);
            }

            results.push_back(result);
        }

        return results;
    }
};
REGISTER_BENCHMARK(BenchColorTwist32f_8u_C3R)

} // namespace npp_benchmark
