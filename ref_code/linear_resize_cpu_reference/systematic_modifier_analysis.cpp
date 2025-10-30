#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include <npp.h>

struct ModifierData {
    float scale;
    float fx;
    float modifier;
    float fx_effective;
};

std::vector<ModifierData> collectModifiers() {
    std::vector<ModifierData> results;

    // Test various resolutions to get different scale and fx combinations
    struct TestConfig {
        int srcW, srcH, dstW, dstH;
    };

    std::vector<TestConfig> configs = {
        // Upscales with different scales
        {2, 1, 3, 1},    // scale=0.667
        {2, 1, 4, 1},    // scale=0.5
        {2, 1, 5, 1},    // scale=0.4
        {2, 1, 7, 1},    // scale=0.286
        {3, 1, 4, 1},    // scale=0.75
        {3, 1, 5, 1},    // scale=0.6
        {3, 1, 7, 1},    // scale=0.429
        {4, 1, 5, 1},    // scale=0.8
        {4, 1, 7, 1},    // scale=0.571
        {5, 1, 7, 1},    // scale=0.714
        {5, 1, 9, 1},    // scale=0.556

        // Downscales
        {3, 1, 2, 1},    // scale=1.5
        {4, 1, 3, 1},    // scale=1.333
        {5, 1, 3, 1},    // scale=1.667
        {5, 1, 4, 1},    // scale=1.25
        {7, 1, 5, 1},    // scale=1.4
    };

    for (auto& cfg : configs) {
        // Use simple pattern: [0, 255]
        std::vector<unsigned char> src(cfg.srcW * cfg.srcH);
        for (int i = 0; i < cfg.srcW; i++) {
            src[i] = (unsigned char)(i * 255 / std::max(1, cfg.srcW - 1));
        }

        // Run NVIDIA NPP
        int channels = 1;
        unsigned char *d_src, *d_dst;
        cudaMalloc(&d_src, cfg.srcW * cfg.srcH * channels);
        cudaMalloc(&d_dst, cfg.dstW * cfg.dstH * channels);
        cudaMemcpy(d_src, src.data(), cfg.srcW * cfg.srcH * channels, cudaMemcpyHostToDevice);

        NppiSize srcSize = {cfg.srcW, cfg.srcH};
        NppiSize dstSize = {cfg.dstW, cfg.dstH};
        NppiRect srcROI = {0, 0, cfg.srcW, cfg.srcH};
        NppiRect dstROI = {0, 0, cfg.dstW, cfg.dstH};

        nppiResize_8u_C1R(d_src, cfg.srcW * channels, srcSize, srcROI,
                          d_dst, cfg.dstW * channels, dstSize, dstROI,
                          NPPI_INTER_LINEAR);

        std::vector<unsigned char> nppResult(cfg.dstW * cfg.dstH * channels);
        cudaMemcpy(nppResult.data(), d_dst, cfg.dstW * cfg.dstH * channels, cudaMemcpyDeviceToHost);

        // Analyze 1D interpolations
        float scaleX = (float)cfg.srcW / cfg.dstW;

        for (int dx = 0; dx < cfg.dstW; dx++) {
            float srcX = (dx + 0.5f) * scaleX - 0.5f;
            srcX = std::max(0.0f, std::min((float)(cfg.srcW - 1), srcX));

            int x0 = (int)std::floor(srcX);
            int x1 = std::min(x0 + 1, cfg.srcW - 1);
            float fx = srcX - x0;

            if (fx < 0.01f || fx > 0.99f) continue;  // Skip near-integer positions

            int p0 = src[x0];
            int p1 = src[x1];
            if (std::abs(p1 - p0) < 2) continue;  // Need sufficient delta

            int npp = nppResult[dx];

            float fx_eff = (npp - p0) / (float)(p1 - p0);
            float modifier = fx_eff / fx;

            ModifierData md;
            md.scale = scaleX;
            md.fx = fx;
            md.modifier = modifier;
            md.fx_effective = fx_eff;

            results.push_back(md);
        }

        cudaFree(d_src);
        cudaFree(d_dst);
    }

    return results;
}

int main() {
    std::cout << "Systematic Modifier Analysis\n";
    std::cout << "=============================\n\n";

    auto data = collectModifiers();

    // Sort by scale, then by fx
    std::sort(data.begin(), data.end(), [](const ModifierData& a, const ModifierData& b) {
        if (std::abs(a.scale - b.scale) > 0.01f) return a.scale < b.scale;
        return a.fx < b.fx;
    });

    // Print results
    std::cout << std::setw(8) << "Scale" << " | "
              << std::setw(6) << "fx" << " | "
              << std::setw(8) << "Modifier" << " | "
              << std::setw(8) << "fx_eff" << "\n";
    std::cout << std::string(40, '-') << "\n";

    for (auto& d : data) {
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(8) << d.scale << " | "
                  << std::setw(6) << d.fx << " | "
                  << std::setw(8) << std::setprecision(4) << d.modifier << " | "
                  << std::setw(8) << std::setprecision(4) << d.fx_effective << "\n";
    }

    // Try to find formula patterns
    std::cout << "\n\nAnalyzing patterns:\n";
    std::cout << std::string(60, '=') << "\n";

    // Group by scale ranges
    std::cout << "\nUPSCALE (scale < 1.0):\n";
    std::cout << std::string(40, '-') << "\n";

    for (auto& d : data) {
        if (d.scale < 1.0f && d.scale >= 0.0f) {
            // Try formula: modifier = a - b * fx
            // or modifier = a * scale + b
            std::cout << "scale=" << std::setw(5) << std::setprecision(3) << d.scale
                      << " fx=" << std::setw(5) << d.fx
                      << " mod=" << std::setw(6) << std::setprecision(4) << d.modifier;

            // Try: mod = 1 - c*fx for some c
            float c = (1.0f - d.modifier) / d.fx;
            std::cout << "  |  (1-mod)/fx=" << std::setw(6) << std::setprecision(4) << c;

            // Try: mod = a * scale
            float a = d.modifier / d.scale;
            std::cout << "  |  mod/scale=" << std::setw(6) << std::setprecision(4) << a;

            std::cout << "\n";
        }
    }

    std::cout << "\nDOWNSCALE (scale >= 1.0):\n";
    std::cout << std::string(40, '-') << "\n";

    for (auto& d : data) {
        if (d.scale >= 1.0f) {
            std::cout << "scale=" << std::setw(5) << std::setprecision(3) << d.scale
                      << " fx=" << std::setw(5) << d.fx
                      << " mod=" << std::setw(6) << std::setprecision(4) << d.modifier
                      << "\n";
        }
    }

    return 0;
}
