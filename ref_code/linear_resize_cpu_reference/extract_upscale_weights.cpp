#include <iostream>
#include <iomanip>
#include <map>
#include <cmath>
#include <npp.h>

// Extract NPP's weight pattern for integer upscales
void extractWeights(int factor) {
    std::cout << "\n========================================\n";
    std::cout << factor << "x UPSCALE WEIGHT EXTRACTION\n";
    std::cout << "========================================\n\n";

    // Use simple 2-pixel source to isolate weight behavior
    int srcW = 2, srcH = 2;
    int dstW = srcW * factor;
    int dstH = srcH * factor;

    uint8_t* srcData = new uint8_t[srcW * srcH];
    // Simple gradient: 0, 100, 0, 100
    srcData[0] = 0;
    srcData[1] = 100;
    srcData[2] = 0;
    srcData[3] = 100;

    std::cout << "Source pattern: [0, 100] in first row\n\n";

    uint8_t* nppResult = new uint8_t[dstW * dstH];
    uint8_t *d_src, *d_dst;
    cudaMalloc(&d_src, srcW * srcH);
    cudaMalloc(&d_dst, dstW * dstH);
    cudaMemcpy(d_src, srcData, srcW * srcH, cudaMemcpyHostToDevice);

    NppiSize srcSz = {srcW, srcH};
    NppiSize dstSz = {dstW, dstH};
    NppiRect srcROI = {0, 0, srcW, srcH};
    NppiRect dstROI = {0, 0, dstW, dstH};

    nppiResize_8u_C1R(d_src, srcW, srcSz, srcROI,
                     d_dst, dstW, dstSz, dstROI,
                     NPPI_INTER_LINEAR);

    cudaMemcpy(nppResult, d_dst, dstW * dstH, cudaMemcpyDeviceToHost);

    // Analyze first row only (Y interpolation doesn't matter here)
    float scale = 1.0f / factor;

    std::cout << "Destination pixels (row 0):\n";
    std::cout << std::setw(5) << "dx" << std::setw(10) << "srcX"
             << std::setw(8) << "fx" << std::setw(10) << "NPP"
             << std::setw(12) << "fx_eff" << std::setw(10) << "ratio\n";
    std::cout << std::string(65, '-') << "\n";

    std::map<float, float> fx_to_ratio;

    for (int dx = 0; dx < dstW; dx++) {
        float srcX = (dx + 0.5f) * scale - 0.5f;
        float fx = srcX - std::floor(srcX);

        // Clamp fx to [0, 1]
        if (fx < 0) fx = 0;
        if (fx > 1) fx = 1;

        int npp = nppResult[dx];

        // Since we know src[0]=0 and src[1]=100
        // NPP value directly tells us the effective weight
        float fx_effective = npp / 100.0f;

        float ratio = 0;
        if (fx > 0.001f) {
            ratio = fx_effective / fx;
        }

        std::cout << std::setw(5) << dx
                 << std::setw(10) << std::fixed << std::setprecision(3) << srcX
                 << std::setw(8) << fx
                 << std::setw(10) << npp
                 << std::setw(12) << fx_effective
                 << std::setw(10) << std::setprecision(2) << ratio << "\n";

        // Collect ratios for pattern analysis
        if (fx > 0.001f && fx < 0.999f) {
            fx_to_ratio[fx] = ratio;
        }
    }

    // Analyze pattern
    std::cout << "\n--- PATTERN ANALYSIS ---\n";
    if (!fx_to_ratio.empty()) {
        std::cout << "\nUnique (fx -> ratio) mappings:\n";
        for (const auto& pair : fx_to_ratio) {
            std::cout << "  fx=" << std::fixed << std::setprecision(3) << pair.first
                     << " -> ratio=" << std::setprecision(2) << pair.second << "\n";
        }

        // Check if ratio depends on fx
        float min_ratio = 100, max_ratio = 0;
        for (const auto& pair : fx_to_ratio) {
            min_ratio = std::min(min_ratio, pair.second);
            max_ratio = std::max(max_ratio, pair.second);
        }

        std::cout << "\nRatio range: [" << std::setprecision(2)
                 << min_ratio << ", " << max_ratio << "]\n";

        if (max_ratio - min_ratio < 0.1f) {
            std::cout << "Pattern: CONSTANT ratio (approximately "
                     << std::setprecision(2) << (min_ratio + max_ratio) / 2 << ")\n";
        } else {
            std::cout << "Pattern: VARIABLE ratio (depends on fx)\n";

            // Try to fit a formula
            std::cout << "\nTrying to identify formula...\n";

            // Check if it's linear: ratio = a + b*fx
            if (fx_to_ratio.size() >= 2) {
                auto it1 = fx_to_ratio.begin();
                auto it2 = fx_to_ratio.rbegin();
                float fx1 = it1->first, r1 = it1->second;
                float fx2 = it2->first, r2 = it2->second;

                float slope = (r2 - r1) / (fx2 - fx1);
                float intercept = r1 - slope * fx1;

                std::cout << "  Linear fit: ratio = " << std::setprecision(3)
                         << intercept << " + " << slope << " * fx\n";
            }
        }
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    delete[] srcData;
    delete[] nppResult;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "INTEGER UPSCALE WEIGHT EXTRACTION\n";
    std::cout << "Isolating NPP's interpolation weights\n";
    std::cout << "========================================\n";

    extractWeights(2);  // Working case
    extractWeights(3);  // Problematic
    extractWeights(4);  // Problematic
    extractWeights(5);  // Problematic
    extractWeights(8);  // Problematic

    return 0;
}
