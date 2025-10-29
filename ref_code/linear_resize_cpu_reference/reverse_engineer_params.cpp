#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <npp.h>

// Reverse engineer NPP's parameters for fractional downscales
void analyzeParameters(const char* name, int srcW, int dstW) {
    std::cout << "\n========================================\n";
    std::cout << "REVERSE ENGINEERING: " << name << " (" << srcW << " -> " << dstW << ")\n";
    std::cout << "Scale: " << std::fixed << std::setprecision(4) << (float)srcW / dstW << "\n";
    std::cout << "========================================\n\n";

    // Create 2D gradient
    int srcH = 2, dstH = 2;
    uint8_t* srcData = new uint8_t[srcW * srcH];
    for (int i = 0; i < srcW; i++) {
        srcData[i] = (i * 255) / (srcW - 1);
        srcData[srcW + i] = (i * 255) / (srcW - 1);
    }

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

    float scale = (float)srcW / dstW;

    std::cout << "Source: ";
    for (int i = 0; i < srcW; i++) {
        std::cout << std::setw(3) << (int)srcData[i] << " ";
    }
    std::cout << "\n\nNPP:    ";
    for (int i = 0; i < dstW; i++) {
        std::cout << std::setw(3) << (int)nppResult[i] << " ";
    }
    std::cout << "\n\n";

    // Analyze each pixel
    std::vector<float> computed_fx_values;
    std::vector<float> original_fx_values;
    std::vector<bool> is_floor_match;

    for (int dx = 0; dx < dstW; dx++) {
        float srcX = (dx + 0.5f) * scale - 0.5f;
        int x0 = (int)std::floor(srcX);
        int x1 = std::min(x0 + 1, srcW - 1);
        x0 = std::max(0, x0);
        float fx = srcX - std::floor(srcX);

        int npp = nppResult[dx];
        int v0 = srcData[x0];
        int v1 = srcData[x1];

        bool floor_match = (npp == v0);
        is_floor_match.push_back(floor_match);
        original_fx_values.push_back(fx);

        if (floor_match) {
            std::cout << "dst[" << dx << "]: NPP=" << std::setw(3) << npp
                     << " = floor(src[" << std::fixed << std::setprecision(3) << srcX << "]) "
                     << "fx=" << fx << " [FLOOR]\n";
            computed_fx_values.push_back(0.0f);
        } else if (v0 != v1) {
            // Reverse calculate fx
            // npp = v0 * (1 - fx_effective) + v1 * fx_effective
            // npp = v0 + (v1 - v0) * fx_effective
            // fx_effective = (npp - v0) / (v1 - v0)
            float fx_effective = (float)(npp - v0) / (v1 - v0);
            computed_fx_values.push_back(fx_effective);

            float bilinear_expected = v0 * (1 - fx) + v1 * fx;
            float attenuation = (fx > 0.001f) ? (fx_effective / fx) : 0.0f;

            std::cout << "dst[" << dx << "]: NPP=" << std::setw(3) << npp
                     << " from [" << v0 << "," << v1 << "] "
                     << "fx_orig=" << std::setprecision(3) << fx
                     << " fx_eff=" << fx_effective
                     << " atten=" << std::setprecision(2) << attenuation
                     << " bilin=" << std::setprecision(1) << bilinear_expected << "\n";
        } else {
            std::cout << "dst[" << dx << "]: NPP=" << std::setw(3) << npp
                     << " (same neighbors) fx=" << fx << "\n";
            computed_fx_values.push_back(fx);
        }
    }

    // Statistical analysis
    std::cout << "\n--- STATISTICAL ANALYSIS ---\n";

    // Find threshold between FLOOR and INTERPOLATE
    float min_interp_fx = 1.0f;
    float max_floor_fx = 0.0f;
    for (int i = 0; i < dstW; i++) {
        if (is_floor_match[i]) {
            max_floor_fx = std::max(max_floor_fx, original_fx_values[i]);
        } else {
            min_interp_fx = std::min(min_interp_fx, original_fx_values[i]);
        }
    }

    std::cout << "FLOOR threshold: fx <= " << std::setprecision(3) << max_floor_fx << "\n";
    std::cout << "INTERP starts at: fx >= " << min_interp_fx << "\n";
    std::cout << "Estimated threshold: " << (max_floor_fx + min_interp_fx) / 2.0f << "\n\n";

    // Calculate average attenuation for interpolated pixels
    std::vector<float> attenuations;
    for (int i = 0; i < dstW; i++) {
        if (!is_floor_match[i] && original_fx_values[i] > 0.001f) {
            float atten = computed_fx_values[i] / original_fx_values[i];
            attenuations.push_back(atten);
        }
    }

    if (!attenuations.empty()) {
        float sum = 0, min_a = 1.0f, max_a = 0.0f;
        for (float a : attenuations) {
            sum += a;
            min_a = std::min(min_a, a);
            max_a = std::max(max_a, a);
        }
        float avg = sum / attenuations.size();
        std::cout << "Attenuation factor:\n";
        std::cout << "  Average: " << std::setprecision(3) << avg << "\n";
        std::cout << "  Range: [" << min_a << ", " << max_a << "]\n";
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    delete[] srcData;
    delete[] nppResult;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "REVERSE ENGINEERING NPP PARAMETERS\n";
    std::cout << "For fractional downscales 1.0 < scale < 2.0\n";
    std::cout << "========================================\n";

    // Fractional downscales that don't match
    analyzeParameters("0.875x (7/8)", 8, 7);
    analyzeParameters("0.8x (4/5)", 10, 8);
    analyzeParameters("0.75x (3/4)", 8, 6);
    analyzeParameters("0.67x (2/3)", 9, 6);
    analyzeParameters("0.6x (3/5)", 10, 6);

    // Compare with working case
    std::cout << "\n========================================\n";
    std::cout << "COMPARISON WITH WORKING CASE\n";
    std::cout << "========================================\n";
    analyzeParameters("0.5x (1/2) [WORKING]", 8, 4);

    return 0;
}
