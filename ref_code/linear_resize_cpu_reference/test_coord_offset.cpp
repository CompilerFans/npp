/**
 * Test if NPP uses a coordinate offset
 */

#include <iostream>
#include <iomanip>
#include <cmath>

int main() {
    std::cout << "Testing coordinate mapping with small offsets\n\n";

    int srcData[2][2] = {{0, 10}, {20, 30}};
    int srcW = 2, srcH = 2;
    int dstW = 3, dstH = 3;

    // Test different coordinate mapping formulas
    struct Formula {
        const char* name;
        float offset;
    };

    Formula formulas[] = {
        {"Standard (0.5)", 0.5f},
        {"Offset 0.4", 0.4f},
        {"Offset 0.45", 0.45f},
        {"Offset 0.3333", 0.3333f},
        {"Offset 0.25", 0.25f},
        {"Offset 0.6", 0.6f},
    };

    int expected[3][3] = {
        {0, 4, 10},
        {8, 13, 18},
        {20, 24, 30}
    };

    for (const auto& formula : formulas) {
        std::cout << "=== " << formula.name << " ===\n";

        float scaleX = (float)srcW / dstW;
        float scaleY = (float)srcH / dstH;

        int matches = 0;
        int total = 0;

        for (int dy = 0; dy < dstH; dy++) {
            for (int dx = 0; dx < dstW; dx++) {
                // Try: srcCoord = (dstCoord + offset) * scale - offset
                float srcX = (dx + formula.offset) * scaleX - formula.offset;
                float srcY = (dy + formula.offset) * scaleY - formula.offset;

                // Clamp
                srcX = std::max(0.0f, std::min(1.0f, srcX));
                srcY = std::max(0.0f, std::min(1.0f, srcY));

                // Interpolate
                int x0 = (int)std::floor(srcX);
                int y0 = (int)std::floor(srcY);
                int x1 = std::min(x0 + 1, 1);
                int y1 = std::min(y0 + 1, 1);

                float fx = srcX - x0;
                float fy = srcY - y0;

                float p00 = srcData[y0][x0];
                float p10 = srcData[y0][x1];
                float p01 = srcData[y1][x0];
                float p11 = srcData[y1][x1];

                float result = p00 * (1-fx) * (1-fy) +
                              p10 * fx * (1-fy) +
                              p01 * (1-fx) * fy +
                              p11 * fx * fy;

                int rounded = (int)(result + 0.5f);

                if (rounded == expected[dy][dx]) {
                    matches++;
                }
                total++;
            }
        }

        std::cout << "Matches: " << matches << "/" << total;
        if (matches == total) {
            std::cout << " ✓ PERFECT MATCH!\n";
        }
        std::cout << "\n\n";
    }

    // Try completely different approach: maybe NPP uses edge-to-edge mapping for non-integer scales?
    std::cout << "=== Edge-to-edge mapping ===\n";

    int matches = 0;
    for (int dy = 0; dy < dstH; dy++) {
        for (int dx = 0; dx < dstW; dx++) {
            // Edge-to-edge: maps corner to corner
            float srcX = dx * (float)(srcW - 1) / (dstW - 1);
            float srcY = dy * (float)(srcH - 1) / (dstH - 1);

            int x0 = (int)std::floor(srcX);
            int y0 = (int)std::floor(srcY);
            int x1 = std::min(x0 + 1, 1);
            int y1 = std::min(y0 + 1, 1);

            float fx = srcX - x0;
            float fy = srcY - y0;

            float p00 = srcData[y0][x0];
            float p10 = srcData[y0][x1];
            float p01 = srcData[y1][x0];
            float p11 = srcData[y1][x1];

            float result = p00 * (1-fx) * (1-fy) +
                          p10 * fx * (1-fy) +
                          p01 * (1-fx) * fy +
                          p11 * fx * fy;

            int rounded = (int)(result + 0.5f);

            std::cout << "dst(" << dx << "," << dy << "): "
                     << "src(" << srcX << "," << srcY << ") -> "
                     << result << " -> " << rounded
                     << " (expected " << expected[dy][dx] << ")";

            if (rounded == expected[dy][dx]) {
                std::cout << " ✓";
                matches++;
            }
            std::cout << "\n";
        }
    }

    std::cout << "\nMatches: " << matches << "/9\n";

    return 0;
}
