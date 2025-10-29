/**
 * Test if NPP uses nearest neighbor for non-integer upscale
 */

#include <iostream>
#include <iomanip>
#include <cmath>

int main() {
    std::cout << "Testing if NPP uses modified nearest neighbor\n\n";

    int srcData[2][2] = {{0, 10}, {20, 30}};

    int expected[3][3] = {
        {0, 4, 10},
        {8, 13, 18},
        {20, 24, 30}
    };

    float scaleX = 2.0f / 3.0f;
    float scaleY = 2.0f / 3.0f;

    std::cout << "Source:\n";
    std::cout << "[  0] [ 10]\n";
    std::cout << "[ 20] [ 30]\n\n";

    std::cout << "Expected NPP output:\n";
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            std::cout << "[" << std::setw(3) << expected[y][x] << "] ";
        }
        std::cout << "\n";
    }

    std::cout << "\nLet's analyze the pattern in NPP's output:\n";
    std::cout << "Row 0: [0, 4, 10]    -> Differences: +4, +6\n";
    std::cout << "Row 1: [8, 13, 18]   -> Differences: +5, +5\n";
    std::cout << "Row 2: [20, 24, 30]  -> Differences: +4, +6\n";
    std::cout << "\nColumn wise:\n";
    std::cout << "Col 0: [0, 8, 20]    -> Differences: +8, +12\n";
    std::cout << "Col 1: [4, 13, 24]   -> Differences: +9, +11\n";
    std::cout << "Col 2: [10, 18, 30]  -> Differences: +8, +12\n";

    std::cout << "\nNotice:\n";
    std::cout << "- Row 0 and Row 2 have same difference pattern\n";
    std::cout << "- Col 0 and Col 2 have same difference pattern\n";
    std::cout << "- Center (13) is not the average of corners (15)\n";

    std::cout << "\nLet's check if NPP is doing weighted averaging:\n";
    std::cout << "If dst(1,0) = 4:\n";
    std::cout << "  Between src[0][0]=0 and src[0][1]=10\n";
    std::cout << "  Weight for src[0][0]: (10-4)/10 = 0.6\n";
    std::cout << "  Weight for src[0][1]: 4/10 = 0.4\n";

    std::cout << "\nIf dst(0,1) = 8:\n";
    std::cout << "  Between src[0][0]=0 and src[1][0]=20\n";
    std::cout << "  Weight for src[0][0]: (20-8)/20 = 0.6\n";
    std::cout << "  Weight for src[1][0]: 8/20 = 0.4\n";

    std::cout << "\nPattern: 0.6 / 0.4 split!\n";
    std::cout << "This suggests NPP might be using fx=0.4 or fy=0.4 instead of 0.5\n\n";

    // Test with fx=fy=0.4
    std::cout << "Testing with modified fractional parts (0.4 instead of 0.5):\n";

    for (int dy = 0; dy < 3; dy++) {
        for (int dx = 0; dx < 3; dx++) {
            float srcX = (dx + 0.5f) * scaleX - 0.5f;
            float srcY = (dy + 0.5f) * scaleY - 0.5f;

            srcX = std::max(0.0f, std::min(1.0f, srcX));
            srcY = std::max(0.0f, std::min(1.0f, srcY));

            int x0 = (int)std::floor(srcX);
            int y0 = (int)std::floor(srcY);
            int x1 = std::min(x0 + 1, 1);
            int y1 = std::min(y0 + 1, 1);

            // Instead of using actual fractional part, use 0.4
            float fx_modified = (srcX > 0 && srcX < 1) ? 0.4f : (srcX - x0);
            float fy_modified = (srcY > 0 && srcY < 1) ? 0.4f : (srcY - y0);

            float p00 = srcData[y0][x0];
            float p10 = srcData[y0][x1];
            float p01 = srcData[y1][x0];
            float p11 = srcData[y1][x1];

            float result = p00 * (1-fx_modified) * (1-fy_modified) +
                          p10 * fx_modified * (1-fy_modified) +
                          p01 * (1-fx_modified) * fy_modified +
                          p11 * fx_modified * fy_modified;

            int rounded = (int)(result + 0.5f);

            std::cout << "dst(" << dx << "," << dy << "): ";
            std::cout << "fx=" << fx_modified << ", fy=" << fy_modified << " -> ";
            std::cout << result << " -> " << rounded;
            std::cout << " (expected " << expected[dy][dx] << ")";
            if (rounded == expected[dy][dx]) {
                std::cout << " ✓";
            }
            std::cout << "\n";
        }
    }

    return 0;
}
