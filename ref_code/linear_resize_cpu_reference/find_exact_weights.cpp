/**
 * Find exact fractional weights that match NPP
 */

#include <iostream>
#include <iomanip>
#include <cmath>

int main() {
    int srcData[2][2] = {{0, 10}, {20, 30}};
    int expected[3][3] = {
        {0, 4, 10},
        {8, 13, 18},
        {20, 24, 30}
    };

    std::cout << "Searching for exact fractional weights...\n\n";

    // For dst(1,0): should be 4
    // p00=0, p10=10, interpolation along x
    // result = p00 * (1-fx) + p10 * fx = 0 + 10*fx = 4
    // fx = 0.4

    float fx_edge = 0.4f;
    std::cout << "For edge pixels (one direction): fx = fy = 0.4\n\n";

    // For dst(1,1): should be 13
    // p00=0, p10=10, p01=20, p11=30
    // result = 0*(1-fx)*(1-fy) + 10*fx*(1-fy) + 20*(1-fx)*fy + 30*fx*fy
    //        = 10*fx*(1-fy) + 20*(1-fx)*fy + 30*fx*fy
    //        = 10*fx - 10*fx*fy + 20*fy - 20*fx*fy + 30*fx*fy
    //        = 10*fx + 20*fy + fx*fy*(30 - 10 - 20)
    //        = 10*fx + 20*fy

    std::cout << "For center pixel dst(1,1):\n";
    std::cout << "Testing different fx and fy values:\n\n";

    bool found = false;
    for (float fx = 0.0f; fx <= 1.0f && !found; fx += 0.01f) {
        for (float fy = 0.0f; fy <= 1.0f && !found; fy += 0.01f) {
            float result = 0 * (1-fx) * (1-fy) +
                          10 * fx * (1-fy) +
                          20 * (1-fx) * fy +
                          30 * fx * fy;

            int rounded = (int)(result + 0.5f);

            if (rounded == 13) {
                std::cout << "fx=" << std::fixed << std::setprecision(2) << fx
                         << ", fy=" << fy << " -> " << result << " -> " << rounded << "\n";

                // Check if this also works for other pixels
                std::cout << "Checking all pixels with fx=" << fx << ", fy=" << fy << ":\n";

                bool all_match = true;
                for (int dy = 0; dy < 3; dy++) {
                    for (int dx = 0; dx < 3; dx++) {
                        float srcX = (dx + 0.5f) * (2.0f/3.0f) - 0.5f;
                        float srcY = (dy + 0.5f) * (2.0f/3.0f) - 0.5f;

                        srcX = std::max(0.0f, std::min(1.0f, srcX));
                        srcY = std::max(0.0f, std::min(1.0f, srcY));

                        int x0 = (int)std::floor(srcX);
                        int y0 = (int)std::floor(srcY);
                        int x1 = std::min(x0 + 1, 1);
                        int y1 = std::min(y0 + 1, 1);

                        // Use custom fx/fy for interpolated pixels
                        float fx_use = (srcX > 0 && srcX < 1 && x0 != x1) ? fx : (srcX - x0);
                        float fy_use = (srcY > 0 && srcY < 1 && y0 != y1) ? fy : (srcY - y0);

                        float p00 = srcData[y0][x0];
                        float p10 = srcData[y0][x1];
                        float p01 = srcData[y1][x0];
                        float p11 = srcData[y1][x1];

                        float res = p00 * (1-fx_use) * (1-fy_use) +
                                   p10 * fx_use * (1-fy_use) +
                                   p01 * (1-fx_use) * fy_use +
                                   p11 * fx_use * fy_use;

                        int rnd = (int)(res + 0.5f);

                        if (rnd != expected[dy][dx]) {
                            all_match = false;
                            break;
                        }
                    }
                    if (!all_match) break;
                }

                if (all_match) {
                    std::cout << "✓ ALL PIXELS MATCH!\n";
                    found = true;
                } else {
                    std::cout << "✗ Not all pixels match\n";
                }

                if (found) break;
            }
        }
    }

    if (!found) {
        std::cout << "\nNo single fx/fy pair matches all pixels.\n";
        std::cout << "NPP likely uses different logic.\n";
    }

    return 0;
}
