/**
 * Test if NPP uses a bias in interpolation
 */

#include <iostream>
#include <iomanip>
#include <cmath>

int main() {
    std::cout << "Testing interpolation with bias/offset\n\n";

    int srcData[2][2] = {{0, 10}, {20, 30}};

    int expected[3][3] = {
        {0, 4, 10},
        {8, 13, 18},
        {20, 24, 30}
    };

    float scaleX = 2.0f / 3.0f;
    float scaleY = 2.0f / 3.0f;

    std::cout << "Testing dst(1,1) which should be 13:\n";
    std::cout << "Standard bilinear gives 15, NPP gives 13, difference = -2\n\n";

    // Standard calculation
    float srcX = (1 + 0.5f) * scaleX - 0.5f;  // 0.5
    float srcY = (1 + 0.5f) * scaleY - 0.5f;  // 0.5

    int x0 = 0, y0 = 0, x1 = 1, y1 = 1;
    float fx = 0.5f, fy = 0.5f;

    float p00 = 0, p10 = 10, p01 = 20, p11 = 30;

    std::cout << "Standard bilinear:\n";
    float std_result = p00 * (1-fx) * (1-fy) +
                       p10 * fx * (1-fy) +
                       p01 * (1-fx) * fy +
                       p11 * fx * fy;
    std::cout << "  Result = " << std_result << " -> " << (int)(std_result + 0.5f) << "\n\n";

    // Try: maybe NPP subtracts a small bias before rounding?
    std::cout << "Testing different biases:\n";
    for (float bias = 0.0f; bias <= 3.0f; bias += 0.1f) {
        int rounded = (int)(std_result - bias + 0.5f);
        if (rounded == 13) {
            std::cout << "  Bias = " << bias << " -> " << rounded << " ✓ MATCH!\n";
        }
    }

    // Or maybe NPP uses a different rounding point?
    std::cout << "\nTesting different rounding points:\n";
    for (float round_point = 0.0f; round_point <= 1.0f; round_point += 0.05f) {
        int rounded = (int)(std_result + round_point);
        if (rounded == 13) {
            std::cout << "  Round point = " << round_point << " -> " << rounded << " ✓ MATCH!\n";
        }
    }

    // Maybe NPP uses integer arithmetic scaled by 256?
    std::cout << "\nTesting fixed-point arithmetic (scale 256):\n";
    int fx_fixed = (int)(fx * 256);  // 128
    int fy_fixed = (int)(fy * 256);  // 128

    int result_fixed = ((int)p00 * (256 - fx_fixed) * (256 - fy_fixed) +
                       (int)p10 * fx_fixed * (256 - fy_fixed) +
                       (int)p01 * (256 - fx_fixed) * fy_fixed +
                       (int)p11 * fx_fixed * fy_fixed) / (256 * 256);

    std::cout << "  fx_fixed = " << fx_fixed << "\n";
    std::cout << "  fy_fixed = " << fy_fixed << "\n";
    std::cout << "  Result = " << result_fixed << "\n";
    if (result_fixed == 13) {
        std::cout << "  ✓ MATCH!\n";
    }

    // Try with rounding in the division
    int result_fixed_round = ((int)p00 * (256 - fx_fixed) * (256 - fy_fixed) +
                              (int)p10 * fx_fixed * (256 - fy_fixed) +
                              (int)p01 * (256 - fx_fixed) * fy_fixed +
                              (int)p11 * fx_fixed * fy_fixed + (256*256/2)) / (256 * 256);

    std::cout << "  With rounding: " << result_fixed_round << "\n";
    if (result_fixed_round == 13) {
        std::cout << "  ✓ MATCH!\n";
    }

    // Let's try to reverse-engineer: if result should be 13, what weights would give that?
    std::cout << "\nReverse engineering for dst(1,1):\n";
    std::cout << "  Need: p00*w00 + p10*w10 + p01*w01 + p11*w11 = 13\n";
    std::cout << "  With p00=0, p10=10, p01=20, p11=30\n";
    std::cout << "  So: 10*w10 + 20*w01 + 30*w11 = 13\n";
    std::cout << "  Standard gives: 10*0.25 + 20*0.25 + 30*0.25 = 2.5 + 5 + 7.5 = 15\n";
    std::cout << "  Need 13, difference = -2\n";
    std::cout << "  If we reduce all contributions proportionally: 13/15 = 0.8667\n";

    // What if NPP uses a different weight calculation?
    // Standard: w = (1-fx) * (1-fy) = 0.5 * 0.5 = 0.25
    // What if NPP uses: w = (1-fx) + (1-fy) - 1 = ?

    std::cout << "\nTrying alternative weight formulas:\n";

    // Separable filter approach with different combination
    float w_x0 = 1 - fx;  // 0.5
    float w_x1 = fx;      // 0.5
    float w_y0 = 1 - fy;  // 0.5
    float w_y1 = fy;      // 0.5

    // What if weights are not multiplied but added/averaged?
    float alt1 = (p00 * w_x0 + p10 * w_x1) * w_y0 +
                 (p01 * w_x0 + p11 * w_x1) * w_y1;
    std::cout << "  Separable (standard): " << alt1 << " -> " << (int)(alt1 + 0.5f) << "\n";

    return 0;
}
