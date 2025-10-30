#include <iostream>
#include <cmath>

int main() {
    std::cout << "Testing NVIDIA rounding method:\n\n";
    
    int src[2][2] = {{0, 100}, {50, 150}};
    float scale = 2.0f / 3.0f;
    
    struct TestCase {
        int dx, dy;
        int expected;
    };
    
    TestCase tests[] = {
        {0, 0, 0}, {1, 0, 42}, {2, 0, 100},
        {0, 1, 21}, {1, 1, 63}, {2, 1, 121},
        {0, 2, 50}, {1, 2, 92}, {2, 2, 150}
    };
    
    for (auto& tc : tests) {
        float sx = (tc.dx + 0.5f) * scale - 0.5f;
        float sy = (tc.dy + 0.5f) * scale - 0.5f;
        
        sx = std::max(0.0f, std::min(1.0f, sx));
        sy = std::max(0.0f, std::min(1.0f, sy));
        
        int x0 = (int)std::floor(sx);
        int y0 = (int)std::floor(sy);
        int x1 = std::min(x0 + 1, 1);
        int y1 = std::min(y0 + 1, 1);
        
        float fx = sx - x0;
        float fy = sy - y0;
        
        // Use 0.85 - 0.04 * fx * fy
        float mod = 0.85f - 0.04f * (fx * fy);
        float fx_mod = fx * mod;
        float fy_mod = fy * mod;
        
        float p00 = src[y0][x0];
        float p10 = src[y0][x1];
        float p01 = src[y1][x0];
        float p11 = src[y1][x1];
        
        float v0 = p00 * (1 - fx_mod) + p10 * fx_mod;
        float v1 = p01 * (1 - fx_mod) + p11 * fx_mod;
        float result = v0 * (1 - fy_mod) + v1 * fy_mod;
        
        int rounded = (int)(result + 0.5f);
        int floored = (int)std::floor(result);
        int ceiled = (int)std::ceil(result);
        
        std::cout << "Pixel[" << tc.dx << "," << tc.dy << "]: result=" << result << "\n";
        std::cout << "  round: " << rounded << (rounded == tc.expected ? " ✓" : " ✗") << "\n";
        std::cout << "  floor: " << floored << (floored == tc.expected ? " ✓" : " ✗") << "\n";
        std::cout << "  ceil:  " << ceiled << (ceiled == tc.expected ? " ✓" : " ✗") << "\n";
        std::cout << "  expected: " << tc.expected << "\n\n";
    }
    
    return 0;
}
