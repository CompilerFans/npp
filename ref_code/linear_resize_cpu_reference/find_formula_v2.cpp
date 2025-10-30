#include <iostream>
#include <cmath>

int main() {
    // Known modifier values:
    // fx=0.5, fy=0 -> mod=0.85
    // fx=0, fy=0.5 -> mod=0.85
    // fx=0.5, fy=0.5 -> mod=0.84
    
    std::cout << "Pattern analysis:\n";
    std::cout << "fx=0.5, fy=0.0: mod=0.85\n";
    std::cout << "fx=0.0, fy=0.5: mod=0.85\n";
    std::cout << "fx=0.5, fy=0.5: mod=0.84\n\n";
    
    // Try: mod = 1 - 0.15 - k*(fx*fy)
    // When fx=0.5, fy=0: mod = 1-0.15-0 = 0.85 ✓
    // When fx=0.5, fy=0.5: mod = 1-0.15-k*0.25
    // We want: 0.84 = 1-0.15-k*0.25
    // So: k*0.25 = 0.01
    // k = 0.04
    
    float k = 0.04f;
    std::cout << "Formula: modifier = 1 - 0.15 - " << k << " * (fx * fy)\n";
    std::cout << "Or:      modifier = 0.85 - " << k << " * (fx * fy)\n\n";
    
    // Test
    struct Case {
        float fx, fy;
        float expected_mod;
    };
    
    Case cases[] = {
        {0.5f, 0.0f, 0.85f},
        {0.0f, 0.5f, 0.85f},
        {0.5f, 0.5f, 0.84f},
        {0.0f, 0.0f, 0.85f}  // Doesn't matter, fx or fy is 0
    };
    
    for (auto& c : cases) {
        float mod = 0.85f - k * (c.fx * c.fy);
        std::cout << "fx=" << c.fx << ", fy=" << c.fy 
                  << " -> mod=" << mod
                  << " (expected=" << c.expected_mod << ")"
                  << (std::abs(mod - c.expected_mod) < 0.001f ? " ✓" : " ✗")
                  << "\n";
    }
    
    // Now verify with all pixels
    std::cout << "\nFull verification:\n";
    int src[2][2] = {{0, 100}, {50, 150}};
    
    struct TestCase {
        int dx, dy;
        int expected;
    };
    
    TestCase tests[] = {
        {0, 0, 0}, {1, 0, 42}, {2, 0, 100},
        {0, 1, 21}, {1, 1, 63}, {2, 1, 121},
        {0, 2, 50}, {1, 2, 92}, {2, 2, 150}
    };
    
    float scale = 2.0f / 3.0f;
    
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
        
        // Apply unified formula
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
        
        std::cout << "Pixel[" << tc.dx << "," << tc.dy << "]: "
                  << "fx=" << fx << ", fy=" << fy
                  << ", mod=" << mod
                  << " -> result=" << rounded
                  << " (expected=" << tc.expected << ")"
                  << (rounded == tc.expected ? " ✓" : " ✗")
                  << "\n";
    }
    
    return 0;
}
