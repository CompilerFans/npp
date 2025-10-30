#include <iostream>
#include <cmath>

float nvidia_upscale_interp(float v0, float v1, float f, bool is_center) {
    // NVIDIA uses modified weights for upscale
    float f_mod = is_center ? (f * 0.84f) : (f * 0.85f);
    return v0 * (1 - f_mod) + v1 * f_mod;
}

int main() {
    // Test all 9 pixels
    struct TestCase {
        int dx, dy;
        int expected;
        bool is_center_x, is_center_y;
    };
    
    TestCase cases[] = {
        {0, 0, 0, false, false},
        {1, 0, 42, true, false},
        {2, 0, 100, false, false},
        {0, 1, 21, false, true},
        {1, 1, 63, true, true},
        {2, 1, 121, false, true},
        {0, 2, 50, false, false},
        {1, 2, 92, true, false},
        {2, 2, 150, false, false}
    };
    
    // Source
    int src[2][2] = {{0, 100}, {50, 150}};
    float scale = 2.0f / 3.0f;
    
    std::cout << "Testing NVIDIA pattern with adaptive weights:\n\n";
    
    for (auto& tc : cases) {
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
        
        float p00 = src[y0][x0];
        float p10 = src[y0][x1];
        float p01 = src[y1][x0];
        float p11 = src[y1][x1];
        
        float v0 = nvidia_upscale_interp(p00, p10, fx, tc.is_center_x);
        float v1 = nvidia_upscale_interp(p01, p11, fx, tc.is_center_x);
        float result = nvidia_upscale_interp(v0, v1, fy, tc.is_center_y);
        
        int rounded = (int)(result + 0.5f);
        
        std::cout << "Pixel[" << tc.dx << "," << tc.dy << "]: "
                  << "expected=" << tc.expected 
                  << ", computed=" << rounded
                  << (rounded == tc.expected ? " ✓" : " ✗")
                  << "\n";
    }
    
    return 0;
}
