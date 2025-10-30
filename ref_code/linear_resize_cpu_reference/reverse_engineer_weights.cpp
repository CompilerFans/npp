#include <iostream>
#include <cmath>

int main() {
    // Source: [0, 100]
    //         [50, 150]
    
    // NVIDIA results for 3x3 output
    struct Result { int x, y, value; };
    Result results[] = {
        {0,0,0}, {1,0,42}, {2,0,100},
        {0,1,21}, {1,1,63}, {2,1,121},
        {0,2,50}, {1,2,92}, {2,2,150}
    };
    
    float scale = 2.0f / 3.0f; // 0.667
    
    std::cout << "Reverse engineering NVIDIA linear interpolation\n";
    std::cout << "Scale factor: " << scale << "\n\n";
    
    for (auto& r : results) {
        // Standard mapping: (dx+0.5)*scale - 0.5
        float sx_std = (r.x + 0.5f) * scale - 0.5f;
        float sy_std = (r.y + 0.5f) * scale - 0.5f;
        
        // Clamp
        sx_std = std::max(0.0f, std::min(1.0f, sx_std));
        sy_std = std::max(0.0f, std::min(1.0f, sy_std));
        
        // Get integer coords
        int x0 = (int)std::floor(sx_std);
        int y0 = (int)std::floor(sy_std);
        int x1 = std::min(x0 + 1, 1);
        int y1 = std::min(y0 + 1, 1);
        
        // Fractional part
        float fx = sx_std - x0;
        float fy = sy_std - y0;
        
        // Source values
        int p00 = (y0 == 0 && x0 == 0) ? 0 : (y0 == 0 && x0 == 1) ? 100 : (y0 == 1 && x0 == 0) ? 50 : 150;
        int p10 = (y0 == 0 && x1 == 0) ? 0 : (y0 == 0 && x1 == 1) ? 100 : (y0 == 1 && x1 == 0) ? 50 : 150;
        int p01 = (y1 == 0 && x0 == 0) ? 0 : (y1 == 0 && x0 == 1) ? 100 : (y1 == 1 && x0 == 0) ? 50 : 150;
        int p11 = (y1 == 0 && x1 == 0) ? 0 : (y1 == 0 && x1 == 1) ? 100 : (y1 == 1 && x1 == 0) ? 50 : 150;
        
        // Standard bilinear
        float v0_std = p00 * (1-fx) + p10 * fx;
        float v1_std = p01 * (1-fx) + p11 * fx;
        float result_std = v0_std * (1-fy) + v1_std * fy;
        
        std::cout << "Pixel[" << r.x << "," << r.y << "] = " << r.value << "\n";
        std::cout << "  Coords: (" << sx_std << ", " << sy_std << "), fx=" << fx << ", fy=" << fy << "\n";
        std::cout << "  Corners: p00=" << p00 << ", p10=" << p10 << ", p01=" << p01 << ", p11=" << p11 << "\n";
        std::cout << "  Std bilinear: " << result_std << " -> " << (int)(result_std+0.5f) << "\n";
        std::cout << "  NVIDIA value: " << r.value << "\n";
        std::cout << "  Difference: " << (result_std - r.value) << "\n";
        
        // Try to find the actual weights NVIDIA used
        if (r.value != (int)(result_std+0.5f)) {
            // NVIDIA value is different, let's solve for the weights
            // Assume: result = v0 * wy0 + v1 * wy1
            // where v0 = p00 * wx0 + p10 * wx1
            //       v1 = p01 * wx0 + p11 * wx1
            
            std::cout << "  Trying to find actual weights...\n";
            
            // For upscale (scale < 1.0), try different weight modifications
            for (float mod = 0.5f; mod <= 1.0f; mod += 0.05f) {
                float fx_mod = fx * mod;
                float fy_mod = fy * mod;
                
                float v0_mod = p00 * (1-fx_mod) + p10 * fx_mod;
                float v1_mod = p01 * (1-fx_mod) + p11 * fx_mod;
                float result_mod = v0_mod * (1-fy_mod) + v1_mod * fy_mod;
                
                if (std::abs(result_mod - r.value) < 0.6f) {
                    std::cout << "    Found: fx_mod=" << fx_mod << " (fx*" << mod << "), fy_mod=" << fy_mod 
                              << " -> " << result_mod << "\n";
                }
            }
        }
        std::cout << "\n";
    }
    
    return 0;
}
