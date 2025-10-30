#include <iostream>
#include <cmath>

int main() {
    // All test cases with their fx, fy values
    struct Case {
        float fx, fy;
        int p00, p10, p01, p11;
        int nvidia_result;
    };
    
    Case cases[] = {
        // [dx,dy] -> fx, fy, corners, nvidia_result
        {0.0f, 0.0f, 0, 100, 50, 150, 0},      // [0,0]
        {0.5f, 0.0f, 0, 100, 50, 150, 42},     // [1,0]
        {0.0f, 0.0f, 100, 100, 150, 150, 100}, // [2,0]
        {0.0f, 0.5f, 0, 100, 50, 150, 21},     // [0,1]
        {0.5f, 0.5f, 0, 100, 50, 150, 63},     // [1,1]
        {0.0f, 0.5f, 100, 100, 150, 150, 121}, // [2,1]
        {0.0f, 0.0f, 50, 150, 50, 150, 50},    // [0,2]
        {0.5f, 0.0f, 50, 150, 50, 150, 92},    // [1,2]
        {0.0f, 0.0f, 150, 150, 150, 150, 150}  // [2,2]
    };
    
    std::cout << "Searching for unified formula...\n\n";
    
    // Try: modifier = 1 - alpha * (fx + fy)
    for (float alpha = 0.14f; alpha <= 0.18f; alpha += 0.002f) {
        int matches = 0;
        for (auto& c : cases) {
            float mod_x = 1.0f - alpha * (c.fx + c.fy);
            float mod_y = mod_x; // Same modifier for both
            
            float fx_mod = c.fx * mod_x;
            float fy_mod = c.fy * mod_y;
            
            float v0 = c.p00 * (1 - fx_mod) + c.p10 * fx_mod;
            float v1 = c.p01 * (1 - fx_mod) + c.p11 * fx_mod;
            float result = v0 * (1 - fy_mod) + v1 * fy_mod;
            
            if (std::abs(result - c.nvidia_result) < 0.6f) {
                matches++;
            }
        }
        if (matches == 9) {
            std::cout << "Found! alpha=" << alpha << " (modifier = 1 - " << alpha << " * (fx+fy))\n";
            
            // Verify each case
            for (int i = 0; i < 9; i++) {
                auto& c = cases[i];
                float mod = 1.0f - alpha * (c.fx + c.fy);
                float fx_mod = c.fx * mod;
                float fy_mod = c.fy * mod;
                
                float v0 = c.p00 * (1 - fx_mod) + c.p10 * fx_mod;
                float v1 = c.p01 * (1 - fx_mod) + c.p11 * fx_mod;
                float result = v0 * (1 - fy_mod) + v1 * fy_mod;
                
                std::cout << "  fx=" << c.fx << ", fy=" << c.fy 
                          << " -> mod=" << mod
                          << " -> result=" << result 
                          << " (rounded=" << (int)(result+0.5f) << ")"
                          << " vs nvidia=" << c.nvidia_result
                          << (std::abs(result - c.nvidia_result) < 0.6f ? " ✓" : " ✗")
                          << "\n";
            }
            break;
        }
    }
    
    return 0;
}
