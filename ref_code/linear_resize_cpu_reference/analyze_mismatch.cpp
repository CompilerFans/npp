#include <iostream>
#include <cmath>

int main() {
    // Source: [0, 127]
    //         [127, 255]
    // Analyzing pixel[2,1] and pixel[1,2] which mismatch
    
    float scale = 2.0f / 3.0f;
    
    // Pixel [2,1]
    int dx = 2, dy = 1;
    float srcX = (dx + 0.5f) * scale - 0.5f;
    float srcY = (dy + 0.5f) * scale - 0.5f;
    
    std::cout << "Analyzing Pixel[2,1] (index 5):\n";
    std::cout << "srcX = " << srcX << ", srcY = " << srcY << "\n";
    
    srcX = std::max(0.0f, std::min(1.0f, srcX));
    srcY = std::max(0.0f, std::min(1.0f, srcY));
    
    int x0 = (int)std::floor(srcX);
    int y0 = (int)std::floor(srcY);
    float fx = srcX - x0;
    float fy = srcY - y0;
    
    std::cout << "x0=" << x0 << ", y0=" << y0 << ", fx=" << fx << ", fy=" << fy << "\n";
    
    // Source values
    int p00 = (y0 == 0 && x0 == 0) ? 0 : (y0 == 0 && x0 == 1) ? 127 : (y0 == 1 && x0 == 0) ? 127 : 255;
    int p10 = (y0 == 0 && x0 + 1 == 1) ? 127 : 255;
    int p01 = (y0 + 1 == 1 && x0 == 0) ? 127 : 255;
    int p11 = 255;
    
    std::cout << "p00=" << p00 << ", p10=" << p10 << ", p01=" << p01 << ", p11=" << p11 << "\n";
    
    // With modifier
    float modifier = 0.85f - 0.04f * (fx * fy);
    float fx_mod = fx * modifier;
    float fy_mod = fy * modifier;
    
    std::cout << "modifier = " << modifier << "\n";
    std::cout << "fx_mod = " << fx_mod << ", fy_mod = " << fy_mod << "\n";
    
    float v0 = p00 * (1 - fx_mod) + p10 * fx_mod;
    float v1 = p01 * (1 - fx_mod) + p11 * fx_mod;
    float result = v0 * (1 - fy_mod) + v1 * fy_mod;
    
    std::cout << "v0 = " << v0 << ", v1 = " << v1 << "\n";
    std::cout << "result = " << result << "\n";
    std::cout << "floor(result) = " << (int)std::floor(result) << "\n";
    std::cout << "round(result) = " << (int)(result + 0.5f) << "\n";
    std::cout << "NPP result = 180\n\n";
    
    // Try different rounding
    std::cout << "Checking different conversions:\n";
    std::cout << "  (int)result = " << (int)result << "\n";
    std::cout << "  floor(result) = " << (int)std::floor(result) << "\n";
    std::cout << "  round(result) = " << (int)(result + 0.5f) << "\n";
    std::cout << "  floor(result + 0.5) = " << (int)std::floor(result + 0.5f) << "\n";
    
    // Check if result is exactly 180.5
    std::cout << "\nIs result exactly 180.5? " << (std::abs(result - 180.5f) < 0.0001f ? "YES" : "NO") << "\n";
    std::cout << "result - 180.5 = " << (result - 180.5f) << "\n";
    
    return 0;
}
