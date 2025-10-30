#include <iostream>
#include <cmath>

int main() {
    // Center point [1,1]: fx=0.5, fy=0.5
    // Corners: p00=0, p10=100, p01=50, p11=150
    // NVIDIA result: 63
    
    float fx = 0.5f, fy = 0.5f;
    int p00 = 0, p10 = 100, p01 = 50, p11 = 150;
    
    std::cout << "Center point analysis:\n";
    std::cout << "fx=" << fx << ", fy=" << fy << "\n";
    std::cout << "p00=" << p00 << ", p10=" << p10 << ", p01=" << p01 << ", p11=" << p11 << "\n\n";
    
    // Try different modifiers
    for (float mod = 0.80f; mod <= 0.90f; mod += 0.01f) {
        float fx_mod = fx * mod;
        float fy_mod = fy * mod;
        
        float v0 = p00 * (1-fx_mod) + p10 * fx_mod;
        float v1 = p01 * (1-fx_mod) + p11 * fx_mod;
        float result = v0 * (1-fy_mod) + v1 * fy_mod;
        
        std::cout << "mod=" << mod << ": fx_mod=" << fx_mod << ", fy_mod=" << fy_mod 
                  << " -> " << result << " (rounded=" << (int)(result+0.5f) << ")\n";
    }
    
    // The exact calculation with 0.84
    float mod = 0.84f;
    float fx_mod = fx * mod;
    float fy_mod = fy * mod;
    
    std::cout << "\nUsing modifier 0.84:\n";
    std::cout << "fx_mod = " << fx_mod << ", fy_mod = " << fy_mod << "\n";
    
    float v0 = p00 * (1-fx_mod) + p10 * fx_mod;
    float v1 = p01 * (1-fx_mod) + p11 * fx_mod;
    
    std::cout << "v0 = " << p00 << "*(1-" << fx_mod << ") + " << p10 << "*" << fx_mod << " = " << v0 << "\n";
    std::cout << "v1 = " << p01 << "*(1-" << fx_mod << ") + " << p11 << "*" << fx_mod << " = " << v1 << "\n";
    
    float result = v0 * (1-fy_mod) + v1 * fy_mod;
    std::cout << "result = " << v0 << "*(1-" << fy_mod << ") + " << v1 << "*" << fy_mod << " = " << result << "\n";
    std::cout << "Rounded: " << (int)(result+0.5f) << "\n";
    std::cout << "NVIDIA: 63\n";
    
    return 0;
}
