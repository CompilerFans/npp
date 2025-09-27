#include <iostream>
#include <cmath>

int main() {
    // Analyze NVIDIA results
    int src1_values[] = {100, 150, 200, 50, 75, 125};
    int nvidia_results[] = {37, 56, 75, 19, 28, 47};
    int alpha1 = 128;
    
    std::cout << "Analyzing NVIDIA ALPHA_IN operation:" << std::endl;
    
    for (int i = 0; i < 6; i++) {
        int src1 = src1_values[i];
        int nvidia = nvidia_results[i];
        
        // Different possible formulas:
        double formula1 = (src1 * alpha1) / 255.0;                    // Direct floating point
        double formula2 = (src1 * alpha1 + 127) / 255.0;             // With rounding
        int formula3 = (src1 * alpha1) / 255;                        // Integer division
        int formula4 = (src1 * alpha1 + 128) / 255;                  // Integer with rounding
        int formula5 = (src1 * alpha1 + 255/2) / 255;                // Integer with proper rounding
        
        std::cout << "src1=" << src1 << ", nvidia=" << nvidia << std::endl;
        std::cout << "  formula1 (float): " << formula1 << std::endl;
        std::cout << "  formula2 (float+round): " << formula2 << std::endl;
        std::cout << "  formula3 (int): " << formula3 << std::endl;
        std::cout << "  formula4 (int+128): " << formula4 << std::endl;
        std::cout << "  formula5 (int+127): " << formula5 << std::endl;
        std::cout << "  Direct calculation: " << src1 << " * " << alpha1 << " = " << (src1 * alpha1) << std::endl;
        std::cout << "  / 255 = " << (src1 * alpha1) / 255 << std::endl;
        std::cout << std::endl;
    }
    
    return 0;
}