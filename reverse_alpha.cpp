#include <iostream>

int main() {
    // Reverse engineer the alpha value
    int src1_values[] = {100, 150, 200, 50, 75, 125};
    int nvidia_results[] = {37, 56, 75, 19, 28, 47};
    
    std::cout << "Reverse engineering alpha value:" << std::endl;
    
    for (int i = 0; i < 6; i++) {
        int src1 = src1_values[i];
        int nvidia = nvidia_results[i];
        
        double effective_alpha = (double)nvidia / src1;
        std::cout << "src1=" << src1 << ", result=" << nvidia << ", effective_alpha=" << effective_alpha << std::endl;
        
        // Test if this matches 96/255 (which is roughly 0.376)
        int test_96 = (src1 * 96) / 255;
        int test_96_round = (src1 * 96 + 127) / 255;
        
        std::cout << "  96/255 test: " << test_96 << ", with rounding: " << test_96_round << std::endl;
        std::cout << std::endl;
    }
    
    // Maybe NPP is using a different alpha interpretation
    // Let's test if alpha1 (128) is being interpreted differently
    std::cout << "Testing different alpha interpretations:" << std::endl;
    std::cout << "128 / 255 = " << (128.0/255.0) << std::endl;
    std::cout << "96 / 255 = " << (96.0/255.0) << std::endl;
    std::cout << "3/8 = " << (3.0/8.0) << std::endl;
    
    // Check if 96 is related to 128 somehow
    std::cout << "96 / 128 = " << (96.0/128.0) << std::endl;
    std::cout << "128 * 3/4 = " << (128 * 3 / 4) << std::endl;
    
    return 0;
}