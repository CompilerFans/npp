#include "npp.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== Debugging specific failing cases ===" << std::endl;
    
    // Case 1: AlphaPremulC_8u_C1IR_InPlaceOperation
    // hostSrc = {100, 150, 200, 50, 75, 125}; alpha = 192
    // Failed at index 5: 125 * 192, expected 93 but got 94
    
    std::cout << "Case 1: 125 * 192 with AlphaPremulC" << std::endl;
    Npp8u src = 125;
    Npp8u alpha = 192;
    
    int srcStep, dstStep;
    Npp8u *d_src = nppiMalloc_8u_C1(1, 1, &srcStep);
    Npp8u *d_dst = nppiMalloc_8u_C1(1, 1, &dstStep);
    
    cudaMemcpy(d_src, &src, sizeof(Npp8u), cudaMemcpyHostToDevice);
    
    NppiSize oSizeROI = {1, 1};
    nppiAlphaPremulC_8u_C1R(d_src, srcStep, alpha, d_dst, dstStep, oSizeROI);
    
    Npp8u result;
    cudaMemcpy(&result, d_dst, sizeof(Npp8u), cudaMemcpyDeviceToHost);
    
    int product = src * alpha;
    int shift8 = product >> 8;
    int div255 = product / 255;
    int div256 = product / 256;
    int rounded255 = (product + 127) / 255;
    int rounded256 = (product + 128) / 256;
    
    std::cout << "src=" << (int)src << " * alpha=" << (int)alpha << " = " << product << std::endl;
    std::cout << "NVIDIA result: " << (int)result << std::endl;
    std::cout << "shift8: " << shift8 << std::endl;
    std::cout << "div255: " << div255 << std::endl;
    std::cout << "div256: " << div256 << std::endl;
    std::cout << "rounded255: " << rounded255 << std::endl;
    std::cout << "rounded256: " << rounded256 << std::endl;
    
    // Test if it's rounding related
    double exactDiv255 = (double)product / 255.0;
    double exactDiv256 = (double)product / 256.0;
    std::cout << "exact div255: " << exactDiv255 << " -> rounded: " << (int)(exactDiv255 + 0.5) << std::endl;
    std::cout << "exact div256: " << exactDiv256 << " -> rounded: " << (int)(exactDiv256 + 0.5) << std::endl;
    
    nppiFree(d_src);
    nppiFree(d_dst);
    
    std::cout << "\n=== Case 2: AC4 pixel alpha ===" << std::endl;
    // AlphaPremul_8u_AC4R_PixelAlpha
    // Failed: 255 * 192, expected 192 but got 191
    
    std::vector<Npp8u> hostSrc = {255, 0, 0, 192}; // R=255, A=192
    
    Npp8u *d_srcAC4 = nppiMalloc_8u_C4(1, 1, &srcStep);
    Npp8u *d_dstAC4 = nppiMalloc_8u_C4(1, 1, &dstStep);
    
    int hostStep = 4 * sizeof(Npp8u);
    cudaMemcpy2D(d_srcAC4, srcStep, hostSrc.data(), hostStep, hostStep, 1, cudaMemcpyHostToDevice);
    
    oSizeROI = {1, 1};
    nppiAlphaPremul_8u_AC4R(d_srcAC4, srcStep, d_dstAC4, dstStep, oSizeROI);
    
    std::vector<Npp8u> resultAC4(4);
    cudaMemcpy2D(resultAC4.data(), hostStep, d_dstAC4, dstStep, hostStep, 1, cudaMemcpyDeviceToHost);
    
    int productAC4 = 255 * 192;
    int shift8AC4 = productAC4 >> 8;
    int div255AC4 = productAC4 / 255;
    int rounded255AC4 = (productAC4 + 127) / 255;
    
    std::cout << "AC4: 255 * 192 = " << productAC4 << std::endl;
    std::cout << "NVIDIA AC4 result R channel: " << (int)resultAC4[0] << std::endl;
    std::cout << "shift8: " << shift8AC4 << std::endl;
    std::cout << "div255: " << div255AC4 << std::endl;
    std::cout << "rounded255: " << rounded255AC4 << std::endl;
    
    double exactDiv255AC4 = (double)productAC4 / 255.0;
    std::cout << "exact div255: " << exactDiv255AC4 << " -> rounded: " << (int)(exactDiv255AC4 + 0.5) << std::endl;
    
    // Also test the failing 128*255 case
    std::vector<Npp8u> hostSrc2 = {128, 255, 32, 128}; // G=255, A=128
    cudaMemcpy2D(d_srcAC4, srcStep, hostSrc2.data(), hostStep, hostStep, 1, cudaMemcpyHostToDevice);
    nppiAlphaPremul_8u_AC4R(d_srcAC4, srcStep, d_dstAC4, dstStep, oSizeROI);
    cudaMemcpy2D(resultAC4.data(), hostStep, d_dstAC4, dstStep, hostStep, 1, cudaMemcpyDeviceToHost);
    
    int product2 = 255 * 128;
    std::cout << "\nAC4: 255 * 128 = " << product2 << std::endl;
    std::cout << "NVIDIA AC4 result G channel: " << (int)resultAC4[1] << std::endl;
    std::cout << "shift8: " << (product2 >> 8) << std::endl;
    std::cout << "div255: " << (product2 / 255) << std::endl;
    std::cout << "rounded255: " << ((product2 + 127) / 255) << std::endl;
    
    nppiFree(d_srcAC4);
    nppiFree(d_dstAC4);
    
    return 0;
}