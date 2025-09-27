#include "npp.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "Analyzing NVIDIA AlphaPremul calculation..." << std::endl;
    
    // Test specific values to understand the formula
    std::vector<Npp8u> testValues = {255, 128, 64, 1, 254, 129, 127};
    Npp8u alpha = 128;

    for (Npp8u src : testValues) {
        // Allocate memory for single pixel test
        int srcStep, dstStep;
        Npp8u *d_src = nppiMalloc_8u_C1(1, 1, &srcStep);
        Npp8u *d_dst = nppiMalloc_8u_C1(1, 1, &dstStep);

        // Copy test value
        cudaMemcpy(d_src, &src, sizeof(Npp8u), cudaMemcpyHostToDevice);

        // Execute NVIDIA NPP operation
        NppiSize oSizeROI = {1, 1};
        nppiAlphaPremulC_8u_C1R(d_src, srcStep, alpha, d_dst, dstStep, oSizeROI);

        // Get result
        Npp8u nvidiaResult;
        cudaMemcpy(&nvidiaResult, d_dst, sizeof(Npp8u), cudaMemcpyDeviceToHost);

        // Calculate different possibilities
        int div255 = (src * alpha) / 255;
        int div256 = (src * alpha) / 256;
        int shift8 = (src * alpha) >> 8;
        int rounded255 = (src * alpha + 127) / 255;
        int rounded256 = (src * alpha + 128) / 256;
        
        // More precise calculation
        double exact255 = (double)(src * alpha) / 255.0;
        double exact256 = (double)(src * alpha) / 256.0;

        std::cout << "src=" << (int)src << " * alpha=" << (int)alpha << " = " << (src * alpha);
        std::cout << " -> NVIDIA=" << (int)nvidiaResult;
        std::cout << ", div255=" << div255;
        std::cout << ", div256=" << div256;
        std::cout << ", shift8=" << shift8;
        std::cout << ", round255=" << rounded255;
        std::cout << ", round256=" << rounded256;
        std::cout << ", exact255=" << exact255;
        std::cout << ", exact256=" << exact256;
        
        // Check which matches
        if (nvidiaResult == div255) std::cout << " [MATCH: div255]";
        if (nvidiaResult == div256) std::cout << " [MATCH: div256]";
        if (nvidiaResult == shift8) std::cout << " [MATCH: shift8]";
        if (nvidiaResult == rounded255) std::cout << " [MATCH: round255]";
        if (nvidiaResult == rounded256) std::cout << " [MATCH: round256]";
        
        std::cout << std::endl;

        // Cleanup
        nppiFree(d_src);
        nppiFree(d_dst);
    }

    // Test AC4 with different alpha values
    std::cout << "\n=== AC4 Analysis ===" << std::endl;
    
    std::vector<Npp8u> testAlphas = {255, 192, 128, 64, 1};
    Npp8u srcValue = 255; // Red channel
    
    for (Npp8u pixelAlpha : testAlphas) {
        std::vector<Npp8u> hostSrc = {srcValue, 0, 0, pixelAlpha}; // R=srcValue, G=0, B=0, A=pixelAlpha

        int srcStep, dstStep;
        Npp8u *d_src = nppiMalloc_8u_C4(1, 1, &srcStep);
        Npp8u *d_dst = nppiMalloc_8u_C4(1, 1, &dstStep);

        int hostStep = 4 * sizeof(Npp8u);
        cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostStep, hostStep, 1, cudaMemcpyHostToDevice);

        NppiSize oSizeROI = {1, 1};
        nppiAlphaPremul_8u_AC4R(d_src, srcStep, d_dst, dstStep, oSizeROI);

        std::vector<Npp8u> result(4);
        cudaMemcpy2D(result.data(), hostStep, d_dst, dstStep, hostStep, 1, cudaMemcpyDeviceToHost);

        int div255 = (srcValue * pixelAlpha) / 255;
        int div256 = (srcValue * pixelAlpha) / 256;
        int shift8 = (srcValue * pixelAlpha) >> 8;
        int rounded255 = (srcValue * pixelAlpha + 127) / 255;
        int rounded256 = (srcValue * pixelAlpha + 128) / 256;

        std::cout << "src=" << (int)srcValue << " * pixel_alpha=" << (int)pixelAlpha;
        std::cout << " -> NVIDIA_R=" << (int)result[0];
        std::cout << ", div255=" << div255;
        std::cout << ", div256=" << div256;
        std::cout << ", shift8=" << shift8;
        std::cout << ", round255=" << rounded255;
        std::cout << ", round256=" << rounded256;

        if (result[0] == div255) std::cout << " [MATCH: div255]";
        if (result[0] == div256) std::cout << " [MATCH: div256]";
        if (result[0] == shift8) std::cout << " [MATCH: shift8]";
        if (result[0] == rounded255) std::cout << " [MATCH: round255]";
        if (result[0] == rounded256) std::cout << " [MATCH: round256]";

        std::cout << std::endl;

        nppiFree(d_src);
        nppiFree(d_dst);
    }

    return 0;
}