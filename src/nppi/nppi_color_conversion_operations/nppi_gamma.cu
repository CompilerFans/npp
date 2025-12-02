#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <mutex>

// ============================================================================
// Constant Memory Lookup Tables (LUT)
// ============================================================================

__constant__ Npp8u d_gamma_fwd_lut[256];
__constant__ Npp8u d_gamma_inv_lut[256];

// ============================================================================
// Host Function: Initialize LUT
// ============================================================================

static std::mutex g_init_mutex;

// NVIDIA NPP Gamma LUT - Extracted directly from NVIDIA NPP library
// These values provide 100% exact match with NVIDIA's implementation
// Reverse-engineered using extract_nvidia_gamma_lut tool

static void initGammaLUT() {
    static bool initialized = false;
    if (initialized) return;

    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (initialized) return;

    // Forward Gamma LUT (Linear -> sRGB encoding)
    // Extracted from NVIDIA NPP - provides exact pixel-perfect matching
    static const Npp8u h_gamma_fwd_lut[256] = {
        0, 4, 9, 13, 18, 22, 26, 30, 33, 36, 40, 42, 45, 48, 50, 53,
        55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 76, 78, 80, 81, 83,
        84, 86, 87, 89, 90, 92, 93, 95, 96, 97, 99, 100, 101, 103, 104, 105,
        106, 108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 123, 124,
        125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
        141, 142, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 151, 152, 153, 154,
        155, 156, 156, 157, 158, 159, 160, 161, 161, 162, 163, 164, 165, 165, 166, 167,
        168, 169, 169, 170, 171, 172, 172, 173, 174, 175, 175, 176, 177, 178, 178, 179,
        180, 180, 181, 182, 183, 183, 184, 185, 185, 186, 187, 188, 188, 189, 190, 190,
        191, 192, 192, 193, 194, 194, 195, 196, 196, 197, 198, 198, 199, 200, 200, 201,
        201, 202, 203, 203, 204, 205, 205, 206, 207, 207, 208, 208, 209, 210, 210, 211,
        211, 212, 213, 213, 214, 214, 215, 216, 216, 217, 217, 218, 219, 219, 220, 220,
        221, 221, 222, 223, 223, 224, 224, 225, 225, 226, 227, 227, 228, 228, 229, 229,
        230, 231, 231, 232, 232, 233, 233, 234, 234, 235, 235, 236, 236, 237, 238, 238,
        239, 239, 240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 245, 245, 246, 246,
        247, 247, 248, 248, 249, 250, 250, 251, 251, 252, 252, 253, 253, 254, 254, 255,
    };

    // Inverse Gamma LUT (sRGB -> Linear decoding)
    // Extracted from NVIDIA NPP - provides exact pixel-perfect matching
    static const Npp8u h_gamma_inv_lut[256] = {
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3,
        3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7,
        7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12,
        12, 13, 13, 14, 14, 15, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19,
        20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 26, 26, 27, 27, 28,
        28, 29, 30, 30, 31, 32, 32, 33, 34, 34, 35, 36, 36, 37, 38, 38,
        39, 40, 41, 41, 42, 43, 44, 44, 45, 46, 47, 48, 48, 49, 50, 51,
        52, 53, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 62, 63, 64, 65,
        66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82,
        83, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 98, 99, 100,
        101, 102, 104, 105, 106, 107, 109, 110, 111, 113, 114, 115, 116, 118, 119, 120,
        122, 123, 124, 126, 127, 129, 130, 131, 133, 134, 136, 137, 139, 140, 141, 143,
        144, 146, 147, 149, 150, 152, 153, 155, 157, 158, 160, 161, 163, 164, 166, 168,
        169, 171, 172, 174, 176, 177, 179, 181, 182, 184, 186, 187, 189, 191, 193, 194,
        196, 198, 200, 201, 203, 205, 207, 209, 210, 212, 214, 216, 218, 220, 221, 223,
        225, 227, 229, 231, 233, 235, 237, 239, 241, 243, 245, 246, 248, 250, 252, 255,
    };

    // Copy to GPU constant memory
    cudaMemcpyToSymbol(d_gamma_fwd_lut, h_gamma_fwd_lut, 256 * sizeof(Npp8u));
    cudaMemcpyToSymbol(d_gamma_inv_lut, h_gamma_inv_lut, 256 * sizeof(Npp8u));

    // CRITICAL: Must synchronize to ensure LUT is ready
    cudaDeviceSynchronize();

    initialized = true;
}

// ============================================================================
// Gamma Operation Functors
// ============================================================================

struct GammaFwdOp {
    __device__ inline Npp8u operator()(Npp8u input) const {
        return d_gamma_fwd_lut[input];
    }
};

struct GammaInvOp {
    __device__ inline Npp8u operator()(Npp8u input) const {
        return d_gamma_inv_lut[input];
    }
};

// ============================================================================
// CUDA Kernels
// ============================================================================

template<int Channels, typename GammaOp>
__global__ void gamma_kernel(
    const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep, int width, int height, GammaOp op
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const Npp8u* src_row = (const Npp8u*)((const char*)pSrc + y * nSrcStep);
        Npp8u* dst_row = (Npp8u*)((char*)pDst + y * nDstStep);

        int pixel_idx = x * Channels;

        // Process RGB channels (first 3 channels)
        #pragma unroll
        for (int c = 0; c < 3; c++) {
            dst_row[pixel_idx + c] = op(src_row[pixel_idx + c]);
        }

        // Set Alpha channel to 0 if AC4 format (matching NVIDIA NPP behavior)
        // NVIDIA NPP clears alpha channel instead of copying it
        if constexpr (Channels == 4) {
            dst_row[pixel_idx + 3] = 0;
        }
    }
}

template<int Channels, typename GammaOp>
__global__ void gamma_inplace_kernel(
    Npp8u* pSrcDst, int nSrcDstStep, int width, int height, GammaOp op
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        Npp8u* row = (Npp8u*)((char*)pSrcDst + y * nSrcDstStep);

        int pixel_idx = x * Channels;

        // In-place modify RGB channels
        #pragma unroll
        for (int c = 0; c < 3; c++) {
            row[pixel_idx + c] = op(row[pixel_idx + c]);
        }

        // Set Alpha channel to 0 if AC4 format (matching NVIDIA NPP behavior)
        if constexpr (Channels == 4) {
            row[pixel_idx + 3] = 0;
        }
    }
}

// ============================================================================
// Executor Functions
// ============================================================================

template<int Channels, typename GammaOp>
NppStatus gamma_execute(
    const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx
) {
    // Parameter validation
    if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;

    // FIX: Use consistent error code - NPP_STEP_ERROR
    int min_step = oSizeROI.width * Channels * sizeof(Npp8u);
    if (nSrcStep < min_step || nDstStep < min_step) return NPP_STEP_ERROR;

    // Initialize LUT (only once)
    initGammaLUT();

    // Configure kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (oSizeROI.width + blockSize.x - 1) / blockSize.x,
        (oSizeROI.height + blockSize.y - 1) / blockSize.y
    );

    GammaOp op;

    // Launch kernel
    gamma_kernel<Channels, GammaOp><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, op
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

    return NPP_SUCCESS;
}

template<int Channels, typename GammaOp>
NppStatus gamma_execute_inplace(
    Npp8u* pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx
) {
    if (!pSrcDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;

    // FIX: Use consistent error code
    int min_step = oSizeROI.width * Channels * sizeof(Npp8u);
    if (nSrcDstStep < min_step) return NPP_STEP_ERROR;

    initGammaLUT();

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (oSizeROI.width + blockSize.x - 1) / blockSize.x,
        (oSizeROI.height + blockSize.y - 1) / blockSize.y
    );

    GammaOp op;

    gamma_inplace_kernel<Channels, GammaOp><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height, op
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;

    return NPP_SUCCESS;
}

// ============================================================================
// External C API Implementations (Ctx versions)
// ============================================================================

extern "C" {

// Forward Gamma: C3R
NppStatus nppiGammaFwd_8u_C3R_Ctx(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return gamma_execute<3, GammaFwdOp>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// Forward Gamma: C3IR
NppStatus nppiGammaFwd_8u_C3IR_Ctx(Npp8u* pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return gamma_execute_inplace<3, GammaFwdOp>(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

// Inverse Gamma: C3R
NppStatus nppiGammaInv_8u_C3R_Ctx(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return gamma_execute<3, GammaInvOp>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// Inverse Gamma: C3IR
NppStatus nppiGammaInv_8u_C3IR_Ctx(Npp8u* pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return gamma_execute_inplace<3, GammaInvOp>(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

// Forward Gamma: AC4R
NppStatus nppiGammaFwd_8u_AC4R_Ctx(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return gamma_execute<4, GammaFwdOp>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// Forward Gamma: AC4IR
NppStatus nppiGammaFwd_8u_AC4IR_Ctx(Npp8u* pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return gamma_execute_inplace<4, GammaFwdOp>(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

// Inverse Gamma: AC4R
NppStatus nppiGammaInv_8u_AC4R_Ctx(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return gamma_execute<4, GammaInvOp>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// Inverse Gamma: AC4IR
NppStatus nppiGammaInv_8u_AC4IR_Ctx(Npp8u* pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return gamma_execute_inplace<4, GammaInvOp>(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

} // extern "C"
