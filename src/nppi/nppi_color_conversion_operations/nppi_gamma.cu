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
// Device Helper Functions: sRGB Standard Gamma Curves
// ============================================================================

__device__ inline float srgb_forward_float(float linear) {
    if (linear <= 0.0031308f)
        return 12.92f * linear;
    else
        return 1.055f * powf(linear, 1.0f/2.4f) - 0.055f;
}

__device__ inline float srgb_inverse_float(float srgb) {
    if (srgb <= 0.04045f)
        return srgb / 12.92f;
    else
        return powf((srgb + 0.055f) / 1.055f, 2.4f);
}

// ============================================================================
// Host Function: Initialize LUT
// ============================================================================

static std::mutex g_init_mutex;

static void initGammaLUT() {
    static bool initialized = false;
    if (initialized) return;

    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (initialized) return;

    Npp8u h_gamma_fwd_lut[256];
    Npp8u h_gamma_inv_lut[256];

    // Precompute Forward Gamma LUT
    for (int i = 0; i < 256; i++) {
        float normalized = i / 255.0f;
        float linear = normalized;
        float gamma_value;
        if (linear <= 0.0031308f)
            gamma_value = 12.92f * linear;
        else
            gamma_value = 1.055f * powf(linear, 1.0f/2.4f) - 0.055f;
        h_gamma_fwd_lut[i] = (Npp8u)(gamma_value * 255.0f + 0.5f);
    }

    // Precompute Inverse Gamma LUT
    for (int i = 0; i < 256; i++) {
        float normalized = i / 255.0f;
        float srgb = normalized;
        float gamma_value;
        if (srgb <= 0.04045f)
            gamma_value = srgb / 12.92f;
        else
            gamma_value = powf((srgb + 0.055f) / 1.055f, 2.4f);
        h_gamma_inv_lut[i] = (Npp8u)(gamma_value * 255.0f + 0.5f);
    }

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

        // Copy Alpha channel if AC4 format
        if constexpr (Channels == 4) {
            dst_row[pixel_idx + 3] = src_row[pixel_idx + 3];
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
        // Alpha channel remains unchanged
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
