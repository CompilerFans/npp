#include "npp.h"
#include <cuda_runtime.h>
#include <cstdint>

/**
 * CUDA Kernels for NV12 to RGB Conversion
 * 
 * NV12 Format:
 * - Y plane: Luminance values (8-bit unsigned)
 * - UV plane: Interleaved U/V chroma values (8-bit unsigned, half resolution)
 * 
 * Conversion Matrices:
 * ITU-R BT.601: Standard definition
 * ITU-R BT.709: High definition
 */

// ITU-R BT.601 coefficients (standard definition)
__constant__ float YUV2RGB_BT601[9] = {
    1.164f,  0.000f,  1.596f,    // R = 1.164*(Y-16) + 1.596*(V-128)
    1.164f, -0.392f, -0.813f,    // G = 1.164*(Y-16) - 0.392*(U-128) - 0.813*(V-128)
    1.164f,  2.017f,  0.000f     // B = 1.164*(Y-16) + 2.017*(U-128)
};

// ITU-R BT.709 coefficients (high definition)
__constant__ float YUV2RGB_BT709[9] = {
    1.164f,  0.000f,  1.793f,    // R = 1.164*(Y-16) + 1.793*(V-128)
    1.164f, -0.213f, -0.533f,    // G = 1.164*(Y-16) - 0.213*(U-128) - 0.533*(V-128)
    1.164f,  2.112f,  0.000f     // B = 1.164*(Y-16) + 2.112*(U-128)
};

/**
 * Convert NV12 to RGB using specified coefficients
 */
__device__ inline void nv12_to_rgb_pixel(
    uint8_t y, uint8_t u, uint8_t v,
    uint8_t& r, uint8_t& g, uint8_t& b,
    const float* coeffs)
{
    // Convert to floating point and apply offset
    float fy = (float)y - 16.0f;
    float fu = (float)u - 128.0f;
    float fv = (float)v - 128.0f;
    
    // Apply conversion matrix
    float fr = coeffs[0] * fy + coeffs[1] * fu + coeffs[2] * fv;
    float fg = coeffs[3] * fy + coeffs[4] * fu + coeffs[5] * fv;
    float fb = coeffs[6] * fy + coeffs[7] * fu + coeffs[8] * fv;
    
    // Clamp and convert back to 8-bit
    r = (uint8_t)fmaxf(0.0f, fminf(255.0f, fr + 0.5f));
    g = (uint8_t)fmaxf(0.0f, fminf(255.0f, fg + 0.5f));
    b = (uint8_t)fmaxf(0.0f, fminf(255.0f, fb + 0.5f));
}

/**
 * NV12 to RGB kernel using BT.601 coefficients
 */
__global__ void nv12_to_rgb_kernel(
    const uint8_t* __restrict__ srcY, int srcYStep,
    const uint8_t* __restrict__ srcUV, int srcUVStep,
    uint8_t* __restrict__ dst, int dstStep,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Get Y value
    uint8_t Y = srcY[y * srcYStep + x];
    
    // Get UV values (chroma is subsampled by 2 in both dimensions)
    int uv_x = (x >> 1) << 1;  // Ensure even coordinate
    int uv_y = y >> 1;          // Chroma is half resolution in Y
    int uv_offset = uv_y * srcUVStep + uv_x;
    uint8_t U = srcUV[uv_offset];
    uint8_t V = srcUV[uv_offset + 1];
    
    // Convert to RGB
    uint8_t R, G, B;
    nv12_to_rgb_pixel(Y, U, V, R, G, B, YUV2RGB_BT601);
    
    // Store RGB result
    int dst_offset = y * dstStep + x * 3;
    dst[dst_offset] = R;
    dst[dst_offset + 1] = G;
    dst[dst_offset + 2] = B;
}

/**
 * NV12 to RGB kernel using BT.709 coefficients
 */
__global__ void nv12_to_rgb_709_kernel(
    const uint8_t* __restrict__ srcY, int srcYStep,
    const uint8_t* __restrict__ srcUV, int srcUVStep,
    uint8_t* __restrict__ dst, int dstStep,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Get Y value
    uint8_t Y = srcY[y * srcYStep + x];
    
    // Get UV values (chroma is subsampled)
    int uv_x = (x >> 1) << 1;  // Align to even coordinate
    int uv_y = y >> 1;          // Chroma is half resolution
    int uv_offset = uv_y * srcUVStep + uv_x;
    uint8_t U = srcUV[uv_offset];
    uint8_t V = srcUV[uv_offset + 1];
    
    // Convert to RGB using BT.709
    uint8_t R, G, B;
    nv12_to_rgb_pixel(Y, U, V, R, G, B, YUV2RGB_BT709);
    
    // Store RGB result
    int dst_offset = y * dstStep + x * 3;
    dst[dst_offset] = R;
    dst[dst_offset + 1] = G;
    dst[dst_offset + 2] = B;
}

/**
 * Host function for NV12 to RGB conversion (BT.601)
 */
extern "C" cudaError_t nppiNV12ToRGB_8u_P2C3R_kernel(
    const Npp8u* pSrcY, int nSrcYStep,
    const Npp8u* pSrcUV, int nSrcUVStep,
    Npp8u* pDst, int nDstStep,
    NppiSize oSizeROI,
    cudaStream_t stream)
{
    // 计算网格和块大小
    dim3 blockSize(16, 16);  // 256 threads per block
    dim3 gridSize(
        (oSizeROI.width + blockSize.x - 1) / blockSize.x,
        (oSizeROI.height + blockSize.y - 1) / blockSize.y
    );
    
    // 启动内核
    nv12_to_rgb_kernel<<<gridSize, blockSize, 0, stream>>>(
        pSrcY, nSrcYStep,
        pSrcUV, nSrcUVStep,
        pDst, nDstStep,
        oSizeROI.width, oSizeROI.height
    );
    
    return cudaGetLastError();
}

/**
 * Host function for NV12 to RGB conversion (BT.709)
 */
extern "C" cudaError_t nppiNV12ToRGB_709CSC_8u_P2C3R_kernel(
    const Npp8u* pSrcY, int nSrcYStep,
    const Npp8u* pSrcUV, int nSrcUVStep,
    Npp8u* pDst, int nDstStep,
    NppiSize oSizeROI,
    cudaStream_t stream)
{
    // 计算网格和块大小
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (oSizeROI.width + blockSize.x - 1) / blockSize.x,
        (oSizeROI.height + blockSize.y - 1) / blockSize.y
    );
    
    // 启动BT.709内核
    nv12_to_rgb_709_kernel<<<gridSize, blockSize, 0, stream>>>(
        pSrcY, nSrcYStep,
        pSrcUV, nSrcUVStep,
        pDst, nDstStep,
        oSizeROI.width, oSizeROI.height
    );
    
    return cudaGetLastError();
}

/**
 * Convert NV12 to RGB using custom color twist matrix
 * Used by TorchCodec for precise BT.709 full range conversion
 */
__device__ inline void nv12_to_rgb_colortwist_pixel(
    uint8_t y, uint8_t u, uint8_t v,
    uint8_t& r, uint8_t& g, uint8_t& b,
    const float* twist)
{
    // Convert to floating point (no offset applied - handled in matrix)
    float fy = (float)y;
    float fu = (float)u;
    float fv = (float)v;
    
    // Apply custom color twist matrix: RGB = [twist] * [Y U V 1]^T
    float fr = twist[0] * fy + twist[1] * fu + twist[2] * fv + twist[3];
    float fg = twist[4] * fy + twist[5] * fu + twist[6] * fv + twist[7];
    float fb = twist[8] * fy + twist[9] * fu + twist[10] * fv + twist[11];
    
    // Clamp and convert back to 8-bit
    r = (uint8_t)fmaxf(0.0f, fminf(255.0f, fr + 0.5f));
    g = (uint8_t)fmaxf(0.0f, fminf(255.0f, fg + 0.5f));
    b = (uint8_t)fmaxf(0.0f, fminf(255.0f, fb + 0.5f));
}

/**
 * NV12 to RGB kernel using custom ColorTwist matrix
 */
__global__ void nv12_to_rgb_colortwist_kernel(
    const uint8_t* __restrict__ srcY, int srcYStep,
    const uint8_t* __restrict__ srcUV, int srcUVStep,
    uint8_t* __restrict__ dst, int dstStep,
    int width, int height,
    const float* __restrict__ twist)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Get Y value
    uint8_t Y = srcY[y * srcYStep + x];
    
    // Get UV values (chroma is subsampled)
    int uv_x = (x >> 1) << 1;  // Align to even coordinate
    int uv_y = y >> 1;          // Chroma is half resolution
    int uv_offset = uv_y * srcUVStep + uv_x;
    uint8_t U = srcUV[uv_offset];
    uint8_t V = srcUV[uv_offset + 1];
    
    // Convert to RGB using custom ColorTwist
    uint8_t R, G, B;
    nv12_to_rgb_colortwist_pixel(Y, U, V, R, G, B, twist);
    
    // Store RGB result
    int dst_offset = y * dstStep + x * 3;
    dst[dst_offset] = R;
    dst[dst_offset + 1] = G;
    dst[dst_offset + 2] = B;
}

/**
 * Host function for NV12 to RGB conversion with ColorTwist matrix
 */
extern "C" cudaError_t nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_kernel(
    const Npp8u* pSrcY, int nSrcYStep,
    const Npp8u* pSrcUV, int nSrcUVStep,
    Npp8u* pDst, int nDstStep,
    NppiSize oSizeROI,
    const Npp32f aTwist[3][4],
    cudaStream_t stream)
{
    // 将ColorTwist矩阵复制到GPU常量内存或设备内存
    // 为了简化，我们将矩阵数据作为参数传递
    float* d_twist;
    cudaError_t err = cudaMalloc(&d_twist, 12 * sizeof(float));
    if (err != cudaSuccess) return err;
    
    // 将3x4矩阵展开为一维数组
    float h_twist[12];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            h_twist[i * 4 + j] = aTwist[i][j];
        }
    }
    
    err = cudaMemcpyAsync(d_twist, h_twist, 12 * sizeof(float), 
                         cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(d_twist);
        return err;
    }
    
    // 计算网格和块大小
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (oSizeROI.width + blockSize.x - 1) / blockSize.x,
        (oSizeROI.height + blockSize.y - 1) / blockSize.y
    );
    
    // 启动ColorTwist内核
    nv12_to_rgb_colortwist_kernel<<<gridSize, blockSize, 0, stream>>>(
        pSrcY, nSrcYStep,
        pSrcUV, nSrcUVStep,
        pDst, nDstStep,
        oSizeROI.width, oSizeROI.height,
        d_twist
    );
    
    err = cudaGetLastError();
    
    // 同步并清理内存
    cudaStreamSynchronize(stream);
    cudaFree(d_twist);
    
    return err;
}