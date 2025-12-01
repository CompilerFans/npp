#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ============================================================================
// Constant Memory Lookup Tables (LUT)
// LUT = Look-Up Table，查找表，用于快速映射输入值到输出值
// 使用__constant__内存可以加速访问，因为它会被缓存
// ============================================================================

__constant__ Npp8u d_gamma_fwd_lut[256];  // sRGB正向Gamma校正表 (线性 -> Gamma空间)
__constant__ Npp8u d_gamma_inv_lut[256];  // sRGB反向Gamma校正表 (Gamma空间 -> 线性)

// ============================================================================
// Device Helper Functions: sRGB Standard Gamma Curves
// sRGB是标准的RGB色彩空间，定义了特定的Gamma曲线
// ============================================================================

__device__ inline float srgb_forward_float(float linear) {
    // 将线性空间的值转换为sRGB Gamma空间
    // 分段函数：小值用线性，大值用幂函数
    if (linear <= 0.0031308f)
        return 12.92f * linear;
    else
        return 1.055f * powf(linear, 1.0f/2.4f) - 0.055f;
}

__device__ inline float srgb_inverse_float(float srgb) {
    // 将sRGB Gamma空间的值转换回线性空间
    if (srgb <= 0.04045f)
        return srgb / 12.92f;
    else
        return powf((srgb + 0.055f) / 1.055f, 2.4f);
}

// ============================================================================
// Host Function: Initialize LUT (executed only once on first call)
// 在第一次调用时，在CPU上预计算整个LUT，然后拷贝到GPU常量内存
// ============================================================================

static void initGammaLUT() {
    static bool initialized = false;
    if (initialized) return;
    
    Npp8u h_gamma_fwd_lut[256];
    Npp8u h_gamma_inv_lut[256];
    
    // 在CPU上预计算sRGB正向Gamma LUT (0-255)
    for (int i = 0; i < 256; i++) {
        float normalized = i / 255.0f;  // 归一化到[0,1]
        float linear = normalized;
        float gamma_value;
        if (linear <= 0.0031308f)
            gamma_value = 12.92f * linear;
        else
            gamma_value = 1.055f * powf(linear, 1.0f/2.4f) - 0.055f;
        h_gamma_fwd_lut[i] = (Npp8u)(gamma_value * 255.0f + 0.5f);  // 反归一化并四舍五入
    }
    
    // 在CPU上预计算sRGB反向Gamma LUT
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
    
    // 拷贝到GPU常量内存（所有CUDA线程都可以快速访问）
    cudaMemcpyToSymbol(d_gamma_fwd_lut, h_gamma_fwd_lut, 256 * sizeof(Npp8u));
    cudaMemcpyToSymbol(d_gamma_inv_lut, h_gamma_inv_lut, 256 * sizeof(Npp8u));
    
    initialized = true;
}

// ============================================================================
// Gamma Operation Functors (Function Objects)
// Functor = 函数对象，可以像函数一样被调用的对象
// 使用Functor可以让kernel代码更通用
// ============================================================================

struct GammaFwdOp {
    __device__ inline Npp8u operator()(Npp8u input) const {
        return d_gamma_fwd_lut[input];  // 直接查表
    }
};

struct GammaInvOp {
    __device__ inline Npp8u operator()(Npp8u input) const {
        return d_gamma_inv_lut[input];
    }
};

// ============================================================================
// CUDA Kernels: Out-of-place (R) - 输出到不同的buffer
// ============================================================================

template<int Channels, typename GammaOp>
__global__ void gamma_kernel(
    const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep, int width, int height, GammaOp op
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程处理的像素x坐标
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 计算当前线程处理的像素y坐标
    
    if (x < width && y < height) {
        // 计算当前行的起始地址（Step是行的字节跨度）
        const Npp8u* src_row = (const Npp8u*)((const char*)pSrc + y * nSrcStep);
        Npp8u* dst_row = (Npp8u*)((char*)pDst + y * nDstStep);
        
        int pixel_idx = x * Channels;  // 计算像素在行中的起始位置
        
        // 处理RGB三个通道（C3或AC4都只处理前3个通道）
        for (int c = 0; c < 3; c++) {
            dst_row[pixel_idx + c] = op(src_row[pixel_idx + c]);
        }
        
        // 如果是AC4格式，Alpha通道不变
        if constexpr (Channels == 4) {
            dst_row[pixel_idx + 3] = src_row[pixel_idx + 3];
        }
    }
}

// ============================================================================
// CUDA Kernels: In-place (IR) - 在原buffer上修改
// ============================================================================

template<int Channels, typename GammaOp>
__global__ void gamma_inplace_kernel(
    Npp8u* pSrcDst, int nSrcDstStep, int width, int height, GammaOp op
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        Npp8u* row = (Npp8u*)((char*)pSrcDst + y * nSrcDstStep);
        
        int pixel_idx = x * Channels;
        
        // 处理RGB三个通道，Alpha不变
        for (int c = 0; c < 3; c++) {
            row[pixel_idx + c] = op(row[pixel_idx + c]);
        }
    }
}

// ============================================================================
// Executor Functions: Template implementations for different channels
// 这些函数是通用的执行器，通过模板参数处理不同通道数和操作
// ============================================================================

template<int Channels, typename GammaOp>
NppStatus gamma_execute(
    const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx
) {
    // 参数验证
    if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
    
    int min_step = oSizeROI.width * Channels * sizeof(Npp8u);
    if (nSrcStep < min_step || nDstStep < min_step) return NPP_STEP_ERROR;
    
    // 初始化LUT（只在第一次调用时执行）
    initGammaLUT();
    
    // 配置CUDA kernel启动参数
    dim3 blockSize(16, 16);  // 每个block 16x16个线程
    dim3 gridSize(
        (oSizeROI.width + blockSize.x - 1) / blockSize.x,
        (oSizeROI.height + blockSize.y - 1) / blockSize.y
    );
    
    GammaOp op;  // 创建操作对象
    
    // 启动kernel，使用nppStreamCtx中的CUDA stream
    gamma_kernel<Channels, GammaOp><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, op
    );
    
    // 检查kernel执行是否成功
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
    
    int min_step = oSizeROI.width * Channels * sizeof(Npp8u);
    if (nSrcDstStep < min_step) return NPP_STRIDE_ERROR;
    
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
// 这些是暴露给外部的C接口，内部调用模板实现
// ============================================================================

extern "C" {

// Forward Gamma: C3R (3 channels, out-of-place)
NppStatus nppiGammaFwd_8u_C3R_Ctx(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return gamma_execute<3, GammaFwdOp>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// Forward Gamma: C3IR (3 channels, in-place)
NppStatus nppiGammaFwd_8u_C3IR_Ctx(Npp8u* pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return gamma_execute_inplace<3, GammaFwdOp>(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

// Inverse Gamma: C3R (3 channels, out-of-place)
NppStatus nppiGammaInv_8u_C3R_Ctx(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return gamma_execute<3, GammaInvOp>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// Inverse Gamma: C3IR (3 channels, in-place)
NppStatus nppiGammaInv_8u_C3IR_Ctx(Npp8u* pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return gamma_execute_inplace<3, GammaInvOp>(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

// Forward Gamma: AC4R (4 channels with alpha, out-of-place)
NppStatus nppiGammaFwd_8u_AC4R_Ctx(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return gamma_execute<4, GammaFwdOp>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// Forward Gamma: AC4IR (4 channels with alpha, in-place)
NppStatus nppiGammaFwd_8u_AC4IR_Ctx(Npp8u* pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return gamma_execute_inplace<4, GammaFwdOp>(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

// Inverse Gamma: AC4R (4 channels with alpha, out-of-place)
NppStatus nppiGammaInv_8u_AC4R_Ctx(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return gamma_execute<4, GammaInvOp>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// Inverse Gamma: AC4IR (4 channels with alpha, in-place)
NppStatus nppiGammaInv_8u_AC4IR_Ctx(Npp8u* pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return gamma_execute_inplace<4, GammaInvOp>(pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

} // extern "C"
