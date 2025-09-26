#include "npp.h"
#include <cuda_runtime.h>
#include <functional>
#include <type_traits>

namespace nppi {
namespace functors {

// Simple type-specific saturating cast
template<typename T>
__device__ __host__ inline T saturate_cast(double value) {
    if constexpr (std::is_same_v<T, Npp8u>) {
        return static_cast<T>(value < 0.0 ? 0.0 : (value > 255.0 ? 255.0 : value));
    } else if constexpr (std::is_same_v<T, Npp16u>) {
        return static_cast<T>(value < 0.0 ? 0.0 : (value > 65535.0 ? 65535.0 : value));
    } else if constexpr (std::is_same_v<T, Npp32f>) {
        return static_cast<T>(value);
    }
    return static_cast<T>(value);
}

// Simple add functor
template<typename T>
struct AddFunctor {
    __device__ __host__ T operator()(T a, T b, int scaleFactor = 0) const {
        double result = static_cast<double>(a) + static_cast<double>(b);
        if (scaleFactor > 0 && std::is_integral_v<T>) {
            result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
        }
        return saturate_cast<T>(result);
    }
};

} // namespace functors
} // namespace nppi

// Kernel implementations
template<typename T>
__global__ void add_kernel_C1(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step,
                               T *pDst, int nDstStep, int width, int height, int scaleFactor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const T *src1Row = (const T *)((const char *)pSrc1 + y * nSrc1Step);
        const T *src2Row = (const T *)((const char *)pSrc2 + y * nSrc2Step);
        T *dstRow = (T *)((char *)pDst + y * nDstStep);

        nppi::functors::AddFunctor<T> op;
        dstRow[x] = op(src1Row[x], src2Row[x], scaleFactor);
    }
}

template<typename T>
__global__ void add_kernel_C3(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step,
                               T *pDst, int nDstStep, int width, int height, int scaleFactor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const T *src1Row = (const T *)((const char *)pSrc1 + y * nSrc1Step);
        const T *src2Row = (const T *)((const char *)pSrc2 + y * nSrc2Step);
        T *dstRow = (T *)((char *)pDst + y * nDstStep);

        nppi::functors::AddFunctor<T> op;
        int idx = x * 3;
        for (int c = 0; c < 3; c++) {
            dstRow[idx + c] = op(src1Row[idx + c], src2Row[idx + c], scaleFactor);
        }
    }
}

// Parameter validation
static NppStatus validateParameters(const void *pSrc1, int nSrc1Step, const void *pSrc2, int nSrc2Step,
                                    const void *pDst, int nDstStep, NppiSize oSizeROI, int elementSize) {
    if (!pSrc1 || !pSrc2 || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width < 0 || oSizeROI.height < 0) {
        return NPP_SIZE_ERROR;
    }
    if (oSizeROI.width == 0 || oSizeROI.height == 0) {
        return NPP_NO_ERROR;
    }
    int minStep = oSizeROI.width * elementSize;
    if (nSrc1Step < minStep || nSrc2Step < minStep || nDstStep < minStep) {
        return NPP_STRIDE_ERROR;
    }
    return NPP_NO_ERROR;
}

// Launch helper
template<typename T>
static NppStatus launchKernel_C1(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step,
                                  T *pDst, int nDstStep, NppiSize oSizeROI, int scaleFactor, 
                                  cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    add_kernel_C1<T><<<gridSize, blockSize, 0, stream>>>(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep,
        oSizeROI.width, oSizeROI.height, scaleFactor);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

template<typename T>
static NppStatus launchKernel_C3(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step,
                                  T *pDst, int nDstStep, NppiSize oSizeROI, int scaleFactor, 
                                  cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    add_kernel_C3<T><<<gridSize, blockSize, 0, stream>>>(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep,
        oSizeROI.width, oSizeROI.height, scaleFactor);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// API implementations for 8u C1 with scaling
extern "C" {
NppStatus nppiAdd_8u_C1RSfs_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                      Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                      NppStreamContext nppStreamCtx) {
    NppStatus status = validateParameters(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, sizeof(Npp8u));
    if (status != NPP_NO_ERROR) return status;
    if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;
    
    return launchKernel_C1<Npp8u>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx.hStream);
}

NppStatus nppiAdd_8u_C3RSfs_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                      Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                      NppStreamContext nppStreamCtx) {
    NppStatus status = validateParameters(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, 3 * sizeof(Npp8u));
    if (status != NPP_NO_ERROR) return status;
    if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;
    
    return launchKernel_C3<Npp8u>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
                                   oSizeROI, nScaleFactor, nppStreamCtx.hStream);
}

// API implementations for 32f (no scaling)
NppStatus nppiAdd_32f_C1R_Ctx_impl(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                                    Npp32f *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    NppStatus status = validateParameters(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, sizeof(Npp32f));
    if (status != NPP_NO_ERROR) return status;
    if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
    
    return launchKernel_C1<Npp32f>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
                                    oSizeROI, 0, nppStreamCtx.hStream);
}

NppStatus nppiAdd_32f_C3R_Ctx_impl(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                                    Npp32f *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    NppStatus status = validateParameters(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, 3 * sizeof(Npp32f));
    if (status != NPP_NO_ERROR) return status;
    if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
    
    return launchKernel_C3<Npp32f>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
                                    oSizeROI, 0, nppStreamCtx.hStream);
}
}

// Public API functions
NppStatus nppiAdd_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                 Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx) {
    return nppiAdd_8u_C1RSfs_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
                                       oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAdd_8u_C1RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                             Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAdd_8u_C1RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
                                  oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAdd_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                 Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx) {
    return nppiAdd_8u_C3RSfs_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
                                       oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAdd_8u_C3RSfs(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                             Npp8u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAdd_8u_C3RSfs_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
                                  oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAdd_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                               Npp32f *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiAdd_32f_C1R_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
                                     oSizeROI, nppStreamCtx);
}

NppStatus nppiAdd_32f_C1R(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                           Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAdd_32f_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
                                oSizeROI, nppStreamCtx);
}

NppStatus nppiAdd_32f_C3R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                               Npp32f *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiAdd_32f_C3R_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
                                     oSizeROI, nppStreamCtx);
}

NppStatus nppiAdd_32f_C3R(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                           Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAdd_32f_C3R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
                                oSizeROI, nppStreamCtx);
}

// In-place versions
NppStatus nppiAdd_8u_C1IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
    return nppiAdd_8u_C1RSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep,
                                  oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAdd_8u_C1IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAdd_8u_C1IRSfs_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAdd_32f_C1IR_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    return nppiAdd_32f_C1R_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep,
                                oSizeROI, nppStreamCtx);
}

NppStatus nppiAdd_32f_C1IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAdd_32f_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}