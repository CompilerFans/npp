#ifndef NPPS_KERNELS_H
#define NPPS_KERNELS_H

#include "npp.h"
#include <cuda_runtime.h>

/**
 * NPPS CUDA Kernel Function Declarations
 * 
 * 这个头文件声明了NPPS模块内部使用的CUDA内核函数。
 * 这些函数仅在NPPS模块内部使用，不对外暴露。
 */

#ifdef __cplusplus
extern "C" {
#endif

// ==============================================================================
// Arithmetic Operations Kernels
// ==============================================================================

cudaError_t nppsAdd_32f_kernel(const Npp32f* pSrc1, const Npp32f* pSrc2, Npp32f* pDst, size_t nLength, cudaStream_t stream);
cudaError_t nppsAdd_16s_kernel(const Npp16s* pSrc1, const Npp16s* pSrc2, Npp16s* pDst, size_t nLength, cudaStream_t stream);
cudaError_t nppsAdd_32fc_kernel(const Npp32fc* pSrc1, const Npp32fc* pSrc2, Npp32fc* pDst, size_t nLength, cudaStream_t stream);

// ==============================================================================
// Statistics Functions Kernels
// ==============================================================================

cudaError_t nppsSum_32f_kernel(const Npp32f* pSrc, size_t nLength, Npp32f* pSum, Npp8u* pDeviceBuffer, cudaStream_t stream);
cudaError_t nppsSum_32fc_kernel(const Npp32fc* pSrc, size_t nLength, Npp32fc* pSum, Npp8u* pDeviceBuffer, cudaStream_t stream);

// ==============================================================================
// Initialization Functions Kernels
// ==============================================================================

cudaError_t nppsSet_8u_kernel(Npp8u nValue, Npp8u* pDst, size_t nLength, cudaStream_t stream);
cudaError_t nppsSet_32f_kernel(Npp32f nValue, Npp32f* pDst, size_t nLength, cudaStream_t stream);
cudaError_t nppsSet_32fc_kernel(Npp32fc nValue, Npp32fc* pDst, size_t nLength, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // NPPS_KERNELS_H