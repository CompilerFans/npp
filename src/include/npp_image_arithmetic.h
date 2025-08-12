#ifndef NPP_IMAGE_ARITHMETIC_H
#define NPP_IMAGE_ARITHMETIC_H

#include "nppdefs.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 8-bit unsigned char channel image add constant, scale, then clamp to saturated value.
 * 
 * @param pSrc1 Source image pointer (device memory)
 * @param nSrc1Step Source image line step in bytes
 * @param nConstant Constant value to add
 * @param pDst Destination image pointer (device memory)
 * @param nDstStep Destination image line step in bytes
 * @param oSizeROI Region of interest (width and height)
 * @param nScaleFactor Scaling factor (right shift amount)
 * @param nppStreamCtx CUDA stream context
 * @return NppStatus error code
 */
NppStatus 
nppiAddC_8u_C1RSfs_Ctx_host(const Npp8u* pSrc1, int nSrc1Step, const Npp8u nConstant,
                                Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                NppStreamContext nppStreamCtx);

/**
 * 8-bit unsigned char channel image add constant, scale, then clamp to saturated value (CUDA version).
 * 
 * @param pSrc1 Source image pointer (device memory)
 * @param nSrc1Step Source image line step in bytes
 * @param nConstant Constant value to add
 * @param pDst Destination image pointer (device memory)
 * @param nDstStep Destination image line step in bytes
 * @param oSizeROI Region of interest (width and height)
 * @param nScaleFactor Scaling factor (right shift amount)
 * @param nppStreamCtx CUDA stream context
 * @return NppStatus error code
 */
NppStatus 
nppiAddC_8u_C1RSfs_Ctx_cuda(const Npp8u* pSrc1, int nSrc1Step, const Npp8u nConstant,
                              Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                              NppStreamContext nppStreamCtx);

/**
 * 8-bit unsigned char channel image add constant, scale, then clamp to saturated value (reference implementation).
 * CPU-based reference implementation for testing.
 * 
 * @param pSrc1 Source image pointer
 * @param nSrc1Step Source image line step in bytes
 * @param nConstant Constant value to add
 * @param pDst Destination image pointer
 * @param nDstStep Destination image line step in bytes
 * @param oSizeROI Region of interest (width and height)
 * @param nScaleFactor Scaling factor (right shift amount)
 * @return NppStatus error code
 */
NppStatus 
nppiAddC_8u_C1RSfs_Ctx_reference(const Npp8u* pSrc1, int nSrc1Step, const Npp8u nConstant,
                                   Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);

#ifdef __cplusplus
}
#endif

#endif // NPP_IMAGE_ARITHMETIC_H