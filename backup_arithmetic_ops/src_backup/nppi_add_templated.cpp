#include "nppi_arithmetic_macros.h"
#include "nppi_arithmetic_functors.h"
#include "npp.h"
#include <string_view>

// Generate add implementations using macros
// 8-bit unsigned with scaling
NPPI_BINARY_OP_IMPL(Add, AddFunctor, 8u, 1, Sfs)
NPPI_BINARY_OP_IMPL(Add, AddFunctor, 8u, 3, Sfs)
NPPI_BINARY_OP_IMPL(Add, AddFunctor, 8u, 4, Sfs)

// 16-bit unsigned with scaling
NPPI_BINARY_OP_IMPL(Add, AddFunctor, 16u, 1, Sfs)
NPPI_BINARY_OP_IMPL(Add, AddFunctor, 16u, 3, Sfs)

// 16-bit signed with scaling
NPPI_BINARY_OP_IMPL(Add, AddFunctor, 16s, 1, Sfs)
NPPI_BINARY_OP_IMPL(Add, AddFunctor, 16s, 3, Sfs)

// 32-bit float (no scaling)
NPPI_BINARY_OP_FLOAT_IMPL(Add, AddFunctor, 32f, 1)
NPPI_BINARY_OP_FLOAT_IMPL(Add, AddFunctor, 32f, 3)
NPPI_BINARY_OP_FLOAT_IMPL(Add, AddFunctor, 32f, 4)

// Multi-channel array support (C1 only, with array of constants)
extern "C" NppStatus nppiAdd_8u_C1RSfs_AC4_Ctx_impl(
    const Npp8u *pSrc1, int nSrc1Step,
    const Npp8u *pSrc2, int nSrc2Step, 
    Npp8u *pDst, int nDstStep,
    NppiSize oSizeROI, int nScaleFactor,
    NppStreamContext nppStreamCtx) {
    
    // AC4 means 4-channel with alpha channel ignored
    // This is equivalent to processing first 3 channels
    return nppiAdd_8u_C3RSfs_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                      pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

// Additional API wrappers for backward compatibility 
NppStatus nppiAdd_8u_C1RSfs_AC4_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                     const Npp8u *pSrc2, int nSrc2Step, 
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, int nScaleFactor,
                                     NppStreamContext nppStreamCtx) {
    return nppiAdd_8u_C1RSfs_AC4_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step,
                                          pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}

NppStatus nppiAdd_8u_C1RSfs_AC4(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp8u *pSrc2, int nSrc2Step,
                                 Npp8u *pDst, int nDstStep,
                                 NppiSize oSizeROI, int nScaleFactor) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAdd_8u_C1RSfs_AC4_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step,
                                     pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx);
}