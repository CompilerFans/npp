#ifndef NPPI_ARITHMETIC_MACROS_H
#define NPPI_ARITHMETIC_MACROS_H

#include "nppi_arithmetic_templates.h"
#include "npp.h"

// Macro to validate scale factor for integer types
#define VALIDATE_SCALE_FACTOR(scaleFactor) \
    if (scaleFactor < 0 || scaleFactor > 31) { \
        return NPP_BAD_ARGUMENT_ERROR; \
    }

// Macro to generate binary operation API implementations
#define NPPI_BINARY_OP_IMPL(OpName, OpFunctor, DataType, Channels, HasScale) \
    extern "C" NppStatus nppi##OpName##_##DataType##_C##Channels##R##HasScale##_Ctx_impl( \
        const Npp##DataType *pSrc1, int nSrc1Step, \
        const Npp##DataType *pSrc2, int nSrc2Step, \
        Npp##DataType *pDst, int nDstStep, \
        NppiSize oSizeROI, \
        int nScaleFactor, \
        NppStreamContext nppStreamCtx) { \
        \
        if constexpr (std::string_view(#HasScale) == "Sfs") { \
            VALIDATE_SCALE_FACTOR(nScaleFactor); \
        } \
        \
        using FunctorType = nppi::functors::OpFunctor<Npp##DataType>; \
        FunctorType functor; \
        \
        return nppi::templates::launchBinaryOp<Npp##DataType, Channels>( \
            pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, \
            oSizeROI, functor, \
            (std::string_view(#HasScale) == "Sfs") ? nScaleFactor : 0, \
            nppStreamCtx.hStream); \
    } \
    \
    NppStatus nppi##OpName##_##DataType##_C##Channels##R##HasScale##_Ctx( \
        const Npp##DataType *pSrc1, int nSrc1Step, \
        const Npp##DataType *pSrc2, int nSrc2Step, \
        Npp##DataType *pDst, int nDstStep, \
        NppiSize oSizeROI, \
        int nScaleFactor, \
        NppStreamContext nppStreamCtx) { \
        \
        return nppi##OpName##_##DataType##_C##Channels##R##HasScale##_Ctx_impl( \
            pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, \
            oSizeROI, nScaleFactor, nppStreamCtx); \
    } \
    \
    NppStatus nppi##OpName##_##DataType##_C##Channels##R##HasScale( \
        const Npp##DataType *pSrc1, int nSrc1Step, \
        const Npp##DataType *pSrc2, int nSrc2Step, \
        Npp##DataType *pDst, int nDstStep, \
        NppiSize oSizeROI, \
        int nScaleFactor) { \
        \
        NppStreamContext nppStreamCtx; \
        nppGetStreamContext(&nppStreamCtx); \
        return nppi##OpName##_##DataType##_C##Channels##R##HasScale##_Ctx( \
            pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, \
            oSizeROI, nScaleFactor, nppStreamCtx); \
    } \
    \
    NppStatus nppi##OpName##_##DataType##_C##Channels##I##HasScale##_Ctx( \
        const Npp##DataType *pSrc, int nSrcStep, \
        Npp##DataType *pSrcDst, int nSrcDstStep, \
        NppiSize oSizeROI, \
        int nScaleFactor, \
        NppStreamContext nppStreamCtx) { \
        \
        return nppi##OpName##_##DataType##_C##Channels##R##HasScale##_Ctx( \
            pSrc, nSrcStep, pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, \
            oSizeROI, nScaleFactor, nppStreamCtx); \
    } \
    \
    NppStatus nppi##OpName##_##DataType##_C##Channels##I##HasScale( \
        const Npp##DataType *pSrc, int nSrcStep, \
        Npp##DataType *pSrcDst, int nSrcDstStep, \
        NppiSize oSizeROI, \
        int nScaleFactor) { \
        \
        NppStreamContext nppStreamCtx; \
        nppGetStreamContext(&nppStreamCtx); \
        return nppi##OpName##_##DataType##_C##Channels##I##HasScale##_Ctx( \
            pSrc, nSrcStep, pSrcDst, nSrcDstStep, \
            oSizeROI, nScaleFactor, nppStreamCtx); \
    }

// Special macro for float types (no scaling)
#define NPPI_BINARY_OP_FLOAT_IMPL(OpName, OpFunctor, DataType, Channels) \
    extern "C" NppStatus nppi##OpName##_##DataType##_C##Channels##R_Ctx_impl( \
        const Npp##DataType *pSrc1, int nSrc1Step, \
        const Npp##DataType *pSrc2, int nSrc2Step, \
        Npp##DataType *pDst, int nDstStep, \
        NppiSize oSizeROI, \
        NppStreamContext nppStreamCtx) { \
        \
        using FunctorType = nppi::functors::OpFunctor<Npp##DataType>; \
        FunctorType functor; \
        \
        return nppi::templates::launchBinaryOp<Npp##DataType, Channels>( \
            pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, \
            oSizeROI, functor, 0, nppStreamCtx.hStream); \
    } \
    \
    NppStatus nppi##OpName##_##DataType##_C##Channels##R_Ctx( \
        const Npp##DataType *pSrc1, int nSrc1Step, \
        const Npp##DataType *pSrc2, int nSrc2Step, \
        Npp##DataType *pDst, int nDstStep, \
        NppiSize oSizeROI, \
        NppStreamContext nppStreamCtx) { \
        \
        return nppi##OpName##_##DataType##_C##Channels##R_Ctx_impl( \
            pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, \
            oSizeROI, nppStreamCtx); \
    } \
    \
    NppStatus nppi##OpName##_##DataType##_C##Channels##R( \
        const Npp##DataType *pSrc1, int nSrc1Step, \
        const Npp##DataType *pSrc2, int nSrc2Step, \
        Npp##DataType *pDst, int nDstStep, \
        NppiSize oSizeROI) { \
        \
        NppStreamContext nppStreamCtx; \
        nppGetStreamContext(&nppStreamCtx); \
        return nppi##OpName##_##DataType##_C##Channels##R_Ctx( \
            pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, \
            oSizeROI, nppStreamCtx); \
    } \
    \
    NppStatus nppi##OpName##_##DataType##_C##Channels##IR_Ctx( \
        const Npp##DataType *pSrc, int nSrcStep, \
        Npp##DataType *pSrcDst, int nSrcDstStep, \
        NppiSize oSizeROI, \
        NppStreamContext nppStreamCtx) { \
        \
        return nppi##OpName##_##DataType##_C##Channels##R_Ctx( \
            pSrc, nSrcStep, pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, \
            oSizeROI, nppStreamCtx); \
    } \
    \
    NppStatus nppi##OpName##_##DataType##_C##Channels##IR( \
        const Npp##DataType *pSrc, int nSrcStep, \
        Npp##DataType *pSrcDst, int nSrcDstStep, \
        NppiSize oSizeROI) { \
        \
        NppStreamContext nppStreamCtx; \
        nppGetStreamContext(&nppStreamCtx); \
        return nppi##OpName##_##DataType##_C##Channels##IR_Ctx( \
            pSrc, nSrcStep, pSrcDst, nSrcDstStep, \
            oSizeROI, nppStreamCtx); \
    }

// Macro to generate unary operation API implementations
#define NPPI_UNARY_OP_IMPL(OpName, OpFunctor, DataType, Channels, HasScale) \
    extern "C" NppStatus nppi##OpName##_##DataType##_C##Channels##R##HasScale##_Ctx_impl( \
        const Npp##DataType *pSrc, int nSrcStep, \
        Npp##DataType *pDst, int nDstStep, \
        NppiSize oSizeROI, \
        int nScaleFactor, \
        NppStreamContext nppStreamCtx) { \
        \
        if constexpr (std::string_view(#HasScale) == "Sfs") { \
            VALIDATE_SCALE_FACTOR(nScaleFactor); \
        } \
        \
        using FunctorType = nppi::functors::OpFunctor<Npp##DataType>; \
        FunctorType functor; \
        \
        return nppi::templates::launchUnaryOp<Npp##DataType, Channels>( \
            pSrc, nSrcStep, pDst, nDstStep, \
            oSizeROI, functor, \
            (std::string_view(#HasScale) == "Sfs") ? nScaleFactor : 0, \
            nppStreamCtx.hStream); \
    } \
    \
    NppStatus nppi##OpName##_##DataType##_C##Channels##R##HasScale##_Ctx( \
        const Npp##DataType *pSrc, int nSrcStep, \
        Npp##DataType *pDst, int nDstStep, \
        NppiSize oSizeROI, \
        int nScaleFactor, \
        NppStreamContext nppStreamCtx) { \
        \
        return nppi##OpName##_##DataType##_C##Channels##R##HasScale##_Ctx_impl( \
            pSrc, nSrcStep, pDst, nDstStep, \
            oSizeROI, nScaleFactor, nppStreamCtx); \
    } \
    \
    NppStatus nppi##OpName##_##DataType##_C##Channels##R##HasScale( \
        const Npp##DataType *pSrc, int nSrcStep, \
        Npp##DataType *pDst, int nDstStep, \
        NppiSize oSizeROI, \
        int nScaleFactor) { \
        \
        NppStreamContext nppStreamCtx; \
        nppGetStreamContext(&nppStreamCtx); \
        return nppi##OpName##_##DataType##_C##Channels##R##HasScale##_Ctx( \
            pSrc, nSrcStep, pDst, nDstStep, \
            oSizeROI, nScaleFactor, nppStreamCtx); \
    } \
    \
    NppStatus nppi##OpName##_##DataType##_C##Channels##I##HasScale##_Ctx( \
        Npp##DataType *pSrcDst, int nSrcDstStep, \
        NppiSize oSizeROI, \
        int nScaleFactor, \
        NppStreamContext nppStreamCtx) { \
        \
        return nppi##OpName##_##DataType##_C##Channels##R##HasScale##_Ctx( \
            pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, \
            oSizeROI, nScaleFactor, nppStreamCtx); \
    } \
    \
    NppStatus nppi##OpName##_##DataType##_C##Channels##I##HasScale( \
        Npp##DataType *pSrcDst, int nSrcDstStep, \
        NppiSize oSizeROI, \
        int nScaleFactor) { \
        \
        NppStreamContext nppStreamCtx; \
        nppGetStreamContext(&nppStreamCtx); \
        return nppi##OpName##_##DataType##_C##Channels##I##HasScale##_Ctx( \
            pSrcDst, nSrcDstStep, \
            oSizeROI, nScaleFactor, nppStreamCtx); \
    }

// Macro to generate constant operation API implementations
#define NPPI_CONST_OP_IMPL(OpName, OpFunctor, DataType, Channels, HasScale, ConstType) \
    extern "C" NppStatus nppi##OpName##_##DataType##_C##Channels##R##HasScale##_Ctx_impl( \
        const Npp##DataType *pSrc, int nSrcStep, \
        const ConstType nConstant, \
        Npp##DataType *pDst, int nDstStep, \
        NppiSize oSizeROI, \
        int nScaleFactor, \
        NppStreamContext nppStreamCtx) { \
        \
        if constexpr (std::string_view(#HasScale) == "Sfs") { \
            VALIDATE_SCALE_FACTOR(nScaleFactor); \
        } \
        \
        using FunctorType = nppi::functors::OpFunctor<Npp##DataType>; \
        FunctorType functor(static_cast<Npp##DataType>(nConstant)); \
        \
        return nppi::templates::launchConstOp<Npp##DataType, Channels>( \
            pSrc, nSrcStep, pDst, nDstStep, \
            oSizeROI, functor, \
            (std::string_view(#HasScale) == "Sfs") ? nScaleFactor : 0, \
            nppStreamCtx.hStream); \
    }

#endif // NPPI_ARITHMETIC_MACROS_H