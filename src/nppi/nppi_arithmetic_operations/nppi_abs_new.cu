#include "nppi_arithmetic_common.h"

using namespace nppi::arithmetic;

// For unary operations, we need a special kernel template
template<typename T, int Channels, typename UnaryOp>
__global__ void unaryOpKernel_C(const T *pSrc, int nSrcStep,
                                T *pDst, int nDstStep,
                                int width, int height,
                                UnaryOp op, int scaleFactor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const T *srcRow = (const T *)((const char *)pSrc + y * nSrcStep);
        T *dstRow = (T *)((char *)pDst + y * nDstStep);

        if constexpr (Channels == 1) {
            dstRow[x] = op(srcRow[x], T(0), scaleFactor);
        } else {
            int idx = x * Channels;
            for (int c = 0; c < Channels; c++) {
                dstRow[idx + c] = op(srcRow[idx + c], T(0), scaleFactor);
            }
        }
    }
}

// Launch helper for unary operations
template<typename T, int Channels, typename UnaryOp>
static NppStatus launchUnaryOpKernel(const T *pSrc, int nSrcStep,
                                     T *pDst, int nDstStep,
                                     NppiSize oSizeROI, int scaleFactor,
                                     cudaStream_t stream, UnaryOp op) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    unaryOpKernel_C<T, Channels, UnaryOp><<<gridSize, blockSize, 0, stream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, op, scaleFactor);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// Implement Abs operations using unary kernel
extern "C" {
NppStatus nppiAbs_8s_C1R_Ctx_impl(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst, int nDstStep,
                                   NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    AbsOp<Npp8s> op;
    return launchUnaryOpKernel<Npp8s, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0, nppStreamCtx.hStream, op);
}

NppStatus nppiAbs_16s_C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    AbsOp<Npp16s> op;
    return launchUnaryOpKernel<Npp16s, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0, nppStreamCtx.hStream, op);
}

NppStatus nppiAbs_32s_C1R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    AbsOp<Npp32s> op;
    return launchUnaryOpKernel<Npp32s, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0, nppStreamCtx.hStream, op);
}

NppStatus nppiAbs_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    AbsOp<Npp32f> op;
    return launchUnaryOpKernel<Npp32f, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0, nppStreamCtx.hStream, op);
}
}

// Public API functions
NppStatus nppiAbs_8s_C1R_Ctx(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    NppStatus status = validateParameters(pSrc, nSrcStep, pDst, nSrcStep, pDst, nDstStep, oSizeROI, sizeof(Npp8s));
    if (status != NPP_NO_ERROR) return status;
    if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
    return nppiAbs_8s_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_8s_C1R(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst, int nDstStep, NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbs_8s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    NppStatus status = validateParameters(pSrc, nSrcStep, pDst, nDstStep, pDst, nDstStep, oSizeROI, sizeof(Npp16s));
    if (status != NPP_NO_ERROR) return status;
    if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
    return nppiAbs_16s_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_16s_C1R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbs_16s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32s_C1R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    NppStatus status = validateParameters(pSrc, nSrcStep, pDst, nDstStep, pDst, nDstStep, oSizeROI, sizeof(Npp32s));
    if (status != NPP_NO_ERROR) return status;
    if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
    return nppiAbs_32s_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32s_C1R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbs_32s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    NppStatus status = validateParameters(pSrc, nSrcStep, pDst, nDstStep, pDst, nDstStep, oSizeROI, sizeof(Npp32f));
    if (status != NPP_NO_ERROR) return status;
    if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
    return nppiAbs_32f_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbs_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    return nppiAbs_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}