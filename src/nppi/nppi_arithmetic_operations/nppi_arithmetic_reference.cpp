#include "npp.h"
#include <algorithm>
#include <cmath>

/**
 * 完整的CPU参考实现 - 所有算术运算
 * 用于3方对比验证：OpenNPP vs NVIDIA NPP vs CPU参考
 */

namespace nppi_reference {

// ============================================================================
// AddC Operations - 图像加常数
// ============================================================================

NppStatus nppiAddC_8u_C1RSfs_reference(const Npp8u* pSrc, int nSrcStep, const Npp8u nConstant,
                                       Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
    if (nSrcStep < oSizeROI.width || nDstStep < oSizeROI.width) return NPP_STRIDE_ERROR;
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;

    for (int y = 0; y < oSizeROI.height; y++) {
        const Npp8u* srcRow = reinterpret_cast<const Npp8u*>(
            reinterpret_cast<const char*>(pSrc) + y * nSrcStep);
        Npp8u* dstRow = reinterpret_cast<Npp8u*>(
            reinterpret_cast<char*>(pDst) + y * nDstStep);
        
        for (int x = 0; x < oSizeROI.width; x++) {
            int result = static_cast<int>(srcRow[x]) + static_cast<int>(nConstant);
            result = result >> nScaleFactor;
            dstRow[x] = static_cast<Npp8u>(std::max(0, std::min(255, result)));
        }
    }
    return NPP_NO_ERROR;
}

NppStatus nppiAddC_16u_C1RSfs_reference(const Npp16u* pSrc, int nSrcStep, const Npp16u nConstant,
                                        Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
    if (nSrcStep < oSizeROI.width * sizeof(Npp16u) || nDstStep < oSizeROI.width * sizeof(Npp16u)) return NPP_STRIDE_ERROR;
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;

    for (int y = 0; y < oSizeROI.height; y++) {
        const Npp16u* srcRow = reinterpret_cast<const Npp16u*>(
            reinterpret_cast<const char*>(pSrc) + y * nSrcStep);
        Npp16u* dstRow = reinterpret_cast<Npp16u*>(
            reinterpret_cast<char*>(pDst) + y * nDstStep);
        
        for (int x = 0; x < oSizeROI.width; x++) {
            int result = static_cast<int>(srcRow[x]) + static_cast<int>(nConstant);
            result = result >> nScaleFactor;
            dstRow[x] = static_cast<Npp16u>(std::max(0, std::min(65535, result)));
        }
    }
    return NPP_NO_ERROR;
}

NppStatus nppiAddC_16s_C1RSfs_reference(const Npp16s* pSrc, int nSrcStep, const Npp16s nConstant,
                                        Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
    if (nSrcStep < oSizeROI.width * sizeof(Npp16s) || nDstStep < oSizeROI.width * sizeof(Npp16s)) return NPP_STRIDE_ERROR;
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;

    for (int y = 0; y < oSizeROI.height; y++) {
        const Npp16s* srcRow = reinterpret_cast<const Npp16s*>(
            reinterpret_cast<const char*>(pSrc) + y * nSrcStep);
        Npp16s* dstRow = reinterpret_cast<Npp16s*>(
            reinterpret_cast<char*>(pDst) + y * nDstStep);
        
        for (int x = 0; x < oSizeROI.width; x++) {
            int result = static_cast<int>(srcRow[x]) + static_cast<int>(nConstant);
            result = result >> nScaleFactor;
            dstRow[x] = static_cast<Npp16s>(std::max(-32768, std::min(32767, result)));
        }
    }
    return NPP_NO_ERROR;
}

NppStatus nppiAddC_32f_C1R_reference(const Npp32f* pSrc, int nSrcStep, const Npp32f nConstant,
                                     Npp32f* pDst, int nDstStep, NppiSize oSizeROI)
{
    if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
    if (nSrcStep < oSizeROI.width * sizeof(Npp32f) || nDstStep < oSizeROI.width * sizeof(Npp32f)) return NPP_STRIDE_ERROR;

    for (int y = 0; y < oSizeROI.height; y++) {
        const Npp32f* srcRow = reinterpret_cast<const Npp32f*>(
            reinterpret_cast<const char*>(pSrc) + y * nSrcStep);
        Npp32f* dstRow = reinterpret_cast<Npp32f*>(
            reinterpret_cast<char*>(pDst) + y * nDstStep);
        
        for (int x = 0; x < oSizeROI.width; x++) {
            dstRow[x] = srcRow[x] + nConstant;
        }
    }
    return NPP_NO_ERROR;
}

// ============================================================================
// SubC Operations - 图像减常数
// ============================================================================

NppStatus nppiSubC_8u_C1RSfs_reference(const Npp8u* pSrc, int nSrcStep, const Npp8u nConstant,
                                       Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
    if (nSrcStep < oSizeROI.width || nDstStep < oSizeROI.width) return NPP_STRIDE_ERROR;
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;

    for (int y = 0; y < oSizeROI.height; y++) {
        const Npp8u* srcRow = reinterpret_cast<const Npp8u*>(
            reinterpret_cast<const char*>(pSrc) + y * nSrcStep);
        Npp8u* dstRow = reinterpret_cast<Npp8u*>(
            reinterpret_cast<char*>(pDst) + y * nDstStep);
        
        for (int x = 0; x < oSizeROI.width; x++) {
            int result = static_cast<int>(srcRow[x]) - static_cast<int>(nConstant);
            result = result >> nScaleFactor;
            dstRow[x] = static_cast<Npp8u>(std::max(0, std::min(255, result)));
        }
    }
    return NPP_NO_ERROR;
}

NppStatus nppiSubC_16u_C1RSfs_reference(const Npp16u* pSrc, int nSrcStep, const Npp16u nConstant,
                                        Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
    if (nSrcStep < oSizeROI.width * sizeof(Npp16u) || nDstStep < oSizeROI.width * sizeof(Npp16u)) return NPP_STRIDE_ERROR;
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;

    for (int y = 0; y < oSizeROI.height; y++) {
        const Npp16u* srcRow = reinterpret_cast<const Npp16u*>(
            reinterpret_cast<const char*>(pSrc) + y * nSrcStep);
        Npp16u* dstRow = reinterpret_cast<Npp16u*>(
            reinterpret_cast<char*>(pDst) + y * nDstStep);
        
        for (int x = 0; x < oSizeROI.width; x++) {
            int result = static_cast<int>(srcRow[x]) - static_cast<int>(nConstant);
            result = result >> nScaleFactor;
            dstRow[x] = static_cast<Npp16u>(std::max(0, std::min(65535, result)));
        }
    }
    return NPP_NO_ERROR;
}

NppStatus nppiSubC_16s_C1RSfs_reference(const Npp16s* pSrc, int nSrcStep, const Npp16s nConstant,
                                        Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
    if (nSrcStep < oSizeROI.width * sizeof(Npp16s) || nDstStep < oSizeROI.width * sizeof(Npp16s)) return NPP_STRIDE_ERROR;
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;

    for (int y = 0; y < oSizeROI.height; y++) {
        const Npp16s* srcRow = reinterpret_cast<const Npp16s*>(
            reinterpret_cast<const char*>(pSrc) + y * nSrcStep);
        Npp16s* dstRow = reinterpret_cast<Npp16s*>(
            reinterpret_cast<char*>(pDst) + y * nDstStep);
        
        for (int x = 0; x < oSizeROI.width; x++) {
            int result = static_cast<int>(srcRow[x]) - static_cast<int>(nConstant);
            result = result >> nScaleFactor;
            dstRow[x] = static_cast<Npp16s>(std::max(-32768, std::min(32767, result)));
        }
    }
    return NPP_NO_ERROR;
}

NppStatus nppiSubC_32f_C1R_reference(const Npp32f* pSrc, int nSrcStep, const Npp32f nConstant,
                                     Npp32f* pDst, int nDstStep, NppiSize oSizeROI)
{
    if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
    if (nSrcStep < oSizeROI.width * sizeof(Npp32f) || nDstStep < oSizeROI.width * sizeof(Npp32f)) return NPP_STRIDE_ERROR;

    for (int y = 0; y < oSizeROI.height; y++) {
        const Npp32f* srcRow = reinterpret_cast<const Npp32f*>(
            reinterpret_cast<const char*>(pSrc) + y * nSrcStep);
        Npp32f* dstRow = reinterpret_cast<Npp32f*>(
            reinterpret_cast<char*>(pDst) + y * nDstStep);
        
        for (int x = 0; x < oSizeROI.width; x++) {
            dstRow[x] = srcRow[x] - nConstant;
        }
    }
    return NPP_NO_ERROR;
}

// ============================================================================
// MulC Operations - 图像乘常数
// ============================================================================

NppStatus nppiMulC_8u_C1RSfs_reference(const Npp8u* pSrc, int nSrcStep, const Npp8u nConstant,
                                       Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
    if (nSrcStep < oSizeROI.width || nDstStep < oSizeROI.width) return NPP_STRIDE_ERROR;
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;

    for (int y = 0; y < oSizeROI.height; y++) {
        const Npp8u* srcRow = reinterpret_cast<const Npp8u*>(
            reinterpret_cast<const char*>(pSrc) + y * nSrcStep);
        Npp8u* dstRow = reinterpret_cast<Npp8u*>(
            reinterpret_cast<char*>(pDst) + y * nDstStep);
        
        for (int x = 0; x < oSizeROI.width; x++) {
            int result = static_cast<int>(srcRow[x]) * static_cast<int>(nConstant);
            result = result >> nScaleFactor;
            dstRow[x] = static_cast<Npp8u>(std::max(0, std::min(255, result)));
        }
    }
    return NPP_NO_ERROR;
}

NppStatus nppiMulC_16u_C1RSfs_reference(const Npp16u* pSrc, int nSrcStep, const Npp16u nConstant,
                                        Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
    if (nSrcStep < oSizeROI.width * sizeof(Npp16u) || nDstStep < oSizeROI.width * sizeof(Npp16u)) return NPP_STRIDE_ERROR;
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;

    for (int y = 0; y < oSizeROI.height; y++) {
        const Npp16u* srcRow = reinterpret_cast<const Npp16u*>(
            reinterpret_cast<const char*>(pSrc) + y * nSrcStep);
        Npp16u* dstRow = reinterpret_cast<Npp16u*>(
            reinterpret_cast<char*>(pDst) + y * nDstStep);
        
        for (int x = 0; x < oSizeROI.width; x++) {
            int result = static_cast<int>(srcRow[x]) * static_cast<int>(nConstant);
            result = result >> nScaleFactor;
            dstRow[x] = static_cast<Npp16u>(std::max(0, std::min(65535, result)));
        }
    }
    return NPP_NO_ERROR;
}

NppStatus nppiMulC_16s_C1RSfs_reference(const Npp16s* pSrc, int nSrcStep, const Npp16s nConstant,
                                        Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
    if (nSrcStep < oSizeROI.width * sizeof(Npp16s) || nDstStep < oSizeROI.width * sizeof(Npp16s)) return NPP_STRIDE_ERROR;
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;

    for (int y = 0; y < oSizeROI.height; y++) {
        const Npp16s* srcRow = reinterpret_cast<const Npp16s*>(
            reinterpret_cast<const char*>(pSrc) + y * nSrcStep);
        Npp16s* dstRow = reinterpret_cast<Npp16s*>(
            reinterpret_cast<char*>(pDst) + y * nDstStep);
        
        for (int x = 0; x < oSizeROI.width; x++) {
            int result = static_cast<int>(srcRow[x]) * static_cast<int>(nConstant);
            result = result >> nScaleFactor;
            dstRow[x] = static_cast<Npp16s>(std::max(-32768, std::min(32767, result)));
        }
    }
    return NPP_NO_ERROR;
}

NppStatus nppiMulC_32f_C1R_reference(const Npp32f* pSrc, int nSrcStep, const Npp32f nConstant,
                                     Npp32f* pDst, int nDstStep, NppiSize oSizeROI)
{
    if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
    if (nSrcStep < oSizeROI.width * sizeof(Npp32f) || nDstStep < oSizeROI.width * sizeof(Npp32f)) return NPP_STRIDE_ERROR;

    for (int y = 0; y < oSizeROI.height; y++) {
        const Npp32f* srcRow = reinterpret_cast<const Npp32f*>(
            reinterpret_cast<const char*>(pSrc) + y * nSrcStep);
        Npp32f* dstRow = reinterpret_cast<Npp32f*>(
            reinterpret_cast<char*>(pDst) + y * nDstStep);
        
        for (int x = 0; x < oSizeROI.width; x++) {
            dstRow[x] = srcRow[x] * nConstant;
        }
    }
    return NPP_NO_ERROR;
}

// ============================================================================
// DivC Operations - 图像除常数
// ============================================================================

NppStatus nppiDivC_8u_C1RSfs_reference(const Npp8u* pSrc, int nSrcStep, const Npp8u nConstant,
                                       Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
    if (nSrcStep < oSizeROI.width || nDstStep < oSizeROI.width) return NPP_STRIDE_ERROR;
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;
    if (nConstant == 0) return NPP_DIVIDE_BY_ZERO_ERROR;

    for (int y = 0; y < oSizeROI.height; y++) {
        const Npp8u* srcRow = reinterpret_cast<const Npp8u*>(
            reinterpret_cast<const char*>(pSrc) + y * nSrcStep);
        Npp8u* dstRow = reinterpret_cast<Npp8u*>(
            reinterpret_cast<char*>(pDst) + y * nDstStep);
        
        for (int x = 0; x < oSizeROI.width; x++) {
            int result = static_cast<int>(srcRow[x]) / static_cast<int>(nConstant);
            result = result << nScaleFactor;  // 除法后左移
            dstRow[x] = static_cast<Npp8u>(std::max(0, std::min(255, result)));
        }
    }
    return NPP_NO_ERROR;
}

NppStatus nppiDivC_16u_C1RSfs_reference(const Npp16u* pSrc, int nSrcStep, const Npp16u nConstant,
                                        Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
    if (nSrcStep < oSizeROI.width * sizeof(Npp16u) || nDstStep < oSizeROI.width * sizeof(Npp16u)) return NPP_STRIDE_ERROR;
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;
    if (nConstant == 0) return NPP_DIVIDE_BY_ZERO_ERROR;

    for (int y = 0; y < oSizeROI.height; y++) {
        const Npp16u* srcRow = reinterpret_cast<const Npp16u*>(
            reinterpret_cast<const char*>(pSrc) + y * nSrcStep);
        Npp16u* dstRow = reinterpret_cast<Npp16u*>(
            reinterpret_cast<char*>(pDst) + y * nDstStep);
        
        for (int x = 0; x < oSizeROI.width; x++) {
            int result = static_cast<int>(srcRow[x]) / static_cast<int>(nConstant);
            result = result << nScaleFactor;
            dstRow[x] = static_cast<Npp16u>(std::max(0, std::min(65535, result)));
        }
    }
    return NPP_NO_ERROR;
}

NppStatus nppiDivC_16s_C1RSfs_reference(const Npp16s* pSrc, int nSrcStep, const Npp16s nConstant,
                                        Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
    if (nSrcStep < oSizeROI.width * sizeof(Npp16s) || nDstStep < oSizeROI.width * sizeof(Npp16s)) return NPP_STRIDE_ERROR;
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;
    if (nConstant == 0) return NPP_DIVIDE_BY_ZERO_ERROR;

    for (int y = 0; y < oSizeROI.height; y++) {
        const Npp16s* srcRow = reinterpret_cast<const Npp16s*>(
            reinterpret_cast<const char*>(pSrc) + y * nSrcStep);
        Npp16s* dstRow = reinterpret_cast<Npp16s*>(
            reinterpret_cast<char*>(pDst) + y * nDstStep);
        
        for (int x = 0; x < oSizeROI.width; x++) {
            int result = static_cast<int>(srcRow[x]) / static_cast<int>(nConstant);
            result = result << nScaleFactor;
            dstRow[x] = static_cast<Npp16s>(std::max(-32768, std::min(32767, result)));
        }
    }
    return NPP_NO_ERROR;
}

NppStatus nppiDivC_32f_C1R_reference(const Npp32f* pSrc, int nSrcStep, const Npp32f nConstant,
                                     Npp32f* pDst, int nDstStep, NppiSize oSizeROI)
{
    if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
    if (nSrcStep < oSizeROI.width * sizeof(Npp32f) || nDstStep < oSizeROI.width * sizeof(Npp32f)) return NPP_STRIDE_ERROR;
    if (std::abs(nConstant) < 1e-7f) return NPP_DIVIDE_BY_ZERO_ERROR;

    for (int y = 0; y < oSizeROI.height; y++) {
        const Npp32f* srcRow = reinterpret_cast<const Npp32f*>(
            reinterpret_cast<const char*>(pSrc) + y * nSrcStep);
        Npp32f* dstRow = reinterpret_cast<Npp32f*>(
            reinterpret_cast<char*>(pDst) + y * nDstStep);
        
        for (int x = 0; x < oSizeROI.width; x++) {
            dstRow[x] = srcRow[x] / nConstant;
        }
    }
    return NPP_NO_ERROR;
}

// ============================================================================
// Multi-Channel Reference Implementations - 多通道参考实现
// ============================================================================

NppStatus nppiAddC_8u_C3RSfs_reference(const Npp8u* pSrc, int nSrcStep, const Npp8u aConstants[3],
                                       Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor)
{
    if (!pSrc || !pDst || !aConstants) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
    if (nSrcStep < oSizeROI.width * 3 || nDstStep < oSizeROI.width * 3) return NPP_STRIDE_ERROR;
    if (nScaleFactor < 0 || nScaleFactor > 31) return NPP_BAD_ARGUMENT_ERROR;

    for (int y = 0; y < oSizeROI.height; y++) {
        const Npp8u* srcRow = reinterpret_cast<const Npp8u*>(
            reinterpret_cast<const char*>(pSrc) + y * nSrcStep);
        Npp8u* dstRow = reinterpret_cast<Npp8u*>(
            reinterpret_cast<char*>(pDst) + y * nDstStep);
        
        for (int x = 0; x < oSizeROI.width; x++) {
            // Process each channel
            for (int c = 0; c < 3; c++) {
                int pixelIdx = x * 3 + c;
                int result = static_cast<int>(srcRow[pixelIdx]) + static_cast<int>(aConstants[c]);
                result = result >> nScaleFactor;
                dstRow[pixelIdx] = static_cast<Npp8u>(std::max(0, std::min(255, result)));
            }
        }
    }
    return NPP_NO_ERROR;
}

} // namespace nppi_reference