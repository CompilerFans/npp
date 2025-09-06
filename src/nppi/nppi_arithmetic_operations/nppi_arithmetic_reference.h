/**
 * @file nppi_arithmetic_reference.h
 * @brief CPU参考实现的算术运算函数声明
 */

#ifndef NPPI_ARITHMETIC_REFERENCE_H
#define NPPI_ARITHMETIC_REFERENCE_H

#include "npp.h"

namespace nppi_reference {

// AddC参考实现
NppStatus nppiAddC_8u_C1RSfs_reference(Npp8u* pSrc, int nSrcStep, Npp8u nConstant, 
                                       Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
NppStatus nppiAddC_16u_C1RSfs_reference(Npp16u* pSrc, int nSrcStep, Npp16u nConstant,
                                        Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
NppStatus nppiAddC_16s_C1RSfs_reference(Npp16s* pSrc, int nSrcStep, Npp16s nConstant,
                                        Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
NppStatus nppiAddC_32f_C1R_reference(Npp32f* pSrc, int nSrcStep, Npp32f nConstant,
                                     Npp32f* pDst, int nDstStep, NppiSize oSizeROI);

// SubC参考实现
NppStatus nppiSubC_8u_C1RSfs_reference(Npp8u* pSrc, int nSrcStep, Npp8u nConstant,
                                       Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
NppStatus nppiSubC_16u_C1RSfs_reference(Npp16u* pSrc, int nSrcStep, Npp16u nConstant,
                                        Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
NppStatus nppiSubC_16s_C1RSfs_reference(Npp16s* pSrc, int nSrcStep, Npp16s nConstant,
                                        Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
NppStatus nppiSubC_32f_C1R_reference(Npp32f* pSrc, int nSrcStep, Npp32f nConstant,
                                     Npp32f* pDst, int nDstStep, NppiSize oSizeROI);

// MulC参考实现
NppStatus nppiMulC_8u_C1RSfs_reference(Npp8u* pSrc, int nSrcStep, Npp8u nConstant,
                                       Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
NppStatus nppiMulC_16u_C1RSfs_reference(Npp16u* pSrc, int nSrcStep, Npp16u nConstant,
                                        Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
NppStatus nppiMulC_16s_C1RSfs_reference(Npp16s* pSrc, int nSrcStep, Npp16s nConstant,
                                        Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
NppStatus nppiMulC_32f_C1R_reference(Npp32f* pSrc, int nSrcStep, Npp32f nConstant,
                                     Npp32f* pDst, int nDstStep, NppiSize oSizeROI);

// DivC参考实现
NppStatus nppiDivC_8u_C1RSfs_reference(Npp8u* pSrc, int nSrcStep, Npp8u nConstant,
                                       Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
NppStatus nppiDivC_16u_C1RSfs_reference(Npp16u* pSrc, int nSrcStep, Npp16u nConstant,
                                        Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
NppStatus nppiDivC_16s_C1RSfs_reference(Npp16s* pSrc, int nSrcStep, Npp16s nConstant,
                                        Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
NppStatus nppiDivC_32f_C1R_reference(Npp32f* pSrc, int nSrcStep, Npp32f nConstant,
                                     Npp32f* pDst, int nDstStep, NppiSize oSizeROI);

} // namespace nppi_reference

#endif // NPPI_ARITHMETIC_REFERENCE_H