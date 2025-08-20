 /* Copyright 2025-2025 MetaX CORPORATION & AFFILIATES.  All rights reserved. 
  * 
  * NOTICE TO LICENSEE: 
  * 
  * The source code and/or documentation ("Licensed Deliverables") are 
  * subject to MetaX intellectual property rights under U.S. and 
  * international Copyright laws. 
  * 
  * The Licensed Deliverables contained herein are PROPRIETARY and 
  * CONFIDENTIAL to MetaX and are being provided under the terms and 
  * conditions of a form of MetaX software license agreement by and 
  * between MetaX and Licensee ("License Agreement") or electronically 
  * accepted by Licensee.  Notwithstanding any terms or conditions to 
  * the contrary in the License Agreement, reproduction or disclosure 
  * of the Licensed Deliverables to any third party without the express 
  * written consent of MetaX is prohibited. 
  * 
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE 
  * LICENSE AGREEMENT, MetaX MAKES NO REPRESENTATION ABOUT THE 
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE 
  * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND. 
  * MetaX DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED 
  * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, 
  * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE. 
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE 
  * LICENSE AGREEMENT, IN NO EVENT SHALL MetaX BE LIABLE FOR ANY 
  * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY 
  * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, 
  * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS 
  * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE 
  * OF THESE LICENSED DELIVERABLES. 
  * 
  * U.S. Government End Users.  These Licensed Deliverables are a 
  * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT 
  * 1995), consisting of "commercial computer software" and "commercial 
  * computer software documentation" as such terms are used in 48 
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government 
  * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and 
  * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all 
  * U.S. Government End Users acquire the Licensed Deliverables with 
  * only those rights set forth herein. 
  * 
  * Any use of the Licensed Deliverables in individual and commercial 
  * software must include, in the user documentation and internal 
  * comments to the code, the above Disclaimer and U.S. Government End 
  * Users Notice. 
  */ 
#ifndef MC_MPPI_THRESHOLD_AND_COMPARE_OPERATIONS_H
#define MC_MPPI_THRESHOLD_AND_COMPARE_OPERATIONS_H
 
/**
 * \file mppi_threshold_and_compare_operations.h
 * MPP Image Processing Functionality.
 */
 
#include "mppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif


/** 
 *  \page image_threshold_and_compare_operations Threshold and Compare Operations
 *  @defgroup image_threshold_and_compare_operations Threshold and Compare Operations
 *  @ingroup mppi
 *
 * Methods for pixel-wise threshold and compare operations.
 *
 * @{
 *
 * These functions can be found in the nppitc library. Linking to only the sub-libraries that you use can significantly
 * save link time, application load time, and MACA runtime startup time when using dynamic libraries.
 *
 */

/** 
 * \section image_thresholding_operations Thresholding Operations
 * @defgroup image_thresholding_operations Thresholding Operations
 * Various types of image thresholding operations.
 *
 * @{
 *
 */

/** 
 * \section image_threshold_operations Threshold Operations
 * @defgroup image_threshold_operations Threshold Operations
 * Threshold image pixels.
 *
 * \subsection CommonThresholdParameters Common parameters for mppiThreshold non-inplace and inplace functions:
 *
 * \param pSrcDst \ref in_place_image_pointer for inplace functions.
 * \param nSrcDstStep \ref in_place_image_line_step for inplace functions.
 * \param pSrc \ref source_image_pointer for non-inplace functions.
 * \param nSrcStep \ref source_image_line_step for non-inplace functions.
 * \param pDst \ref destination_image_pointer for non-inplace functions.
 * \param nDstStep \ref destination_image_line_step for non-inplace functions.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: MPP_CMP_LESS and MPP_CMP_GREATER.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, or MPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_8u_C1R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                         Mpp8u * pDst, int nDstStep, 
                                   MppiSize oSizeROI, 
                                   const Mpp8u nThreshold, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_8u_C1R(const Mpp8u * pSrc, int nSrcStep,
                                     Mpp8u * pDst, int nDstStep, 
                               MppiSize oSizeROI, 
                               const Mpp8u nThreshold, MppCmpOp eComparisonOperation); 

/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_8u_C1IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp8u nThreshold, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_8u_C1IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                                MppiSize oSizeROI, 
                                const Mpp8u nThreshold, MppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16u_C1R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                          Mpp16u * pDst, int nDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp16u nThreshold, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16u_C1R(const Mpp16u * pSrc, int nSrcStep,
                                      Mpp16u * pDst, int nDstStep, 
                                MppiSize oSizeROI, 
                                const Mpp16u nThreshold, MppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16u_C1IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp16u nThreshold, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16u_C1IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                                 MppiSize oSizeROI, 
                                 const Mpp16u nThreshold, MppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16s_C1R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                          Mpp16s * pDst, int nDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp16s nThreshold, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16s_C1R(const Mpp16s * pSrc, int nSrcStep,
                                      Mpp16s * pDst, int nDstStep, 
                                MppiSize oSizeROI, 
                                const Mpp16s nThreshold, MppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16s_C1IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp16s nThreshold, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16s_C1IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                                 MppiSize oSizeROI, 
                                 const Mpp16s nThreshold, MppCmpOp eComparisonOperation); 

/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_32f_C1R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                          Mpp32f * pDst, int nDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp32f nThreshold, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_32f_C1R(const Mpp32f * pSrc, int nSrcStep,
                                      Mpp32f * pDst, int nDstStep, 
                                MppiSize oSizeROI, 
                                const Mpp32f nThreshold, MppCmpOp eComparisonOperation); 

/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_32f_C1IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp32f nThreshold, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_32f_C1IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                                 MppiSize oSizeROI, 
                                 const Mpp32f nThreshold, MppCmpOp eComparisonOperation); 

/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                         Mpp8u * pDst, int nDstStep, 
                                   MppiSize oSizeROI, 
                                   const Mpp8u rThresholds[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_8u_C3R(const Mpp8u * pSrc, int nSrcStep,
                                     Mpp8u * pDst, int nDstStep, 
                               MppiSize oSizeROI, 
                               const Mpp8u rThresholds[3], MppCmpOp eComparisonOperation); 

/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_8u_C3IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp8u rThresholds[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_8u_C3IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                                MppiSize oSizeROI, 
                                const Mpp8u rThresholds[3], MppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16u_C3R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                          Mpp16u * pDst, int nDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp16u rThresholds[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16u_C3R(const Mpp16u * pSrc, int nSrcStep,
                                      Mpp16u * pDst, int nDstStep, 
                                MppiSize oSizeROI, 
                                const Mpp16u rThresholds[3], MppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16u_C3IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp16u rThresholds[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16u_C3IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                                 MppiSize oSizeROI, 
                                 const Mpp16u rThresholds[3], MppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16s_C3R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                          Mpp16s * pDst, int nDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp16s rThresholds[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16s_C3R(const Mpp16s * pSrc, int nSrcStep,
                                      Mpp16s * pDst, int nDstStep, 
                                MppiSize oSizeROI, 
                                const Mpp16s rThresholds[3], MppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16s_C3IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp16s rThresholds[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16s_C3IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                                 MppiSize oSizeROI, 
                                 const Mpp16s rThresholds[3], MppCmpOp eComparisonOperation); 

/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_32f_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                          Mpp32f * pDst, int nDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp32f rThresholds[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_32f_C3R(const Mpp32f * pSrc, int nSrcStep,
                                      Mpp32f * pDst, int nDstStep, 
                                MppiSize oSizeROI, 
                                const Mpp32f rThresholds[3], MppCmpOp eComparisonOperation); 

/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_32f_C3IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp32f rThresholds[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_32f_C3IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                                 MppiSize oSizeROI, 
                                 const Mpp32f rThresholds[3], MppCmpOp eComparisonOperation); 

/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                          Mpp8u * pDst, int nDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp8u rThresholds[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_8u_AC4R(const Mpp8u * pSrc, int nSrcStep,
                                      Mpp8u * pDst, int nDstStep, 
                                MppiSize oSizeROI, 
                                const Mpp8u rThresholds[3], MppCmpOp eComparisonOperation);

/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_8u_AC4IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp8u rThresholds[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_8u_AC4IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                                 MppiSize oSizeROI, 
                                 const Mpp8u rThresholds[3], MppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16u_AC4R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                           Mpp16u * pDst, int nDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp16u rThresholds[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16u_AC4R(const Mpp16u * pSrc, int nSrcStep,
                                       Mpp16u * pDst, int nDstStep, 
                                 MppiSize oSizeROI, 
                                 const Mpp16u rThresholds[3], MppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16u_AC4IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp16u rThresholds[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16u_AC4IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                                  MppiSize oSizeROI, 
                                  const Mpp16u rThresholds[3], MppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16s_AC4R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                           Mpp16s * pDst, int nDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp16s rThresholds[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16s_AC4R(const Mpp16s * pSrc, int nSrcStep,
                                       Mpp16s * pDst, int nDstStep, 
                                 MppiSize oSizeROI, 
                                 const Mpp16s rThresholds[3], MppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16s_AC4IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp16s rThresholds[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_16s_AC4IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                                  MppiSize oSizeROI, 
                                  const Mpp16s rThresholds[3], MppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_32f_AC4R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                           Mpp32f * pDst, int nDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp32f rThresholds[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_32f_AC4R(const Mpp32f * pSrc, int nSrcStep,
                                       Mpp32f * pDst, int nDstStep, 
                                 MppiSize oSizeROI, 
                                 const Mpp32f rThresholds[3], MppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_32f_AC4IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp32f rThresholds[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdParameters.
 *
 */
MppStatus mppiThreshold_32f_AC4IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                                  MppiSize oSizeROI, 
                                  const Mpp32f rThresholds[3], MppCmpOp eComparisonOperation);

/** @} image_threshold_operations */

/** 
 * \section image_threshold_greater_than_operations Threshold Greater Than Operations
 * @defgroup image_threshold_greater_than_operations Threshold Greater Than Operations
 * Threshold greater than image pixels.
 *
 * \subsection CommonThresholdGreaterThanParameters Common parameters for mppiThreshold_GT non-inplace and inplace functions:
 *
 * \param pSrcDst \ref in_place_image_pointer for inplace functions.
 * \param nSrcDstStep \ref in_place_image_line_step for inplace functions.
 * \param pSrc \ref source_image_pointer for non-inplace functions.
 * \param nSrcStep \ref source_image_line_step for non-inplace functions.
 * \param pDst \ref destination_image_pointer for non-inplace functions.
 * \param nDstStep \ref destination_image_line_step for non-inplace functions.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_8u_C1R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                            Mpp8u * pDst, int nDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp8u nThreshold, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_8u_C1R(const Mpp8u * pSrc, int nSrcStep,
                                        Mpp8u * pDst, int nDstStep, 
                                  MppiSize oSizeROI, 
                                  const Mpp8u nThreshold); 

/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_8u_C1IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp8u nThreshold, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_8u_C1IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                                   MppiSize oSizeROI, 
                                   const Mpp8u nThreshold); 

/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16u_C1R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                             Mpp16u * pDst, int nDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp16u nThreshold, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16u_C1R(const Mpp16u * pSrc, int nSrcStep,
                                         Mpp16u * pDst, int nDstStep, 
                                   MppiSize oSizeROI, 
                                   const Mpp16u nThreshold); 

/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16u_C1IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp16u nThreshold, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16u_C1IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp16u nThreshold); 

/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16s_C1R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                             Mpp16s * pDst, int nDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp16s nThreshold, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16s_C1R(const Mpp16s * pSrc, int nSrcStep,
                                         Mpp16s * pDst, int nDstStep, 
                                   MppiSize oSizeROI, 
                                   const Mpp16s nThreshold); 

/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16s_C1IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp16s nThreshold, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16s_C1IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp16s nThreshold); 

/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_32f_C1R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                             Mpp32f * pDst, int nDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp32f nThreshold, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_32f_C1R(const Mpp32f * pSrc, int nSrcStep,
                                         Mpp32f * pDst, int nDstStep, 
                                   MppiSize oSizeROI, 
                                   const Mpp32f nThreshold); 

/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_32f_C1IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp32f nThreshold, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_32f_C1IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp32f nThreshold); 

/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                            Mpp8u * pDst, int nDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp8u rThresholds[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_8u_C3R(const Mpp8u * pSrc, int nSrcStep,
                                        Mpp8u * pDst, int nDstStep, 
                                  MppiSize oSizeROI, 
                                  const Mpp8u rThresholds[3]); 

/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_8u_C3IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp8u rThresholds[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_8u_C3IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                                   MppiSize oSizeROI, 
                                   const Mpp8u rThresholds[3]); 

/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16u_C3R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                             Mpp16u * pDst, int nDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp16u rThresholds[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16u_C3R(const Mpp16u * pSrc, int nSrcStep,
                                         Mpp16u * pDst, int nDstStep, 
                                   MppiSize oSizeROI, 
                                   const Mpp16u rThresholds[3]); 

/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16u_C3IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp16u rThresholds[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16u_C3IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp16u rThresholds[3]); 

/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16s_C3R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                             Mpp16s * pDst, int nDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp16s rThresholds[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16s_C3R(const Mpp16s * pSrc, int nSrcStep,
                                         Mpp16s * pDst, int nDstStep, 
                                   MppiSize oSizeROI, 
                                   const Mpp16s rThresholds[3]); 

/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16s_C3IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp16s rThresholds[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16s_C3IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp16s rThresholds[3]); 

/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_32f_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                             Mpp32f * pDst, int nDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp32f rThresholds[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_32f_C3R(const Mpp32f * pSrc, int nSrcStep,
                                         Mpp32f * pDst, int nDstStep, 
                                   MppiSize oSizeROI, 
                                   const Mpp32f rThresholds[3]); 

/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_32f_C3IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp32f rThresholds[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_32f_C3IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp32f rThresholds[3]); 

/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                             Mpp8u * pDst, int nDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp8u rThresholds[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_8u_AC4R(const Mpp8u * pSrc, int nSrcStep,
                                         Mpp8u * pDst, int nDstStep, 
                                   MppiSize oSizeROI, 
                                   const Mpp8u rThresholds[3]);

/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_8u_AC4IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp8u rThresholds[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_8u_AC4IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp8u rThresholds[3]);

/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16u_AC4R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                              Mpp16u * pDst, int nDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp16u rThresholds[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16u_AC4R(const Mpp16u * pSrc, int nSrcStep,
                                          Mpp16u * pDst, int nDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp16u rThresholds[3]);

/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16u_AC4IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                                         MppiSize oSizeROI, 
                                         const Mpp16u rThresholds[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16u_AC4IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp16u rThresholds[3]);

/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16s_AC4R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                              Mpp16s * pDst, int nDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp16s rThresholds[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16s_AC4R(const Mpp16s * pSrc, int nSrcStep,
                                          Mpp16s * pDst, int nDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp16s rThresholds[3]);

/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16s_AC4IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                                         MppiSize oSizeROI, 
                                         const Mpp16s rThresholds[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_16s_AC4IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp16s rThresholds[3]);

/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_32f_AC4R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                              Mpp32f * pDst, int nDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp32f rThresholds[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_32f_AC4R(const Mpp32f * pSrc, int nSrcStep,
                                          Mpp32f * pDst, int nDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp32f rThresholds[3]);

/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_32f_AC4IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                                         MppiSize oSizeROI, 
                                         const Mpp32f rThresholds[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanParameters.
 *
 */
MppStatus mppiThreshold_GT_32f_AC4IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp32f rThresholds[3]);

/** @} image_threshold_greater_than_operations */

/** 
 * \section image_threshold_less_than_operations Threshold Less Than Operations
 * @defgroup image_threshold_less_than_operations Threshold Less Than Operations
 * Threshold less than image pixels.
 *
 * \subsection CommonThresholdLessThanParameters Common parameters for mppiThreshold_LT non-inplace and inplace functions:
 *
 * \param pSrcDst \ref in_place_image_pointer for inplace functions.
 * \param nSrcDstStep \ref in_place_image_line_step for inplace functions.
 * \param pSrc \ref source_image_pointer for non-inplace functions.
 * \param nSrcStep \ref source_image_line_step for non-inplace functions.
 * \param pDst \ref destination_image_pointer for non-inplace functions.
 * \param nDstStep \ref destination_image_line_step for non-inplace functions.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_8u_C1R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                            Mpp8u * pDst, int nDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp8u nThreshold, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_8u_C1R(const Mpp8u * pSrc, int nSrcStep,
                                        Mpp8u * pDst, int nDstStep, 
                                  MppiSize oSizeROI, 
                                  const Mpp8u nThreshold); 

/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_8u_C1IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp8u nThreshold, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_8u_C1IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                                   MppiSize oSizeROI, 
                                   const Mpp8u nThreshold); 

/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16u_C1R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                             Mpp16u * pDst, int nDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp16u nThreshold, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16u_C1R(const Mpp16u * pSrc, int nSrcStep,
                                         Mpp16u * pDst, int nDstStep, 
                                   MppiSize oSizeROI, 
                                   const Mpp16u nThreshold); 

/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16u_C1IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp16u nThreshold, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16u_C1IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp16u nThreshold); 

/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16s_C1R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                             Mpp16s * pDst, int nDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp16s nThreshold, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16s_C1R(const Mpp16s * pSrc, int nSrcStep,
                                         Mpp16s * pDst, int nDstStep, 
                                   MppiSize oSizeROI, 
                                   const Mpp16s nThreshold); 

/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16s_C1IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp16s nThreshold, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16s_C1IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp16s nThreshold); 

/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_32f_C1R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                             Mpp32f * pDst, int nDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp32f nThreshold, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_32f_C1R(const Mpp32f * pSrc, int nSrcStep,
                                         Mpp32f * pDst, int nDstStep, 
                                   MppiSize oSizeROI, 
                                   const Mpp32f nThreshold); 

/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_32f_C1IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp32f nThreshold, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_32f_C1IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp32f nThreshold); 

/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                            Mpp8u * pDst, int nDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp8u rThresholds[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_8u_C3R(const Mpp8u * pSrc, int nSrcStep,
                                        Mpp8u * pDst, int nDstStep, 
                                  MppiSize oSizeROI, 
                                  const Mpp8u rThresholds[3]); 

/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_8u_C3IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp8u rThresholds[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_8u_C3IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                                   MppiSize oSizeROI, 
                                   const Mpp8u rThresholds[3]); 

/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16u_C3R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                             Mpp16u * pDst, int nDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp16u rThresholds[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16u_C3R(const Mpp16u * pSrc, int nSrcStep,
                                         Mpp16u * pDst, int nDstStep, 
                                   MppiSize oSizeROI, 
                                   const Mpp16u rThresholds[3]); 

/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16u_C3IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp16u rThresholds[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16u_C3IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp16u rThresholds[3]); 

/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16s_C3R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                             Mpp16s * pDst, int nDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp16s rThresholds[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16s_C3R(const Mpp16s * pSrc, int nSrcStep,
                                         Mpp16s * pDst, int nDstStep, 
                                   MppiSize oSizeROI, 
                                   const Mpp16s rThresholds[3]); 

/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16s_C3IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp16s rThresholds[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16s_C3IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp16s rThresholds[3]); 

/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_32f_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                             Mpp32f * pDst, int nDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp32f rThresholds[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_32f_C3R(const Mpp32f * pSrc, int nSrcStep,
                                         Mpp32f * pDst, int nDstStep, 
                                   MppiSize oSizeROI, 
                                   const Mpp32f rThresholds[3]); 

/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_32f_C3IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp32f rThresholds[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_32f_C3IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp32f rThresholds[3]); 

/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                             Mpp8u * pDst, int nDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp8u rThresholds[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_8u_AC4R(const Mpp8u * pSrc, int nSrcStep,
                                         Mpp8u * pDst, int nDstStep, 
                                   MppiSize oSizeROI, 
                                   const Mpp8u rThresholds[3]);

/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_8u_AC4IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp8u rThresholds[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_8u_AC4IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp8u rThresholds[3]);

/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16u_AC4R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                              Mpp16u * pDst, int nDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp16u rThresholds[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16u_AC4R(const Mpp16u * pSrc, int nSrcStep,
                                          Mpp16u * pDst, int nDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp16u rThresholds[3]);

/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16u_AC4IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                                         MppiSize oSizeROI, 
                                         const Mpp16u rThresholds[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16u_AC4IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp16u rThresholds[3]);

/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16s_AC4R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                              Mpp16s * pDst, int nDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp16s rThresholds[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16s_AC4R(const Mpp16s * pSrc, int nSrcStep,
                                          Mpp16s * pDst, int nDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp16s rThresholds[3]);

/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16s_AC4IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                                         MppiSize oSizeROI, 
                                         const Mpp16s rThresholds[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_16s_AC4IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp16s rThresholds[3]);

/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_32f_AC4R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                              Mpp32f * pDst, int nDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp32f rThresholds[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_32f_AC4R(const Mpp32f * pSrc, int nSrcStep,
                                          Mpp32f * pDst, int nDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp32f rThresholds[3]);

/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_32f_AC4IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                                         MppiSize oSizeROI, 
                                         const Mpp32f rThresholds[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanParameters.
 *
 */
MppStatus mppiThreshold_LT_32f_AC4IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp32f rThresholds[3]);

/** @} image_threshold_less_than_operations */

/** 
 * \section image_threshold_value_operations Threshold Value Operations
 * @defgroup image_threshold_value_operations Threshold Value Operations
 * Replace thresholded image pixels with a value.
 *
 * \subsection CommonThresholdValueParameters Common parameters for mppiThreshold_Val non-inplace and inplace functions:
 *
 * \param pSrcDst \ref in_place_image_pointer for inplace functions.
 * \param nSrcDstStep \ref in_place_image_line_step for inplace functions.
 * \param pSrc \ref source_image_pointer for non-inplace functions.
 * \param nSrcStep \ref source_image_line_step for non-inplace functions.
 * \param pDst \ref destination_image_pointer for non-inplace functions.
 * \param nDstStep \ref destination_image_line_step for non-inplace functions.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param nValue The threshold replacement value.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: MPP_CMP_LESS and MPP_CMP_GREATER.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, or MPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_8u_C1R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                             Mpp8u * pDst, int nDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp8u nThreshold, const Mpp8u nValue, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_8u_C1R(const Mpp8u * pSrc, int nSrcStep,
                                         Mpp8u * pDst, int nDstStep, 
                                   MppiSize oSizeROI, 
                                   const Mpp8u nThreshold, const Mpp8u nValue, MppCmpOp eComparisonOperation); 

/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_8u_C1IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp8u nThreshold, const Mpp8u nValue, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_8u_C1IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp8u nThreshold, const Mpp8u nValue, MppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_16u_C1R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                              Mpp16u * pDst, int nDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp16u nThreshold, const Mpp16u nValue, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_16u_C1R(const Mpp16u * pSrc, int nSrcStep,
                                          Mpp16u * pDst, int nDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp16u nThreshold, const Mpp16u nValue, MppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_16u_C1IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                                         MppiSize oSizeROI, 
                                         const Mpp16u nThreshold, const Mpp16u nValue, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_16u_C1IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp16u nThreshold, const Mpp16u nValue, MppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_16s_C1R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                              Mpp16s * pDst, int nDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp16s nThreshold, const Mpp16s nValue, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_16s_C1R(const Mpp16s * pSrc, int nSrcStep,
                                          Mpp16s * pDst, int nDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp16s nThreshold, const Mpp16s nValue, MppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_16s_C1IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                                         MppiSize oSizeROI, 
                                         const Mpp16s nThreshold, const Mpp16s nValue, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */                                 
MppStatus mppiThreshold_Val_16s_C1IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp16s nThreshold, const Mpp16s nValue, MppCmpOp eComparisonOperation); 
                                     
/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_32f_C1R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                              Mpp32f * pDst, int nDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp32f nThreshold, const Mpp32f nValue, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_32f_C1R(const Mpp32f * pSrc, int nSrcStep,
                                          Mpp32f * pDst, int nDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp32f nThreshold, const Mpp32f nValue, MppCmpOp eComparisonOperation); 

/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_32f_C1IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                                         MppiSize oSizeROI, 
                                         const Mpp32f nThreshold, const Mpp32f nValue, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_32f_C1IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp32f nThreshold, const Mpp32f nValue, MppCmpOp eComparisonOperation); 

/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                             Mpp8u * pDst, int nDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp8u rThresholds[3], const Mpp8u rValues[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_8u_C3R(const Mpp8u * pSrc, int nSrcStep,
                                         Mpp8u * pDst, int nDstStep, 
                                   MppiSize oSizeROI, 
                                   const Mpp8u rThresholds[3], const Mpp8u rValues[3], MppCmpOp eComparisonOperation); 

/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_8u_C3IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp8u rThresholds[3], const Mpp8u rValues[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_8u_C3IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp8u rThresholds[3], const Mpp8u rValues[3], MppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_16u_C3R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                              Mpp16u * pDst, int nDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp16u rThresholds[3], const Mpp16u rValues[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_16u_C3R(const Mpp16u * pSrc, int nSrcStep,
                                          Mpp16u * pDst, int nDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp16u rThresholds[3], const Mpp16u rValues[3], MppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_16u_C3IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                                         MppiSize oSizeROI, 
                                         const Mpp16u rThresholds[3], const Mpp16u rValues[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_16u_C3IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp16u rThresholds[3], const Mpp16u rValues[3], MppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_16s_C3R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                              Mpp16s * pDst, int nDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp16s rThresholds[3], const Mpp16s rValues[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_16s_C3R(const Mpp16s * pSrc, int nSrcStep,
                                          Mpp16s * pDst, int nDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp16s rThresholds[3], const Mpp16s rValues[3], MppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_16s_C3IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                                         MppiSize oSizeROI, 
                                         const Mpp16s rThresholds[3], const Mpp16s rValues[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_16s_C3IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp16s rThresholds[3], const Mpp16s rValues[3], MppCmpOp eComparisonOperation); 

/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_32f_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                              Mpp32f * pDst, int nDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp32f rThresholds[3], const Mpp32f rValues[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_32f_C3R(const Mpp32f * pSrc, int nSrcStep,
                                          Mpp32f * pDst, int nDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp32f rThresholds[3], const Mpp32f rValues[3], MppCmpOp eComparisonOperation); 

/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_32f_C3IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                                         MppiSize oSizeROI, 
                                         const Mpp32f rThresholds[3], const Mpp32f rValues[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_32f_C3IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp32f rThresholds[3], const Mpp32f rValues[3], MppCmpOp eComparisonOperation); 


/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                              Mpp8u * pDst, int nDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp8u rThresholds[3], const Mpp8u rValues[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_8u_AC4R(const Mpp8u * pSrc, int nSrcStep,
                                          Mpp8u * pDst, int nDstStep, 
                                    MppiSize oSizeROI, 
                                    const Mpp8u rThresholds[3], const Mpp8u rValues[3], MppCmpOp eComparisonOperation);

/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_8u_AC4IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                                         MppiSize oSizeROI, 
                                         const Mpp8u rThresholds[3], const Mpp8u rValues[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_8u_AC4IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp8u rThresholds[3], const Mpp8u rValues[3], MppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_16u_AC4R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                               Mpp16u * pDst, int nDstStep, 
                                         MppiSize oSizeROI, 
                                         const Mpp16u rThresholds[3], const Mpp16u rValues[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_16u_AC4R(const Mpp16u * pSrc, int nSrcStep,
                                           Mpp16u * pDst, int nDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp16u rThresholds[3], const Mpp16u rValues[3], MppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_16u_AC4IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                                          MppiSize oSizeROI, 
                                          const Mpp16u rThresholds[3], const Mpp16u rValues[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_16u_AC4IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp16u rThresholds[3], const Mpp16u rValues[3], MppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_16s_AC4R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                               Mpp16s * pDst, int nDstStep, 
                                         MppiSize oSizeROI, 
                                         const Mpp16s rThresholds[3], const Mpp16s rValues[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_16s_AC4R(const Mpp16s * pSrc, int nSrcStep,
                                           Mpp16s * pDst, int nDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp16s rThresholds[3], const Mpp16s rValues[3], MppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_16s_AC4IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                                          MppiSize oSizeROI, 
                                          const Mpp16s rThresholds[3], const Mpp16s rValues[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_16s_AC4IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp16s rThresholds[3], const Mpp16s rValues[3], MppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_32f_AC4R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                               Mpp32f * pDst, int nDstStep, 
                                         MppiSize oSizeROI, 
                                         const Mpp32f rThresholds[3], const Mpp32f rValues[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_32f_AC4R(const Mpp32f * pSrc, int nSrcStep,
                                           Mpp32f * pDst, int nDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp32f rThresholds[3], const Mpp32f rValues[3], MppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_32f_AC4IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                                          MppiSize oSizeROI, 
                                          const Mpp32f rThresholds[3], const Mpp32f rValues[3], MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdValueParameters.
 *
 */
MppStatus mppiThreshold_Val_32f_AC4IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp32f rThresholds[3], const Mpp32f rValues[3], MppCmpOp eComparisonOperation);

/** @} image_threshold_value_operations */

/** 
 * \section image_threshold_greater_than_value_operations Threshold Greater Than Value Operations
 * @defgroup image_threshold_greater_than_value_operations Threshold Greater Than Value Operations
 * Replace image pixels greater than threshold with a value.
 *
 * \subsection CommonThresholdGreaterThanValueParameters Common parameters for mppiThreshold_GTVal non-inplace and inplace functions:
 *
 * \param pSrcDst \ref in_place_image_pointer for inplace functions.
 * \param nSrcDstStep \ref in_place_image_line_step for inplace functions.
 * \param pSrc \ref source_image_pointer for non-inplace functions.
 * \param nSrcStep \ref source_image_line_step for non-inplace functions.
 * \param pDst \ref destination_image_pointer for non-inplace functions.
 * \param nDstStep \ref destination_image_line_step for non-inplace functions.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param nValue The threshold replacement value.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 *
 * @{
 *
 */



/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_8u_C1R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                               Mpp8u * pDst, int nDstStep, 
                                         MppiSize oSizeROI, 
                                         const Mpp8u nThreshold, const Mpp8u nValue, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_8u_C1R(const Mpp8u * pSrc, int nSrcStep,
                                           Mpp8u * pDst, int nDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp8u nThreshold, const Mpp8u nValue); 

/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_8u_C1IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                                          MppiSize oSizeROI, 
                                          const Mpp8u nThreshold, const Mpp8u nValue, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_8u_C1IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp8u nThreshold, const Mpp8u nValue); 

/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_16u_C1R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                                Mpp16u * pDst, int nDstStep, 
                                          MppiSize oSizeROI, 
                                          const Mpp16u nThreshold, const Mpp16u nValue, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_16u_C1R(const Mpp16u * pSrc, int nSrcStep,
                                            Mpp16u * pDst, int nDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp16u nThreshold, const Mpp16u nValue); 

/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_16u_C1IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp16u nThreshold, const Mpp16u nValue, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_16u_C1IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp16u nThreshold, const Mpp16u nValue); 

/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_16s_C1R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                                Mpp16s * pDst, int nDstStep, 
                                          MppiSize oSizeROI, 
                                          const Mpp16s nThreshold, const Mpp16s nValue, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_16s_C1R(const Mpp16s * pSrc, int nSrcStep,
                                            Mpp16s * pDst, int nDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp16s nThreshold, const Mpp16s nValue); 

/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_16s_C1IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp16s nThreshold, const Mpp16s nValue, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_16s_C1IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp16s nThreshold, const Mpp16s nValue); 

/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_32f_C1R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                                Mpp32f * pDst, int nDstStep, 
                                          MppiSize oSizeROI, 
                                          const Mpp32f nThreshold, const Mpp32f nValue, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_32f_C1R(const Mpp32f * pSrc, int nSrcStep,
                                            Mpp32f * pDst, int nDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp32f nThreshold, const Mpp32f nValue); 

/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_32f_C1IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp32f nThreshold, const Mpp32f nValue, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_32f_C1IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp32f nThreshold, const Mpp32f nValue); 

/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                               Mpp8u * pDst, int nDstStep, 
                                         MppiSize oSizeROI, 
                                         const Mpp8u rThresholds[3], const Mpp8u rValues[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_8u_C3R(const Mpp8u * pSrc, int nSrcStep,
                                           Mpp8u * pDst, int nDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp8u rThresholds[3], const Mpp8u rValues[3]); 

/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_8u_C3IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                                          MppiSize oSizeROI, 
                                          const Mpp8u rThresholds[3], const Mpp8u rValues[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_8u_C3IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp8u rThresholds[3], const Mpp8u rValues[3]); 

/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_16u_C3R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                                Mpp16u * pDst, int nDstStep, 
                                          MppiSize oSizeROI, 
                                          const Mpp16u rThresholds[3], const Mpp16u rValues[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_16u_C3R(const Mpp16u * pSrc, int nSrcStep,
                                            Mpp16u * pDst, int nDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp16u rThresholds[3], const Mpp16u rValues[3]); 

/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_16u_C3IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp16u rThresholds[3], const Mpp16u rValues[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */                                      
MppStatus mppiThreshold_GTVal_16u_C3IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp16u rThresholds[3], const Mpp16u rValues[3]); 
                                       
/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_16s_C3R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                                Mpp16s * pDst, int nDstStep, 
                                          MppiSize oSizeROI, 
                                          const Mpp16s rThresholds[3], const Mpp16s rValues[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_16s_C3R(const Mpp16s * pSrc, int nSrcStep,
                                            Mpp16s * pDst, int nDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp16s rThresholds[3], const Mpp16s rValues[3]); 

/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_16s_C3IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp16s rThresholds[3], const Mpp16s rValues[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_16s_C3IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp16s rThresholds[3], const Mpp16s rValues[3]); 

/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_32f_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                                Mpp32f * pDst, int nDstStep, 
                                          MppiSize oSizeROI, 
                                          const Mpp32f rThresholds[3], const Mpp32f rValues[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_32f_C3R(const Mpp32f * pSrc, int nSrcStep,
                                            Mpp32f * pDst, int nDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp32f rThresholds[3], const Mpp32f rValues[3]); 

/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_32f_C3IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp32f rThresholds[3], const Mpp32f rValues[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_32f_C3IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp32f rThresholds[3], const Mpp32f rValues[3]); 

/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                                Mpp8u * pDst, int nDstStep, 
                                          MppiSize oSizeROI, 
                                          const Mpp8u rThresholds[3], const Mpp8u rValues[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_8u_AC4R(const Mpp8u * pSrc, int nSrcStep,
                                            Mpp8u * pDst, int nDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp8u rThresholds[3], const Mpp8u rValues[3]);

/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_8u_AC4IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp8u rThresholds[3], const Mpp8u rValues[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_8u_AC4IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp8u rThresholds[3], const Mpp8u rValues[3]);

/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_16u_AC4R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                                 Mpp16u * pDst, int nDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp16u rThresholds[3], const Mpp16u rValues[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_16u_AC4R(const Mpp16u * pSrc, int nSrcStep,
                                             Mpp16u * pDst, int nDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp16u rThresholds[3], const Mpp16u rValues[3]);

/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_16u_AC4IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                                            MppiSize oSizeROI, 
                                            const Mpp16u rThresholds[3], const Mpp16u rValues[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_16u_AC4IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp16u rThresholds[3], const Mpp16u rValues[3]);

/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_16s_AC4R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                                 Mpp16s * pDst, int nDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp16s rThresholds[3], const Mpp16s rValues[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_16s_AC4R(const Mpp16s * pSrc, int nSrcStep,
                                             Mpp16s * pDst, int nDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp16s rThresholds[3], const Mpp16s rValues[3]);

/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_16s_AC4IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                                            MppiSize oSizeROI, 
                                            const Mpp16s rThresholds[3], const Mpp16s rValues[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_16s_AC4IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp16s rThresholds[3], const Mpp16s rValues[3]);

/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_32f_AC4R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                                 Mpp32f * pDst, int nDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp32f rThresholds[3], const Mpp32f rValues[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_32f_AC4R(const Mpp32f * pSrc, int nSrcStep,
                                             Mpp32f * pDst, int nDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp32f rThresholds[3], const Mpp32f rValues[3]);

/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_32f_AC4IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                                            MppiSize oSizeROI, 
                                            const Mpp32f rThresholds[3], const Mpp32f rValues[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_GTVal_32f_AC4IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp32f rThresholds[3], const Mpp32f rValues[3]);

/** @} image_threshold_greater_than_value_operations */


/** 
 * \section image_fused_absdiff_threshold_greater_than_value_operations Fused AbsDiff Threshold Greater Than Value Operations
 * @defgroup image_fused_absdiff_threshold_greater_than_value_operations Fused AbsDiff Threshold Greater Than Value Operations
 * Replace image pixels greater than threshold with a value.
 *
 * Supported data types include MPP_8U, MPP_16U, MPP_16S, MPP_32F. 
 * Supported channel counts include MPP_CH_1, MPP_CH_3, MPP_CH_A4. 
 * \param eSrcDstType image data type.
 * \param eSrcDstChannels image channels. 
 * \param pSrcDst \ref in_place_image_pointer for inplace functions.
 * \param nSrcDstStep \ref in_place_image_line_step for inplace functions.
 * \param pSrc1 \ref source_image_pointer for non-inplace functions.
 * \param nSrc1Step \ref source_image_line_step for non-inplace functions.
 * \param pSrc2 \ref source_image_pointer to second source image for non-inplace functions.
 * \param nSrc2Step \ref source_image_line_step of second source image for non-inplace functions.
 * \param pDst \ref destination_image_pointer for non-inplace functions.
 * \param nDstStep \ref destination_image_line_step for non-inplace functions.
 * \param oSizeROI \ref roi_specification.
 * \param pThreshold The threshold value.
 * \param pValue The threshold replacement value.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 *
 * @{
 *
 */



/** 
 * Image fused absdiff and greater than threshold value.
 * If for a comparison operations absdiff of sourcePixels is greater than pThreshold is true, the output pixel is set
 * to pValue, otherwise it is set to absdiff of sourcePixels.
 * 
 */
MppStatus 
mppiFusedAbsDiff_Threshold_GTVal_Ctx(MppDataType eSrcDstType, MppiChannels eSrcDstChannels,
                                     const void * pSrc1, int nSrc1Step,
                                     const void * pSrc2, int nSrc2Step,
                                     void * pDst, int nDstStep, MppiSize oSizeROI, 
                                     const void * pThreshold, const void * pvalue,
                                     MppStreamContext mppStreamCtx); 

/** 
 * In place fused absdiff image greater than threshold value.
 * If for a comparison operations absdiff of sourcePixels is greater than pThreshold is true, the output pixel is set
 * to pValue, otherwise it is set to absdiff of sourcePixels.
 * 
 */
MppStatus 
mppiFusedAbsDiff_Threshold_GTVal_I_Ctx(MppDataType eSrcDstType, MppiChannels eSrcDstChannels,
                                       void * pSrcDst, int nSrcDstStep,  
                                       const void * pSrc2, int nSrc2Step,
                                       MppiSize oSizeROI,
                                       const void * pThreshold, const void * pvalue,
                                       MppStreamContext mppStreamCtx); 

/** @} image_fused_absdiff_threshold_greater_than_value_operations */

/** 
 * \section image_threshold_less_than_value_operations Threshold Less Than Value Operations
 * @defgroup image_threshold_less_than_value_operations Threshold Less Than Value Operations
 * Replace image pixels less than threshold with a value.
 *
 * \subsection CommonThresholdLessThanValueParameters Common parameters for mppiThreshold_LTVal non-inplace and inplace functions:
 *
 * \param pSrcDst \ref in_place_image_pointer for inplace functions.
 * \param nSrcDstStep \ref in_place_image_line_step for inplace functions.
 * \param pSrc \ref source_image_pointer for non-inplace functions.
 * \param nSrcStep \ref source_image_line_step for non-inplace functions.
 * \param pDst \ref destination_image_pointer for non-inplace functions.
 * \param nDstStep \ref destination_image_line_step for non-inplace functions.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param nValue The threshold replacement value.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_8u_C1R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                               Mpp8u * pDst, int nDstStep, 
                                         MppiSize oSizeROI, 
                                         const Mpp8u nThreshold, const Mpp8u nValue, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_8u_C1R(const Mpp8u * pSrc, int nSrcStep,
                                           Mpp8u * pDst, int nDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp8u nThreshold, const Mpp8u nValue); 

/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_8u_C1IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                                          MppiSize oSizeROI, 
                                          const Mpp8u nThreshold, const Mpp8u nValue, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_8u_C1IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp8u nThreshold, const Mpp8u nValue); 

/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16u_C1R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                                Mpp16u * pDst, int nDstStep, 
                                          MppiSize oSizeROI, 
                                          const Mpp16u nThreshold, const Mpp16u nValue, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16u_C1R(const Mpp16u * pSrc, int nSrcStep,
                                            Mpp16u * pDst, int nDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp16u nThreshold, const Mpp16u nValue); 

/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16u_C1IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp16u nThreshold, const Mpp16u nValue, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16u_C1IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp16u nThreshold, const Mpp16u nValue); 

/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16s_C1R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                                Mpp16s * pDst, int nDstStep, 
                                          MppiSize oSizeROI, 
                                          const Mpp16s nThreshold, const Mpp16s nValue, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16s_C1R(const Mpp16s * pSrc, int nSrcStep,
                                            Mpp16s * pDst, int nDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp16s nThreshold, const Mpp16s nValue); 

/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16s_C1IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp16s nThreshold, const Mpp16s nValue, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16s_C1IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp16s nThreshold, const Mpp16s nValue); 

/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_32f_C1R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                                Mpp32f * pDst, int nDstStep, 
                                          MppiSize oSizeROI, 
                                          const Mpp32f nThreshold, const Mpp32f nValue, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_32f_C1R(const Mpp32f * pSrc, int nSrcStep,
                                            Mpp32f * pDst, int nDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp32f nThreshold, const Mpp32f nValue); 

/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_32f_C1IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp32f nThreshold, const Mpp32f nValue, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_32f_C1IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp32f nThreshold, const Mpp32f nValue); 

/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                               Mpp8u * pDst, int nDstStep, 
                                         MppiSize oSizeROI, 
                                         const Mpp8u rThresholds[3], const Mpp8u rValues[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_8u_C3R(const Mpp8u * pSrc, int nSrcStep,
                                           Mpp8u * pDst, int nDstStep, 
                                     MppiSize oSizeROI, 
                                     const Mpp8u rThresholds[3], const Mpp8u rValues[3]); 

/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_8u_C3IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                                          MppiSize oSizeROI, 
                                          const Mpp8u rThresholds[3], const Mpp8u rValues[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_8u_C3IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp8u rThresholds[3], const Mpp8u rValues[3]); 

/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16u_C3R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                                Mpp16u * pDst, int nDstStep, 
                                          MppiSize oSizeROI, 
                                          const Mpp16u rThresholds[3], const Mpp16u rValues[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16u_C3R(const Mpp16u * pSrc, int nSrcStep,
                                            Mpp16u * pDst, int nDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp16u rThresholds[3], const Mpp16u rValues[3]); 

/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16u_C3IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp16u rThresholds[3], const Mpp16u rValues[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16u_C3IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp16u rThresholds[3], const Mpp16u rValues[3]); 

/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16s_C3R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                                Mpp16s * pDst, int nDstStep, 
                                          MppiSize oSizeROI, 
                                          const Mpp16s rThresholds[3], const Mpp16s rValues[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16s_C3R(const Mpp16s * pSrc, int nSrcStep,
                                            Mpp16s * pDst, int nDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp16s rThresholds[3], const Mpp16s rValues[3]); 

/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16s_C3IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp16s rThresholds[3], const Mpp16s rValues[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16s_C3IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp16s rThresholds[3], const Mpp16s rValues[3]); 

/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_32f_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                                Mpp32f * pDst, int nDstStep, 
                                          MppiSize oSizeROI, 
                                          const Mpp32f rThresholds[3], const Mpp32f rValues[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_32f_C3R(const Mpp32f * pSrc, int nSrcStep,
                                            Mpp32f * pDst, int nDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp32f rThresholds[3], const Mpp32f rValues[3]); 

/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_32f_C3IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp32f rThresholds[3], const Mpp32f rValues[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_32f_C3IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp32f rThresholds[3], const Mpp32f rValues[3]); 

/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                                Mpp8u * pDst, int nDstStep, 
                                          MppiSize oSizeROI, 
                                          const Mpp8u rThresholds[3], const Mpp8u rValues[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_8u_AC4R(const Mpp8u * pSrc, int nSrcStep,
                                            Mpp8u * pDst, int nDstStep, 
                                      MppiSize oSizeROI, 
                                      const Mpp8u rThresholds[3], const Mpp8u rValues[3]);

/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_8u_AC4IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp8u rThresholds[3], const Mpp8u rValues[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_8u_AC4IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp8u rThresholds[3], const Mpp8u rValues[3]);

/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16u_AC4R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                                 Mpp16u * pDst, int nDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp16u rThresholds[3], const Mpp16u rValues[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16u_AC4R(const Mpp16u * pSrc, int nSrcStep,
                                             Mpp16u * pDst, int nDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp16u rThresholds[3], const Mpp16u rValues[3]);

/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16u_AC4IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                                            MppiSize oSizeROI, 
                                            const Mpp16u rThresholds[3], const Mpp16u rValues[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16u_AC4IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp16u rThresholds[3], const Mpp16u rValues[3]);

/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16s_AC4R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                                 Mpp16s * pDst, int nDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp16s rThresholds[3], const Mpp16s rValues[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16s_AC4R(const Mpp16s * pSrc, int nSrcStep,
                                             Mpp16s * pDst, int nDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp16s rThresholds[3], const Mpp16s rValues[3]);

/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16s_AC4IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                                            MppiSize oSizeROI, 
                                            const Mpp16s rThresholds[3], const Mpp16s rValues[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_16s_AC4IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp16s rThresholds[3], const Mpp16s rValues[3]);

/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_32f_AC4R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                                 Mpp32f * pDst, int nDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp32f rThresholds[3], const Mpp32f rValues[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_32f_AC4R(const Mpp32f * pSrc, int nSrcStep,
                                             Mpp32f * pDst, int nDstStep, 
                                       MppiSize oSizeROI, 
                                       const Mpp32f rThresholds[3], const Mpp32f rValues[3]);

/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_32f_AC4IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                                            MppiSize oSizeROI, 
                                            const Mpp32f rThresholds[3], const Mpp32f rValues[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTVal_32f_AC4IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                                        MppiSize oSizeROI, 
                                        const Mpp32f rThresholds[3], const Mpp32f rValues[3]);

/** @} image_threshold_less_than_value_operations */

/** 
 * \section image_threshold_less_than_value_greater_than_value_operations Threshold Less Than Value Or Greater Than Value Operations
 * @defgroup image_threshold_less_than_value_greater_than_value_operations Threshold Less Than Value Or Greater Than Value Operations
 * Replace image pixels less than thresholdLT or greater than thresholdGT with with valueLT or valueGT respectively.
 *
 * \subsection CommonThresholdLessThanValueGreaterThanValueParameters Common parameters for mppiThreshold_LTValGTVal non-inplace and inplace functions:
 *
 * \param pSrcDst \ref in_place_image_pointer for inplace functions.
 * \param nSrcDstStep \ref in_place_image_line_step for inplace functions.
 * \param pSrc \ref source_image_pointer for non-inplace functions.
 * \param nSrcStep \ref source_image_line_step for non-inplace functions.
 * \param pDst \ref destination_image_pointer for non-inplace functions.
 * \param nDstStep \ref destination_image_line_step for non-inplace functions.
 * \param oSizeROI \ref roi_specification.
 * \param nThresholdLT The thresholdLT value.
 * \param nValueLT The thresholdLT replacement value.
 * \param nThresholdGT The thresholdGT value.
 * \param nValueGT The thresholdGT replacement value.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 *
 * @{
 *
 */


/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_8u_C1R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                                    Mpp8u * pDst, int nDstStep, 
                                              MppiSize oSizeROI, 
                                              const Mpp8u nThresholdLT, const Mpp8u nValueLT, const Mpp8u nThresholdGT, const Mpp8u nValueGT, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_8u_C1R(const Mpp8u * pSrc, int nSrcStep,
                                                Mpp8u * pDst, int nDstStep, 
                                          MppiSize oSizeROI, 
                                          const Mpp8u nThresholdLT, const Mpp8u nValueLT, const Mpp8u nThresholdGT, const Mpp8u nValueGT); 

/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_8u_C1IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                                               MppiSize oSizeROI, 
                                               const Mpp8u nThresholdLT, const Mpp8u nValueLT, const Mpp8u nThresholdGT, const Mpp8u nValueGT, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_8u_C1IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp8u nThresholdLT, const Mpp8u nValueLT, const Mpp8u nThresholdGT, const Mpp8u nValueGT); 

/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16u_C1R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                                     Mpp16u * pDst, int nDstStep, 
                                               MppiSize oSizeROI, 
                                               const Mpp16u nThresholdLT, const Mpp16u nValueLT, const Mpp16u nThresholdGT, const Mpp16u nValueGT, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16u_C1R(const Mpp16u * pSrc, int nSrcStep,
                                                 Mpp16u * pDst, int nDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp16u nThresholdLT, const Mpp16u nValueLT, const Mpp16u nThresholdGT, const Mpp16u nValueGT); 

/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16u_C1IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                                                MppiSize oSizeROI, 
                                                const Mpp16u nThresholdLT, const Mpp16u nValueLT, const Mpp16u nThresholdGT, const Mpp16u nValueGT, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16u_C1IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                                            MppiSize oSizeROI, 
                                            const Mpp16u nThresholdLT, const Mpp16u nValueLT, const Mpp16u nThresholdGT, const Mpp16u nValueGT); 

/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16s_C1R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                                     Mpp16s * pDst, int nDstStep, 
                                               MppiSize oSizeROI, 
                                               const Mpp16s nThresholdLT, const Mpp16s nValueLT, const Mpp16s nThresholdGT, const Mpp16s nValueGT, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16s_C1R(const Mpp16s * pSrc, int nSrcStep,
                                                 Mpp16s * pDst, int nDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp16s nThresholdLT, const Mpp16s nValueLT, const Mpp16s nThresholdGT, const Mpp16s nValueGT); 

/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16s_C1IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                                                MppiSize oSizeROI, 
                                                const Mpp16s nThresholdLT, const Mpp16s nValueLT, const Mpp16s nThresholdGT, const Mpp16s nValueGT, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16s_C1IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                                            MppiSize oSizeROI, 
                                            const Mpp16s nThresholdLT, const Mpp16s nValueLT, const Mpp16s nThresholdGT, const Mpp16s nValueGT); 

/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_32f_C1R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                                     Mpp32f * pDst, int nDstStep, 
                                               MppiSize oSizeROI, 
                                               const Mpp32f nThresholdLT, const Mpp32f nValueLT, const Mpp32f nThresholdGT, const Mpp32f nValueGT, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_32f_C1R(const Mpp32f * pSrc, int nSrcStep,
                                                 Mpp32f * pDst, int nDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp32f nThresholdLT, const Mpp32f nValueLT, const Mpp32f nThresholdGT, const Mpp32f nValueGT); 

/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_32f_C1IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                                                MppiSize oSizeROI, 
                                                const Mpp32f nThresholdLT, const Mpp32f nValueLT, const Mpp32f nThresholdGT, const Mpp32f nValueGT, MppStreamContext mppStreamCtx); 
/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_32f_C1IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                                            MppiSize oSizeROI, 
                                            const Mpp32f nThresholdLT, const Mpp32f nValueLT, const Mpp32f nThresholdGT, const Mpp32f nValueGT); 

/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                                    Mpp8u * pDst, int nDstStep, 
                                              MppiSize oSizeROI, 
                                              const Mpp8u rThresholdsLT[3], const Mpp8u rValuesLT[3], const Mpp8u rThresholdsGT[3], const Mpp8u rValuesGT[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_8u_C3R(const Mpp8u * pSrc, int nSrcStep,
                                                Mpp8u * pDst, int nDstStep, 
                                          MppiSize oSizeROI, 
                                          const Mpp8u rThresholdsLT[3], const Mpp8u rValuesLT[3], const Mpp8u rThresholdsGT[3], const Mpp8u rValuesGT[3]); 

/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_8u_C3IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                                               MppiSize oSizeROI, 
                                               const Mpp8u rThresholdsLT[3], const Mpp8u rValuesLT[3], const Mpp8u rThresholdsGT[3], const Mpp8u rValuesGT[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_8u_C3IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp8u rThresholdsLT[3], const Mpp8u rValuesLT[3], const Mpp8u rThresholdsGT[3], const Mpp8u rValuesGT[3]); 

/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16u_C3R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                                     Mpp16u * pDst, int nDstStep, 
                                               MppiSize oSizeROI, 
                                               const Mpp16u rThresholdsLT[3], const Mpp16u rValuesLT[3], const Mpp16u rThresholdsGT[3], const Mpp16u rValuesGT[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16u_C3R(const Mpp16u * pSrc, int nSrcStep,
                                                 Mpp16u * pDst, int nDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp16u rThresholdsLT[3], const Mpp16u rValuesLT[3], const Mpp16u rThresholdsGT[3], const Mpp16u rValuesGT[3]); 

/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16u_C3IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                                                MppiSize oSizeROI, 
                                                const Mpp16u rThresholdsLT[3], const Mpp16u rValuesLT[3], const Mpp16u rThresholdsGT[3], const Mpp16u rValuesGT[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16u_C3IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                                            MppiSize oSizeROI, 
                                            const Mpp16u rThresholdsLT[3], const Mpp16u rValuesLT[3], const Mpp16u rThresholdsGT[3], const Mpp16u rValuesGT[3]); 

/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16s_C3R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                                     Mpp16s * pDst, int nDstStep, 
                                               MppiSize oSizeROI, 
                                               const Mpp16s rThresholdsLT[3], const Mpp16s rValuesLT[3], const Mpp16s rThresholdsGT[3], const Mpp16s rValuesGT[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16s_C3R(const Mpp16s * pSrc, int nSrcStep,
                                                 Mpp16s * pDst, int nDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp16s rThresholdsLT[3], const Mpp16s rValuesLT[3], const Mpp16s rThresholdsGT[3], const Mpp16s rValuesGT[3]); 

/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16s_C3IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                                                MppiSize oSizeROI, 
                                                const Mpp16s rThresholdsLT[3], const Mpp16s rValuesLT[3], const Mpp16s rThresholdsGT[3], const Mpp16s rValuesGT[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16s_C3IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                                            MppiSize oSizeROI, 
                                            const Mpp16s rThresholdsLT[3], const Mpp16s rValuesLT[3], const Mpp16s rThresholdsGT[3], const Mpp16s rValuesGT[3]); 

/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_32f_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                                     Mpp32f * pDst, int nDstStep, 
                                               MppiSize oSizeROI, 
                                               const Mpp32f rThresholdsLT[3], const Mpp32f rValuesLT[3], const Mpp32f rThresholdsGT[3], const Mpp32f rValuesGT[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_32f_C3R(const Mpp32f * pSrc, int nSrcStep,
                                                 Mpp32f * pDst, int nDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp32f rThresholdsLT[3], const Mpp32f rValuesLT[3], const Mpp32f rThresholdsGT[3], const Mpp32f rValuesGT[3]); 

/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_32f_C3IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                                                MppiSize oSizeROI, 
                                                const Mpp32f rThresholdsLT[3], const Mpp32f rValuesLT[3], const Mpp32f rThresholdsGT[3], const Mpp32f rValuesGT[3], MppStreamContext mppStreamCtx); 
/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_32f_C3IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                                            MppiSize oSizeROI, 
                                            const Mpp32f rThresholdsLT[3], const Mpp32f rValuesLT[3], const Mpp32f rThresholdsGT[3], const Mpp32f rValuesGT[3]); 

/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                                     Mpp8u * pDst, int nDstStep, 
                                               MppiSize oSizeROI, 
                                               const Mpp8u rThresholdsLT[3], const Mpp8u rValuesLT[3], const Mpp8u rThresholdsGT[3], const Mpp8u rValuesGT[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_8u_AC4R(const Mpp8u * pSrc, int nSrcStep,
                                                 Mpp8u * pDst, int nDstStep, 
                                           MppiSize oSizeROI, 
                                           const Mpp8u rThresholdsLT[3], const Mpp8u rValuesLT[3], const Mpp8u rThresholdsGT[3], const Mpp8u rValuesGT[3]);

/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_8u_AC4IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                                                MppiSize oSizeROI, 
                                                const Mpp8u rThresholdsLT[3], const Mpp8u rValuesLT[3], const Mpp8u rThresholdsGT[3], const Mpp8u rValuesGT[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_8u_AC4IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                                            MppiSize oSizeROI, 
                                            const Mpp8u rThresholdsLT[3], const Mpp8u rValuesLT[3], const Mpp8u rThresholdsGT[3], const Mpp8u rValuesGT[3]);

/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16u_AC4R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                                      Mpp16u * pDst, int nDstStep, 
                                                MppiSize oSizeROI, 
                                                const Mpp16u rThresholdsLT[3], const Mpp16u rValuesLT[3], const Mpp16u rThresholdsGT[3], const Mpp16u rValuesGT[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16u_AC4R(const Mpp16u * pSrc, int nSrcStep,
                                                  Mpp16u * pDst, int nDstStep, 
                                            MppiSize oSizeROI, 
                                            const Mpp16u rThresholdsLT[3], const Mpp16u rValuesLT[3], const Mpp16u rThresholdsGT[3], const Mpp16u rValuesGT[3]);

/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16u_AC4IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                                                 MppiSize oSizeROI, 
                                                 const Mpp16u rThresholdsLT[3], const Mpp16u rValuesLT[3], const Mpp16u rThresholdsGT[3], const Mpp16u rValuesGT[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16u_AC4IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                                             MppiSize oSizeROI, 
                                             const Mpp16u rThresholdsLT[3], const Mpp16u rValuesLT[3], const Mpp16u rThresholdsGT[3], const Mpp16u rValuesGT[3]);

/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16s_AC4R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                                      Mpp16s * pDst, int nDstStep, 
                                                MppiSize oSizeROI, 
                                                const Mpp16s rThresholdsLT[3], const Mpp16s rValuesLT[3], const Mpp16s rThresholdsGT[3], const Mpp16s rValuesGT[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16s_AC4R(const Mpp16s * pSrc, int nSrcStep,
                                                  Mpp16s * pDst, int nDstStep, 
                                            MppiSize oSizeROI, 
                                            const Mpp16s rThresholdsLT[3], const Mpp16s rValuesLT[3], const Mpp16s rThresholdsGT[3], const Mpp16s rValuesGT[3]);

/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16s_AC4IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                                                 MppiSize oSizeROI, 
                                                 const Mpp16s rThresholdsLT[3], const Mpp16s rValuesLT[3], const Mpp16s rThresholdsGT[3], const Mpp16s rValuesGT[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_16s_AC4IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                                             MppiSize oSizeROI, 
                                             const Mpp16s rThresholdsLT[3], const Mpp16s rValuesLT[3], const Mpp16s rThresholdsGT[3], const Mpp16s rValuesGT[3]);

/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_32f_AC4R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                                      Mpp32f * pDst, int nDstStep, 
                                                MppiSize oSizeROI, 
                                                const Mpp32f rThresholdsLT[3], const Mpp32f rValuesLT[3], const Mpp32f rThresholdsGT[3], const Mpp32f rValuesGT[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_32f_AC4R(const Mpp32f * pSrc, int nSrcStep,
                                                  Mpp32f * pDst, int nDstStep, 
                                            MppiSize oSizeROI, 
                                            const Mpp32f rThresholdsLT[3], const Mpp32f rValuesLT[3], const Mpp32f rThresholdsGT[3], const Mpp32f rValuesGT[3]);

/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_32f_AC4IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                                                 MppiSize oSizeROI, 
                                                 const Mpp32f rThresholdsLT[3], const Mpp32f rValuesLT[3], const Mpp32f rThresholdsGT[3], const Mpp32f rValuesGT[3], MppStreamContext mppStreamCtx);
/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see \ref CommonThresholdLessThanValueGreaterThanValueParameters.
 *
 */
MppStatus mppiThreshold_LTValGTVal_32f_AC4IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                                             MppiSize oSizeROI, 
                                             const Mpp32f rThresholdsLT[3], const Mpp32f rValuesLT[3], const Mpp32f rThresholdsGT[3], const Mpp32f rValuesGT[3]);

/** @} image_threshold_less_than_value_greater_than_value_operations */

/** @} image_thresholding_operations */

/** 
 * \section image_comparison_operations Comparison Operations
 * @defgroup image_comparison_operations Comparison Operations
 * Compare the pixels of two images or one image and a constant value and create a binary result image. In case of multi-channel
 * image types, the condition must be fulfilled for all channels, otherwise the comparison
 * is considered false.
 * The "binary" result image is of type 8u_C1. False is represented by 0, true by MPP_MAX_8U.
 *
 * @{
 *
 */


/** 
 * \section compare_images_operations Compare Images Operations
 * @defgroup compare_images_operations Compare Images OperationsCompare Images Operations
 * Compare the pixels of two images and create a binary result image. In case of multi-channel
 * image types, the condition must be fulfilled for all channels, otherwise the comparison
 * is considered false.
 * The "binary" result image is of type 8u_C1. False is represented by 0, true by MPP_MAX_8U.
 *
 * \subsection CommonCompareImagesParameters Common parameters for mppiCompare functions:
 *
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned char image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_8u_C1R_Ctx(const Mpp8u * pSrc1, int nSrc1Step,
                                 const Mpp8u * pSrc2, int nSrc2Step,
                                       Mpp8u * pDst,  int nDstStep,
                                 MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 1 channel 8-bit unsigned char image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_8u_C1R(const Mpp8u * pSrc1, int nSrc1Step,
                             const Mpp8u * pSrc2, int nSrc2Step,
                                   Mpp8u * pDst,  int nDstStep,
                             MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 3 channel 8-bit unsigned char image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_8u_C3R_Ctx(const Mpp8u * pSrc1, int nSrc1Step,
                                 const Mpp8u * pSrc2, int nSrc2Step,
                                       Mpp8u * pDst,  int nDstStep,
                                 MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 3 channel 8-bit unsigned char image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_8u_C3R(const Mpp8u * pSrc1, int nSrc1Step,
                             const Mpp8u * pSrc2, int nSrc2Step,
                                   Mpp8u * pDst,  int nDstStep,
                             MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 4 channel 8-bit unsigned char image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_8u_C4R_Ctx(const Mpp8u * pSrc1, int nSrc1Step,
                                 const Mpp8u * pSrc2, int nSrc2Step,
                                       Mpp8u * pDst,  int nDstStep,
                                 MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 8-bit unsigned char image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_8u_C4R(const Mpp8u * pSrc1, int nSrc1Step,
                             const Mpp8u * pSrc2, int nSrc2Step,
                                   Mpp8u * pDst,  int nDstStep,
                             MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 4 channel 8-bit unsigned char image compare, not affecting Alpha.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_8u_AC4R_Ctx(const Mpp8u * pSrc1, int nSrc1Step,
                                  const Mpp8u * pSrc2, int nSrc2Step,
                                        Mpp8u * pDst,  int nDstStep,
                                  MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 8-bit unsigned char image compare, not affecting Alpha.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_8u_AC4R(const Mpp8u * pSrc1, int nSrc1Step,
                              const Mpp8u * pSrc2, int nSrc2Step,
                                    Mpp8u * pDst,  int nDstStep,
                              MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 1 channel 16-bit unsigned short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_16u_C1R_Ctx(const Mpp16u * pSrc1, int nSrc1Step,
                                  const Mpp16u * pSrc2, int nSrc2Step,
                                        Mpp8u * pDst,  int nDstStep,
                                  MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 1 channel 16-bit unsigned short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_16u_C1R(const Mpp16u * pSrc1, int nSrc1Step,
                              const Mpp16u * pSrc2, int nSrc2Step,
                                    Mpp8u * pDst,  int nDstStep,
                              MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 3 channel 16-bit unsigned short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_16u_C3R_Ctx(const Mpp16u * pSrc1, int nSrc1Step,
                                  const Mpp16u * pSrc2, int nSrc2Step,
                                        Mpp8u * pDst,  int nDstStep,
                                  MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 3 channel 16-bit unsigned short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_16u_C3R(const Mpp16u * pSrc1, int nSrc1Step,
                              const Mpp16u * pSrc2, int nSrc2Step,
                                    Mpp8u * pDst,  int nDstStep,
                              MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_16u_C4R_Ctx(const Mpp16u * pSrc1, int nSrc1Step,
                                  const Mpp16u * pSrc2, int nSrc2Step,
                                        Mpp8u * pDst,  int nDstStep,
                                  MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit unsigned short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_16u_C4R(const Mpp16u * pSrc1, int nSrc1Step,
                              const Mpp16u * pSrc2, int nSrc2Step,
                                    Mpp8u * pDst,  int nDstStep,
                              MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short image compare, not affecting Alpha.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_16u_AC4R_Ctx(const Mpp16u * pSrc1, int nSrc1Step,
                                   const Mpp16u * pSrc2, int nSrc2Step,
                                         Mpp8u * pDst,  int nDstStep,
                                   MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit unsigned short image compare, not affecting Alpha.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_16u_AC4R(const Mpp16u * pSrc1, int nSrc1Step,
                               const Mpp16u * pSrc2, int nSrc2Step,
                                     Mpp8u * pDst,  int nDstStep,
                               MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 1 channel 16-bit signed short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_16s_C1R_Ctx(const Mpp16s * pSrc1, int nSrc1Step,
                                  const Mpp16s * pSrc2, int nSrc2Step,
                                        Mpp8u * pDst,  int nDstStep,
                                  MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 1 channel 16-bit signed short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_16s_C1R(const Mpp16s * pSrc1, int nSrc1Step,
                              const Mpp16s * pSrc2, int nSrc2Step,
                                    Mpp8u * pDst,  int nDstStep,
                              MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 3 channel 16-bit signed short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_16s_C3R_Ctx(const Mpp16s * pSrc1, int nSrc1Step,
                                  const Mpp16s * pSrc2, int nSrc2Step,
                                        Mpp8u * pDst,  int nDstStep,
                                  MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 3 channel 16-bit signed short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_16s_C3R(const Mpp16s * pSrc1, int nSrc1Step,
                              const Mpp16s * pSrc2, int nSrc2Step,
                                    Mpp8u * pDst,  int nDstStep,
                              MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_16s_C4R_Ctx(const Mpp16s * pSrc1, int nSrc1Step,
                                  const Mpp16s * pSrc2, int nSrc2Step,
                                        Mpp8u * pDst,  int nDstStep,
                                  MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit signed short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_16s_C4R(const Mpp16s * pSrc1, int nSrc1Step,
                              const Mpp16s * pSrc2, int nSrc2Step,
                                    Mpp8u * pDst,  int nDstStep,
                              MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short image compare, not affecting Alpha.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_16s_AC4R_Ctx(const Mpp16s * pSrc1, int nSrc1Step,
                                   const Mpp16s * pSrc2, int nSrc2Step,
                                         Mpp8u * pDst,  int nDstStep,
                                   MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit signed short image compare, not affecting Alpha.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_16s_AC4R(const Mpp16s * pSrc1, int nSrc1Step,
                               const Mpp16s * pSrc2, int nSrc2Step,
                                     Mpp8u * pDst,  int nDstStep,
                               MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 1 channel 32-bit floating point image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_32f_C1R_Ctx(const Mpp32f * pSrc1, int nSrc1Step,
                                  const Mpp32f * pSrc2, int nSrc2Step,
                                        Mpp8u * pDst,  int nDstStep,
                                  MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 1 channel 32-bit floating point image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_32f_C1R(const Mpp32f * pSrc1, int nSrc1Step,
                              const Mpp32f * pSrc2, int nSrc2Step,
                                    Mpp8u * pDst,  int nDstStep,
                              MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 3 channel 32-bit floating point image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_32f_C3R_Ctx(const Mpp32f * pSrc1, int nSrc1Step,
                                  const Mpp32f * pSrc2, int nSrc2Step,
                                        Mpp8u * pDst,  int nDstStep,
                                  MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 3 channel 32-bit floating point image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_32f_C3R(const Mpp32f * pSrc1, int nSrc1Step,
                              const Mpp32f * pSrc2, int nSrc2Step,
                                    Mpp8u * pDst,  int nDstStep,
                              MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit floating point image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_32f_C4R_Ctx(const Mpp32f * pSrc1, int nSrc1Step,
                                  const Mpp32f * pSrc2, int nSrc2Step,
                                        Mpp8u * pDst,  int nDstStep,
                                  MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 32-bit floating point image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_32f_C4R(const Mpp32f * pSrc1, int nSrc1Step,
                              const Mpp32f * pSrc2, int nSrc2Step,
                                    Mpp8u * pDst,  int nDstStep,
                              MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit signed floating point compare, not affecting Alpha.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_32f_AC4R_Ctx(const Mpp32f * pSrc1, int nSrc1Step,
                                   const Mpp32f * pSrc2, int nSrc2Step,
                                         Mpp8u * pDst,  int nDstStep,
                                   MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 32-bit signed floating point compare, not affecting Alpha.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImagesParameters.
 *
 */
MppStatus mppiCompare_32f_AC4R(const Mpp32f * pSrc1, int nSrc1Step,
                               const Mpp32f * pSrc2, int nSrc2Step,
                                     Mpp8u * pDst,  int nDstStep,
                               MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** @} compare_images_operations */

/** 
 * \section compare_image_with_constant_operations Compare Image With Constant Operations
 * @defgroup compare_image_with_constant_operations Compare Image With Constant Operations
 * Compare the pixels of an image with a constant value and create a binary result image. In case of multi-channel
 * image types, the condition must be fulfilled for all channels, otherwise the comparison
 * is considered false.
 * The "binary" result image is of type 8u_C1. False is represented by 0, true by MPP_MAX_8U.
 *
 * \subsection CommonCompareImageWithConstantParameters Common parameters for mppiCompareC functions:
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param nConstant constant value for single channel functions.
 * \param pConstants pointer to a list of constant values, one per color channel for multi-channel functions.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned char image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_8u_C1R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                  const Mpp8u nConstant,
                                        Mpp8u * pDst,  int nDstStep,
                                  MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 1 channel 8-bit unsigned char image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_8u_C1R(const Mpp8u * pSrc, int nSrcStep,
                              const Mpp8u nConstant,
                                    Mpp8u * pDst,  int nDstStep,
                              MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 3 channel 8-bit unsigned char image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                  const Mpp8u * pConstants,
                                        Mpp8u * pDst,  int nDstStep,
                                  MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 3 channel 8-bit unsigned char image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_8u_C3R(const Mpp8u * pSrc, int nSrcStep,
                              const Mpp8u * pConstants,
                                    Mpp8u * pDst,  int nDstStep,
                              MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 4 channel 8-bit unsigned char image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_8u_C4R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                  const Mpp8u * pConstants,
                                        Mpp8u * pDst,  int nDstStep,
                                  MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 8-bit unsigned char image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_8u_C4R(const Mpp8u * pSrc, int nSrcStep,
                              const Mpp8u * pConstants,
                                    Mpp8u * pDst,  int nDstStep,
                              MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 4 channel 8-bit unsigned char image compare, not affecting Alpha.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep,
                                   const Mpp8u * pConstants,
                                         Mpp8u * pDst,  int nDstStep,
                                   MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 8-bit unsigned char image compare, not affecting Alpha.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_8u_AC4R(const Mpp8u * pSrc, int nSrcStep,
                               const Mpp8u * pConstants,
                                     Mpp8u * pDst,  int nDstStep,
                               MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 1 channel 16-bit unsigned short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_16u_C1R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                   const Mpp16u nConstant,
                                         Mpp8u * pDst,  int nDstStep,
                                   MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 1 channel 16-bit unsigned short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_16u_C1R(const Mpp16u * pSrc, int nSrcStep,
                               const Mpp16u nConstant,
                                     Mpp8u * pDst,  int nDstStep,
                               MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 3 channel 16-bit unsigned short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_16u_C3R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                   const Mpp16u * pConstants,
                                         Mpp8u * pDst,  int nDstStep,
                                   MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 3 channel 16-bit unsigned short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_16u_C3R(const Mpp16u * pSrc, int nSrcStep,
                               const Mpp16u * pConstants,
                                     Mpp8u * pDst,  int nDstStep,
                               MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_16u_C4R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                   const Mpp16u * pConstants,
                                         Mpp8u * pDst,  int nDstStep,
                                   MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit unsigned short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_16u_C4R(const Mpp16u * pSrc, int nSrcStep,
                               const Mpp16u * pConstants,
                                     Mpp8u * pDst,  int nDstStep,
                               MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short image compare, not affecting Alpha.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_16u_AC4R_Ctx(const Mpp16u * pSrc, int nSrcStep,
                                    const Mpp16u * pConstants,
                                          Mpp8u * pDst,  int nDstStep,
                                    MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit unsigned short image compare, not affecting Alpha.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_16u_AC4R(const Mpp16u * pSrc, int nSrcStep,
                                const Mpp16u * pConstants,
                                      Mpp8u * pDst,  int nDstStep,
                                MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 1 channel 16-bit signed short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_16s_C1R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                   const Mpp16s nConstant,
                                         Mpp8u * pDst,  int nDstStep,
                                   MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 1 channel 16-bit signed short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_16s_C1R(const Mpp16s * pSrc, int nSrcStep,
                               const Mpp16s nConstant,
                                     Mpp8u * pDst,  int nDstStep,
                               MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 3 channel 16-bit signed short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_16s_C3R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                   const Mpp16s * pConstants,
                                         Mpp8u * pDst,  int nDstStep,
                                   MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 3 channel 16-bit signed short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_16s_C3R(const Mpp16s * pSrc, int nSrcStep,
                               const Mpp16s * pConstants,
                                     Mpp8u * pDst,  int nDstStep,
                               MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_16s_C4R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                   const Mpp16s * pConstants,
                                         Mpp8u * pDst,  int nDstStep,
                                   MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit signed short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_16s_C4R(const Mpp16s * pSrc, int nSrcStep,
                               const Mpp16s * pConstants,
                                     Mpp8u * pDst,  int nDstStep,
                               MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short image compare, not affecting Alpha.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_16s_AC4R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                    const Mpp16s * pConstants,
                                          Mpp8u * pDst,  int nDstStep,
                                    MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 16-bit signed short image compare, not affecting Alpha.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_16s_AC4R(const Mpp16s * pSrc, int nSrcStep,
                                const Mpp16s * pConstants,
                                      Mpp8u * pDst,  int nDstStep,
                                MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 1 channel 32-bit floating point image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_32f_C1R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                   const Mpp32f nConstant,
                                         Mpp8u * pDst,  int nDstStep,
                                   MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 1 channel 32-bit floating point image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_32f_C1R(const Mpp32f * pSrc, int nSrcStep,
                               const Mpp32f nConstant,
                                     Mpp8u * pDst,  int nDstStep,
                               MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 3 channel 32-bit floating point image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_32f_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                   const Mpp32f * pConstants,
                                         Mpp8u * pDst,  int nDstStep,
                                   MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 3 channel 32-bit floating point image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_32f_C3R(const Mpp32f * pSrc, int nSrcStep,
                               const Mpp32f * pConstants,
                                     Mpp8u * pDst,  int nDstStep,
                               MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit floating point image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_32f_C4R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                   const Mpp32f * pConstants,
                                         Mpp8u * pDst,  int nDstStep,
                                   MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 32-bit floating point image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_32f_C4R(const Mpp32f * pSrc, int nSrcStep,
                               const Mpp32f * pConstants,
                                     Mpp8u * pDst,  int nDstStep,
                               MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit signed floating point compare, not affecting Alpha.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_32f_AC4R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                    const Mpp32f * pConstants,
                                          Mpp8u * pDst,  int nDstStep,
                                    MppiSize oSizeROI, MppCmpOp eComparisonOperation, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 32-bit signed floating point compare, not affecting Alpha.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageWithConstantParameters.
 *
 */
MppStatus mppiCompareC_32f_AC4R(const Mpp32f * pSrc, int nSrcStep,
                                const Mpp32f * pConstants,
                                      Mpp8u * pDst,  int nDstStep,
                                MppiSize oSizeROI, MppCmpOp eComparisonOperation);

/** @} compare_image_with_constant_operations */

/** 
 * \section compare_image_differences_with_epsilon_operations Compare Image Differences With Epsilon Operations
 * @defgroup compare_image_differences_with_epsilon_operations Compare Image Differences With Epsilon Operations
 * Compare the pixels value differences of two images with an epsilon value and create a binary result image. In case of multi-channel
 * image types, the condition must be fulfilled for all channels, otherwise the comparison
 * is considered false.
 * The "binary" result image is of type 8u_C1. False is represented by 0, true by MPP_MAX_8U.
 *
 * \subsection CommonCompareImageDifferencesWithEpsilonParameters Common parameters for mppiCompareEqualEps functions include:
 *
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nEpsilon epsilon tolerance value to compare to pixel absolute differences
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * 1 channel 32-bit floating point image compare whether two images are equal within epsilon.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageDifferencesWithEpsilonParameters.
 *
 */
MppStatus mppiCompareEqualEps_32f_C1R_Ctx(const Mpp32f * pSrc1, int nSrc1Step,
                                          const Mpp32f * pSrc2, int nSrc2Step,
                                                Mpp8u * pDst,   int nDstStep,
                                          MppiSize oSizeROI, Mpp32f nEpsilon, MppStreamContext mppStreamCtx);
/** 
 * 1 channel 32-bit floating point image compare whether two images are equal within epsilon.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageDifferencesWithEpsilonParameters.
 *
 */
MppStatus mppiCompareEqualEps_32f_C1R(const Mpp32f * pSrc1, int nSrc1Step,
                                      const Mpp32f * pSrc2, int nSrc2Step,
                                            Mpp8u * pDst,   int nDstStep,
                                      MppiSize oSizeROI, Mpp32f nEpsilon);

/** 
 * 3 channel 32-bit floating point image compare whether two images are equal within epsilon.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageDifferencesWithEpsilonParameters.
 *
 */
MppStatus mppiCompareEqualEps_32f_C3R_Ctx(const Mpp32f * pSrc1, int nSrc1Step,
                                          const Mpp32f * pSrc2, int nSrc2Step,
                                                Mpp8u * pDst,   int nDstStep,
                                          MppiSize oSizeROI, Mpp32f nEpsilon, MppStreamContext mppStreamCtx);
/** 
 * 3 channel 32-bit floating point image compare whether two images are equal within epsilon.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageDifferencesWithEpsilonParameters.
 *
 */
MppStatus mppiCompareEqualEps_32f_C3R(const Mpp32f * pSrc1, int nSrc1Step,
                                      const Mpp32f * pSrc2, int nSrc2Step,
                                            Mpp8u * pDst,   int nDstStep,
                                      MppiSize oSizeROI, Mpp32f nEpsilon);

/** 
 * 4 channel 32-bit floating point image compare whether two images are equal within epsilon.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageDifferencesWithEpsilonParameters.
 *
 */
MppStatus mppiCompareEqualEps_32f_C4R_Ctx(const Mpp32f * pSrc1, int nSrc1Step,
                                          const Mpp32f * pSrc2, int nSrc2Step,
                                                Mpp8u * pDst,   int nDstStep,
                                          MppiSize oSizeROI, Mpp32f nEpsilon, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 32-bit floating point image compare whether two images are equal within epsilon.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageDifferencesWithEpsilonParameters.
 *
 */
MppStatus mppiCompareEqualEps_32f_C4R(const Mpp32f * pSrc1, int nSrc1Step,
                                      const Mpp32f * pSrc2, int nSrc2Step,
                                            Mpp8u * pDst,   int nDstStep,
                                      MppiSize oSizeROI, Mpp32f nEpsilon);

/** 
 * 4 channel 32-bit signed floating point compare whether two images are equal within epsilon, not affecting Alpha.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageDifferencesWithEpsilonParameters.
 *
 */
MppStatus mppiCompareEqualEps_32f_AC4R_Ctx(const Mpp32f * pSrc1, int nSrc1Step,
                                           const Mpp32f * pSrc2, int nSrc2Step,
                                                 Mpp8u * pDst,   int nDstStep,
                                           MppiSize oSizeROI, Mpp32f nEpsilon, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 32-bit signed floating point compare whether two images are equal within epsilon, not affecting Alpha.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageDifferencesWithEpsilonParameters.
 *
 */
MppStatus mppiCompareEqualEps_32f_AC4R(const Mpp32f * pSrc1, int nSrc1Step,
                                       const Mpp32f * pSrc2, int nSrc2Step,
                                             Mpp8u * pDst,   int nDstStep,
                                       MppiSize oSizeROI, Mpp32f nEpsilon);

/** @} compare_image_differences_with_epsilon_operations */


/** 
 * \section compare_image_difference_to_constant_with_epsilon_operations Compare Image Difference With Constant Within Epsilon Operations
 * @defgroup compare_image_difference_to_constant_with_epsilon_operations Compare Image Difference With Constant Within Epsilon Operations
 * Compare differences between image pixels and constant within an epsilon value and create a binary result image. In case of multi-channel
 * image types, the condition must be fulfilled for all channels, otherwise the comparison
 * is considered false.
 * The "binary" result image is of type 8u_C1. False is represented by 0, true by MPP_MAX_8U.
 *
 * \subsection CommonCompareImageDifferenceWithConstantWithinEpsilonParameters Common parameters for mppiCompareEqualEpsC functions:
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param nConstant constant value for single channel functions.
 * \param pConstants pointer to a list of constants, one per color channel for multi-channel image functions.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nEpsilon epsilon tolerance value to compare to per color channel pixel absolute differences
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * 1 channel 32-bit floating point image compare whether image and constant are equal within epsilon.
 * Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageDifferenceWithConstantWithinEpsilonParameters.
 *
 */
MppStatus mppiCompareEqualEpsC_32f_C1R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                           const Mpp32f nConstant,
                                                 Mpp8u * pDst,  int nDstStep,
                                           MppiSize oSizeROI, Mpp32f nEpsilon, MppStreamContext mppStreamCtx);
/** 
 * 1 channel 32-bit floating point image compare whether image and constant are equal within epsilon.
 * Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageDifferenceWithConstantWithinEpsilonParameters.
 *
 */
MppStatus mppiCompareEqualEpsC_32f_C1R(const Mpp32f * pSrc, int nSrcStep,
                                       const Mpp32f nConstant,
                                             Mpp8u * pDst,  int nDstStep,
                                       MppiSize oSizeROI, Mpp32f nEpsilon);

/** 
 * 3 channel 32-bit floating point image compare whether image and constant are equal within epsilon.
 * Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageDifferenceWithConstantWithinEpsilonParameters.
 *
 */
MppStatus mppiCompareEqualEpsC_32f_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                           const Mpp32f * pConstants,
                                                 Mpp8u * pDst,  int nDstStep,
                                           MppiSize oSizeROI, Mpp32f nEpsilon, MppStreamContext mppStreamCtx);
/** 
 * 3 channel 32-bit floating point image compare whether image and constant are equal within epsilon.
 * Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageDifferenceWithConstantWithinEpsilonParameters.
 *
 */
MppStatus mppiCompareEqualEpsC_32f_C3R(const Mpp32f * pSrc, int nSrcStep,
                                       const Mpp32f * pConstants,
                                             Mpp8u * pDst,  int nDstStep,
                                       MppiSize oSizeROI, Mpp32f nEpsilon);

/** 
 * 4 channel 32-bit floating point image compare whether image and constant are equal within epsilon.
 * Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageDifferenceWithConstantWithinEpsilonParameters.
 *
 */
MppStatus mppiCompareEqualEpsC_32f_C4R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                           const Mpp32f * pConstants,
                                                 Mpp8u * pDst,  int nDstStep,
                                           MppiSize oSizeROI, Mpp32f nEpsilon, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 32-bit floating point image compare whether image and constant are equal within epsilon.
 * Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageDifferenceWithConstantWithinEpsilonParameters.
 *
 */
MppStatus mppiCompareEqualEpsC_32f_C4R(const Mpp32f * pSrc, int nSrcStep,
                                       const Mpp32f * pConstants,
                                             Mpp8u * pDst,  int nDstStep,
                                       MppiSize oSizeROI, Mpp32f nEpsilon);

/** 
 * 4 channel 32-bit signed floating point compare whether image and constant are equal within epsilon, not affecting Alpha.
 * Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageDifferenceWithConstantWithinEpsilonParameters.
 *
 */
MppStatus mppiCompareEqualEpsC_32f_AC4R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                            const Mpp32f * pConstants,
                                                  Mpp8u * pDst,  int nDstStep,
                                            MppiSize oSizeROI, Mpp32f nEpsilon, MppStreamContext mppStreamCtx);
/** 
 * 4 channel 32-bit signed floating point compare whether image and constant are equal within epsilon, not affecting Alpha.
 * Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon. 
 * 
 * For common parameter descriptions, see \ref CommonCompareImageDifferenceWithConstantWithinEpsilonParameters.
 *
 */
MppStatus mppiCompareEqualEpsC_32f_AC4R(const Mpp32f * pSrc, int nSrcStep,
                                        const Mpp32f * pConstants,
                                              Mpp8u * pDst,  int nDstStep,
                                        MppiSize oSizeROI, Mpp32f nEpsilon);

/** @} compare_image_difference_to_constant_within_epsilon_operations */

/** @} image_comparison_operations */

/** @} image_threshold_and_compare_operations */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MC_MPPI_THRESHOLD_AND_COMPARE_OPERATIONS_H */
