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
#ifndef MC_MPPI_COLOR_CONVERSION_H
#define MC_MPPI_COLOR_CONVERSION_H
 
/**
 * \file mppi_color_conversion.h
 * MPP Image Processing Functionality.
 */
 
#include "mppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif


/** 
 *  \page image_color_conversion Color and Sampling Conversion
 *  @defgroup image_color_conversion Color and Sampling Conversion
 *  @ingroup mppi
 *
 * Routines manipulating an image's color model and sampling format.
 *
 * @{
 *
 * These functions can be found in the nppicc library. Linking to only the sub-libraries that you use can significantly
 * save link time, application load time, and MACA runtime startup time when using dynamic libraries.
 *
 */

/** 
 *  @defgroup image_color_model_conversion Color Model Conversion
 *
 * Routines for converting between various image color models.
 *
 * @{ 
 *
 */

/** @defgroup rgbtoyuv RGBToYUV 
 *  RGB to YUV color conversion.
 *
 *  Here is how MPP converts gamma corrected RGB or BGR to YUV. For digital RGB values in the range [0..255], 
 *  Y has the range [0..255], U varies in the range [-112..+112], 
 *  and V in the range [-157..+157]. To fit in the range of [0..255], a constant value
 *  of 128 is added to computed U and V values, and V is then saturated.
 *
 *  \code   
 *  Mpp32f nY =  0.299F * R + 0.587F * G + 0.114F * B; 
 *  Mpp32f nU = (0.492F * ((Mpp32f)nB - nY)) + 128.0F;
 *  Mpp32f nV = (0.877F * ((Mpp32f)nR - nY)) + 128.0F;
 *  if (nV > 255.0F) 
 *      nV = 255.0F;
 *  \endcode
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed YUV color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYUV_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed YUV color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYUV_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed YUV color conversion with alpha, not affecting alpha.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYUV_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed YUV color conversion with alpha, not affecting alpha.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYUV_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YUV color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYUV_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YUV color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYUV_8u_P3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YUV color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYUV_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YUV color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYUV_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned planar YUV color conversion with alpha.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYUV_8u_AC4P4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[4], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned planar YUV color conversion with alpha.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYUV_8u_AC4P4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[4], int nDstStep, MppiSize oSizeROI);

/** @} rgbtoyuv */

/** @defgroup bgrtoyuv BGRToYUV 
 *  BGR to YUV color conversion.
 *
 *  Here is how MPP converts gamma corrected RGB or BGR to YUV. For digital RGB values in the range [0..255], 
 *  Y has the range [0..255], U varies in the range [-112..+112], 
 *  and V in the range [-157..+157]. To fit in the range of [0..255], a constant value
 *  of 128 is added to computed U and V values, and V is then saturated.
 *
 *  \code   
 *  Mpp32f nY =  0.299F * R + 0.587F * G + 0.114F * B; 
 *  Mpp32f nU = (0.492F * ((Mpp32f)nB - nY)) + 128.0F;
 *  Mpp32f nV = (0.877F * ((Mpp32f)nR - nY)) + 128.0F;
 *  if (nV > 255.0F) 
 *      nV = 255.0F;
 *  \endcode
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned packed YUV color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYUV_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned packed YUV color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYUV_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 4 channel 8-bit unsigned packed YUV color conversion with alpha, not affecting alpha.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYUV_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 4 channel 8-bit unsigned packed YUV color conversion with alpha, not affecting alpha.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYUV_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar YUV color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYUV_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar YUV color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYUV_8u_P3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YUV color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYUV_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YUV color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYUV_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 4 channel 8-bit unsigned planar YUV color conversion with alpha.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYUV_8u_AC4P4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[4], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 4 channel 8-bit unsigned planar YUV color conversion with alpha.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYUV_8u_AC4P4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[4], int nDstStep, MppiSize oSizeROI);

/** @} bgrtoyuv */

/** @defgroup yuvtorgb YUVToRGB 
 *  YUV to RGB color conversion.
 *
 *  Here is how MPP converts YUV to gamma corrected RGB or BGR.
 *
 *  \code
 *  Mpp32f nY = (Mpp32f)Y;
 *  Mpp32f nU = (Mpp32f)U - 128.0F;
 *  Mpp32f nV = (Mpp32f)V - 128.0F;
 *  Mpp32f nR = nY + 1.140F * nV; 
 *  if (nR < 0.0F)
 *      nR = 0.0F;
 *  if (nR > 255.0F)
 *      nR = 255.0F;    
 *  Mpp32f nG = nY - 0.394F * nU - 0.581F * nV;
 *  if (nG < 0.0F)
 *      nG = 0.0F;
 *  if (nG > 255.0F)
 *      nG = 255.0F;    
 *  Mpp32f nB = nY + 2.032F * nU;
 *  if (nB < 0.0F)
 *      nB = 0.0F;
 *  if (nB > 255.0F)
 *      nB = 255.0F;    
 *  \endcode
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned packed YUV to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToRGB_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed YUV to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToRGB_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit packed YUV with alpha to 4 channel 8-bit unsigned packed RGB color conversion with alpha, not affecting alpha.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToRGB_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit packed YUV with alpha to 4 channel 8-bit unsigned packed RGB color conversion with alpha, not affecting alpha.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToRGB_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned planar RGB color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToRGB_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned planar RGB color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToRGB_8u_P3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToRGB_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToRGB_8u_P3C3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} yuvtorgb */

/** @defgroup yuvtorgbbatch YUVToRGBBatch
 *  YUV to RGB batch color conversion with a single \ref roi_specification for all pairs of input/output images provided in batches.
 *
 *  MPP converts YUV to gamma corrected RGB the same way as in \ref yuvtorgb.
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned packed YUV to 3 channel 8-bit unsigned packed RGB batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input and output images passed in pSrcBatchList and pSrcBatchList
 * arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the borders of any of provided images.
 *
 * \param pSrcBatchList \ref source_batch_images_pointer.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToRGBBatch_8u_C3R_Ctx(const MppiImageDescriptor* pSrcBatchList, MppiImageDescriptor* pDstBatchList, int nBatchSize, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed YUV to 3 channel 8-bit unsigned packed RGB batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input and output images passed in pSrcBatchList and pSrcBatchList
 * arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the borders of any of provided images.
 *
 * \param pSrcBatchList \ref source_batch_images_pointer.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToRGBBatch_8u_C3R(const MppiImageDescriptor* pSrcBatchList, MppiImageDescriptor* pDstBatchList, int nBatchSize, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned packed RGB batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of U planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToRGBBatch_8u_P3C3R_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned packed RGB batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of U planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToRGBBatch_8u_P3C3R(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI);

/** @} yuvtorgbbatch */

/** @defgroup yuvtorgbbatchadvanced YUVToRGBBatchAdvanced
 *  YUV to RGB batch color conversion where each pair of input/output images from provided batches has own \ref roi_specification.
 *
 *  MPP converts YUV to gamma corrected RGB the same way as in \ref yuvtorgb.
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned packed YUV to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair of input/output images has own ROI.
 * Provided oMaxSizeROI must contain the maximum width and the maximum height of all ROIs defined in pDstBatchList. API user must ensure that
 * ROI from pDstBatchList for each pair of input and output images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList \ref source_batch_images_pointer.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \param mppStreamCtx \ref application_managed_stream_context.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToRGBBatch_8u_C3R_Advanced_Ctx(const MppiImageDescriptor* pSrcBatchList, MppiImageDescriptor* pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed YUV to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair of input/output images has own ROI.
 * Provided oMaxSizeROI must contain the maximum width and the maximum height of all ROIs defined in pDstBatchList. API user must ensure that
 * ROI from pDstBatchList for each pair of input and output images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList \ref source_batch_images_pointer.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToRGBBatch_8u_C3R_Advanced(const MppiImageDescriptor* pSrcBatchList, MppiImageDescriptor* pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI);

/**
 * 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of U planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToRGBBatch_8u_P3C3R_Advanced_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of U planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToRGBBatch_8u_P3C3R_Advanced(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI);

/** @} yuvtorgbbatchadvanced */

/** @defgroup yuvtobgr YUVToBGR 
 *  YUV to BGR color conversion.
 *
 *  Here is how MPP converts YUV to gamma corrected RGB or BGR.
 *
 *  \code
 *  Mpp32f nY = (Mpp32f)Y;
 *  Mpp32f nU = (Mpp32f)U - 128.0F;
 *  Mpp32f nV = (Mpp32f)V - 128.0F;
 *  Mpp32f nR = nY + 1.140F * nV; 
 *  if (nR < 0.0F)
 *      nR = 0.0F;
 *  if (nR > 255.0F)
 *      nR = 255.0F;    
 *  Mpp32f nG = nY - 0.394F * nU - 0.581F * nV;
 *  if (nG < 0.0F)
 *      nG = 0.0F;
 *  if (nG > 255.0F)
 *      nG = 255.0F;    
 *  Mpp32f nB = nY + 2.032F * nU;
 *  if (nB < 0.0F)
 *      nB = 0.0F;
 *  if (nB > 255.0F)
 *      nB = 255.0F;    
 *  \endcode
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned packed YUV to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToBGR_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed YUV to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToBGR_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit packed YUV with alpha to 4 channel 8-bit unsigned packed BGR color conversion with alpha, not affecting alpha.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToBGR_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit packed YUV with alpha to 4 channel 8-bit unsigned packed BGR color conversion with alpha, not affecting alpha.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToBGR_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned planar BGR color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToBGR_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned planar BGR color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToBGR_8u_P3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToBGR_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToBGR_8u_P3C3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} yuvtobgr */

/** @defgroup yuvtobgrbatch YUVToBGRBatch
 *  YUV to BGR batch color conversion with a single \ref roi_specification for all pairs of input/output images provided in batches.
 *
 *  MPP converts YUV to gamma corrected BGR the same way as in \ref yuvtobgr.
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned packed YUV to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input and output images passed in pSrcBatchList and pSrcBatchList
 * arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the borders of any of provided images.
 *
 * \param pSrcBatchList \ref source_batch_images_pointer.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToBGRBatch_8u_C3R_Ctx(const MppiImageDescriptor* pSrcBatchList, MppiImageDescriptor* pDstBatchList, int nBatchSize, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed YUV to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input and output images passed in pSrcBatchList and pSrcBatchList
 * arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the borders of any of provided images.
 *
 * \param pSrcBatchList \ref source_batch_images_pointer.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToBGRBatch_8u_C3R(const MppiImageDescriptor* pSrcBatchList, MppiImageDescriptor* pDstBatchList, int nBatchSize, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of U planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToBGRBatch_8u_P3C3R_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of U planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToBGRBatch_8u_P3C3R(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI);

/** @} yuvtobgrbatch */

/** @defgroup yuvtobgrbatchadvanced YUVToBGRBatchAdvanced
 *  YUV to BGR batch color conversion where each pair of input/output images from provided batches has own \ref roi_specification.
 *
 *  MPP converts YUV to gamma corrected BGR the same way as in \ref yuvtobgr.
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned packed YUV to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair of input/output images has own ROI.
 * Provided oMaxSizeROI must contain the maximum width and the maximum height of all ROIs defined in pDstBatchList. API user must ensure that
 * ROI from pDstBatchList for each pair of input and output images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList \ref source_batch_images_pointer.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \param mppStreamCtx \ref application_managed_stream_context.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToBGRBatch_8u_C3R_Advanced_Ctx(const MppiImageDescriptor* pSrcBatchList, MppiImageDescriptor* pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed YUV to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair of input/output images has own ROI.
 * Provided oMaxSizeROI must contain the maximum width and the maximum height of all ROIs defined in pDstBatchList. API user must ensure that
 * ROI from pDstBatchList for each pair of input and output images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList \ref source_batch_images_pointer.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToBGRBatch_8u_C3R_Advanced(const MppiImageDescriptor* pSrcBatchList, MppiImageDescriptor* pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI);

/**
 * 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of U planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToBGRBatch_8u_P3C3R_Advanced_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of U planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUVToBGRBatch_8u_P3C3R_Advanced(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI);

/** @} yuvtobgrbatchadvanced */

/** @defgroup rgbtoyuv422 RGBToYUV422 
 *  RGB to YUV422 color conversion.
 *
 *  MPP converts YUV to gamma corrected BGR the same way as in \ref yuvtobgr.
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned packed RGB to 2 channel 8-bit unsigned packed YUV422 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYUV422_8u_C3C2R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 2 channel 8-bit unsigned packed YUV422 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYUV422_8u_C3C2R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YUV422 color conversion.
 * images.
 *                         
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYUV422_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YUV422 color conversion.
 * images.
 *                         
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYUV422_8u_P3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YUV422 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYUV422_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YUV422 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYUV422_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} rgbtoyuv422 */

/** @defgroup yuv422torgb YUV422ToRGB 
 *  YUV422 to RGB color conversion.
 *
 * @{
 *
 */

/**
 * 2 channel 8-bit unsigned packed YUV422 to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV422ToRGB_8u_C2C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed YUV422 to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV422ToRGB_8u_C2C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned planar RGB color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV422ToRGB_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned planar RGB color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV422ToRGB_8u_P3R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV422ToRGB_8u_P3C3R_Ctx(const Mpp8u* const pSrc[3], int rSrcStep[3], Mpp8u* pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV422ToRGB_8u_P3C3R(const Mpp8u* const pSrc[3], int rSrcStep[3], Mpp8u* pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YUV422 to 4 channel 8-bit unsigned packed RGB color conversion with alpha.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV422ToRGB_8u_P3AC4R_Ctx(const Mpp8u* const pSrc[3], int rSrcStep[3], Mpp8u* pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV422 to 4 channel 8-bit unsigned packed RGB color conversion with alpha.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV422ToRGB_8u_P3AC4R(const Mpp8u* const pSrc[3], int rSrcStep[3], Mpp8u* pDst, int nDstStep, MppiSize oSizeROI);

/** @} yuv422torgb */

/** @defgroup yuv422torgbbatch YUV422ToRGBBatch
 *  Planar YUV422 to packed RGB batch color conversion with a single \ref roi_specification for all pairs of input/output images provided in batches.
 *
 * @{
 *
 */
 
/**
 * 3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned packed RGB batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of U planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV422ToRGBBatch_8u_P3C3R_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned packed RGB batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of U planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV422ToRGBBatch_8u_P3C3R(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI);

/** @} yuv422torgbbatch */

/** @defgroup yuv422torgbbatchadvanced YUV422ToRGBBatchAdvanced
 *  Planar YUV422 to packed RGB batch color conversion where each pair of input/output images from provided batches has own \ref roi_specification.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of U planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \param mppStreamCtx \ref application_managed_stream_context.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV422ToRGBBatch_8u_P3C3R_Advanced_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of U planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV422ToRGBBatch_8u_P3C3R_Advanced(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI);

/** @} yuv422torgbbatchadvanced */

/** @defgroup yuv422tobgrbatch YUV422ToBGRBatch
 *  Planar YUV422 to packed BGR batch color conversion with a single \ref roi_specification for all pairs of input/output images provided in batches.
 *
 * @{
 *
 */
 
/**
 * 3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of U planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV422ToBGRBatch_8u_P3C3R_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of U planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV422ToBGRBatch_8u_P3C3R(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI);

/** @} yuv422tobgrbatch */

/** @defgroup yuv422tobgrbatchadvanced YUV422ToBGRBatchAdvanced
 *  Planar YUV422 to packed BGR batch color conversion where each pair of input/output images from provided batches has own \ref roi_specification.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of U planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV422ToBGRBatch_8u_P3C3R_Advanced_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of U planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV422ToBGRBatch_8u_P3C3R_Advanced(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI);

/** @} yuv422tobgrbatchadvanced */

/** @defgroup rgbtoyuv420 RGBToYUV420 
 *  RGB to YUV420 color conversion.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YUV420 color conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYUV420_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YUV420 color conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYUV420_8u_P3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YUV420 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYUV420_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YUV420 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYUV420_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} rgbtoyuv420 */

/** @defgroup yuv420torgb YUV420ToRGB 
 *  YUV420 to RGB color conversion.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned planar RGB color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV420ToRGB_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned planar RGB color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV420ToRGB_8u_P3R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV420ToRGB_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV420ToRGB_8u_P3C3R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YUV420 to 4 channel 8-bit unsigned packed RGB color conversion with constant alpha (0xFF).
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV420ToRGB_8u_P3C4R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV420 to 4 channel 8-bit unsigned packed RGB color conversion with constant alpha (0xFF).
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV420ToRGB_8u_P3C4R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YUV420 to 4 channel 8-bit unsigned packed RGB color conversion with alpha.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV420ToRGB_8u_P3AC4R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV420 to 4 channel 8-bit unsigned packed RGB color conversion with alpha.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV420ToRGB_8u_P3AC4R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} yuv420torgb */

/** @defgroup yuv420torgbbatch YUV420ToRGBBatch
 *  Planar YUV420 to packed RGB batch color conversion with a single \ref roi_specification for all pairs of input/output images provided in batches.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed RGB batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents Y planes.
 *        The second element of array (pSrcBatchList[1]) represents U planes. The third element of array (pSrcBatchList[2])
 *        represents V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV420ToRGBBatch_8u_P3C3R_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed RGB batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents Y planes.
 *        The second element of array (pSrcBatchList[1]) represents U planes. The third element of array (pSrcBatchList[2])
 *        represents V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV420ToRGBBatch_8u_P3C3R(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI);

/** @} yuv420torgbbatch */

/** @defgroup yuv420torgbbatchadvanced YUV420ToRGBBatchAdvanced
 *  Planar YUV420 to packed RGB batch color conversion where each pair of input/output images from provided batches has own \ref roi_specification.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of U planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV420ToRGBBatch_8u_P3C3R_Advanced_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of U planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV420ToRGBBatch_8u_P3C3R_Advanced(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI);

/** @} yuv420torgbbatchadvanced */

/** @defgroup nv12torgb NV12ToRGB 
 *  NV12 to RGB color conversion.
 *
 * @{
 *
 */

/**
 *  2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit
 *  unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array (one for Y
 *        plane, one for UV plane).
 * \param rSrcStep \ref source_planar_image_line_step. Same value is used for each source plane.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiNV12ToRGB_8u_P2C3R_Ctx(const Mpp8u * const pSrc[2], int rSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 *  2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit
 *  unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array (one for Y
 *        plane, one for UV plane).
 * \param rSrcStep \ref source_planar_image_line_step. Same value is used for each source plane.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiNV12ToRGB_8u_P2C3R(const Mpp8u * const pSrc[2], int rSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 *  2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit
 *  unsigned packed RGB 709 HDTV full color conversion. Note that HDTV conversion assumes full color range of 0 - 255, use CSC version for limited range color.
 *
 * \param pSrc \ref source_planar_image_pointer_array (one for Y
 *        plane, one for UV plane).
 * \param rSrcStep \ref source_planar_image_line_step. Same value is used for each plane.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(const Mpp8u * const pSrc[2], int rSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 *  2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit
 *  unsigned packed RGB 709 HDTV full color conversion. Note that HDTV conversion assumes full color range of 0 - 255, use CSC version for limited range color.
 *
 * \param pSrc \ref source_planar_image_pointer_array (one for Y
 *        plane, one for UV plane).
 * \param rSrcStep \ref source_planar_image_line_step. Same value is used for each plane.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiNV12ToRGB_709HDTV_8u_P2C3R(const Mpp8u * const pSrc[2], int rSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 *  2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit
 *  unsigned packed RGB 709 CSC color conversion. Note that HDTV conversion assumes full color range of 0 - 255, use CSC version for limited range color.
 *
 * \param pSrc \ref source_planar_image_pointer_array (one for Y
 *        plane, one for UV plane).
 * \param rSrcStep \ref source_planar_image_line_step. Same value is used for each plane.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiNV12ToRGB_709CSC_8u_P2C3R_Ctx(const Mpp8u * const pSrc[2], int rSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 *  2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit
 *  unsigned packed RGB 709 CSC color conversion. Note that HDTV conversion assumes full color range of 0 - 255, use CSC version for limited range color.
 *
 * \param pSrc \ref source_planar_image_pointer_array (one for Y
 *        plane, one for UV plane).
 * \param rSrcStep \ref source_planar_image_line_step. Same value is used for each plane.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiNV12ToRGB_709CSC_8u_P2C3R(const Mpp8u * const pSrc[2], int rSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} nv12torgb */

/** @defgroup nv21torgb NV21ToRGB 
 *  NV21 to RGB color conversion.
 *
 * @{
 *
 */

/**
 *  2 channel 8-bit unsigned planar NV21 to 4 channel 8-bit unsigned packed RGBA color conversion with constant alpha (0xFF).
 *
 * \param pSrc \ref source_planar_image_pointer_array (one for Y plane, one for VU plane).
 * \param rSrcStep \ref source_planar_image_line_step. Same value is used for each plane.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiNV21ToRGB_8u_P2C4R_Ctx(const Mpp8u * const pSrc[2], int rSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 *  2 channel 8-bit unsigned planar NV21 to 4 channel 8-bit unsigned packed RGBA color conversion with constant alpha (0xFF).
 *
 * \param pSrc \ref source_planar_image_pointer_array (one for Y plane, one for VU plane).
 * \param rSrcStep \ref source_planar_image_line_step. Same value is used for each plane.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiNV21ToRGB_8u_P2C4R(const Mpp8u * const pSrc[2], int rSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} nv21torgb */

/** @defgroup bgrtoyuv420 BGRToYUV420 
 *  BGR to YUV420 color conversion.
 *
 * @{
 *
 */

/**
 * 4 channel 8-bit unsigned pacmed BGR with alpha to 3 channel 8-bit unsigned planar YUV420 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYUV420_8u_AC4P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);
 
/**
 * 4 channel 8-bit unsigned pacmed BGR with alpha to 3 channel 8-bit unsigned planar YUV420 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYUV420_8u_AC4P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);
 
/** @} bgrtoyuv420 */

/** @defgroup yuv420tobgr YUV420ToBGR 
 *  YUV420 to BGR color conversion.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV420ToBGR_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV420ToBGR_8u_P3C3R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YUV420 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha (0xFF).
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV420ToBGR_8u_P3C4R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV420 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha (0xFF).
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV420ToBGR_8u_P3C4R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} yuv420tobgr */

/** @defgroup yuv420tobgrbatch YUV420ToBGRBatch
 *  Planar YUV420 to packed BGR batch color conversion with a single \ref roi_specification for all pairs of input/output images provided in batches.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents Y planes.
 *        The second element of array (pSrcBatchList[1]) represents U planes. The third element of array (pSrcBatchList[2])
 *        represents V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV420ToBGRBatch_8u_P3C3R_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents Y planes.
 *        The second element of array (pSrcBatchList[1]) represents U planes. The third element of array (pSrcBatchList[2])
 *        represents V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV420ToBGRBatch_8u_P3C3R(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI);

/** @} yuv420tobgrbatch */

/** @defgroup yuv420tobgrbatchadvanced YUV420ToBGRBatchAdvanced
 *  Planar YUV420 to packed BGR batch color conversion where each pair of input/output images from provided batches has own \ref roi_specification.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of U planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV420ToBGRBatch_8u_P3C3R_Advanced_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of U planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of V planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV420ToBGRBatch_8u_P3C3R_Advanced(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI);

/** @} yuv420tobgrbatchadvanced */

/** @defgroup nv12tobgr NV12ToBGR 
 *  NV12 to BGR color conversion.
 *
 * @{
 *
 */

/**
 *  2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit
 *  unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array (one for Y
 *        plane, one for UV plane).
 * \param rSrcStep \ref source_planar_image_line_step.  Same value is used for each plane.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiNV12ToBGR_8u_P2C3R_Ctx(const Mpp8u * const pSrc[2], int rSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 *  2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit
 *  unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array (one for Y
 *        plane, one for UV plane).
 * \param rSrcStep \ref source_planar_image_line_step.  Same value is used for each plane.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiNV12ToBGR_8u_P2C3R(const Mpp8u * const pSrc[2], int rSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 *  2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit
 *  unsigned packed RGB 709 HDTV full color conversion. Note that HDTV conversion assumes full color range of 0 - 255, use CSC version for limited range color.
 *
 * \param pSrc \ref source_planar_image_pointer_array (one for Y
 *        plane, one for UV plane).
 * \param rSrcStep \ref source_planar_image_line_step. Same value is used for each plane.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx(const Mpp8u * const pSrc[2], int rSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 *  2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit
 *  unsigned packed RGB 709 HDTV full color conversion. Note that HDTV conversion assumes full color range of 0 - 255, use CSC version for limited range color.
 *
 * \param pSrc \ref source_planar_image_pointer_array (one for Y
 *        plane, one for UV plane).
 * \param rSrcStep \ref source_planar_image_line_step. Same value is used for each plane.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiNV12ToBGR_709HDTV_8u_P2C3R(const Mpp8u * const pSrc[2], int rSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 *  2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit
 *  unsigned packed RGB 709 CSC color conversion. Note that HDTV conversion assumes full color range of 0 - 255, use CSC version for limited range color.
 *
 * \param pSrc \ref source_planar_image_pointer_array (one for Y
 *        plane, one for UV plane).
 * \param rSrcStep \ref source_planar_image_line_step. Same value is used for each plane.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiNV12ToBGR_709CSC_8u_P2C3R_Ctx(const Mpp8u * const pSrc[2], int rSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 *  2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit
 *  unsigned packed RGB 709 CSC color conversion. Note that HDTV conversion assumes full color range of 0 - 255, use CSC version for limited range color.
 *
 * \param pSrc \ref source_planar_image_pointer_array (one for Y
 *        plane, one for UV plane).
 * \param rSrcStep \ref source_planar_image_line_step. Same value is used for each plane.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiNV12ToBGR_709CSC_8u_P2C3R(const Mpp8u * const pSrc[2], int rSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} nv12tobgr */

/** @defgroup nv21tobgr NV21ToBGR 
 *  NV21 to BGR color conversion.
 *
 * @{
 *
 */

/**
 *  2 channel 8-bit unsigned planar NV21 to 4 channel 8-bit unsigned packed BGRA color conversion with constant alpha (0xFF).
 *
 * \param pSrc \ref source_planar_image_pointer_array (one for Y plane, one for VU plane).
 * \param rSrcStep \ref source_planar_image_line_step.  Same value is used for each plane.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiNV21ToBGR_8u_P2C4R_Ctx(const Mpp8u * const pSrc[2], int rSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 *  2 channel 8-bit unsigned planar NV21 to 4 channel 8-bit unsigned packed BGRA color conversion with constant alpha (0xFF).
 *
 * \param pSrc \ref source_planar_image_pointer_array (one for Y plane, one for VU plane).
 * \param rSrcStep \ref source_planar_image_line_step.  Same value is used for each plane.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiNV21ToBGR_8u_P2C4R(const Mpp8u * const pSrc[2], int rSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} nv21tobgr */

/** @defgroup rgbtoycbcr RGBToYCbCr 
 *  RGB to YCbCr color conversion.
 *
 *  Here is how MPP converts gamma corrected RGB or BGR to YCbCr.  In the YCbCr model, 
 *  Y is defined to have a nominal range [16..235], while Cb and Cr are defined
 *  to have a range [16..240], with the value of 128 as corresponding to zero.
 *
 *  \code
 *  Mpp32f nY  =  0.257F * R + 0.504F * G + 0.098F * B + 16.0F; 
 *  Mpp32f nCb = -0.148F * R - 0.291F * G + 0.439F * B + 128.0F;
 *  Mpp32f nCr =  0.439F * R - 0.368F * G - 0.071F * B + 128.0F;
 *  \endcode
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel unsigned 8-bit packed YCbCr color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiRGBToYCbCr_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel unsigned 8-bit packed YCbCr color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiRGBToYCbCr_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel unsigned 8-bit packed YCbCr with alpha color conversion, not affecting alpha.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel unsigned 8-bit packed YCbCr with alpha color conversion, not affecting alpha.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel planar 8-bit unsigned RGB to 3 channel planar 8-bit YCbCr color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel planar 8-bit unsigned RGB to 3 channel planar 8-bit YCbCr color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr_8u_P3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel unsigned 8-bit planar YCbCr color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel unsigned 8-bit planar YCbCr color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 3 channel 8-bit unsigned planar YCbCr color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr_8u_AC4P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 3 channel 8-bit unsigned planar YCbCr color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr_8u_AC4P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/** @} rgbtoycbcr */

/** 
 * @defgroup ycbcrtorgb YCbCr To RGB 
 *  YCbCr to RGB color conversion.
 *
 *  Here is how MPP converts YCbCr to gamma corrected RGB or BGR.  The output RGB values are saturated to the range [0..255].
 *
 *  \code
 *  Mpp32f nY = 1.164F * ((Mpp32f)Y - 16.0F);
 *  Mpp32f nR = ((Mpp32f)Cr - 128.0F); Mpp32f nB = ((Mpp32f)Cb
 *  - 128.0F); Mpp32f nG = nY - 0.813F * nR - 0.392F * nB; if (nG > 255.0F)
 *      nG = 255.0F;
 *  nR = nY + 1.596F * nR; 
 *  if (nR > 255.0F)
 *      nR = 255.0F;
 *  nB = nY + 2.017F * nB;
 *  if (nB > 255.0F)
 *      nB = 255.0F;
 *  \endcode
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned packed YCbCr to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToRGB_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed YCbCr to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToRGB_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed YCbCr with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion, not affecting alpha.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToRGB_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed YCbCr with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion, not affecting alpha.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToRGB_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned planar RGB color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToRGB_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned planar RGB color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToRGB_8u_P3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToRGB_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToRGB_8u_P3C3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr to 4 channel 8-bit unsigned packed RGB color conversion with constant alpha.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAval 8-bit unsigned alpha constant.                                         
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToRGB_8u_P3C4R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst , int nDstStep, MppiSize oSizeROI, Mpp8u nAval, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr to 4 channel 8-bit unsigned packed RGB color conversion with constant alpha.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAval 8-bit unsigned alpha constant.                                         
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToRGB_8u_P3C4R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst , int nDstStep, MppiSize oSizeROI, Mpp8u nAval);

/** @} ycbcrtorgb */

/** @defgroup ycbcrtorgbbatch YCbCrToRGBBatch
 *  YCbCr to RGB batch color conversion with a single \ref roi_specification for all pairs of input/output images provided in batches.
 *
 *  MPP converts YCbCr to gamma corrected RGB the same way as in \ref ycbcrtorgb.
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned packed YCbCr to 3 channel 8-bit unsigned packed RGB batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input and output images passed in pSrcBatchList and pSrcBatchList
 * arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the borders of any of provided images.
 *
 * \param pSrcBatchList \ref source_batch_images_pointer.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToRGBBatch_8u_C3R_Ctx(const MppiImageDescriptor* pSrcBatchList, MppiImageDescriptor* pDstBatchList, int nBatchSize, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed YCbCr to 3 channel 8-bit unsigned packed RGB batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input and output images passed in pSrcBatchList and pSrcBatchList
 * arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the borders of any of provided images.
 *
 * \param pSrcBatchList \ref source_batch_images_pointer.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToRGBBatch_8u_C3R(const MppiImageDescriptor* pSrcBatchList, MppiImageDescriptor* pDstBatchList, int nBatchSize, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed RGB batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToRGBBatch_8u_P3C3R_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed RGB batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToRGBBatch_8u_P3C3R(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI);

/** @} ycbcrtorgbbatch */

/** @defgroup ycbcrtorgbbatchadvanced YCbCrToRGBBatchAdvanced
 *  YCbCr to RGB batch color conversion where each pair of input/output images from provided batches has own \ref roi_specification.
 *
 *  MPP converts YCbCr to gamma corrected RGB the same way as in \ref ycbcrtorgb.
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned packed YCbCr to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair of input/output images has own ROI.
 * Provided oMaxSizeROI must contain the maximum width and the maximum height of all ROIs defined in pDstBatchList. API user must ensure that
 * ROI from pDstBatchList for each pair of input and output images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList \ref source_batch_images_pointer.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \param mppStreamCtx \ref application_managed_stream_context.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToRGBBatch_8u_C3R_Advanced_Ctx(const MppiImageDescriptor* pSrcBatchList, MppiImageDescriptor* pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed YCbCr to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair of input/output images has own ROI.
 * Provided oMaxSizeROI must contain the maximum width and the maximum height of all ROIs defined in pDstBatchList. API user must ensure that
 * ROI from pDstBatchList for each pair of input and output images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList \ref source_batch_images_pointer.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToRGBBatch_8u_C3R_Advanced(const MppiImageDescriptor* pSrcBatchList, MppiImageDescriptor* pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToRGBBatch_8u_P3C3R_Advanced_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToRGBBatch_8u_P3C3R_Advanced(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI);

/** @} ycbcrtorgbbatchadvanced */

/** @defgroup ycbcrtobgr YCbCrToBGR 
 *  YCbCr to BGR color conversion.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToBGR_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToBGR_8u_P3C3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAval 8-bit unsigned alpha constant.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToBGR_8u_P3C4R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nAval, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAval 8-bit unsigned alpha constant.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToBGR_8u_P3C4R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nAval);

/** @} ycbcrtobgr */

/** @defgroup ycbcrtobgrbatch YCbCrToBGRBatch
 *  YCbCr to BGR batch color conversion with a single \ref roi_specification for all pairs of input/output images provided in batches.
 *
 *  MPP converts YCbCr to gamma corrected BGR the same way as in \ref ycbcrtobgr.
 * @{
 *
 */
 
/**
 * 3 channel 8-bit unsigned packed YCbCr to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input and output images passed in pSrcBatchList and pSrcBatchList
 * arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the borders of any of provided images.
 *
 * \param pSrcBatchList \ref source_batch_images_pointer.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToBGRBatch_8u_C3R_Ctx(const MppiImageDescriptor* pSrcBatchList, MppiImageDescriptor* pDstBatchList, int nBatchSize, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed YCbCr to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input and output images passed in pSrcBatchList and pSrcBatchList
 * arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the borders of any of provided images.
 *
 * \param pSrcBatchList \ref source_batch_images_pointer.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToBGRBatch_8u_C3R(const MppiImageDescriptor* pSrcBatchList, MppiImageDescriptor* pDstBatchList, int nBatchSize, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this  call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToBGRBatch_8u_P3C3R_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI, MppStreamContext mppStreamCtx);
 
/**
 * 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this  call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToBGRBatch_8u_P3C3R(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI);

/** @} ycbcrtobgrbatch */

/** @defgroup ycbcrtobgrbatchadvanced YCbCrToBGRBatchAdvanced
 *  YCbCr to BGR batch color conversion where each pair of input/output images from provided batches has own \ref roi_specification.
 *
 *  MPP converts YCbCr to gamma corrected BGR the same way as in \ref ycbcrtobgr.
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned packed YCbCr to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair of input/output images has own ROI.
 * Provided oMaxSizeROI must contain the maximum width and the maximum height of all ROIs defined in pDstBatchList. API user must ensure that
 * ROI from pDstBatchList for each pair of input and output images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList \ref source_batch_images_pointer.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \param mppStreamCtx \ref application_managed_stream_context.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToBGRBatch_8u_C3R_Advanced_Ctx(const MppiImageDescriptor* pSrcBatchList, MppiImageDescriptor* pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed YCbCr to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair of input/output images has own ROI.
 * Provided oMaxSizeROI must contain the maximum width and the maximum height of all ROIs defined in pDstBatchList. API user must ensure that
 * ROI from pDstBatchList for each pair of input and output images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList \ref source_batch_images_pointer.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToBGRBatch_8u_C3R_Advanced(const MppiImageDescriptor* pSrcBatchList, MppiImageDescriptor* pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToBGRBatch_8u_P3C3R_Advanced_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToBGRBatch_8u_P3C3R_Advanced(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI);

/** @} ycbcrtobgrbatchadvanced */

/** @defgroup ycbcrtobgr709 YCbCrToBGR_709CSC 
 *  YCbCr to BGR_709CSC color conversion.
 *
 * @{
 *
 */
 
/**
 * 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed BGR_709CSC color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToBGR_709CSC_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed BGR_709CSC color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToBGR_709CSC_8u_P3C3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr to 4 channel 8-bit unsigned packed BGR_709CSC color conversion with constant alpha.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAval 8-bit unsigned alpha constant.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToBGR_709CSC_8u_P3C4R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nAval, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr to 4 channel 8-bit unsigned packed BGR_709CSC color conversion with constant alpha.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAval 8-bit unsigned alpha constant.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCrToBGR_709CSC_8u_P3C4R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nAval);

/** @} ycbcrtobgr709 */

/** 
 *  
 * @defgroup rgbToycbcr422 RGB To YCbCr422 
 *  RGB to YCbCr422 color conversion.
 *
 * @{
 *
 */


/**
 * 3 channel 8-bit unsigned packed RGB to 2 channel 8-bit unsigned packed YCbCr422 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr422_8u_C3C2R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 2 channel 8-bit unsigned packed YCbCr422 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr422_8u_C3C2R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr422_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr422_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar RGB to 2 channel 8-bit unsigned packed YCbCr422 color conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr422_8u_P3C2R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar RGB to 2 channel 8-bit unsigned packed YCbCr422 color conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr422_8u_P3C2R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} rgbtoycbcr422 */

/** @defgroup ycbcr422torgb YCbCr422 To RGB 
 *  YCbCr422 to RGB color conversion.
 *
 * @{
 *
 */
 
/**
 * 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned packed RGB color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToRGB_8u_C2C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx); 

/**
 * 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned packed RGB color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToRGB_8u_C2C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI); 

/**
 * 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar RGB color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToRGB_8u_C2P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar RGB color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToRGB_8u_C2P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed RGB color conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToRGB_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed RGB color conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToRGB_8u_P3C3R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} ycbcr422torgb */

/** @defgroup ycbcr422torgbbatch YCbCr422ToRGBBatch
 *  Planar YCbCr422 to packed RGB batch color conversion with a single \ref roi_specification for all pairs of input/output images provided in batches.
 *
 * @{
 *
 */
 
/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed RGB batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToRGBBatch_8u_P3C3R_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed RGB batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToRGBBatch_8u_P3C3R(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI);

/** @} ycbcr422torgbbatch */

/** @defgroup ycbcr422torgbbatchadvanced YCbCr422ToRGBBatchAdvanced
 *  Planar YCbCr422 to packed RGB batch color conversion where each pair of input/output images from provided batches has own \ref roi_specification.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToRGBBatch_8u_P3C3R_Advanced_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToRGBBatch_8u_P3C3R_Advanced(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI);

/** @} ycbcr422torgbbatchadvanced */

/** @defgroup rgbtoycrcb422 RGBToYCrCb422 
 *  RGB to YCrCb422 color conversion.
 *
 * @{
 *
 */
 
/**
 * 3 channel 8-bit unsigned packed RGB to 2 channel 8-bit unsigned packed YCrCb422 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCrCb422_8u_C3C2R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 2 channel 8-bit unsigned packed YCrCb422 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCrCb422_8u_C3C2R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar RGB to 2 channel 8-bit unsigned packed YCrCb422 color conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCrCb422_8u_P3C2R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar RGB to 2 channel 8-bit unsigned packed YCrCb422 color conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCrCb422_8u_P3C2R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} rgbtoycrcb422 */

/** @defgroup ycrcb422torgb YCrCb422ToRGB 
 *  YCrCb422 to RGB color conversion.
 *
 * @{
 *
 */

/**
 * 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned packed RGB color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCrCb422ToRGB_8u_C2C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx); 

/**
 * 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned packed RGB color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCrCb422ToRGB_8u_C2C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI); 

/**
 * 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar RGB color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCrCb422ToRGB_8u_C2P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar RGB color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCrCb422ToRGB_8u_C2P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/** @} ycrcb422torgb */

/** @defgroup bgrtoycbcr422 BGRToYCbCr422 
 *  BGR to YCbCr422 color conversion.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned packed BGR to 2 channel 8-bit unsigned packed YCrCb422 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr422_8u_C3C2R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed BGR to 2 channel 8-bit unsigned packed YCrCb422 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr422_8u_C3C2R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 2 channel 8-bit unsigned packed YCrCb422 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr422_8u_AC4C2R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 2 channel 8-bit unsigned packed YCrCb422 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr422_8u_AC4C2R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr422_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr422_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr422_8u_AC4P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr422_8u_AC4P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} bgrtoycbcr422 */

/** @defgroup ycbcr422tobgr YCbCr422ToBGR 
 *  YCbCr422 to BGR color conversion.
 *
 * @{
 *
 */

/**
 * 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned packed BGR color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToBGR_8u_C2C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned packed BGR color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToBGR_8u_C2C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 2 channel 8-bit unsigned packed YCrCb422 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAval 8-bit unsigned alpha constant.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToBGR_8u_C2C4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nAval, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed YCrCb422 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAval 8-bit unsigned alpha constant.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToBGR_8u_C2C4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nAval);

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed BGR color conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToBGR_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed BGR color conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToBGR_8u_P3C3R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} ycbcr422tobgr */

/** @defgroup ycbcr422tobgrbatch YCbCr422ToBGRBatch
 *  Planar YCbCr422 to packed BGR batch color conversion with a single \ref roi_specification for all pairs of input/output images provided in batches.
 *
 * @{
 *
 */
 
/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this  call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToBGRBatch_8u_P3C3R_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI, MppStreamContext mppStreamCtx);
 
/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this  call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToBGRBatch_8u_P3C3R(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI);

/** @} ycbcr422tobgrbatch */

/** @defgroup ycbcr422tobgrbatchadvanced YCbCr422ToBGRBatchAdvanced
 *  Planar YCbCr422 to packed BGR batch color conversion where each pair of input/output images from provided batches has own \ref roi_specification.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToBGRBatch_8u_P3C3R_Advanced_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToBGRBatch_8u_P3C3R_Advanced(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI);

/** @} ycbcr422tobgrbatchadvanced */

/** @defgroup rgbtocbycr422 RGBToCbYCr422 
 *  RGB to CbYCr422 color conversion.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned packed RGB to 2 channel 8-bit unsigned packed CbYCr422 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToCbYCr422_8u_C3C2R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 2 channel 8-bit unsigned packed CbYCr422 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToCbYCr422_8u_C3C2R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned packed RGB first gets forward gamma corrected then converted to 2 channel 8-bit unsigned packed CbYCr422 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToCbYCr422Gamma_8u_C3C2R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB first gets forward gamma corrected then converted to 2 channel 8-bit unsigned packed CbYCr422 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToCbYCr422Gamma_8u_C3C2R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} rgbtocbycr422 */

/** @defgroup cbcr422torgb CbYCr422ToRGB 
 *  CbYCr422 to RGB color conversion.
 *
 * @{
 *
 */

/**
 * 2 channel 8-bit unsigned packed CbYCrC22 to 3 channel 8-bit unsigned packed RGB color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCbYCr422ToRGB_8u_C2C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed CbYCrC22 to 3 channel 8-bit unsigned packed RGB color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCbYCr422ToRGB_8u_C2C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} cbcr422torgb */

/** @defgroup bgrtocbycr422 BGRToCbYCr422 
 *  BGR to CbYCr422 color conversion.
 *
 * @{
 *
 */

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 2 channel 8-bit unsigned packed CbYCr422 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToCbYCr422_8u_AC4C2R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 2 channel 8-bit unsigned packed CbYCr422 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToCbYCr422_8u_AC4C2R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} bgrtocbycr422 */

/** @defgroup bgrtocbycr422709HDTV BGRToCbYCr422_709HDTV 
 *  BGR to CbYCr422_709HDTV color conversion.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned packed BGR to 2 channel 8-bit unsigned packed CbYCr422_709HDTV color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToCbYCr422_709HDTV_8u_C3C2R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed BGR to 2 channel 8-bit unsigned packed CbYCr422_709HDTV color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToCbYCr422_709HDTV_8u_C3C2R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 2 channel 8-bit unsigned packed CbYCr422_709HDTV color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToCbYCr422_709HDTV_8u_AC4C2R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 2 channel 8-bit unsigned packed CbYCr422_709HDTV color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToCbYCr422_709HDTV_8u_AC4C2R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} bgrtocbycr422709HDTV */

/** @defgroup cbycr422tobgr CbYCr422ToBGR 
 *  CbYCr422 to BGR color conversion.
 *
 * @{
 *
 */

/**
 * 2 channel 8-bit unsigned packed CbYCr422 to 4 channel 8-bit unsigned packed BGR color conversion with alpha.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAval 8-bit unsigned alpha constant.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCbYCr422ToBGR_8u_C2C4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nAval, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed CbYCr422 to 4 channel 8-bit unsigned packed BGR color conversion with alpha.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAval 8-bit unsigned alpha constant.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCbYCr422ToBGR_8u_C2C4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nAval);

/** @} cbycr422tobgr */

/** @defgroup cbycr422tobgr709hdtv CbYCr422ToBGR_709HDTV 
 *  CbYCr422 to BGR_709HDTV color conversion.
 *
 * @{
 *
 */

/**
 * 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned packed BGR_709HDTV color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCbYCr422ToBGR_709HDTV_8u_C2C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned packed BGR_709HDTV color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCbYCr422ToBGR_709HDTV_8u_C2C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 2 channel 8-bit unsigned packed CbYCr422 to 4 channel 8-bit unsigned packed BGR_709HDTV color conversion with constant alpha.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAval 8-bit unsigned alpha constant.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCbYCr422ToBGR_709HDTV_8u_C2C4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nAval, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed CbYCr422 to 4 channel 8-bit unsigned packed BGR_709HDTV color conversion with constant alpha.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAval 8-bit unsigned alpha constant.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCbYCr422ToBGR_709HDTV_8u_C2C4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nAval);

/** @} cbycr422tobgr709hdtv */

/** @defgroup rgbtoycbcr420 RGBToYCbCr420 
 *  RGB to YCbCr420 color conversion.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr420_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr420_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} rgbtoycbcr420 */

/** @defgroup ycbcr420torgb YCbCr420ToRGB 
 *  YCbCr420 to RGB color conversion.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToRGB_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToRGB_8u_P3C3R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} ycbcr420torgb */

/** @defgroup ycbcr420torgbbatch YCbCr420ToRGBBatch
 *  Planar YCbCr420 to packed RGB batch color conversion with a single \ref roi_specification for all pairs of input/output images provided in batches.
 *
 * @{
 *
 */
 
/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed RGB batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToRGBBatch_8u_P3C3R_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed RGB batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToRGBBatch_8u_P3C3R(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI);

/** @} ycbcr420torgbbatch */

/** @defgroup ycbcr420torgbbatchadvanced YCbCr420ToRGBBatchAdvanced
 *  Planar YCbCr420 to packed RGB batch color conversion where each pair of input/output images from provided batches has own \ref roi_specification.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToRGBBatch_8u_P3C3R_Advanced_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToRGBBatch_8u_P3C3R_Advanced(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI);

/** @} ycbcr420torgbbatchadvanced */

/** @defgroup rgbtoycrcb420 RGBToYCrCb420 
 *  RGB to YCrCb420 color conversion.
 *
 * @{
 *
 */

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 3 channel 8-bit unsigned planar YCrCb420 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCrCb420_8u_AC4P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 3 channel 8-bit unsigned planar YCrCb420 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCrCb420_8u_AC4P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} rgbtoycrcb420 */

/** @defgroup ycrcb420torgb YCrCb420ToRGB 
 *  YCrCb420 to RGB color conversion.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned planar YCrCb420 to 4 channel 8-bit unsigned packed RGB color conversion with constant alpha.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAval 8-bit unsigned alpha constant.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCrCb420ToRGB_8u_P3C4R_Ctx(const Mpp8u * const pSrc[3],int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nAval, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCrCb420 to 4 channel 8-bit unsigned packed RGB color conversion with constant alpha.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAval 8-bit unsigned alpha constant.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCrCb420ToRGB_8u_P3C4R(const Mpp8u * const pSrc[3],int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nAval);

/** @} ycrcb420torgb */

/** @defgroup bgrtoycbcr420 BGRToYCbCr420 
 *  BGR to YCbCr420 color conversion.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr420_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr420_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr420_8u_AC4P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr420_8u_AC4P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} bgrtoycbcr420 */

/** @defgroup bgrtoycbcr420709csc BGRToYCbCr420_709CSC 
 *  BGR to YCbCr420_709CSC color conversion.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr420_709CSC color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr420_709CSC_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr420_709CSC color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr420_709CSC_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr420_709CSC color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr420_709CSC_8u_AC4P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr420_709CSC color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr420_709CSC_8u_AC4P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} bgrtoycbcr420709csc */

/** @defgroup bgrtoycbcr420709hdtv BGRToYCbCr420_709HDTV 
 *  BGR to YCbCr420_709HDTV color conversion.
 *
 * @{
 *
 */

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr420_709HDTV color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr420_709HDTV_8u_AC4P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr420_709HDTV color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr420_709HDTV_8u_AC4P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} bgrtoycbcr420709hdtv */

/** @defgroup bgrtoycrdb420709csc BGRToYCrCb420_709CSC 
 *  BGR to YCrCb420_709CSC color conversion.
 * @{
 */

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCrCb420_709CSC color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCrCb420_709CSC_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCrCb420_709CSC color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCrCb420_709CSC_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCrCb420_709CSC color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCrCb420_709CSC_8u_AC4P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCrCb420_709CSC color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCrCb420_709CSC_8u_AC4P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} bgrtoycrdb420709csc */

/** @defgroup ycbcr420tobgr YCbCr420ToBGR 
 *  YCbCr420 to BGR color conversion.
 * @{
 */
/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToBGR_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToBGR_8u_P3C3R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAval 8-bit unsigned alpha constant.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToBGR_8u_P3C4R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3],Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nAval, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAval 8-bit unsigned alpha constant.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToBGR_8u_P3C4R(const Mpp8u * const pSrc[3], int rSrcStep[3],Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nAval);

/** @} ycbcr420tobgr */

/** @defgroup ycbcr420tobgrbatch YCbCr420ToBGRBatch
 *  Planar YCbCr420 to packed BGR batch color conversion with a single \ref roi_specification for all pairs of input/output images provided in batches.
 *
 * @{
 *
 */
 
/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this  call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToBGRBatch_8u_P3C3R_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI, MppStreamContext mppStreamCtx);
 
/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
 * Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
 * pSrcBatchList and pSrcBatchList arguments. API user must ensure that provided ROI (oSizeROI) does not go beyond the
 * borders of any of provided images.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize A number of \ref MppiImageDescriptor structures processed in this  call (must be > 1).
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToBGRBatch_8u_P3C3R(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oSizeROI);

/** @} ycbcr420tobgrbatch */

/** @defgroup ycbcr420tobgrbatchadvanced YCbCr420ToBGRBatchAdvanced
 *  Planar YCbCr420 to packed BGR batch color conversion where each pair of input/output images from provided batches has own \ref roi_specification.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToBGRBatch_8u_P3C3R_Advanced_Ctx(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair
 * of input/output images has own ROI. Provided oMaxSizeROI must contain the maximum width and the maximum height of all
 * ROIs defined in pDstBatchList. API user must ensure that ROI from pDstBatchList for each pair of input and output
 * images does not go beyond the borders of images in each pair.
 *
 * \param pSrcBatchList An array where each element is a batch of images representing one of planes in planar images,
 *        \ref source_batch_images_pointer. The first element of array (pSrcBatchList[0]) represents a batch of Y planes.
 *        The second element of array (pSrcBatchList[1]) represents a batch of Cb planes. The third element of array
 *        (pSrcBatchList[2]) represents a batch of Cr planes.
 * \param pDstBatchList \ref destination_batch_images_pointer.
 * \param nBatchSize Number of \ref MppiImageDescriptor structures processed in this call (must be > 1).
 * \param oMaxSizeROI \ref roi_specification, must contain the maximum width and the maximum height from all destination ROIs used for processing data.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToBGRBatch_8u_P3C3R_Advanced(const MppiImageDescriptor * const pSrcBatchList[3], MppiImageDescriptor * pDstBatchList, int nBatchSize, MppiSize oMaxSizeROI);

/** @} ycbcr420tobgrbatchadvanced */

/** @defgroup ycbcr420tobgr709csc YCbCr420ToBGR_709CSC 
 *  YCbCr420_709CSC to BGR color conversion.
 * @{
 */
/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed BGR_709CSC color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToBGR_709CSC_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed BGR_709CSC color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToBGR_709CSC_8u_P3C3R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} ycbcr420tobgr709csc */

/** @defgroup ycbcr420tobgr709hdtv YCbCr420ToBGR_709HDTV 
 *  YCbCr420_709HDTV to BGR color conversion.
 * @{
 */
/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 4 channel 8-bit unsigned packed BGR_709HDTV color conversion with constant alpha.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAval 8-bit unsigned alpha constant.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToBGR_709HDTV_8u_P3C4R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nAval, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 4 channel 8-bit unsigned packed BGR_709HDTV color conversion with constant alpha.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAval 8-bit unsigned alpha constant.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToBGR_709HDTV_8u_P3C4R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nAval);

/** @} ycbcr420tobgr709hdtv */

/** @defgroup bgrtoycrcb420 BGRToYCrCb420 
 *  BGR to YCrCb420 color conversion.
 * @{
 */
/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCrCb420 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCrCb420_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCrCb420 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCrCb420_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCrCb420 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCrCb420_8u_AC4P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCrCb420 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCrCb420_8u_AC4P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} bgrtoycrcb420 */

/** @defgroup bgrtoycbcr411 BGRToYCbCr411 
 *  BGR to YCbCr411 color conversion.
 * @{
 */
/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr411 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr411_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr411 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr411_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr411 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr411_8u_AC4P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr411 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr411_8u_AC4P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} bgrtoycbcr411 */

/** @defgroup rgbtoycbcr411 RGBToYCbCr411 
 *  RGB to YCbCr411 color conversion.
 * @{
 */
/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr411 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr411_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr411 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr411_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 3 channel 8-bit unsigned planar YCbCr411 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr411_8u_AC4P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 3 channel 8-bit unsigned planar YCbCr411 color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr411_8u_AC4P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} rgbtoycbcr411 */

/** @defgroup bgrtoycbcr BGRToYCbCr 
 *  BGR to YCbCr color conversion.
 * @{
 */
/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr_8u_AC4P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr_8u_AC4P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 4 channel 8-bit unsigned planar YCbCr color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr_8u_AC4P4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[4], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 4 channel 8-bit unsigned planar YCbCr color conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr_8u_AC4P4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[4], int nDstStep, MppiSize oSizeROI);

/** @} bgrtoycbcr */

/** @defgroup ycbcr411tobgr YCbCr411ToBGR 
 *  YCbCr411 to BGR color conversion.
 * @{
 */
/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToBGR_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToBGR_8u_P3C3R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAval 8-bit unsigned alpha constant.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToBGR_8u_P3C4R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nAval, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAval 8-bit unsigned alpha constant.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToBGR_8u_P3C4R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nAval);

/** @} ycbcr411tobgr */

/** @defgroup ycbcr411torgb YCbCr411ToRGB 
 *  YCbCr411 to RGB color conversion.
 * @{
 */
/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToRGB_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToRGB_8u_P3C3R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 4 channel 8-bit unsigned packed RGB color conversion with constant alpha.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAval 8-bit unsigned alpha constant.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToRGB_8u_P3C4R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nAval, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 4 channel 8-bit unsigned packed RGB color conversion with constant alpha.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAval 8-bit unsigned alpha constant.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToRGB_8u_P3C4R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nAval);

/** @} ycbcr411torgb */

/** @defgroup rgbtoxyz RGBToXYZ 
 *  RGB to XYZ color conversion.
 *
 *  Here is how MPP converts gamma corrected RGB or BGR to XYZ.
 *
 *  \code
 *  Mpp32f nNormalizedR = (Mpp32f)R * 0.003921569F; // / 255.0F
 *  Mpp32f nNormalizedG = (Mpp32f)G * 0.003921569F;
 *  Mpp32f nNormalizedB = (Mpp32f)B * 0.003921569F;
 *  Mpp32f nX = 0.412453F * nNormalizedR + 0.35758F  * nNormalizedG + 0.180423F * nNormalizedB; 
 *  if (nX > 1.0F)
 *      nX = 1.0F;
 *  Mpp32f nY = 0.212671F * nNormalizedR + 0.71516F  * nNormalizedG + 0.072169F * nNormalizedB;
 *  if (nY > 1.0F)
 *      nY = 1.0F;
 *  Mpp32f nZ = 0.019334F * nNormalizedR + 0.119193F * nNormalizedG + 0.950227F * nNormalizedB;
 *  if (nZ > 1.0F)
 *      nZ = 1.0F;
 *  X = (Mpp8u)(nX * 255.0F);
 *  Y = (Mpp8u)(nY * 255.0F);
 *  Z = (Mpp8u)(nZ * 255.0F);
 *  \endcode
 *
 * @{
 */
/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed XYZ color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToXYZ_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed XYZ color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToXYZ_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed XYZ with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToXYZ_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed XYZ with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToXYZ_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} rgbtoxyz */

/** @defgroup xyztorgb XYZToRGB 
 *  XYZ to RGB color conversion.
 *
 *  Here is how MPP converts XYZ to gamma corrected RGB or BGR.  The code assumes that X,Y, and Z values are in the range [0..1].
 *
 *  \code
 *  Mpp32f nNormalizedX = (Mpp32f)X * 0.003921569F; // / 255.0F
 *  Mpp32f nNormalizedY = (Mpp32f)Y * 0.003921569F;
 *  Mpp32f nNormalizedZ = (Mpp32f)Z * 0.003921569F;
 *  Mpp32f nR = 3.240479F * nNormalizedX - 1.53715F  * nNormalizedY - 0.498535F * nNormalizedZ; 
 *  if (nR > 1.0F)
 *      nR = 1.0F;
 *  Mpp32f nG = -0.969256F * nNormalizedX + 1.875991F  * nNormalizedY + 0.041556F * nNormalizedZ;
 *  if (nG > 1.0F)
 *      nG = 1.0F;
 *  Mpp32f nB = 0.055648F * nNormalizedX - 0.204043F * nNormalizedY + 1.057311F * nNormalizedZ;
 *  if (nB > 1.0F)
 *      nB = 1.0F;
 *  R = (Mpp8u)(nR * 255.0F);
 *  G = (Mpp8u)(nG * 255.0F);
 *  B = (Mpp8u)(nB * 255.0F);
 *  \endcode
 *
 * @{
 */
/**
 * 3 channel 8-bit unsigned packed XYZ to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiXYZToRGB_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed XYZ to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiXYZToRGB_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed XYZ with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiXYZToRGB_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed XYZ with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiXYZToRGB_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} xyztorgb */

/** @defgroup rgbtoluv RGBToLUV 
 *  RGB to LUV color conversion.
 *
 *  Here is how MPP converts gamma corrected RGB or BGR to CIE LUV using the CIE XYZ D65 white point with a Y luminance of 1.0.
 *  The computed values of the L component are in the range [0..100], U component in the range [-134..220], 
 *  and V component in the range [-140..122]. The code uses cbrtf() the 32 bit floating point cube root math function.
 *
 *  \code
 *  // use CIE D65 chromaticity coordinates
 *  #define nCIE_XYZ_D65_xn 0.312713F
 *  #define nCIE_XYZ_D65_yn 0.329016F
 *  #define nn_DIVISOR (-2.0F * nCIE_XYZ_D65_xn + 12.0F * nCIE_XYZ_D65_yn + 3.0F)
 *  #define nun (4.0F * nCIE_XYZ_D65_xn / nn_DIVISOR)
 *  #define nvn (9.0F * nCIE_XYZ_D65_yn / nn_DIVISOR)
 *    
 *  // First convert to XYZ
 *  Mpp32f nNormalizedR = (Mpp32f)R * 0.003921569F; // / 255.0F
 *  Mpp32f nNormalizedG = (Mpp32f)G * 0.003921569F;
 *  Mpp32f nNormalizedB = (Mpp32f)B * 0.003921569F;
 *  Mpp32f nX = 0.412453F * nNormalizedR + 0.35758F  * nNormalizedG + 0.180423F * nNormalizedB; 
 *  Mpp32f nY = 0.212671F * nNormalizedR + 0.71516F  * nNormalizedG + 0.072169F * nNormalizedB;
 *  Mpp32f nZ = 0.019334F * nNormalizedR + 0.119193F * nNormalizedG + 0.950227F * nNormalizedB;
 *  // Now calculate LUV from the XYZ value
 *  Mpp32f nTemp = nX + 15.0F * nY + 3.0F * nZ;
 *  Mpp32f nu = 4.0F * nX / nTemp;
 *  Mpp32f nv = 9.0F * nY / nTemp;
 *  Mpp32f nL = 116.0F * cbrtf(nY) - 16.0F;
 *  if (nL < 0.0F)
 *      nL = 0.0F;
 *  if (nL > 100.0F)
 *      nL = 100.0F;
 *  nTemp = 13.0F * nL;
 *  Mpp32f nU = nTemp * (nu - nun);
 *  if (nU < -134.0F)
 *      nU = -134.0F;
 *  if (nU > 220.0F)
 *      nU = 220.0F;
 *  Mpp32f nV = nTemp * (nv - nvn);
 *  if (nV < -140.0F)
 *      nV = -140.0F;
 *  if (nV > 122.0F)
 *      nV = 122.0F;
 *  L = (Mpp8u)(nL * 255.0F * 0.01F); // / 100.0F
 *  U = (Mpp8u)((nU + 134.0F) * 255.0F * 0.0028249F); // / 354.0F
 *  V = (Mpp8u)((nV + 140.0F) * 255.0F * 0.0038168F); // / 262.0F
 *  \endcode
 *
 * @{
 */
/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed LUV color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToLUV_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed LUV color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToLUV_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed LUV with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToLUV_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed LUV with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToLUV_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} rgbtoluv */

/** @defgroup luvtorgb LUVToRGB 
 *  LUV to RGB color conversion.
 *
 *  Here is how MPP converts CIE LUV to gamma corrected RGB or BGR using the CIE XYZ D65 white point with a Y luminance of 1.0.
 *  The code uses powf() the 32 bit floating point power math function. 
 *
 *  \code
 *  // use CIE D65 chromaticity coordinates
 *  #define nCIE_XYZ_D65_xn 0.312713F
 *  #define nCIE_XYZ_D65_yn 0.329016F
 *  #define nn_DIVISOR (-2.0F * nCIE_XYZ_D65_xn + 12.0F * nCIE_XYZ_D65_yn + 3.0F)
 *  #define nun (4.0F * nCIE_XYZ_D65_xn / nn_DIVISOR)
 *  #define nvn (9.0F * nCIE_XYZ_D65_yn / nn_DIVISOR)
 *    
 *  // First convert normalized LUV back to original CIE LUV range
 *  Mpp32f nL = (Mpp32f)L * 100.0F * 0.003921569F;  // / 255.0F
 *  Mpp32f nU = ((Mpp32f)U * 354.0F * 0.003921569F) - 134.0F;
 *  Mpp32f nV = ((Mpp32f)V * 262.0F * 0.003921569F) - 140.0F;
 *  // Now convert LUV to CIE XYZ
 *  Mpp32f nTemp = 13.0F * nL;
 *  Mpp32f nu = nU / nTemp + nun;
 *  Mpp32f nv = nV / nTemp + nvn;
 *  Mpp32f nNormalizedY;
 *  if (nL > 7.9996248F)
 *  {
 *      nNormalizedY = (nL + 16.0F) * 0.008621F; // / 116.0F
 *      nNormalizedY = powf(nNormalizedY, 3.0F);
 *  }
 *  else
 *  {    
 *      nNormalizedY = nL * 0.001107F; // / 903.3F
 *  }    
 *  Mpp32f nNormalizedX = (-9.0F * nNormalizedY * nu) / ((nu - 4.0F) * nv - nu * nv);
 *  Mpp32f nNormalizedZ = (9.0F * nNormalizedY - 15.0F * nv * nNormalizedY - nv * nNormalizedX) / (3.0F * nv);
 *  Mpp32f nR = 3.240479F * nNormalizedX - 1.53715F  * nNormalizedY - 0.498535F * nNormalizedZ; 
 *  if (nR > 1.0F)
 *      nR = 1.0F;
 *  if (nR < 0.0F)
 *      nR = 0.0F;
 *  Mpp32f nG = -0.969256F * nNormalizedX + 1.875991F  * nNormalizedY + 0.041556F * nNormalizedZ;
 *  if (nG > 1.0F)
 *      nG = 1.0F;
 *  if (nG < 0.0F)
 *      nG = 0.0F;
 *  Mpp32f nB = 0.055648F * nNormalizedX - 0.204043F * nNormalizedY + 1.057311F * nNormalizedZ;
 *  if (nB > 1.0F)
 *      nB = 1.0F;
 *  if (nB < 0.0F)
 *      nB = 0.0F;
 *  R = (Mpp8u)(nR * 255.0F);
 *  G = (Mpp8u)(nG * 255.0F);
 *  B = (Mpp8u)(nB * 255.0F);
 *  \endcode
 *
 * @{
 */
/**
 * 3 channel 8-bit unsigned packed LUV to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiLUVToRGB_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed LUV to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiLUVToRGB_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed LUV with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiLUVToRGB_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed LUV with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiLUVToRGB_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} luvtorgb */

/** @defgroup bgrtolab BGRToLab 
 *  BGR to Lab color conversion.
 *
 *  This is how MPP converts gamma corrected BGR or RGB to Lab using the CIE Lab D65 white point with a Y luminance of 1.0.
 *  The computed values of the L component are in the range [0..100], a and b component values are in the range [-128..127].
 *  The code uses cbrtf() the 32 bit floating point cube root math function.
 *
 *  \code
 *  // use CIE Lab chromaticity coordinates
 *  #define nCIE_LAB_D65_xn 0.950455F
 *  #define nCIE_LAB_D65_yn 1.0F
 *  #define nCIE_LAB_D65_zn 1.088753F
 *  // First convert to XYZ
 *  Mpp32f nNormalizedR = (Mpp32f)R * 0.003921569F; // / 255.0F
 *  Mpp32f nNormalizedG = (Mpp32f)G * 0.003921569F;
 *  Mpp32f nNormalizedB = (Mpp32f)B * 0.003921569F;
 *  Mpp32f nX = 0.412453F * nNormalizedR + 0.35758F  * nNormalizedG + 0.180423F * nNormalizedB; 
 *  Mpp32f nY = 0.212671F * nNormalizedR + 0.71516F  * nNormalizedG + 0.072169F * nNormalizedB;
 *  Mpp32f nZ = 0.019334F * nNormalizedR + 0.119193F * nNormalizedG + 0.950227F * nNormalizedB;
 *  Mpp32f nL = cbrtf(nY);
 *  Mpp32f nA; Mpp32f nB; Mpp32f nfX = nX * 1.052128F; // / nCIE_LAB_D65_xn; Mpp32f nfY = nY; Mpp32f nfZ = nZ * 0.918482F; // /
 *  nCIE_LAB_D65_zn; nfY = nL - 16.0F; nL = 116.0F * nL - 16.0F; nA = cbrtf(nfX) - 16.0F; nA = 500.0F
 *  * (nA - nfY); nB = cbrtf(nfZ) - 16.0F; nB = 200.0F * (nfY - nB); // Now scale Lab range nL = nL * 255.0F
 *  * 0.01F; // / 100.0F nA = nA + 128.0F; nB = nB + 128.0F; L = (Mpp8u)nL; a = (Mpp8u)nA; b = (Mpp8u)nB;
 *  \endcode
 * 
 * @{
 */
/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned packed Lab color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToLab_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned packed Lab color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToLab_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} bgrtolab */

/** @defgroup labtobgr LabToBGR 
 *  Lab to BGR color conversion.
 *
 *  This is how MPP converts Lab to gamma corrected BGR or RGB using the CIE Lab D65 white point with a Y luminance of 1.0.
 *  The code uses powf() the 32 bit floating point power math function. 
 *
 *  \code
 *  // use CIE Lab chromaticity coordinates
 *  #define nCIE_LAB_D65_xn 0.950455F
 *  #define nCIE_LAB_D65_yn 1.0F
 *  #define nCIE_LAB_D65_zn 1.088753F
 *  // First convert Lab back to original range then to CIE XYZ
 *  Mpp32f nL = (Mpp32f)L * 100.0F * 0.003921569F;  // / 255.0F
 *  Mpp32f nA = (Mpp32f)a - 128.0F;
 *  Mpp32f nB = (Mpp32f)b - 128.0F;
 *  Mpp32f nP = (nL + 16.0F) * 0.008621F; // / 116.0F
 *  Mpp32f nNormalizedY = nP * nP * nP; // powf(nP, 3.0F);
 *  Mpp32f nNormalizedX = nCIE_LAB_D65_xn * powf((nP + nA * 0.002F), 3.0F); // / 500.0F
 *  Mpp32f nNormalizedZ = nCIE_LAB_D65_zn * powf((nP - nB * 0.005F), 3.0F); // / 200.0F
 *  Mpp32f nR = 3.240479F * nNormalizedX - 1.53715F  * nNormalizedY - 0.498535F * nNormalizedZ; 
 *  if (nR > 1.0F)
 *      nR = 1.0F;
 *  Mpp32f nG = -0.969256F * nNormalizedX + 1.875991F  * nNormalizedY + 0.041556F * nNormalizedZ;
 *  if (nG > 1.0F)
 *      nG = 1.0F;
 *  nB = 0.055648F * nNormalizedX - 0.204043F * nNormalizedY + 1.057311F * nNormalizedZ;
 *  if (nB > 1.0F)
 *      nB = 1.0F;
 *  R = (Mpp8u)(nR * 255.0F);
 *  G = (Mpp8u)(nG * 255.0F);
 *  B = (Mpp8u)(nB * 255.0F);
 *  \endcode
 *
 * @{
 */
/**
 * 3 channel 8-bit unsigned packed Lab to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiLabToBGR_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed Lab to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiLabToBGR_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} labtobgr */

/** @defgroup rgbtoycc RGBToYCC 
 *  RGB to PhotoYCC color conversion.
 *
 *  This is how MPP converts gamma corrected BGR or RGB to PhotoYCC.
 *  The computed Y, C1, C2 values are then quantized and converted to fit in the range [0..1] before expanding to 8 bits.
 *
 *  \code
 *  Mpp32f nNormalizedR = (Mpp32f)R * 0.003921569F; // / 255.0F
 *  Mpp32f nNormalizedG = (Mpp32f)G * 0.003921569F;
 *  Mpp32f nNormalizedB = (Mpp32f)B * 0.003921569F;
 *  Mpp32f nY = 0.299F * nNormalizedR + 0.587F  * nNormalizedG + 0.114F * nNormalizedB; 
 *  Mpp32f nC1 = nNormalizedB - nY;
 *  nC1 = 111.4F * 0.003921569F * nC1 + 156.0F * 0.003921569F;
 *  Mpp32f nC2 = nNormalizedR - nY;
 *  nC2 = 135.64F * 0.003921569F * nC2 + 137.0F * 0.003921569F;
 *  nY = 1.0F * 0.713267F * nY; // / 1.402F
 *  Y = (Mpp8u)(nY  * 255.0F);
 *  C1 = (Mpp8u)(nC1 * 255.0F);
 *  C2 = (Mpp8u)(nC2 * 255.0F);
 *  \endcode
 * 
 * @{
 */
/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed YCC color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCC_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed YCC color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCC_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed YCC with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCC_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed YCC with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCC_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} rgbtoycc */

/** @defgroup ycctorgb YCCToRGB 
 *  PhotoYCC to RGB color conversion.
 *
 *  This is how MPP converts PhotoYCC to gamma corrected RGB or BGR.
 *
 *  \code
 *  Mpp32f nNormalizedY  = ((Mpp32f)Y * 0.003921569F) * 1.3584F;  // / 255.0F
 *  Mpp32f nNormalizedC1 = (((Mpp32f)C1 * 0.003921569F) - 156.0F * 0.003921569F) * 2.2179F;
 *  Mpp32f nNormalizedC2 = (((Mpp32f)C2 * 0.003921569F) - 137.0F * 0.003921569F) * 1.8215F;
 *  Mpp32f nR = nNormalizedY + nNormalizedC2;
 *  if (nR > 1.0F)
 *      nR = 1.0F;
 *  Mpp32f nG = nNormalizedY - 0.194F  * nNormalizedC1 - 0.509F * nNormalizedC2;
 *  if (nG > 1.0F)
 *      nG = 1.0F;
 *  Mpp32f nB = nNormalizedY + nNormalizedC1;
 *  if (nB > 1.0F)
 *      nB = 1.0F;
 *  R = (Mpp8u)(nR * 255.0F);
 *  G = (Mpp8u)(nG * 255.0F);
 *  B = (Mpp8u)(nB * 255.0F);
 *  \endcode
 *
 * @{
 */
/**
 * 3 channel 8-bit unsigned packed YCC to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCCToRGB_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed YCC to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCCToRGB_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed YCC with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCCToRGB_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed YCC with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCCToRGB_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} ycctorgb */

/** @defgroup yccktocmyk601 YCCKToCMYK_JPEG 
 *  This function partially converts JPEG YCCK to CMYK.
 *
 *  This is how MPP converts JPEG YCCK to CMYK. MPP only performs and initial YCC to RGB conversion using the 601 conversion coefficients
 *  and the RGB to CMY inversion leaving K unmodified.  To complete this conversion to useful RGB values an additional RGB conversion needs
 *  to follow this function using the color profile contained in the YCCK JPEG file metadata section.  MPP does not directly support
 *  this conversion but potentially mppiColorTwist can be used to perform it once the conversion coefficients are known.
 *
 *  \code
 *  Mpp32f nY  = static_cast<Mpp32f>(Y);
 *  Mpp32f nC1 = static_cast<Mpp32f>(Cb);
 *  Mpp32f nC2 = static_cast<Mpp32f>(Cr);
 *  Mpp32f nR = nY + 1.402F * nC2 - 179.456F;
 *  Mpp32f nG = nY - 0.34414F * nC1 - 0.71414F * nC2 + 135.45984F;
 *  Mpp32f nB = nY + 1.772F * nC1 - 226.816F;
 *
 *  Mpp8u nC = static_cast<Mpp8u>(255.0F - nR);
 *  Mpp8u nM = static_cast<Mpp8u>(255.0F - nG);
 *  Mpp8u nM = static_cast<Mpp8u>(255.0F - nB);
 *  Mpp8u nK = K;
 *  \endcode
 *
 * @{
 */
/**
 * 4 channel 8-bit unsigned planar JPEG YCCK color format to 4 channel 8-bit unsigned planar CMYK color conversion using 601 RGB color coefficients and CMY inversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCCKToCMYK_JPEG_601_8u_P4R_Ctx(const Mpp8u * pSrc[4], int nSrcStep, Mpp8u * pDst[4], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned planar JPEG YCCK color format to 4 channel 8-bit unsigned planar CMYK color conversion using 601 RGB color coefficients and CMY inversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCCKToCMYK_JPEG_601_8u_P4R(const Mpp8u * pSrc[4], int nSrcStep, Mpp8u * pDst[4], int nDstStep, MppiSize oSizeROI);

/** @} yccktocmyk601 */

/** @defgroup ycckorcmyktorgb CMYK_OR_YCCK_JPEG_TO_RGB 
 *  These functions convert JPEG CMYK or YCCK color format images to either planar or packed RGB images for images which need no color profile conversion.
 *
 * @{
 */

/**
 * 4 channel 8-bit unsigned planar JPEG CMYK or YCCK color model image to 3 channel 8-bit unsigned planar RGB color model image without color profile conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */

MppStatus mppiCMYKOrYCCKToRGB_JPEG_8u_P4P3R_Ctx(const Mpp8u * pSrc[4], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned planar JPEG CMYK or YCCK color model image to 3 channel 8-bit unsigned planar RGB color model image without color profile conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCMYKOrYCCKToRGB_JPEG_8u_P4P3R(const Mpp8u * pSrc[4], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned planar JPEG CMYK or YCCK color model image to 3 channel 8-bit unsigned packed RGB color model image without color profile conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */

MppStatus mppiCMYKOrYCCKToRGB_JPEG_8u_P4C3R_Ctx(const Mpp8u * pSrc[4], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned planar JPEG CMYK or YCCK color model image to 3 channel 8-bit unsigned packed RGB color model image without color profile conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCMYKOrYCCKToRGB_JPEG_8u_P4C3R(const Mpp8u * pSrc[4], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} ycckorcmyktorgb */

/** @defgroup ycckorcmyktobgr CMYK_OR_YCCK_JPEG_TO_BGR 
 *  These functions convert JPEG CMYK or YCCK color format images to either planar or packed BGR images for images which need no color profile conversion.
 *
 * @{
 */

/**
 * 4 channel 8-bit unsigned planar JPEG CMYK or YCCK color model image to 3 channel 8-bit unsigned planar BGR color model image without color profile conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */

MppStatus mppiCMYKOrYCCKToBGR_JPEG_8u_P4P3R_Ctx(const Mpp8u * pSrc[4], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned planar JPEG CMYK or YCCK color model image to 3 channel 8-bit unsigned planar BGR color model image without color profile conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCMYKOrYCCKToBGR_JPEG_8u_P4P3R(const Mpp8u * pSrc[4], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned planar JPEG CMYK or YCCK color model image to 3 channel 8-bit unsigned packed BGR color model image without color profile conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */

MppStatus mppiCMYKOrYCCKToBGR_JPEG_8u_P4C3R_Ctx(const Mpp8u * pSrc[4], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned planar JPEG CMYK or YCCK color model image to 3 channel 8-bit unsigned packed BGR color model image without color profile conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCMYKOrYCCKToBGR_JPEG_8u_P4C3R(const Mpp8u * pSrc[4], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} ycckorcmyktobgr */

/** @defgroup rgbtohls RGBToHLS 
 *  RGB to HLS color conversion.
 *
 *  This is how MPP converts gamma corrected RGB or BGR to HLS.
 *  This code uses the fmaxf() and fminf() 32 bit floating point math functions.
 *
 *  \code
 *  Mpp32f nNormalizedR = (Mpp32f)R * 0.003921569F;  // / 255.0F
 *  Mpp32f nNormalizedG = (Mpp32f)G * 0.003921569F;
 *  Mpp32f nNormalizedB = (Mpp32f)B * 0.003921569F;
 *  Mpp32f nS;
 *  Mpp32f nH;
 *  // Lightness
 *  Mpp32f nMax = fmaxf(nNormalizedR, nNormalizedG);
 *         nMax = fmaxf(nMax, nNormalizedB);
 *  Mpp32f nMin = fminf(nNormalizedR, nNormalizedG);
 *         nMin = fminf(nMin, nNormalizedB);
 *  Mpp32f nL = (nMax + nMin) * 0.5F;
 *  Mpp32f nDivisor = nMax - nMin;
 *  // Saturation
 *  if (nDivisor == 0.0F) // achromatics case
 *  {
 *      nS = 0.0F;
 *      nH = 0.0F;
 *  }
 *  else // chromatics case
 *  {
 *      if (nL > 0.5F)
 *          nS = nDivisor / (1.0F - (nMax + nMin - 1.0F));
 *      else
 *          nS = nDivisor / (nMax + nMin);
 *  }
 *  // Hue
 *  Mpp32f nCr = (nMax - nNormalizedR) / nDivisor;
 *  Mpp32f nCg = (nMax - nNormalizedG) / nDivisor;
 *  Mpp32f nCb = (nMax - nNormalizedB) / nDivisor;
 *  if (nNormalizedR == nMax)
 *      nH = nCb - nCg;
 *  else if (nNormalizedG == nMax)
 *      nH = 2.0F + nCr - nCb;
 *  else if (nNormalizedB == nMax)
 *      nH = 4.0F + nCg - nCr;
 *  nH = nH * 0.166667F; // / 6.0F       
 *  if (nH < 0.0F)
 *      nH = nH + 1.0F;
 *  H = (Mpp8u)(nH * 255.0F);
 *  L = (Mpp8u)(nL * 255.0F);
 *  S = (Mpp8u)(nS * 255.0F);
 *  \endcode
 *
 * @{
 */
/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed HLS color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToHLS_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed HLS color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToHLS_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed HLS with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToHLS_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed HLS with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToHLS_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} rgbtohls */

/** @defgroup hlstorgb HLSToRGB 
 *  HLS to RGB color conversion.
 *
 *  This is how MPP converts HLS to gamma corrected RGB or BGR.
 *
 *  \code
 *  Mpp32f nNormalizedH = (Mpp32f)H * 0.003921569F;  // / 255.0F
 *  Mpp32f nNormalizedL = (Mpp32f)L * 0.003921569F;
 *  Mpp32f nNormalizedS = (Mpp32f)S * 0.003921569F;
 *  Mpp32f nM1;
 *  Mpp32f nM2;
 *  Mpp32f nR;
 *  Mpp32f nG;
 *  Mpp32f nB;
 *  Mpp32f nh = 0.0F;
 *  if (nNormalizedL <= 0.5F)
 *      nM2 = nNormalizedL * (1.0F + nNormalizedS);
 *  else
 *      nM2 = nNormalizedL + nNormalizedS - nNormalizedL * nNormalizedS;
 *  nM1 = 2.0F * nNormalizedL - nM2;
 *  if (nNormalizedS == 0.0F)
 *      nR = nG = nB = nNormalizedL;
 *  else
 *  {
 *      nh = nNormalizedH + 0.3333F;
 *      if (nh > 1.0F)
 *          nh -= 1.0F;
 *  }
 *  Mpp32f nMDiff = nM2 - nM1;
 *  if (0.6667F < nh)
 *      nR = nM1;
 *  else
 *  {    
 *      if (nh < 0.1667F)
 *          nR = (nM1 + nMDiff * nh * 6.0F); // / 0.1667F
 *      else if (nh < 0.5F)
 *          nR = nM2;
 *      else
 *          nR = nM1 + nMDiff * ( 0.6667F - nh ) * 6.0F; // / 0.1667F
 *  }
 *  if (nR > 1.0F)
 *      nR = 1.0F;     
 *  nh = nNormalizedH;
 *  if (0.6667F < nh)
 *      nG = nM1;
 *  else
 *  {
 *      if (nh < 0.1667F)
 *          nG = (nM1 + nMDiff * nh * 6.0F); // / 0.1667F
 *      else if (nh < 0.5F)
 *          nG = nM2;
 *      else
 *          nG = nM1 + nMDiff * (0.6667F - nh ) * 6.0F; // / 0.1667F
 *  }
 *  if (nG > 1.0F)
 *      nG = 1.0F;     
 *  nh = nNormalizedH - 0.3333F;
 *  if (nh < 0.0F)
 *      nh += 1.0F;
 *  if (0.6667F < nh)
 *      nB = nM1;
 *  else
 *  {
 *      if (nh < 0.1667F)
 *          nB = (nM1 + nMDiff * nh * 6.0F); // / 0.1667F
 *      else if (nh < 0.5F)
 *          nB = nM2;
 *      else
 *          nB = nM1 + nMDiff * (0.6667F - nh ) * 6.0F; // / 0.1667F
 *  }        
 *  if (nB > 1.0F)
 *      nB = 1.0F;     
 *  R = (Mpp8u)(nR * 255.0F);
 *  G = (Mpp8u)(nG * 255.0F);
 *  B = (Mpp8u)(nB * 255.0F);
 *  \endcode
 *
 * @{
 */
/**
 * 3 channel 8-bit unsigned packed HLS to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiHLSToRGB_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed HLS to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiHLSToRGB_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed HLS with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiHLSToRGB_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed HLS with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiHLSToRGB_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} hlstorgb */

/** @defgroup bgrtohls BGRToHLS 
 *  BGR to HLS color conversion.
 * @{
 */
/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 4 channel 8-bit unsigned packed HLS with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToHLS_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 4 channel 8-bit unsigned packed HLS with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToHLS_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar HLS color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToHLS_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar HLS color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToHLS_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 4 channel 8-bit unsigned planar HLS with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToHLS_8u_AC4P4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[4], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed BGR with alpha to 4 channel 8-bit unsigned planar HLS with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToHLS_8u_AC4P4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[4], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned packed HLS color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToHLS_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned packed HLS color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToHLS_8u_P3C3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned planar BGR with alpha to 4 channel 8-bit unsigned packed HLS with alpha color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToHLS_8u_AP4C4R_Ctx(const Mpp8u * const pSrc[4], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned planar BGR with alpha to 4 channel 8-bit unsigned packed HLS with alpha color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToHLS_8u_AP4C4R(const Mpp8u * const pSrc[4], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar HLS color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToHLS_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar HLS color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToHLS_8u_P3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned planar BGR with alpha to 4 channel 8-bit unsigned planar HLS with alpha color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToHLS_8u_AP4R_Ctx(const Mpp8u * const pSrc[4], int nSrcStep, Mpp8u * pDst[4], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned planar BGR with alpha to 4 channel 8-bit unsigned planar HLS with alpha color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToHLS_8u_AP4R(const Mpp8u * const pSrc[4], int nSrcStep, Mpp8u * pDst[4], int nDstStep, MppiSize oSizeROI);

/** @} bgrtohls */

/** @defgroup hlstobgr HLSToBGR 
 *  HLS to BGR color conversion.
 * @{
 */
/**
 * 3 channel 8-bit unsigned packed HLS to 3 channel 8-bit unsigned planar BGR color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiHLSToBGR_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed HLS to 3 channel 8-bit unsigned planar BGR color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiHLSToBGR_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed HLS with alpha to 4 channel 8-bit unsigned planar BGR with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiHLSToBGR_8u_AC4P4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[4], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed HLS with alpha to 4 channel 8-bit unsigned planar BGR with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiHLSToBGR_8u_AC4P4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[4], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar HLS to 3 channel 8-bit unsigned planar BGR color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiHLSToBGR_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar HLS to 3 channel 8-bit unsigned planar BGR color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiHLSToBGR_8u_P3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned planar HLS with alpha to 4 channel 8-bit unsigned planar BGR with alpha color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiHLSToBGR_8u_AP4R_Ctx(const Mpp8u * const pSrc[4], int nSrcStep, Mpp8u * pDst[4], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned planar HLS with alpha to 4 channel 8-bit unsigned planar BGR with alpha color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiHLSToBGR_8u_AP4R(const Mpp8u * const pSrc[4], int nSrcStep, Mpp8u * pDst[4], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar HLS to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiHLSToBGR_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar HLS to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiHLSToBGR_8u_P3C3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned planar HLS with alpha to 4 channel 8-bit unsigned packed BGR with alpha color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiHLSToBGR_8u_AP4C4R_Ctx(const Mpp8u * const pSrc[4], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned planar HLS with alpha to 4 channel 8-bit unsigned packed BGR with alpha color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiHLSToBGR_8u_AP4C4R(const Mpp8u * const pSrc[4], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} hlstobgr */

/** @defgroup rgbtohsv RGBToHSV 
 *  RGB to HSV color conversion.
 *
 *  This is how MPP converts gamma corrected RGB or BGR to HSV.
 *  This code uses the fmaxf() and fminf() 32 bit floating point math functions.
 *
 * \code
 *  Mpp32f nNormalizedR = (Mpp32f)R * 0.003921569F; // / 255.0F
 *  Mpp32f nNormalizedG = (Mpp32f)G * 0.003921569F;
 *  Mpp32f nNormalizedB = (Mpp32f)B * 0.003921569F;
 *  Mpp32f nS;
 *  Mpp32f nH;
 *  // Value
 *  Mpp32f nV = fmaxf(nNormalizedR, nNormalizedG);
 *         nV = fmaxf(nV, nNormalizedB);
 *  // Saturation
 *  Mpp32f nTemp = fminf(nNormalizedR, nNormalizedG);
 *         nTemp = fminf(nTemp, nNormalizedB);
 *  Mpp32f nDivisor = nV - nTemp;
 *  if (nV == 0.0F) // achromatics case
 *  {
 *      nS = 0.0F;
 *      nH = 0.0F;
 *  }    
 *  else // chromatics case
 *      nS = nDivisor / nV;
 *  // Hue:
 *  Mpp32f nCr = (nV - nNormalizedR) / nDivisor;
 *  Mpp32f nCg = (nV - nNormalizedG) / nDivisor;
 *  Mpp32f nCb = (nV - nNormalizedB) / nDivisor;
 *  if (nNormalizedR == nV)
 *      nH = nCb - nCg;
 *  else if (nNormalizedG == nV)
 *      nH = 2.0F + nCr - nCb;
 *  else if (nNormalizedB == nV)
 *      nH = 4.0F + nCg - nCr;
 *  nH = nH * 0.166667F; // / 6.0F       
 *  if (nH < 0.0F)
 *      nH = nH + 1.0F;
 *  H = (Mpp8u)(nH * 255.0F);
 *  S = (Mpp8u)(nS * 255.0F);
 *  V = (Mpp8u)(nV * 255.0F);
 * \endcode
 *
 * @{
 */
/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed HSV color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToHSV_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed HSV color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToHSV_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed HSV with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToHSV_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed HSV with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToHSV_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} rgbtohsv */

/** @defgroup hsvtorgb HSVToRGB 
 *  HSV to RGB color conversion.
 *
 *  This is how MPP converts HSV to gamma corrected RGB or BGR.
 *  This code uses the floorf() 32 bit floating point math function.
 *
 *  \code
 *  Mpp32f nNormalizedH = (Mpp32f)H * 0.003921569F; // / 255.0F
 *  Mpp32f nNormalizedS = (Mpp32f)S * 0.003921569F;
 *  Mpp32f nNormalizedV = (Mpp32f)V * 0.003921569F;
 *  Mpp32f nR;
 *  Mpp32f nG;
 *  Mpp32f nB;
 *  if (nNormalizedS == 0.0F)
 *  {
 *      nR = nG = nB = nNormalizedV;
 *  }
 *  else
 *  {
 *      if (nNormalizedH == 1.0F)
 *          nNormalizedH = 0.0F;
 *      else
 *          nNormalizedH = nNormalizedH * 6.0F; // / 0.1667F
 *  }
 *  Mpp32f nI = floorf(nNormalizedH);
 *  Mpp32f nF = nNormalizedH - nI;
 *  Mpp32f nM = nNormalizedV * (1.0F - nNormalizedS);
 *  Mpp32f nN = nNormalizedV * (1.0F - nNormalizedS * nF);
 *  Mpp32f nK = nNormalizedV * (1.0F - nNormalizedS * (1.0F - nF));
 *  if (nI == 0.0F)
 *      { nR = nNormalizedV; nG = nK; nB = nM; }
 *  else if (nI == 1.0F)
 *      { nR = nN; nG = nNormalizedV; nB = nM; }
 *  else if (nI == 2.0F)
 *      { nR = nM; nG = nNormalizedV; nB = nK; }
 *  else if (nI == 3.0F)
 *      { nR = nM; nG = nN; nB = nNormalizedV; }
 *  else if (nI == 4.0F)
 *      { nR = nK; nG = nM; nB = nNormalizedV; }
 *  else if (nI == 5.0F)
 *      { nR = nNormalizedV; nG = nM; nB = nN; }
 *  R = (Mpp8u)(nR * 255.0F);
 *  G = (Mpp8u)(nG * 255.0F);
 *  B = (Mpp8u)(nB * 255.0F);
 *  \endcode
 *
 * @{
 */
/**
 * 3 channel 8-bit unsigned packed HSV to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiHSVToRGB_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed HSV to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiHSVToRGB_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed HSV with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiHSVToRGB_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed HSV with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiHSVToRGB_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} hsvtorgb */

/** 
 * @defgroup image_JPEG_color_conversion JPEG Color Conversion
 * The set of JPEG color conversion functions available in the library.
 * @{
 *
 */

/** @name RGBToYCbCr_JPEG Planar to planar.
 *  JPEG RGB to YCbCr color conversion.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr420_JPEG_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr420_JPEG_8u_P3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr422_JPEG_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr422_JPEG_8u_P3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YCbCr411 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr411_JPEG_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YCbCr411 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr411_JPEG_8u_P3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YCbCr444 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr444_JPEG_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YCbCr444 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr444_JPEG_8u_P3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr420_JPEG_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr420_JPEG_8u_P3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr422_JPEG_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr422_JPEG_8u_P3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar YCbCr411 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr411_JPEG_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar YCbCr411 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr411_JPEG_8u_P3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar YCbCr444 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr444_JPEG_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar YCbCr444 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */

MppStatus mppiBGRToYCbCr444_JPEG_8u_P3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToRGB_JPEG_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToRGB_JPEG_8u_P3R(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned planar RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToRGB_JPEG_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned planar RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToRGB_JPEG_8u_P3R(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToRGB_JPEG_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToRGB_JPEG_8u_P3R(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr444 to 3 channel 8-bit unsigned planar RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr444ToRGB_JPEG_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr444 to 3 channel 8-bit unsigned planar RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr444ToRGB_JPEG_8u_P3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar BGR color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToBGR_JPEG_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar BGR color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToBGR_JPEG_8u_P3R(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned planar BGR color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToBGR_JPEG_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned planar BGR color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToBGR_JPEG_8u_P3R(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar BGR color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToBGR_JPEG_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar BGR color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToBGR_JPEG_8u_P3R(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr444 to 3 channel 8-bit unsigned planar BGR color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr444ToBGR_JPEG_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr444 to 3 channel 8-bit unsigned planar BGR color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr444ToBGR_JPEG_8u_P3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/** @} */

/** @name RGBToYCbCr_JPEG Planar to packed.
 *  JPEG RGB to YCbCr color conversion.
 *
 * @{
 *
 */


/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr420_JPEG_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr420_JPEG_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr422_JPEG_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr422_JPEG_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr411 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr411_JPEG_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr411 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr411_JPEG_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr444 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr444_JPEG_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr444 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYCbCr444_JPEG_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr420_JPEG_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr420_JPEG_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr422_JPEG_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr422_JPEG_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr411 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr411_JPEG_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr411 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr411_JPEG_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr444 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr444_JPEG_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr444 color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiBGRToYCbCr444_JPEG_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned packed YCbCr420 to 3 channel 8-bit unsigned planar RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToRGB_JPEG_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed YCbCr420 to 3 channel 8-bit unsigned planar RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToRGB_JPEG_8u_P3C3R(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToRGB_JPEG_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToRGB_JPEG_8u_P3C3R(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned packed YCbCr411 to 3 channel 8-bit unsigned planar RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToRGB_JPEG_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed YCbCr411 to 3 channel 8-bit unsigned planar RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToRGB_JPEG_8u_P3C3R(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr444 to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr444ToRGB_JPEG_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr444 to 3 channel 8-bit unsigned packed RGB color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr444ToRGB_JPEG_8u_P3C3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToBGR_JPEG_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToBGR_JPEG_8u_P3C3R(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToBGR_JPEG_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToBGR_JPEG_8u_P3C3R(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToBGR_JPEG_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToBGR_JPEG_8u_P3C3R(const Mpp8u * const pSrc[3], int aSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr444 to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr444ToBGR_JPEG_8u_P3C3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr444 to 3 channel 8-bit unsigned packed BGR color conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr444ToBGR_JPEG_8u_P3C3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} */

/** @} image_JPEG_color_conversion */

/** @} image_color_model_conversion */ 

/** 
 * @defgroup image_color_to_gray ColorToGray Conversion
 * Routines for converting color images to grayscale.
 *
 * @{ 
 *
 */

/** @name RGBToGray 
 *  RGB to CCIR601 Gray conversion.
 *
 *  Here is how MPP converts gamma corrected RGB to CCIR601 Gray.
 *
 *  \code   
 *   nGray =  0.299F * R + 0.587F * G + 0.114F * B; 
 *  \endcode
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned packed RGB to 1 channel 8-bit unsigned packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToGray_8u_C3C1R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 1 channel 8-bit unsigned packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToGray_8u_C3C1R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 1 channel 8-bit unsigned packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToGray_8u_AC4C1R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 1 channel 8-bit unsigned packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */MppStatus mppiRGBToGray_8u_AC4C1R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 16-bit unsigned packed RGB to 1 channel 16-bit unsigned packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToGray_16u_C3C1R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit unsigned packed RGB to 1 channel 16-bit unsigned packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToGray_16u_C3C1R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 16-bit unsigned packed RGB with alpha to 1 channel 16-bit unsigned packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToGray_16u_AC4C1R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit unsigned packed RGB with alpha to 1 channel 16-bit unsigned packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToGray_16u_AC4C1R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 16-bit signed packed RGB to 1 channel 16-bit signed packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToGray_16s_C3C1R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit signed packed RGB to 1 channel 16-bit signed packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToGray_16s_C3C1R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 16-bit signed packed RGB with alpha to 1 channel 16-bit signed packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToGray_16s_AC4C1R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit signed packed RGB with alpha to 1 channel 16-bit signed packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToGray_16s_AC4C1R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 32-bit floating point packed RGB to 1 channel 32-bit floating point packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToGray_32f_C3C1R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit floating point packed RGB to 1 channel 32-bit floating point packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToGray_32f_C3C1R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 32-bit floating point packed RGB with alpha to 1 channel 32-bit floating point packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToGray_32f_AC4C1R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point packed RGB with alpha to 1 channel 32-bit floating point packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToGray_32f_AC4C1R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** @} */

/** @name ColorToGray 
 *  RGB Color to Gray conversion using user supplied conversion coefficients.
 *
 *  Here is how MPP converts gamma corrected RGB Color to Gray using user supplied conversion coefficients.
 *
 *  \code   
 *   nGray =  aCoeffs[0] * R + aCoeffs[1] * G + aCoeffs[2] * B;
 *  \endcode
 *
 *  For the C4C1R versions of the functions the calculations are as follows.  
 *  For BGRA or other formats with alpha just rearrange the coefficients accordingly.
 *
 *  \code   
 *   nGray =  aCoeffs[0] * R + aCoeffs[1] * G + aCoeffs[2] * B + aCoeffs[3] * A;
 *  \endcode
 *
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned packed RGB to 1 channel 8-bit unsigned packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_8u_C3C1R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 1 channel 8-bit unsigned packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_8u_C3C1R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[3]);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 1 channel 8-bit unsigned packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_8u_AC4C1R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed RGB with alpha to 1 channel 8-bit unsigned packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_8u_AC4C1R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[3]);

/**
 * 4 channel 8-bit unsigned packed RGBA to 1 channel 8-bit unsigned packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_8u_C4C1R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed RGBA to 1 channel 8-bit unsigned packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_8u_C4C1R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[4]);

/**
 * 3 channel 16-bit unsigned packed RGB to 1 channel 16-bit unsigned packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_16u_C3C1R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit unsigned packed RGB to 1 channel 16-bit unsigned packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_16u_C3C1R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[3]);

/**
 * 4 channel 16-bit unsigned packed RGB with alpha to 1 channel 16-bit unsigned packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_16u_AC4C1R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit unsigned packed RGB with alpha to 1 channel 16-bit unsigned packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_16u_AC4C1R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[3]);

/**
 * 4 channel 16-bit unsigned packed RGBA to 1 channel 16-bit unsigned packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_16u_C4C1R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit unsigned packed RGBA to 1 channel 16-bit unsigned packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_16u_C4C1R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[4]);

/**
 * 3 channel 16-bit signed packed RGB to 1 channel 16-bit signed packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_16s_C3C1R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit signed packed RGB to 1 channel 16-bit signed packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_16s_C3C1R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[3]);

/**
 * 4 channel 16-bit signed packed RGB with alpha to 1 channel 16-bit signed packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_16s_AC4C1R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit signed packed RGB with alpha to 1 channel 16-bit signed packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_16s_AC4C1R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[3]);

/**
 * 4 channel 16-bit signed packed RGBA to 1 channel 16-bit signed packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_16s_C4C1R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit signed packed RGBA to 1 channel 16-bit signed packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_16s_C4C1R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[4]);

/**
 * 3 channel 32-bit floating point packed RGB to 1 channel 32-bit floating point packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_32f_C3C1R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit floating point packed RGB to 1 channel 32-bit floating point packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_32f_C3C1R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[3]);

/**
 * 4 channel 32-bit floating point packed RGB with alpha to 1 channel 32-bit floating point packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_32f_AC4C1R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point packed RGB with alpha to 1 channel 32-bit floating point packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_32f_AC4C1R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[3]);

/**
 * 4 channel 32-bit floating point packed RGBA to 1 channel 32-bit floating point packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_32f_C4C1R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point packed RGBA to 1 channel 32-bit floating point packed Gray conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aCoeffs fixed size array of constant floating point conversion coefficient values, one per color channel.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiColorToGray_32f_C4C1R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, const Mpp32f aCoeffs[4]);

/** @} */

/** @name GradientColorToGray 
 *  RGB Color to Gray Gradient conversion using user selected gradient distance method.
 *
 * @{
 *
 */

/**
 * 3 channel 8-bit unsigned packed RGB to 1 channel 8-bit unsigned packed Gray Gradient conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eNorm Gradient distance method to use.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGradientColorToGray_8u_C3C1R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppiNorm eNorm, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed RGB to 1 channel 8-bit unsigned packed Gray Gradient conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eNorm Gradient distance method to use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGradientColorToGray_8u_C3C1R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppiNorm eNorm);

/**
 * 3 channel 16-bit unsigned packed RGB to 1 channel 16-bit unsigned packed Gray Gradient conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eNorm Gradient distance method to use.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGradientColorToGray_16u_C3C1R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppiNorm eNorm, MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit unsigned packed RGB to 1 channel 16-bit unsigned packed Gray Gradient conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eNorm Gradient distance method to use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGradientColorToGray_16u_C3C1R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppiNorm eNorm);

/**
 * 3 channel 16-bit signed packed RGB to 1 channel 16-bit signed packed Gray Gradient conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eNorm Gradient distance method to use.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGradientColorToGray_16s_C3C1R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppiNorm eNorm, MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit signed packed RGB to 1 channel 16-bit signed packed Gray Gradient conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eNorm Gradient distance method to use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGradientColorToGray_16s_C3C1R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppiNorm eNorm);

/**
 * 3 channel 32-bit floating point packed RGB to 1 channel 32-bit floating point packed Gray Gradient conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eNorm Gradient distance method to use.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGradientColorToGray_32f_C3C1R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppiNorm eNorm, MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit floating point packed RGB to 1 channel 32-bit floating point packed Gray Gradient conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eNorm Gradient distance method to use.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGradientColorToGray_32f_C3C1R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppiNorm eNorm);

/** @} */

/** @} image_color_to_gray */

/** 
 *  @defgroup image_color_debayer ColorDebayer
 *  Grayscale Color Filter Array to RGB Color Debayer conversion. 
 *
 *  Generates one RGB color pixel for every grayscale source pixel.
 *  Source and destination images must have even width and height.  Missing pixel colors are generated using bilinear interpolation
 *  with chroma correlation of generated green values (eInterpolation MUST be set to 0). eGrid allows the user to specify the Bayer grid 
 *  registration position at source image location oSrcROI.x, oSrcROI.y relative to pSrc. Possible registration positions are:
 *
 *  \code
 *  MPPI_BAYER_BGGR  MPPI_BAYER_RGGB  MPPI_BAYER_GBRG  MPPI_BAYER_GRBG
 *
 *        B G              R G              G B              G R
 *        G R              G B              R G              B G
 *
 *  \endcode
 *
 *  If it becomes necessary to access source pixels outside source image then the source image borders are mirrored.
 *
 *  Here is how the algorithm works.  R, G, and B base pixels from the source image are used unmodified.  To generate R values for those
 *  G pixels, the average of R(x - 1, y) and R(x + 1, y) or R(x, y - 1) and R(x, y + 1) is used depending on whether the left and right
 *  or top and bottom pixels are R base pixels.  To generate B values for those G pixels, the same algorithm is used using nearest B values.
 *  For an R base pixel, if there are no B values in the upper, lower, left, or right adjacent pixels then B is the average of B values
 *  in the 4 diagonal (G base) pixels.  The same algorithm is used using R values to generate the R value of a B base pixel. 
 *  Chroma correlation is applied to generated G values only, for a B base pixel G(x - 1, y) and G(x + 1, y) are averaged or G(x, y - 1)
 *  and G(x, y + 1) are averaged depending on whether the absolute difference between B(x, y) and the average of B(x - 2, y) and B(x + 2, y)
 *  is smaller than the absolute difference between B(x, y) and the average of B(x, y - 2) and B(x, y + 2). For an R base pixel the same
 *  algorithm is used testing against the surrounding R values at those offsets.  If the horizontal and vertical differences are the same
 *  at one of those pixels then the average of the four left, right, upper and lower G values is used instead.
 *  
 *
 * @{
 *
 */

/**
 * 1 channel 8-bit unsigned packed CFA grayscale Bayer pattern to 3 channel 8-bit unsigned packed RGB conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize full source image width and height relative to pSrc.
 * \param oSrcROI rectangle specifying starting source image pixel x and y location relative to pSrc and ROI width and height. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param eGrid enumeration value specifying bayer grid registration position at location oSrcROI.x, oSrcROI.y relative to pSrc.
 * \param eInterpolation MUST be MPPI_INTER_UNDEFINED
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiCFAToRGB_8u_C1C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, MppiSize oSrcSize, MppiRect oSrcROI, 
                                Mpp8u * pDst, int nDstStep, MppiBayerGridPosition eGrid, MppiInterpolationMode eInterpolation, MppStreamContext mppStreamCtx);

/**
 * 1 channel 8-bit unsigned packed CFA grayscale Bayer pattern to 3 channel 8-bit unsigned packed RGB conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize full source image width and height relative to pSrc.
 * \param oSrcROI rectangle specifying starting source image pixel x and y location relative to pSrc and ROI width and height. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param eGrid enumeration value specifying bayer grid registration position at location oSrcROI.x, oSrcROI.y relative to pSrc.
 * \param eInterpolation MUST be MPPI_INTER_UNDEFINED
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiCFAToRGB_8u_C1C3R(const Mpp8u * pSrc, int nSrcStep, MppiSize oSrcSize, MppiRect oSrcROI, 
                            Mpp8u * pDst, int nDstStep, MppiBayerGridPosition eGrid, MppiInterpolationMode eInterpolation);

/**
 * 1 channel 8-bit unsigned packed CFA grayscale Bayer pattern to 4 channel 8-bit unsigned packed RGB conversion with alpha.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize full source image width and height relative to pSrc.
 * \param oSrcROI rectangle specifying starting source image pixel x and y location relative to pSrc and ROI width and height. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param eGrid enumeration value specifying bayer grid registration position at location oSrcROI.x, oSrcROI.y relative to pSrc.
 * \param eInterpolation MUST be MPPI_INTER_UNDEFINED
 * \param nAlpha constant alpha value to be written to each destination pixel
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiCFAToRGBA_8u_C1AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, MppiSize oSrcSize, MppiRect oSrcROI, 
                                  Mpp8u * pDst, int nDstStep, MppiBayerGridPosition eGrid, MppiInterpolationMode eInterpolation, Mpp8u nAlpha, MppStreamContext mppStreamCtx);

/**
 * 1 channel 8-bit unsigned packed CFA grayscale Bayer pattern to 4 channel 8-bit unsigned packed RGB conversion with alpha.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize full source image width and height relative to pSrc.
 * \param oSrcROI rectangle specifying starting source image pixel x and y location relative to pSrc and ROI width and height. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param eGrid enumeration value specifying bayer grid registration position at location oSrcROI.x, oSrcROI.y relative to pSrc.
 * \param eInterpolation MUST be MPPI_INTER_UNDEFINED
 * \param nAlpha constant alpha value to be written to each destination pixel
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiCFAToRGBA_8u_C1AC4R(const Mpp8u * pSrc, int nSrcStep, MppiSize oSrcSize, MppiRect oSrcROI, 
                              Mpp8u * pDst, int nDstStep, MppiBayerGridPosition eGrid, MppiInterpolationMode eInterpolation, Mpp8u nAlpha);

/**
 * 1 channel 16-bit unsigned packed CFA grayscale Bayer pattern to 3 channel 16-bit unsigned packed RGB conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize full source image width and height relative to pSrc.
 * \param oSrcROI rectangle specifying starting source image pixel x and y location relative to pSrc and ROI width and height. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param eGrid enumeration value specifying bayer grid registration position at location oSrcROI.x, oSrcROI.y relative to pSrc.
 * \param eInterpolation MUST be MPPI_INTER_UNDEFINED
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiCFAToRGB_16u_C1C3R_Ctx(const Mpp16u * pSrc, int nSrcStep, MppiSize oSrcSize, MppiRect oSrcROI, 
                                 Mpp16u * pDst, int nDstStep, MppiBayerGridPosition eGrid, MppiInterpolationMode eInterpolation, MppStreamContext mppStreamCtx);

/**
 * 1 channel 16-bit unsigned packed CFA grayscale Bayer pattern to 3 channel 16-bit unsigned packed RGB conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize full source image width and height relative to pSrc.
 * \param oSrcROI rectangle specifying starting source image pixel x and y location relative to pSrc and ROI width and height. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param eGrid enumeration value specifying bayer grid registration position at location oSrcROI.x, oSrcROI.y relative to pSrc.
 * \param eInterpolation MUST be MPPI_INTER_UNDEFINED
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus  
mppiCFAToRGB_16u_C1C3R(const Mpp16u * pSrc, int nSrcStep, MppiSize oSrcSize, MppiRect oSrcROI, 
                             Mpp16u * pDst, int nDstStep, MppiBayerGridPosition eGrid, MppiInterpolationMode eInterpolation);

/**
 * 1 channel 16-bit unsigned packed CFA grayscale Bayer pattern to 4 channel 16-bit unsigned packed RGB conversion with alpha.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize full source image width and height relative to pSrc.
 * \param oSrcROI rectangle specifying starting source image pixel x and y location relative to pSrc and ROI width and height. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param eGrid enumeration value specifying bayer grid registration position at location oSrcROI.x, oSrcROI.y relative to pSrc.
 * \param eInterpolation MUST be MPPI_INTER_UNDEFINED
 * \param nAlpha constant alpha value to be written to each destination pixel
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiCFAToRGBA_16u_C1AC4R_Ctx(const Mpp16u * pSrc, int nSrcStep, MppiSize oSrcSize, MppiRect oSrcROI, 
                                   Mpp16u * pDst, int nDstStep, MppiBayerGridPosition eGrid, MppiInterpolationMode eInterpolation, Mpp16u nAlpha, MppStreamContext mppStreamCtx);

/**
 * 1 channel 16-bit unsigned packed CFA grayscale Bayer pattern to 4 channel 16-bit unsigned packed RGB conversion with alpha.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize full source image width and height relative to pSrc.
 * \param oSrcROI rectangle specifying starting source image pixel x and y location relative to pSrc and ROI width and height. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param eGrid enumeration value specifying bayer grid registration position at location oSrcROI.x, oSrcROI.y relative to pSrc.
 * \param eInterpolation MUST be MPPI_INTER_UNDEFINED
 * \param nAlpha constant alpha value to be written to each destination pixel
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiCFAToRGBA_16u_C1AC4R(const Mpp16u * pSrc, int nSrcStep, MppiSize oSrcSize, MppiRect oSrcROI, 
                               Mpp16u * pDst, int nDstStep, MppiBayerGridPosition eGrid, MppiInterpolationMode eInterpolation, Mpp16u nAlpha);

//32u


/**
 * 1 channel 32-bit unsigned packed CFA grayscale Bayer pattern to 3 channel 32-bit unsigned packed RGB conversion.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize full source image width and height relative to pSrc.
 * \param oSrcROI rectangle specifying starting source image pixel x and y location relative to pSrc and ROI width and height.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param eGrid enumeration value specifying bayer grid registration position at location oSrcROI.x, oSrcROI.y relative to pSrc.
 * \param eInterpolation MUST be MPPI_INTER_UNDEFINED
 * \param mppStreamCtx \ref application_managed_stream_context.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus
mppiCFAToRGB_32u_C1C3R_Ctx(const Mpp32u * pSrc, int nSrcStep, MppiSize oSrcSize, MppiRect oSrcROI,
                                 Mpp32u * pDst, int nDstStep, MppiBayerGridPosition eGrid, MppiInterpolationMode eInterpolation, MppStreamContext mppStreamCtx);


/**
 * 1 channel 32-bit unsigned packed CFA grayscale Bayer pattern to 4 channel 32-bit unsigned packed RGB conversion with alpha.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize full source image width and height relative to pSrc.
 * \param oSrcROI rectangle specifying starting source image pixel x and y location relative to pSrc and ROI width and height.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param eGrid enumeration value specifying bayer grid registration position at location oSrcROI.x, oSrcROI.y relative to pSrc.
 * \param eInterpolation MUST be MPPI_INTER_UNDEFINED
 * \param nAlpha constant alpha value to be written to each destination pixel
 * \param mppStreamCtx \ref application_managed_stream_context.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus
mppiCFAToRGBA_32u_C1AC4R_Ctx(const Mpp32u * pSrc, int nSrcStep, MppiSize oSrcSize, MppiRect oSrcROI,
                                   Mpp32u * pDst, int nDstStep, MppiBayerGridPosition eGrid, MppiInterpolationMode eInterpolation, Mpp32u nAlpha, MppStreamContext mppStreamCtx);






/** @} image_color_debayer */


/** 
 * @defgroup image_color_sampling_format_conversion Color Sampling Format Conversion
 *
 * Routines for converting between various image color sampling formats.
 *
 *
 * @{                                         
 */

/** @defgroup ycbcr420toycbcr411 YCbCr420 To YCbCr411 
 *  YCbCr420 to YCbCr411 sampling format conversion.
 * @{
 */
/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDstY \ref destination_planar_image_pointer.
 * \param nDstYStep \ref destination_planar_image_line_step.
 * \param pDstCbCr \ref destination_planar_image_pointer.
 * \param nDstCbCrStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToYCbCr411_8u_P3P2R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u* pDstY, int nDstYStep, Mpp8u* pDstCbCr, int nDstCbCrStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDstY \ref destination_planar_image_pointer.
 * \param nDstYStep \ref destination_planar_image_line_step.
 * \param pDstCbCr \ref destination_planar_image_pointer.
 * \param nDstCbCrStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToYCbCr411_8u_P3P2R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u* pDstY, int nDstYStep, Mpp8u* pDstCbCr, int nDstCbCrStep, MppiSize oSizeROI);

/**
 * 2 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
 *
 * \param pSrcY \ref source_planar_image_pointer.
 * \param nSrcYStep \ref source_planar_image_line_step.
 * \param pSrcCbCr \ref source_planar_image_pointer.
 * \param nSrcCbCrStep \ref source_planar_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToYCbCr411_8u_P2P3R_Ctx(const Mpp8u * pSrcY, int nSrcYStep, const Mpp8u* pSrcCbCr, int nSrcCbCrStep, Mpp8u* pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
 *
 * \param pSrcY \ref source_planar_image_pointer.
 * \param nSrcYStep \ref source_planar_image_line_step.
 * \param pSrcCbCr \ref source_planar_image_pointer.
 * \param nSrcCbCrStep \ref source_planar_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToYCbCr411_8u_P2P3R(const Mpp8u * pSrcY, int nSrcYStep, const Mpp8u* pSrcCbCr, int nSrcCbCrStep, Mpp8u* pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} */

/** @defgroup ycbcr422toycbcr422 YCbCr422 To YCbCr422 
 *  YCbCr422 to YCbCr422 sampling format conversion.
 * @{
 */
/**
 * 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422_8u_C2P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422_8u_C2P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422_8u_P3C2R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422_8u_P3C2R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} */

/** @defgroup ycbcr422toycrcb422 YCbCr422 To YCrCb422 
 *  YCbCr422 to YCrCb422 sampling format conversion.
 * @{
 */
/**
 * 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned packed YCrCb422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToYCrCb422_8u_C2R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned packed YCrCb422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToYCrCb422_8u_C2R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 2 channel 8-bit unsigned packed YCrCb422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToYCrCb422_8u_P3C2R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 2 channel 8-bit unsigned packed YCrCb422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToYCrCb422_8u_P3C2R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} */

/** @defgroup ycbcr422tocbycr422 YCbCr422 To CbYCr422 
 *  YCbCr422 to CbYCr422 sampling format conversion.
 * @{
 */
/**
 * 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned packed CbYCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToCbYCr422_8u_C2R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned packed CbYCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToCbYCr422_8u_C2R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} */

/** @defgroup cbycr422toycbcr411 CbYCr422 To YCbCr411 
 *  CbYCr422 to YCbCr411 sampling format conversion.
 * @{
 */
/**
 * 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCbYCr422ToYCbCr411_8u_C2P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCbYCr422ToYCbCr411_8u_C2P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} */

/** @defgroup ycbcr422toycbcr420 YCbCr422 To YCbCr420 
 *  YCbCr422 to YCbCr420 sampling format conversion.
 * @{
 */
/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToYCbCr420_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst[3], int nDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToYCbCr420_8u_P3R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst[3], int nDstStep[3], MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDstY \ref destination_planar_image_pointer.
 * \param nDstYStep \ref destination_planar_image_line_step.
 * \param pDstCbCr \ref destination_planar_image_pointer.
 * \param nDstCbCrStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToYCbCr420_8u_P3P2R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDstY, int nDstYStep, Mpp8u * pDstCbCr, int nDstCbCrStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDstY \ref destination_planar_image_pointer.
 * \param nDstYStep \ref destination_planar_image_line_step.
 * \param pDstCbCr \ref destination_planar_image_pointer.
 * \param nDstCbCrStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToYCbCr420_8u_P3P2R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDstY, int nDstYStep, Mpp8u * pDstCbCr, int nDstCbCrStep, MppiSize oSizeROI);

/**
 * 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToYCbCr420_8u_C2P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToYCbCr420_8u_C2P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/**
 * 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDstY \ref destination_planar_image_pointer.
 * \param nDstYStep \ref destination_planar_image_line_step.
 * \param pDstCbCr \ref destination_planar_image_pointer.
 * \param nDstCbCrStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToYCbCr420_8u_C2P2R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDstY, int nDstYStep, Mpp8u * pDstCbCr, int nDstCbCrStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDstY \ref destination_planar_image_pointer.
 * \param nDstYStep \ref destination_planar_image_line_step.
 * \param pDstCbCr \ref destination_planar_image_pointer.
 * \param nDstCbCrStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToYCbCr420_8u_C2P2R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDstY, int nDstYStep, Mpp8u * pDstCbCr, int nDstCbCrStep, MppiSize oSizeROI);

/** @} */

/** @defgroup ycrcb420toycbcr422 YCrCb420 To YCbCr422 
 *  YCrCb420 to YCbCr422 sampling format conversion.
 * @{
 */
/**
 * 3 channel 8-bit unsigned planar YCrCb420 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCrCb420ToYCbCr422_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCrCb420 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCrCb420ToYCbCr422_8u_P3R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCrCb420 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCrCb420ToYCbCr422_8u_P3C2R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCrCb420 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCrCb420ToYCbCr422_8u_P3C2R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} */

/** @defgroup ycbcr422toycrcb420 YCbCr422 To YCrCb420 
 *  YCbCr422 to YCrCb420 sampling format conversion.
 * @{
 */
/**
 * 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCrCb420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToYCrCb420_8u_C2P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCrCb420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToYCrCb420_8u_C2P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} */

/** @defgroup ycbcr422toycbcr411 YCbCr422 To YCbCr411 
 *  YCbCr422 to YCbCr411 sampling format conversion.
 * @{
 */
/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToYCbCr411_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToYCbCr411_8u_P3R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDstY \ref destination_planar_image_pointer.
 * \param nDstYStep \ref destination_planar_image_line_step.
 * \param pDstCbCr \ref destination_planar_image_pointer.
 * \param nDstCbCrStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToYCbCr411_8u_P3P2R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDstY, int nDstYStep, Mpp8u * pDstCbCr, int nDstCbCrStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr422 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDstY \ref destination_planar_image_pointer.
 * \param nDstYStep \ref destination_planar_image_line_step.
 * \param pDstCbCr \ref destination_planar_image_pointer.
 * \param nDstCbCrStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToYCbCr411_8u_P3P2R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDstY, int nDstYStep, Mpp8u * pDstCbCr, int nDstCbCrStep, MppiSize oSizeROI);

/**
 * 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToYCbCr411_8u_C2P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToYCbCr411_8u_C2P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/**
 * 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDstY \ref destination_planar_image_pointer.
 * \param nDstYStep \ref destination_planar_image_line_step.
 * \param pDstCbCr \ref destination_planar_image_pointer.
 * \param nDstCbCrStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToYCbCr411_8u_C2P2R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDstY, int nDstYStep, Mpp8u * pDstCbCr, int nDstCbCrStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDstY \ref destination_planar_image_pointer.
 * \param nDstYStep \ref destination_planar_image_line_step.
 * \param pDstCbCr \ref destination_planar_image_pointer.
 * \param nDstCbCrStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr422ToYCbCr411_8u_C2P2R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDstY, int nDstYStep, Mpp8u * pDstCbCr, int nDstCbCrStep, MppiSize oSizeROI);

/** @} */

/** @defgroup ycrcb422toycbcr422 YCrCb422 To YCbCr422 
 *  YCrCb422 to YCbCr422 sampling format conversion.
 * @{
 */
/**
 * 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCrCb422ToYCbCr422_8u_C2P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCrCb422ToYCbCr422_8u_C2P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} */

/** @defgroup ycrcb422toycbcr420 YCrCb422 To YCbCr420 
 *  YCrCb422 to YCbCr420 sampling format conversion.
 * @{
 */
/**
 * 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCrCb422ToYCbCr420_8u_C2P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCrCb422ToYCbCr420_8u_C2P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} */

/** @defgroup ycrcb422toycbcr411 YCrCb422 To YCbCr411 
 *  YCrCb422 to YCbCr411 sampling format conversion.
 * @{
 */
/**
 * 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCrCb422ToYCbCr411_8u_C2P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCrCb422ToYCbCr411_8u_C2P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} */

/** @defgroup cbycr422toycbcr422 CbYCr422 To YCbCr422 
 *  CbYCr422 to YCbCr422 sampling format conversion.
 * @{
 */
/**
 * 2 channel 8-bit unsigned packed CbYCr422 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCbYCr422ToYCbCr422_8u_C2R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed CbYCr422 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCbYCr422ToYCbCr422_8u_C2R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCbYCr422ToYCbCr422_8u_C2P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCbYCr422ToYCbCr422_8u_C2P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} */

/** @defgroup cbycr422toycbcr420 CbYCr422 To YCbCr420 
 *  CbYCr422 to YCbCr420 sampling format conversion.
 * @{
 */
/**
 * 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCbYCr422ToYCbCr420_8u_C2P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCbYCr422ToYCbCr420_8u_C2P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/**
 * 2 channel 8-bit unsigned packed CbYCr422 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDstY \ref destination_planar_image_pointer.
 * \param nDstYStep \ref destination_planar_image_line_step.
 * \param pDstCbCr \ref destination_planar_image_pointer.
 * \param nDstCbCrStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCbYCr422ToYCbCr420_8u_C2P2R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDstY, int nDstYStep, Mpp8u * pDstCbCr, int nDstCbCrStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed CbYCr422 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDstY \ref destination_planar_image_pointer.
 * \param nDstYStep \ref destination_planar_image_line_step.
 * \param pDstCbCr \ref destination_planar_image_pointer.
 * \param nDstCbCrStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCbYCr422ToYCbCr420_8u_C2P2R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDstY, int nDstYStep, Mpp8u * pDstCbCr, int nDstCbCrStep, MppiSize oSizeROI);

/** @} */

/** @defgroup cbycr422toycrcb420 CbYCr422 To YCrCb420 
 *  CbYCr422 to YCrCb420 sampling format conversion.
 * @{
 */
/**
 * 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCrCb420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCbYCr422ToYCrCb420_8u_C2P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCrCb420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCbYCr422ToYCrCb420_8u_C2P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} */

/** @defgroup ycbcr420toycbcr420 YCbCr420 To YCbCr420 
 *  YCbCr420 to YCbCr420 sampling format conversion.
 * @{
 */
/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDstY \ref destination_planar_image_pointer.
 * \param nDstYStep \ref destination_planar_image_line_step.
 * \param pDstCbCr \ref destination_planar_image_pointer.
 * \param nDstCbCrStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420_8u_P3P2R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDstY, int nDstYStep, Mpp8u * pDstCbCr, int nDstCbCrStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDstY \ref destination_planar_image_pointer.
 * \param nDstYStep \ref destination_planar_image_line_step.
 * \param pDstCbCr \ref destination_planar_image_pointer.
 * \param nDstCbCrStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420_8u_P3P2R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDstY, int nDstYStep, Mpp8u * pDstCbCr, int nDstCbCrStep, MppiSize oSizeROI);

/**
 * 2 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 *
 * \param pSrcY \ref source_planar_image_pointer.
 * \param nSrcYStep \ref source_planar_image_line_step.
 * \param pSrcCbCr \ref source_planar_image_pointer.
 * \param nSrcCbCrStep \ref source_planar_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420_8u_P2P3R_Ctx(const Mpp8u * const pSrcY, int nSrcYStep, const Mpp8u * pSrcCbCr, int nSrcCbCrStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);
 
/**
 * 2 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 *
 * \param pSrcY \ref source_planar_image_pointer.
 * \param nSrcYStep \ref source_planar_image_line_step.
 * \param pSrcCbCr \ref source_planar_image_pointer.
 * \param nSrcCbCrStep \ref source_planar_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420_8u_P2P3R(const Mpp8u * const pSrcY, int nSrcYStep, const Mpp8u * pSrcCbCr, int nSrcCbCrStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);
 
/** @} */

/** @defgroup ycbcr420toycbcr422 YCbCr420 To YCbCr422 
 *  YCbCr420 to YCbCr422 sampling format conversion.
 * @{
 */
/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToYCbCr422_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst[3], int nDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToYCbCr422_8u_P3R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst[3], int nDstStep[3], MppiSize oSizeROI);

/**
 * 2 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
 *
 * \param pSrcY \ref source_planar_image_pointer.
 * \param nSrcYStep \ref source_planar_image_line_step.
 * \param pSrcCbCr \ref source_planar_image_pointer.
 * \param nSrcCbCrStep \ref source_planar_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToYCbCr422_8u_P2P3R_Ctx(const Mpp8u * pSrcY, int nSrcYStep, const Mpp8u * pSrcCbCr, int nSrcCbCrStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
 *
 * \param pSrcY \ref source_planar_image_pointer.
 * \param nSrcYStep \ref source_planar_image_line_step.
 * \param pSrcCbCr \ref source_planar_image_pointer.
 * \param nSrcCbCrStep \ref source_planar_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToYCbCr422_8u_P2P3R(const Mpp8u * pSrcY, int nSrcYStep, const Mpp8u * pSrcCbCr, int nSrcCbCrStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/**
 * 2 channel 8-bit unsigned planar YCbCr420 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
 *
 * \param pSrcY \ref source_planar_image_pointer.
 * \param nSrcYStep \ref source_planar_image_line_step.
 * \param pSrcCbCr \ref source_planar_image_pointer.
 * \param nSrcCbCrStep \ref source_planar_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToYCbCr422_8u_P2C2R_Ctx(const Mpp8u * pSrcY, int nSrcYStep, const Mpp8u * pSrcCbCr, int nSrcCbCrStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned planar YCbCr420 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
 *
 * \param pSrcY \ref source_planar_image_pointer.
 * \param nSrcYStep \ref source_planar_image_line_step.
 * \param pSrcCbCr \ref source_planar_image_pointer.
 * \param nSrcCbCrStep \ref source_planar_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToYCbCr422_8u_P2C2R(const Mpp8u * pSrcY, int nSrcYStep, const Mpp8u * pSrcCbCr, int nSrcCbCrStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} */

/** @defgroup ycbcr420tocbycr422 YCbCr420 To CbYCr422 
 *  YCbCr420 to CbYCr422 sampling format conversion.
 * @{
 */
/**
 * 2 channel 8-bit unsigned planar YCbCr420 to 2 channel 8-bit unsigned packed CbYCr422 sampling format conversion.
 *
 * \param pSrcY \ref source_planar_image_pointer.
 * \param nSrcYStep \ref source_planar_image_line_step.
 * \param pSrcCbCr \ref source_planar_image_pointer.
 * \param nSrcCbCrStep \ref source_planar_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToCbYCr422_8u_P2C2R_Ctx(const Mpp8u * pSrcY, int nSrcYStep, const Mpp8u * pSrcCbCr, int nSrcCbCrStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned planar YCbCr420 to 2 channel 8-bit unsigned packed CbYCr422 sampling format conversion.
 *
 * \param pSrcY \ref source_planar_image_pointer.
 * \param nSrcYStep \ref source_planar_image_line_step.
 * \param pSrcCbCr \ref source_planar_image_pointer.
 * \param nSrcCbCrStep \ref source_planar_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToCbYCr422_8u_P2C2R(const Mpp8u * pSrcY, int nSrcYStep, const Mpp8u * pSrcCbCr, int nSrcCbCrStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} */

/** @defgroup ycbcr420toycrcb420 YCbCr420 To YCrCb420 
 *  YCbCr420 to YCrCb420 sampling format conversion.
 * @{
 */
/**
 * 2 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCrCb420 sampling format conversion.
 *
 * \param pSrcY \ref source_planar_image_pointer.
 * \param nSrcYStep \ref source_planar_image_line_step.
 * \param pSrcCbCr \ref source_planar_image_pointer.
 * \param nSrcCbCrStep \ref source_planar_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToYCrCb420_8u_P2P3R_Ctx(const Mpp8u * pSrcY, int nSrcYStep, const Mpp8u * pSrcCbCr, int nSrcCbCrStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCrCb420 sampling format conversion.
 *
 * \param pSrcY \ref source_planar_image_pointer.
 * \param nSrcYStep \ref source_planar_image_line_step.
 * \param pSrcCbCr \ref source_planar_image_pointer.
 * \param nSrcCbCrStep \ref source_planar_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr420ToYCrCb420_8u_P2P3R(const Mpp8u * pSrcY, int nSrcYStep, const Mpp8u * pSrcCbCr, int nSrcCbCrStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} */

/** @defgroup ycrcb420tocbycr422 YCrCb420 To CbYCr422 
 *  YCrCb420 to CbYCr422 sampling format conversion.
 * @{
 */
/**
 * 3 channel 8-bit unsigned planar YCrCb420 to 2 channel 8-bit unsigned packed CbYCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCrCb420ToCbYCr422_8u_P3C2R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCrCb420 to 2 channel 8-bit unsigned packed CbYCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCrCb420ToCbYCr422_8u_P3C2R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} */

/** @defgroup ycrcb420toycbcr420 YCrCb420 To YCbCr420 
 *  YCrCb420 to YCbCr420 sampling format conversion.
 * @{
 */
/**
 * 3 channel 8-bit unsigned planar YCrCb420 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDstY \ref destination_planar_image_pointer.
 * \param nDstYStep \ref destination_planar_image_line_step.
 * \param pDstCbCr \ref destination_planar_image_pointer.
 * \param nDstCbCrStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCrCb420ToYCbCr420_8u_P3P2R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDstY, int nDstYStep, Mpp8u * pDstCbCr, int nDstCbCrStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCrCb420 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDstY \ref destination_planar_image_pointer.
 * \param nDstYStep \ref destination_planar_image_line_step.
 * \param pDstCbCr \ref destination_planar_image_pointer.
 * \param nDstCbCrStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCrCb420ToYCbCr420_8u_P3P2R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDstY, int nDstYStep, Mpp8u * pDstCbCr, int nDstCbCrStep, MppiSize oSizeROI);

/** @} */

/** @defgroup ycrcb420toycbcr411 YCrCb420 To YCbCr411 
 *  YCrCb420 to YCbCr411 sampling format conversion.
 * @{
 */
/**
 * 3 channel 8-bit unsigned planar YCrCb420 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDstY \ref destination_planar_image_pointer.
 * \param nDstYStep \ref destination_planar_image_line_step.
 * \param pDstCbCr \ref destination_planar_image_pointer.
 * \param nDstCbCrStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCrCb420ToYCbCr411_8u_P3P2R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDstY, int nDstYStep, Mpp8u * pDstCbCr, int nDstCbCrStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCrCb420 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDstY \ref destination_planar_image_pointer.
 * \param nDstYStep \ref destination_planar_image_line_step.
 * \param pDstCbCr \ref destination_planar_image_pointer.
 * \param nDstCbCrStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCrCb420ToYCbCr411_8u_P3P2R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDstY, int nDstYStep, Mpp8u * pDstCbCr, int nDstCbCrStep, MppiSize oSizeROI);

/** @} */

/** @defgroup ycbcr411toycbcr411 YCbCr411 To YCbCr411 
 *  YCbCr411 to YCbCr411 sampling format conversion.
 * @{
 */
/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDstY \ref destination_planar_image_pointer.
 * \param nDstYStep \ref destination_planar_image_line_step.
 * \param pDstCbCr \ref destination_planar_image_pointer.
 * \param nDstCbCrStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411_8u_P3P2R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDstY, int nDstYStep, Mpp8u * pDstCbCr, int nDstCbCrStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDstY \ref destination_planar_image_pointer.
 * \param nDstYStep \ref destination_planar_image_line_step.
 * \param pDstCbCr \ref destination_planar_image_pointer.
 * \param nDstCbCrStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411_8u_P3P2R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDstY, int nDstYStep, Mpp8u * pDstCbCr, int nDstCbCrStep, MppiSize oSizeROI);

/**
 * 2 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
 *
 * \param pSrcY \ref source_planar_image_pointer.
 * \param nSrcYStep \ref source_planar_image_line_step.
 * \param pSrcCbCr \ref source_planar_image_pointer.
 * \param nSrcCbCrStep \ref source_planar_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411_8u_P2P3R_Ctx(const Mpp8u * pSrcY, int nSrcYStep, const Mpp8u * pSrcCbCr, int nSrcCbCrStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
 *
 * \param pSrcY \ref source_planar_image_pointer.
 * \param nSrcYStep \ref source_planar_image_line_step.
 * \param pSrcCbCr \ref source_planar_image_pointer.
 * \param nSrcCbCrStep \ref source_planar_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411_8u_P2P3R(const Mpp8u * pSrcY, int nSrcYStep, const Mpp8u * pSrcCbCr, int nSrcCbCrStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} */

/** @defgroup ycbcr411toycbcr422 YCbCr411 To YCbCr422 
 *  YCbCr411 to YCbCr422 sampling format conversion.
 * @{
 */
/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToYCbCr422_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst[3], int nDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToYCbCr422_8u_P3R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst[3], int nDstStep[3], MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToYCbCr422_8u_P3C2R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToYCbCr422_8u_P3C2R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 2 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
 *
 * \param pSrcY \ref source_planar_image_pointer.
 * \param nSrcYStep \ref source_planar_image_line_step.
 * \param pSrcCbCr \ref source_planar_image_pointer.
 * \param nSrcCbCrStep \ref source_planar_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToYCbCr422_8u_P2P3R_Ctx(const Mpp8u * const pSrcY, int nSrcYStep, const Mpp8u * pSrcCbCr, int nSrcCbCrStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
 *
 * \param pSrcY \ref source_planar_image_pointer.
 * \param nSrcYStep \ref source_planar_image_line_step.
 * \param pSrcCbCr \ref source_planar_image_pointer.
 * \param nSrcCbCrStep \ref source_planar_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToYCbCr422_8u_P2P3R(const Mpp8u * const pSrcY, int nSrcYStep, const Mpp8u * pSrcCbCr, int nSrcCbCrStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/**
 * 2 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
 *
 * \param pSrcY \ref source_planar_image_pointer.
 * \param nSrcYStep \ref source_planar_image_line_step.
 * \param pSrcCbCr \ref source_planar_image_pointer.
 * \param nSrcCbCrStep \ref source_planar_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToYCbCr422_8u_P2C2R_Ctx(const Mpp8u * pSrcY, int nSrcYStep, const Mpp8u * pSrcCbCr, int nSrcCbCrStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
 *
 * \param pSrcY \ref source_planar_image_pointer.
 * \param nSrcYStep \ref source_planar_image_line_step.
 * \param pSrcCbCr \ref source_planar_image_pointer.
 * \param nSrcCbCrStep \ref source_planar_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToYCbCr422_8u_P2C2R(const Mpp8u * pSrcY, int nSrcYStep, const Mpp8u * pSrcCbCr, int nSrcCbCrStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} */

/** @defgroup ycbcr411toycrcb422 YCbCr411 To YCrCb422 
 *  YCbCr411 to YCrCb422 sampling format conversion.
 * @{
 */
/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCrCb422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToYCrCb422_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst[3], int nDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCrCb422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToYCrCb422_8u_P3R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst[3], int nDstStep[3], MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned packed YCrCb422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToYCrCb422_8u_P3C2R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned packed YCrCb422 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToYCrCb422_8u_P3C2R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** @} */

/** @defgroup ycbcr411toycbcr420 YCbCr411 To YCbCr420 
 *  YCbCr411 to YCbCr420 sampling format conversion.
 * @{
 */
/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToYCbCr420_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst[3], int nDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToYCbCr420_8u_P3R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDst[3], int nDstStep[3], MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDstY \ref destination_planar_image_pointer.
 * \param nDstYStep \ref destination_planar_image_line_step.
 * \param pDstCbCr \ref destination_planar_image_pointer.
 * \param nDstCbCrStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToYCbCr420_8u_P3P2R_Ctx(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDstY, int nDstYStep, Mpp8u * pDstCbCr, int nDstCbCrStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 * images.
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param rSrcStep \ref source_planar_image_line_step_array.
 * \param pDstY \ref destination_planar_image_pointer.
 * \param nDstYStep \ref destination_planar_image_line_step.
 * \param pDstCbCr \ref destination_planar_image_pointer.
 * \param nDstCbCrStep \ref destination_planar_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToYCbCr420_8u_P3P2R(const Mpp8u * const pSrc[3], int rSrcStep[3], Mpp8u * pDstY, int nDstYStep, Mpp8u * pDstCbCr, int nDstCbCrStep, MppiSize oSizeROI);

/**
 * 2 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 *
 * \param pSrcY \ref source_planar_image_pointer.
 * \param nSrcYStep \ref source_planar_image_line_step.
 * \param pSrcCbCr \ref source_planar_image_pointer.
 * \param nSrcCbCrStep \ref source_planar_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToYCbCr420_8u_P2P3R_Ctx(const Mpp8u * pSrcY, int nSrcYStep, const Mpp8u * pSrcCbCr, int nSrcCbCrStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
 *
 * \param pSrcY \ref source_planar_image_pointer.
 * \param nSrcYStep \ref source_planar_image_line_step.
 * \param pSrcCbCr \ref source_planar_image_pointer.
 * \param nSrcCbCrStep \ref source_planar_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToYCbCr420_8u_P2P3R(const Mpp8u * pSrcY, int nSrcYStep, const Mpp8u * pSrcCbCr, int nSrcCbCrStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} */

/** @defgroup ycbcr411toycrcb420 YCbCr411 To YCrCb420 
 *  YCbCr411 to YCrCb420 sampling format conversion.
 * @{
 */
/**
 * 2 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCrCb420 sampling format conversion.
 *
 * \param pSrcY \ref source_planar_image_pointer.
 * \param nSrcYStep \ref source_planar_image_line_step.
 * \param pSrcCbCr \ref source_planar_image_pointer.
 * \param nSrcCbCrStep \ref source_planar_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToYCrCb420_8u_P2P3R_Ctx(const Mpp8u * pSrcY, int nSrcYStep, const Mpp8u * pSrcCbCr, int nSrcCbCrStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCrCb420 sampling format conversion.
 *
 * \param pSrcY \ref source_planar_image_pointer.
 * \param nSrcYStep \ref source_planar_image_line_step.
 * \param pSrcCbCr \ref source_planar_image_pointer.
 * \param nSrcCbCrStep \ref source_planar_image_line_step.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param rDstStep \ref destination_planar_image_line_step_array.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYCbCr411ToYCrCb420_8u_P2P3R(const Mpp8u * pSrcY, int nSrcYStep, const Mpp8u * pSrcCbCr, int nSrcCbCrStep, Mpp8u * pDst[3], int rDstStep[3], MppiSize oSizeROI);

/** @} */

/** @defgroup nv12toyuv420 NV12 To YUV420
 *  NV12 to YUV420 color conversion.
 *
 * @{
 *
 */

/**
 *  2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit unsigned planar YUV420 color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array (one for Y plane, one for UV plane).
 * \param nSrcStep \ref source_planar_image_line_step.  Same value is used for each plane.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiNV12ToYUV420_8u_P2P3R_Ctx(const Mpp8u * const pSrc[2], int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 *  2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit unsigned planar YUV420 color conversion.
 *
 * \param pSrc \ref source_planar_image_pointer_array (one for Y plane, one for UV plane).
 * \param nSrcStep \ref source_planar_image_line_step.  Same value is used for each plane.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiNV12ToYUV420_8u_P2P3R(const Mpp8u * const pSrc[2], int nSrcStep, Mpp8u * pDst[3], int aDstStep[3], MppiSize oSizeROI);

/** @} */

/** @} image_color_sampling_format_conversion */ 

/** 
 * @defgroup image_color_processing Color Processing
 *
 * Routines for performing image color manipulation.
 *
 * @{                                         
 *
 */

/** 
 * @defgroup image_color_gamma_correction Color Gamma Correction
 *
 * Routines for correcting image color gamma.
 *
 * @{                                         
 *
 */

/** @name GammaFwd 
 *  Forward gamma correction.
 * @{
 */

#ifndef MPP_PLUS
/**
 * 3 channel 8-bit unsigned packed color not in place forward gamma correction.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaFwd_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed color not in place forward gamma correction.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaFwd_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned packed color in place forward gamma correction.
 *
 * \param pSrcDst in place packed pixel image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaFwd_8u_C3IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed color in place forward gamma correction.
 *
 * \param pSrcDst in place packed pixel image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaFwd_8u_C3IR(Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed color with alpha not in place forward gamma correction.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaFwd_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed color with alpha not in place forward gamma correction.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaFwd_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed color with alpha in place forward gamma correction.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaFwd_8u_AC4IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed color with alpha in place forward gamma correction.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaFwd_8u_AC4IR(Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar color not in place forward gamma correction.
 *
 * \param pSrc source planar pixel format image pointer array.
 * \param nSrcStep source planar pixel format image line step.
 * \param pDst destination planar pixel format image pointer array.
 * \param nDstStep destination planar pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaFwd_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar color not in place forward gamma correction.
 *
 * \param pSrc source planar pixel format image pointer array.
 * \param nSrcStep source planar pixel format image line step.
 * \param pDst destination planar pixel format image pointer array.
 * \param nDstStep destination planar pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaFwd_8u_P3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar color in place forward gamma correction.
 *
 * \param pSrcDst in place planar pixel format image pointer array.
 * \param nSrcDstStep in place planar pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaFwd_8u_IP3R_Ctx(Mpp8u * const pSrcDst[3], int nSrcDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar color in place forward gamma correction.
 *
 * \param pSrcDst in place planar pixel format image pointer array.
 * \param nSrcDstStep in place planar pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaFwd_8u_IP3R(Mpp8u * const pSrcDst[3], int nSrcDstStep, MppiSize oSizeROI);

#endif

/** @} */

/** @name GammaInv 
 *  Inverse gamma correction.
 * @{
 */

#ifndef MPP_PLUS

/**
 * 3 channel 8-bit unsigned packed color not in place inverse gamma correction.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaInv_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed color not in place inverse gamma correction.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaInv_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned packed color in place inverse gamma correction.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaInv_8u_C3IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed color in place inverse gamma correction.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaInv_8u_C3IR(Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed color with alpha not in place inverse gamma correction.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaInv_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed color with alpha not in place inverse gamma correction.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaInv_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/**
 * 4 channel 8-bit unsigned packed color with alpha in place inverse gamma correction.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaInv_8u_AC4IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed color with alpha in place inverse gamma correction.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaInv_8u_AC4IR(Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar color not in place inverse gamma correction.
 *
 * \param pSrc source planar pixel format image pointer array.
 * \param nSrcStep source planar pixel format image line step.
 * \param pDst destination planar pixel format image pointer array.
 * \param nDstStep destination planar pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaInv_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar color not in place inverse gamma correction.
 *
 * \param pSrc source planar pixel format image pointer array.
 * \param nSrcStep source planar pixel format image line step.
 * \param pDst destination planar pixel format image pointer array.
 * \param nDstStep destination planar pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaInv_8u_P3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * pDst[3], int nDstStep, MppiSize oSizeROI);

/**
 * 3 channel 8-bit unsigned planar color in place inverse gamma correction.
 *
 * \param pSrcDst in place planar pixel format image pointer array.
 * \param nSrcDstStep in place planar pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaInv_8u_IP3R_Ctx(Mpp8u * const pSrcDst[3], int nSrcDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar color in place inverse gamma correction.
 *
 * \param pSrcDst in place planar pixel format image pointer array.
 * \param nSrcDstStep in place planar pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiGammaInv_8u_IP3R(Mpp8u * const pSrcDst[3], int nSrcDstStep, MppiSize oSizeROI);

#endif

/** @} */

/** @} image_color_gamma_correction */ 

/** 
 * @defgroup image_complement_color_key Complement Color Key
 *
 * Routines for performing complement color key replacement.
 *
 * @{                                         
 *
 */

/** @name CompColorKey 
 *  Complement color key replacement.
 * @{
 */
/**
 * 1 channel 8-bit unsigned packed color complement color key replacement of source image 1 by source image 2.
 *
 * \param pSrc1 source1 packed pixel format image pointer.
 * \param nSrc1Step source1 packed pixel format image line step.
 * \param pSrc2 source2 packed pixel format image pointer.
 * \param nSrc2Step source2 packed pixel format image line step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nColorKeyConst color key constant
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCompColorKey_8u_C1R_Ctx(const Mpp8u * pSrc1, int nSrc1Step, const Mpp8u * pSrc2, int nSrc2Step, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nColorKeyConst, MppStreamContext mppStreamCtx);

/**
 * 1 channel 8-bit unsigned packed color complement color key replacement of source image 1 by source image 2.
 *
 * \param pSrc1 source1 packed pixel format image pointer.
 * \param nSrc1Step source1 packed pixel format image line step.
 * \param pSrc2 source2 packed pixel format image pointer.
 * \param nSrc2Step source2 packed pixel format image line step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nColorKeyConst color key constant
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCompColorKey_8u_C1R(const Mpp8u * pSrc1, int nSrc1Step, const Mpp8u * pSrc2, int nSrc2Step, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nColorKeyConst);

/**
 * 3 channel 8-bit unsigned packed color complement color key replacement of source image 1 by source image 2.
 *
 * \param pSrc1 source1 packed pixel format image pointer.
 * \param nSrc1Step source1 packed pixel format image line step.
 * \param pSrc2 source2 packed pixel format image pointer.
 * \param nSrc2Step source2 packed pixel format image line step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nColorKeyConst color key constant array
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCompColorKey_8u_C3R_Ctx(const Mpp8u * pSrc1, int nSrc1Step, const Mpp8u * pSrc2, int nSrc2Step, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nColorKeyConst[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned packed color complement color key replacement of source image 1 by source image 2.
 *
 * \param pSrc1 source1 packed pixel format image pointer.
 * \param nSrc1Step source1 packed pixel format image line step.
 * \param pSrc2 source2 packed pixel format image pointer.
 * \param nSrc2Step source2 packed pixel format image line step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nColorKeyConst color key constant array
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCompColorKey_8u_C3R(const Mpp8u * pSrc1, int nSrc1Step, const Mpp8u * pSrc2, int nSrc2Step, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nColorKeyConst[3]);

/**
 * 4 channel 8-bit unsigned packed color complement color key replacement of source image 1 by source image 2.
 *
 * \param pSrc1 source1 packed pixel format image pointer.
 * \param nSrc1Step source1 packed pixel format image line step.
 * \param pSrc2 source2 packed pixel format image pointer.
 * \param nSrc2Step source2 packed pixel format image line step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nColorKeyConst color key constant array
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCompColorKey_8u_C4R_Ctx(const Mpp8u * pSrc1, int nSrc1Step, const Mpp8u * pSrc2, int nSrc2Step, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nColorKeyConst[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed color complement color key replacement of source image 1 by source image 2.
 *
 * \param pSrc1 source1 packed pixel format image pointer.
 * \param nSrc1Step source1 packed pixel format image line step.
 * \param pSrc2 source2 packed pixel format image pointer.
 * \param nSrc2Step source2 packed pixel format image line step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nColorKeyConst color key constant array
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiCompColorKey_8u_C4R(const Mpp8u * pSrc1, int nSrc1Step, const Mpp8u * pSrc2, int nSrc2Step, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp8u nColorKeyConst[4]);

/**
 * 4 channel 8-bit unsigned packed color complement color key replacement of source image 1 by source image 2 with alpha blending.
 *
 * \param pSrc1 source1 packed pixel format image pointer.
 * \param nSrc1Step source1 packed pixel format image line step.
 * \param nAlpha1 source1 image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 source2 packed pixel format image pointer.
 * \param nSrc2Step source2 packed pixel format image line step.
 * \param nAlpha2 source2 image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nColorKeyConst color key constant array
 * \param nppAlphaOp MppiAlphaOp alpha compositing operation selector  (excluding premul ops).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiAlphaCompColorKey_8u_AC4R_Ctx(const Mpp8u * pSrc1, int nSrc1Step, Mpp8u nAlpha1, const Mpp8u * pSrc2, int nSrc2Step, Mpp8u nAlpha2, Mpp8u * pDst, int nDstStep, 
                                            MppiSize oSizeROI, Mpp8u nColorKeyConst[4], MppiAlphaOp nppAlphaOp, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned packed color complement color key replacement of source image 1 by source image 2 with alpha blending.
 *
 * \param pSrc1 source1 packed pixel format image pointer.
 * \param nSrc1Step source1 packed pixel format image line step.
 * \param nAlpha1 source1 image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 source2 packed pixel format image pointer.
 * \param nSrc2Step source2 packed pixel format image line step.
 * \param nAlpha2 source2 image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nColorKeyConst color key constant array
 * \param nppAlphaOp MppiAlphaOp alpha compositing operation selector  (excluding premul ops).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiAlphaCompColorKey_8u_AC4R(const Mpp8u * pSrc1, int nSrc1Step, Mpp8u nAlpha1, const Mpp8u * pSrc2, int nSrc2Step, Mpp8u nAlpha2, Mpp8u * pDst, int nDstStep, 
                                        MppiSize oSizeROI, Mpp8u nColorKeyConst[4], MppiAlphaOp nppAlphaOp);

/** @} */

/** @} image_complement_color_key */ 

/** 
 * @defgroup image_color_twist ColorTwist
 *
 * Routines for converting between various image color models using user supplied matrix coefficients.
 *
 * @{ 
 *
 */

/** @name ColorTwist
 * 
 *  Perform color twist pixel processing.  Color twist consists of applying the following formula to each
 *  image pixel using coefficients from the user supplied color twist host matrix array as follows where 
 *  dst[x] and src[x] represent destination pixel and source pixel channel or plane x. The full sized
 *  coefficient matrix should be sent for all pixel channel sizes, the function will process the appropriate
 *  coefficients and channels for the corresponding pixel size.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3]
 *  \endcode
 *
 * @{
 */

/**
 * 1 channel 8-bit unsigned color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_C1R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, 
                             MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 1 channel 8-bit unsigned color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_C1R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, 
                         MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 1 channel 8-bit unsigned in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_C1IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 1 channel 8-bit unsigned in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_C1IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 2 channel 8-bit unsigned color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_C2R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, 
                             MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_C2R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, 
                         MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 2 channel 8-bit unsigned in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_C2IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit unsigned in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_C2IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 3 channel 8-bit unsigned color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, 
                             MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, 
                         MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 3 channel 8-bit unsigned in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_C3IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_C3IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 4 channel 8-bit unsigned color twist, with alpha copy.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is copied unmodified from the source pixel to the destination pixel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_C4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, 
                             MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned color twist, with alpha copy.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is copied unmodified from the source pixel to the destination pixel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_C4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, 
                         MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 4 channel 8-bit unsigned in place color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is unmodified.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_C4IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned in place color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is unmodified.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_C4IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 4 channel 8-bit unsigned color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 4 channel 8-bit unsigned in place color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_AC4IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                               MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned in place color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_AC4IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                           MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 4 channel 8-bit unsigned color twist with 4x4 matrix and constant vector addition.
 *
 * An input 4x4 color twist matrix with floating-point coefficient values with an additional constant vector addition
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aConstants[0]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aConstants[1]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aConstants[2]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aConstants[3]
 *  \endcode
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param aConstants fixed size array of constant values, one per channel..
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32fC_8u_C4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[4][4], const Mpp32f aConstants[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned color twist with 4x4 matrix and constant vector addition.
 *
 * An input 4x4 color twist matrix with floating-point coefficient values with an additional constant vector addition
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aConstants[0]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aConstants[1]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aConstants[2]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aConstants[3]
 *  \endcode
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param aConstants fixed size array of constant values, one per channel..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32fC_8u_C4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[4][4], const Mpp32f aConstants[4]);

/**
 * 4 channel 8-bit unsigned in place color twist with 4x4 matrix and an additional constant vector addition.
 *
 * An input 4x4 color twist matrix with floating-point coefficient values with an additional constant vector addition
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aConstants[0]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aConstants[1]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aConstants[2]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aConstants[3]
 *  \endcode
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param aConstants fixed size array of constant values, one per channel..
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32fC_8u_C4IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, 
                               MppiSize oSizeROI, const Mpp32f aTwist[4][4], const Mpp32f aConstants[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned in place color twist with 4x4 matrix and an additional constant vector addition.
 *
 * An input 4x4 color twist matrix with floating-point coefficient values with an additional constant vector addition
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aConstants[0]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aConstants[1]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aConstants[2]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aConstants[3]
 *  \endcode
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param aConstants fixed size array of constant values, one per channel..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32fC_8u_C4IR(Mpp8u * pSrcDst, int nSrcDstStep, 
                           MppiSize oSizeROI, const Mpp32f aTwist[4][4], const Mpp32f aConstants[4]);

/**
 * 3 channel 8-bit unsigned planar color twist.
 *
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_P3R_Ctx(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * const pDst[3], int nDstStep, 
                             MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar color twist.
 *
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_P3R(const Mpp8u * const pSrc[3], int nSrcStep, Mpp8u * const pDst[3], int nDstStep, 
                         MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 3 channel 8-bit unsigned planar in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place planar pixel format image pointer array, one pointer per plane.
 * \param nSrcDstStep in place planar pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_IP3R_Ctx(Mpp8u * const pSrcDst[3], int nSrcDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned planar in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place planar pixel format image pointer array, one pointer per plane.
 * \param nSrcDstStep in place planar pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8u_IP3R(Mpp8u * const pSrcDst[3], int nSrcDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 1 channel 8-bit signed color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_C1R_Ctx(const Mpp8s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, 
                             MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 1 channel 8-bit signed color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_C1R(const Mpp8s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, 
                         MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 1 channel 8-bit signed in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_C1IR_Ctx(Mpp8s * pSrcDst, int nSrcDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 1 channel 8-bit signed in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_C1IR(Mpp8s * pSrcDst, int nSrcDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 2 channel 8-bit signed color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_C2R_Ctx(const Mpp8s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, 
                             MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit signed color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_C2R(const Mpp8s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, 
                         MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 2 channel 8-bit signed in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_C2IR_Ctx(Mpp8s * pSrcDst, int nSrcDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 2 channel 8-bit signed in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_C2IR(Mpp8s * pSrcDst, int nSrcDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 3 channel 8-bit signed color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_C3R_Ctx(const Mpp8s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, 
                             MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit signed color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_C3R(const Mpp8s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, 
                         MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 3 channel 8-bit signed in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_C3IR_Ctx(Mpp8s * pSrcDst, int nSrcDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit signed in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_C3IR(Mpp8s * pSrcDst, int nSrcDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 4 channel 8-bit signed color twist, with alpha copy.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is copied unmodified from the source pixel to the destination pixel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_C4R_Ctx(const Mpp8s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, 
                             MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit signed color twist, with alpha copy.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is copied unmodified from the source pixel to the destination pixel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_C4R(const Mpp8s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, 
                         MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 4 channel 8-bit signed in place color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is unmodified.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_C4IR_Ctx(Mpp8s * pSrcDst, int nSrcDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit signed in place color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is unmodified.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_C4IR(Mpp8s * pSrcDst, int nSrcDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 4 channel 8-bit signed color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_AC4R_Ctx(const Mpp8s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit signed color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_AC4R(const Mpp8s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 4 channel 8-bit signed in place color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_AC4IR_Ctx(Mpp8s * pSrcDst, int nSrcDstStep, 
                               MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit signed in place color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_AC4IR(Mpp8s * pSrcDst, int nSrcDstStep, 
                           MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 3 channel 8-bit signed planar color twist.
 *
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_P3R_Ctx(const Mpp8s * const pSrc[3], int nSrcStep, Mpp8s * const pDst[3], int nDstStep, 
                             MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit signed planar color twist.
 *
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_P3R(const Mpp8s * const pSrc[3], int nSrcStep, Mpp8s * const pDst[3], int nDstStep, 
                         MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 3 channel 8-bit signed planar in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place planar pixel format image pointer array, one pointer per plane.
 * \param nSrcDstStep in place planar pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_IP3R_Ctx(Mpp8s * const pSrcDst[3], int nSrcDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit signed planar in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place planar pixel format image pointer array, one pointer per plane.
 * \param nSrcDstStep in place planar pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_8s_IP3R(Mpp8s * const pSrcDst[3], int nSrcDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);


/**
 * 1 channel 16-bit unsigned color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16u_C1R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 1 channel 16-bit unsigned color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16u_C1R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 1 channel 16-bit unsigned in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16u_C1IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                               MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 1 channel 16-bit unsigned in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16u_C1IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                           MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 2 channel 16-bit unsigned color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16u_C2R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 2 channel 16-bit unsigned color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16u_C2R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 2 channel 16-bit unsigned in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16u_C2IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                               MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 2 channel 16-bit unsigned in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16u_C2IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                           MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 3 channel 16-bit unsigned color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16u_C3R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit unsigned color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16u_C3R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 3 channel 16-bit unsigned in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16u_C3IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                               MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit unsigned in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16u_C3IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                           MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 4 channel 16-bit unsigned color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16u_AC4R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, 
                               MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit unsigned color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16u_AC4R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, 
                           MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 4 channel 16-bit unsigned in place color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16u_AC4IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, 
                                MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit unsigned in place color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16u_AC4IR(Mpp16u * pSrcDst, int nSrcDstStep, 
                            MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 3 channel 16-bit unsigned planar color twist.
 *
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16u_P3R_Ctx(const Mpp16u * const pSrc[3], int nSrcStep, Mpp16u * const pDst[3], int nDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit unsigned planar color twist.
 *
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16u_P3R(const Mpp16u * const pSrc[3], int nSrcStep, Mpp16u * const pDst[3], int nDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 3 channel 16-bit unsigned planar in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place planar pixel format image pointer array, one pointer per plane.
 * \param nSrcDstStep in place planar pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16u_IP3R_Ctx(Mpp16u * const pSrcDst[3], int nSrcDstStep, 
                               MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit unsigned planar in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place planar pixel format image pointer array, one pointer per plane.
 * \param nSrcDstStep in place planar pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16u_IP3R(Mpp16u * const pSrcDst[3], int nSrcDstStep, 
                           MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 1 channel 16-bit signed color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16s_C1R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 1 channel 16-bit signed color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16s_C1R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 1 channel 16-bit signed in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16s_C1IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                               MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 1 channel 16-bit signed in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16s_C1IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                           MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 2 channel 16-bit signed color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16s_C2R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 2 channel 16-bit signed color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16s_C2R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 2 channel 16-bit signed in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16s_C2IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                               MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 2 channel 16-bit signed in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16s_C2IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                           MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 3 channel 16-bit signed color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16s_C3R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit signed color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16s_C3R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 3 channel 16-bit signed in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16s_C3IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                               MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit signed in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16s_C3IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                           MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 4 channel 16-bit signed color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16s_AC4R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, 
                               MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit signed color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16s_AC4R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, 
                           MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 4 channel 16-bit signed in place color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16s_AC4IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, 
                                MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit signed in place color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16s_AC4IR(Mpp16s * pSrcDst, int nSrcDstStep, 
                            MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 3 channel 16-bit signed planar color twist.
 *
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16s_P3R_Ctx(const Mpp16s * const pSrc[3], int nSrcStep, Mpp16s * const pDst[3], int nDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit signed planar color twist.
 *
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16s_P3R(const Mpp16s * const pSrc[3], int nSrcStep, Mpp16s * const pDst[3], int nDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 3 channel 16-bit signed planar in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place planar pixel format image pointer array, one pointer per plane.
 * \param nSrcDstStep in place planar pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16s_IP3R_Ctx(Mpp16s * const pSrcDst[3], int nSrcDstStep, 
                               MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit signed planar in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place planar pixel format image pointer array, one pointer per plane.
 * \param nSrcDstStep in place planar pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16s_IP3R(Mpp16s * const pSrcDst[3], int nSrcDstStep, 
                           MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 1 channel 16-bit floating point color twist.
 * 
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16f_C1R_Ctx(const Mpp16f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 1 channel 16-bit floating point color twist.
 * 
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16f_C1R(const Mpp16f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 1 channel 16-bit floating point in place color twist.
 * 
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16f_C1IR_Ctx(Mpp16f * pSrcDst, int nSrcDstStep, 
                               MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 1 channel 16-bit floating point in place color twist.
 * 
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16f_C1IR(Mpp16f * pSrcDst, int nSrcDstStep, 
                           MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 2 channel 16-bit floating point color twist.
 * 
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16f_C2R_Ctx(const Mpp16f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 2 channel 16-bit floating point color twist.
 * 
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16f_C2R(const Mpp16f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 2 channel 16-bit floating point in place color twist.
 * 
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16f_C2IR_Ctx(Mpp16f * pSrcDst, int nSrcDstStep, 
                               MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 2 channel 16-bit floating point in place color twist.
 * 
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16f_C2IR(Mpp16f * pSrcDst, int nSrcDstStep, 
                           MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 3 channel 16-bit floating point color twist.
 * 
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16f_C3R_Ctx(const Mpp16f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit floating point color twist.
 * 
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16f_C3R(const Mpp16f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 3 channel 16-bit floating point in place color twist.
 * 
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16f_C3IR_Ctx(Mpp16f * pSrcDst, int nSrcDstStep, 
                               MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit floating point in place color twist.
 * 
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16f_C3IR(Mpp16f * pSrcDst, int nSrcDstStep, 
                           MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 4 channel 16-bit floating point color twist, with alpha copy.
 *
 * An input color twist matrix with 32-bit floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is copied unmodified from the source pixel to the destination pixel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16f_C4R_Ctx(const Mpp16f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, 
                              MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit floating point color twist, with alpha copy.
 *
 * An input color twist matrix with 32-bit floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is copied unmodified from the source pixel to the destination pixel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16f_C4R(const Mpp16f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, 
                          MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 4 channel 16-bit floating point in place color twist, not affecting Alpha.
 *
 * An input color twist matrix with 32-bit floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not modified.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16f_C4IR_Ctx(Mpp16f * pSrcDst, int nSrcDstStep, 
                               MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit floating point in place color twist, not affecting Alpha.
 *
 * An input color twist matrix with 32-bit floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not modified.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32f_16f_C4IR(Mpp16f * pSrcDst, int nSrcDstStep, 
                           MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 4 channel 16-bit floating point color twist with 4x4 matrix and constant vector addition.
 *
 * An input 4x4 color twist matrix with 32-bit floating-point coefficient values with an additional 32-bit floating point constant vector addition
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aConstants[0]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aConstants[1]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aConstants[2]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aConstants[3]
 *  \endcode
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param aConstants fixed size array of constant values, one per channel..
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32fC_16f_C4R_Ctx(const Mpp16f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, 
                               MppiSize oSizeROI, const Mpp32f aTwist[4][4], const Mpp32f aConstants[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit floating point color twist with 4x4 matrix and constant vector addition.
 *
 * An input 4x4 color twist matrix with 32-bit floating-point coefficient values with an additional 32-bit floating point constant vector addition
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aConstants[0]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aConstants[1]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aConstants[2]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aConstants[3]
 *  \endcode
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param aConstants fixed size array of constant values, one per channel..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32fC_16f_C4R(const Mpp16f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, 
                           MppiSize oSizeROI, const Mpp32f aTwist[4][4], const Mpp32f aConstants[4]);

/**
 * 4 channel 16-bit floating point in place color twist with 4x4 matrix and an additional constant vector addition.
 *
 * An input 4x4 color twist matrix with 32-bit floating-point coefficient values with an additional 32-bit floating point constant vector addition
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aConstants[0]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aConstants[1]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aConstants[2]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aConstants[3]
 *  \endcode
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param aConstants fixed size array of constant values, one per channel..
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32fC_16f_C4IR_Ctx(Mpp16f * pSrcDst, int nSrcDstStep, 
                                MppiSize oSizeROI, const Mpp32f aTwist[4][4], const Mpp32f aConstants[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit floating point in place color twist with 4x4 matrix and an additional constant vector addition.
 *
 * An input 4x4 color twist matrix with 32-bit floating-point coefficient values with an additional 32-bit floating point constant vector addition
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aConstants[0]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aConstants[1]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aConstants[2]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aConstants[3]
 *  \endcode
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param aConstants fixed size array of constant values, one per channel..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist32fC_16f_C4IR(Mpp16f * pSrcDst, int nSrcDstStep, 
                            MppiSize oSizeROI, const Mpp32f aTwist[4][4], const Mpp32f aConstants[4]);

/**
 * 1 channel 32-bit floating point color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_C1R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, 
                           MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 1 channel 32-bit floating point color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_C1R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, 
                       MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 1 channel 32-bit floating point in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_C1IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                            MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 1 channel 32-bit floating point in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_C1IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                        MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 2 channel 32-bit floating point color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_C2R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, 
                           MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 2 channel 32-bit floating point color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_C2R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, 
                       MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 2 channel 32-bit floating point in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_C2IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                            MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 2 channel 32-bit floating point in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_C2IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                        MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 3 channel 32-bit floating point color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, 
                           MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit floating point color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_C3R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, 
                       MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 3 channel 32-bit floating point in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_C3IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                            MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit floating point in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_C3IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                        MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 4 channel 32-bit floating point color twist, with alpha copy.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is copied unmodified from the source pixel to the destination pixel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_C4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, 
                           MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point color twist, with alpha copy.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is copied unmodified from the source pixel to the destination pixel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_C4R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, 
                       MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 4 channel 32-bit floating point in place color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not modified.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_C4IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                            MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point in place color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not modified.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_C4IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                        MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 4 channel 32-bit floating point color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_AC4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, 
                            MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_AC4R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, 
                        MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 4 channel 32-bit floating point in place color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_AC4IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                             MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point in place color twist, not affecting Alpha.
 *
 * An input color twist matrix with floating-point coefficient values is applied with
 * in ROI.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_AC4IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                         MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 4 channel 32-bit floating point color twist with 4x4 matrix and constant vector addition.
 *
 * An input 4x4 color twist matrix with floating-point coefficient values with an additional constant vector addition
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aConstants[0]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aConstants[1]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aConstants[2]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aConstants[3]
 *  \endcode
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param aConstants fixed size array of constant values, one per channel..
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32fC_C4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, 
                            MppiSize oSizeROI, const Mpp32f aTwist[4][4], const Mpp32f aConstants[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point color twist with 4x4 matrix and constant vector addition.
 *
 * An input 4x4 color twist matrix with floating-point coefficient values with an additional constant vector addition
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aConstants[0]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aConstants[1]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aConstants[2]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aConstants[3]
 *  \endcode
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param aConstants fixed size array of constant values, one per channel..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32fC_C4R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, 
                        MppiSize oSizeROI, const Mpp32f aTwist[4][4], const Mpp32f aConstants[4]);

/**
 * 4 channel 32-bit floating point in place color twist with 4x4 matrix and an additional constant vector addition.
 *
 * An input 4x4 color twist matrix with floating-point coefficient values with an additional constant vector addition
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aConstants[0]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aConstants[1]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aConstants[2]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aConstants[3]
 *  \endcode
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param aConstants fixed size array of constant values, one per channel..
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32fC_C4IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, 
                             MppiSize oSizeROI, const Mpp32f aTwist[4][4], const Mpp32f aConstants[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point in place color twist with 4x4 matrix and an additional constant vector addition.
 *
 * An input 4x4 color twist matrix with floating-point coefficient values with an additional constant vector addition
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aConstants[0]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aConstants[1]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aConstants[2]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aConstants[3]
 *  \endcode
 *
 * \param pSrcDst in place packed pixel format image pointer.
 * \param nSrcDstStep in place packed pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param aConstants fixed size array of constant values, one per channel..
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32fC_C4IR(Mpp32f * pSrcDst, int nSrcDstStep, 
                         MppiSize oSizeROI, const Mpp32f aTwist[4][4], const Mpp32f aConstants[4]);

/**
 * 3 channel 32-bit floating point planar color twist.
 *
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_P3R_Ctx(const Mpp32f * const pSrc[3], int nSrcStep, Mpp32f * const pDst[3], int nDstStep, 
                           MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit floating point planar color twist.
 *
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_P3R(const Mpp32f * const pSrc[3], int nSrcStep, Mpp32f * const pDst[3], int nDstStep, 
                       MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/**
 * 3 channel 32-bit floating point planar in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place planar pixel format image pointer array, one pointer per plane.
 * \param nSrcDstStep in place planar pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_IP3R_Ctx(Mpp32f * const pSrcDst[3], int nSrcDstStep, 
                            MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);
/**
 * 3 channel 32-bit floating point planar in place color twist.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI.
 *
 * \param pSrcDst in place planar pixel format image pointer array, one pointer per plane.
 * \param nSrcDstStep in place planar pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwist_32f_IP3R(Mpp32f * const pSrcDst[3], int nSrcDstStep, 
                        MppiSize oSizeROI, const Mpp32f aTwist[3][4]);

/** @} color_twist */

/** @} image_color_twist */

/** 
 * @defgroup image_color_twist_batch ColorTwistBatch
 *
 * Routines for converting between various image color models using user supplied matrix coefficients on batches of images.
 *
 * @{ 
 *
 */
 
/** @name ColorTwistBatch
 * 
 * Perform color twist pixel batch processing.  Color twist consists of applying the following formula to each
 * image pixel using coefficients from one or more user supplied color twist device memory matrix arrays as follows where 
 * dst[x] and src[x] represent destination pixel and source pixel channel or plane x. The full sized
 * coefficient matrix should be sent for all pixel channel sizes, the function will process the appropriate
 * coefficients and channels for the corresponding pixel size.  ColorTwistBatch generally takes the same parameter list as ColorTwist
 * except that there is a list of N instances of those parameters (N > 1) and that list is passed in device memory;  The matrix
 * pointers referenced for each image in the batch also need to point to device memory matrix values. A convenient
 * data structure is provided that allows for easy initialization of the parameter lists.  The only restriction on these functions is
 * that there is one single ROI which is applied respectively to each image in the batch.  The primary purpose of this function is to
 * provide improved performance for batches of smaller images as long as GPU resources are available.  Therefore it is recommended
 * that the function not be used for very large images as there may not be resources available for processing several large images
 * simultaneously.  
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3]
 *  \endcode
 *
 * @{
 */

/**
 * 1 channel 8-bit unsigned integer color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within the ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_8u_C1R_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 1 channel 8-bit unsigned integer color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within the ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_8u_C1R(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 1 channel 8-bit unsigned integer in place color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_8u_C1IR_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 1 channel 8-bit unsigned integer in place color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_8u_C1IR(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 3 channel 8-bit unsigned integer color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within the ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_8u_C3R_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned integer color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within the ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_8u_C3R(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 3 channel 8-bit unsigned integer in place color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_8u_C3IR_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned integer in place color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_8u_C3IR(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 4 channel 8-bit unsigned integer color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within the ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_8u_C4R_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned integer color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within the ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_8u_C4R(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 4 channel 8-bit unsigned integer in place color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_8u_C4IR_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned integer in place color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_8u_C4IR(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 4 channel 8-bit unsigned integer color twist batch, not affecting Alpha.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within the ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_8u_AC4R_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned integer color twist batch, not affecting Alpha.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within the ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_8u_AC4R(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 4 channel 8-bit unsigned integer in place color twist batch, not affecting Alpha.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_8u_AC4IR_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned integer in place color twist batch, not affecting Alpha.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_8u_AC4IR(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 4 channel 8-bit unsigned integer color twist with 4x5 matrix including a constant vector (20 coefficients total).
 *
 * An input 4x5 color twist matrix with floating-point coefficient values including a constant (in the fourth column) vector
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aTwist[0][4]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aTwist[1][4]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aTwist[2][4]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aTwist[3][4]
 *  \endcode
 *
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32fC_8u_C4R_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned integer color twist with 4x5 matrix including a constant vector (20 coefficients total).
 *
 * An input 4x5 color twist matrix with floating-point coefficient values including a constant (in the fourth column) vector
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aTwist[0][4]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aTwist[1][4]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aTwist[2][4]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aTwist[3][4]
 *  \endcode
 *
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32fC_8u_C4R(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 4 channel 8-bit unsigned integer in place color twist with 4x5 matrix including a constant vector (20 coefficients total).
 *
 * An input 4x5 color twist matrix with floating-point coefficient values including a constant (in the fourth column) vector
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aTwist[0][4]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aTwist[1][4]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aTwist[2][4]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aTwist[3][4]
 *  \endcode
 *
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32fC_8u_C4IR_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned integer in place color twist with 4x5 matrix including a constant vector (20 coefficients total).
 *
 * An input 4x5 color twist matrix with floating-point coefficient values including a constant (in the fourth column) vector
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aTwist[0][4]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aTwist[1][4]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aTwist[2][4]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aTwist[3][4]
 *  \endcode
 *
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32fC_8u_C4IR(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 1 channel 32-bit floating point color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within the ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch_32f_C1R_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 1 channel 32-bit floating point color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within the ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch_32f_C1R(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 1 channel 32-bit floating point in place color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch_32f_C1IR_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 1 channel 32-bit floating point in place color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch_32f_C1IR(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 3 channel 32-bit floating point color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within the ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch_32f_C3R_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit floating point color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within the ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch_32f_C3R(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 3 channel 32-bit floating point in place color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch_32f_C3IR_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit floating point in place color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch_32f_C3IR(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 4 channel 32-bit floating point color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within the ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch_32f_C4R_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within the ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch_32f_C4R(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 4 channel 32-bit floating point in place color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch_32f_C4IR_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point in place color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch_32f_C4IR(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 4 channel 32-bit floating point color twist batch, not affecting Alpha.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within the ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch_32f_AC4R_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point color twist batch, not affecting Alpha.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within the ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch_32f_AC4R(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 4 channel 32-bit floating point in place color twist batch, not affecting Alpha.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch_32f_AC4IR_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point in place color twist batch, not affecting Alpha.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch_32f_AC4IR(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 4 channel 32-bit floating point color twist with 4x5 matrix including a constant vector (20 coefficients total).
 *
 * An input 4x5 color twist matrix with floating-point coefficient values including a constant (in the fourth column) vector
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aTwist[0][4]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aTwist[1][4]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aTwist[2][4]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aTwist[3][4]
 *  \endcode
 *
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch_32fC_C4R_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point color twist with 4x5 matrix including a constant vector (20 coefficients total).
 *
 * An input 4x5 color twist matrix with floating-point coefficient values including a constant (in the fourth column) vector
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aTwist[0][4]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aTwist[1][4]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aTwist[2][4]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aTwist[3][4]
 *  \endcode
 *
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch_32fC_C4R(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 4 channel in place 32-bit floating point color twist with 4x5 matrix including a constant vector (20 coefficients total).
 *
 * An input 4x5 color twist matrix with floating-point coefficient values including a constant (in the fourth column) vector
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aTwist[0][4]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aTwist[1][4]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aTwist[2][4]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aTwist[3][4]
 *  \endcode
 *
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch_32fC_C4IR_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 4 channel in place 32-bit floating point color twist with 4x5 matrix including a constant vector (20 coefficients total).
 *
 * An input 4x5 color twist matrix with floating-point coefficient values including a constant (in the fourth column) vector
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aTwist[0][4]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aTwist[1][4]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aTwist[2][4]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aTwist[3][4]
 *  \endcode
 *
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch_32fC_C4IR(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 1 channel 16-bit floating point color twist batch.
 * 
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within the ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_16f_C1R_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 1 channel 16-bit floating point color twist batch.
 * 
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within the ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_16f_C1R(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 1 channel 16-bit floating point in place color twist batch.
 * 
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_16f_C1IR_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 1 channel 16-bit floating point in place color twist batch.
 * 
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_16f_C1IR(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 3 channel 16-bit floating point color twist batch.
 * 
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within the ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_16f_C3R_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit floating point color twist batch.
 * 
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within the ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_16f_C3R(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 3 channel 16-bit floating point in place color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_16f_C3IR_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit floating point in place color twist batch.
 * 
 * An input color twist matrix with floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_16f_C3IR(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 4 channel 16-bit floating point color twist batch.
 * 
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within the ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_16f_C4R_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit floating point color twist batch.
 * 
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within the ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_16f_C4R(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 4 channel 16-bit floating point in place color twist batch.
 * 
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_16f_C4IR_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit floating point in place color twist batch.
 * 
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32f_16f_C4IR(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 4 channel 16-bit floating point color twist with 4x5 matrix including a constant vector (20 coefficients total).
 *
 * An input 4x5 color twist matrix with 32-bit floating-point coefficient values including a constant (in the fourth column) vector
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aTwist[0][4]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aTwist[1][4]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aTwist[2][4]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aTwist[3][4]
 *  \endcode
 *
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32fC_16f_C4R_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit floating point color twist with 4x5 matrix including a constant vector (20 coefficients total).
 *
 * An input 4x5 color twist matrix with 32-bit floating-point coefficient values including a constant (in the fourth column) vector
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aTwist[0][4]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aTwist[1][4]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aTwist[2][4]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aTwist[3][4]
 *  \endcode
 *
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32fC_16f_C4R(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/**
 * 4 channel in place 16-bit floating point color twist with 4x5 matrix including a constant vector (20 coefficients total).
 *
 * An input 4x5 color twist matrix with 32-bitfloating-point coefficient values including a constant (in the fourth column) vector
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aTwist[0][4]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aTwist[1][4]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aTwist[2][4]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aTwist[3][4]
 *  \endcode
 *
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32fC_16f_C4IR_Ctx(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize, MppStreamContext mppStreamCtx);

/**
 * 4 channel in place 16-bit floating point color twist with 4x5 matrix including a constant vector (20 coefficients total).
 *
 * An input 4x5 color twist matrix with 32-bitfloating-point coefficient values including a constant (in the fourth column) vector
 * is applied within ROI.  For this particular version of the function the result is generated as shown below.
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3] * src[3] + aTwist[0][4]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3] * src[3] + aTwist[1][4]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3] * src[3] + aTwist[2][4]
 *      dst[3] = aTwist[3][0] * src[0] + aTwist[3][1] * src[1] + aTwist[3][2] * src[2] + aTwist[3][3] * src[3] + aTwist[3][4]
 *  \endcode
 *
 * An input color twist matrix with 32-bit floating-point coefficient values is applied
 * within ROI for each image in batch.  Color twist matrix can vary per image. The same ROI is applied to each image.
 *
 * \param nMin Minimum clamp value.
 * \param nMax Maximum saturation and clamp value.
 * \param oSizeROI \ref roi_specification.
 * \param pBatchList Device memory pointer to nBatchSize list of MppiColorTwistBatchCXR structures.
 * \param nBatchSize Number of MppiColorTwistBatchCXR structures in this call (must be > 1).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus 
mppiColorTwistBatch32fC_16f_C4IR(Mpp32f nMin, Mpp32f nMax, MppiSize oSizeROI, MppiColorTwistBatchCXR * pBatchList, int nBatchSize);

/** @} color_twist_batch */

/** @} image_color_twist_batch */

/** 
 * @defgroup image_color_LUT ColorLUT
 *  Perform image color processing using members of various types of color look up tables.
 * 
 * @{
 */

/**
 * 8-bit unsigned look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_8u_C1R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                   const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 8-bit unsigned look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_8u_C1R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
               const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels);

/**
 * 8-bit unsigned look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_8u_C1IR_Ctx(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                    const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 8-bit unsigned look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_8u_C1IR(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels);

/**
 * 3 channel 8-bit unsigned look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                   const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
               const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 3 channel 8-bit unsigned look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_8u_C3IR_Ctx(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                    const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_8u_C3IR(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 4 channel 8-bit unsigned look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_8u_C4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                   const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_8u_C4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
               const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4]);

/**
 * 4 channel 8-bit unsigned look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_8u_C4IR_Ctx(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                    const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_8u_C4IR(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4]);

/**
 * 4 channel 8-bit unsigned look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                    const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 4 channel 8-bit unsigned look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_8u_AC4IR_Ctx(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                     const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_8u_AC4IR(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                 const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 16-bit unsigned look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16u_C1R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                    const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 16-bit unsigned look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16u_C1R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels);

/**
 * 16-bit unsigned look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16u_C1IR_Ctx(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                     const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 16-bit unsigned look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16u_C1IR(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                 const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels);

/**
 * 3 channel 16-bit unsigned look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16u_C3R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                    const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit unsigned look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16u_C3R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 3 channel 16-bit unsigned look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16u_C3IR_Ctx(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                     const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit unsigned look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16u_C3IR(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                 const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 4 channel 16-bit unsigned look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16u_C4R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                    const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit unsigned look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16u_C4R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4]);

/**
 * 4 channel 16-bit unsigned look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16u_C4IR_Ctx(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                     const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit unsigned look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16u_C4IR(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                 const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4]);

/**
 * 4 channel 16-bit unsigned look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16u_AC4R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                     const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit unsigned look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16u_AC4R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                 const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 4 channel 16-bit unsigned look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16u_AC4IR_Ctx(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                      const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit unsigned look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16u_AC4IR(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                  const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 16-bit signed look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16s_C1R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                    const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 16-bit signed look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16s_C1R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels);

/**
 * 16-bit signed look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16s_C1IR_Ctx(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                     const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 16-bit signed look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16s_C1IR(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                 const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels);

/**
 * 3 channel 16-bit signed look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16s_C3R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                    const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit signed look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16s_C3R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 3 channel 16-bit signed look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16s_C3IR_Ctx(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                     const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit signed look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16s_C3IR(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                 const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 4 channel 16-bit signed look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16s_C4R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                    const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit signed look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16s_C4R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4]);

/**
 * 4 channel 16-bit signed look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16s_C4IR_Ctx(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                     const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit signed look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16s_C4IR(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                 const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4]);

/**
 * 4 channel 16-bit signed look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16s_AC4R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                     const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit signed look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16s_AC4R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                 const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 4 channel 16-bit signed look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16s_AC4IR_Ctx(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                      const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit signed look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_16s_AC4IR(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                  const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 32-bit floating point look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_32f_C1R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                    const Mpp32f * pValues, const Mpp32f * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 32-bit floating point look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_32f_C1R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                const Mpp32f * pValues, const Mpp32f * pLevels, int nLevels);

/**
 * 32-bit floating point look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_32f_C1IR_Ctx(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                     const Mpp32f * pValues, const Mpp32f * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 32-bit floating point look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_32f_C1IR(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                 const Mpp32f * pValues, const Mpp32f * pLevels, int nLevels);

/**
 * 3 channel 32-bit floating point look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_32f_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                    const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit floating point look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_32f_C3R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3]);

/**
 * 3 channel 32-bit floating point look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_32f_C3IR_Ctx(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                     const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit floating point look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_32f_C3IR(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                 const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3]);

/**
 * 4 channel 32-bit floating point look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_32f_C4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                    const Mpp32f * pValues[4], const Mpp32f * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points with no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_32f_C4R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                const Mpp32f * pValues[4], const Mpp32f * pLevels[4], int nLevels[4]);

/**
 * 4 channel 32-bit floating point look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_32f_C4IR_Ctx(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                     const Mpp32f * pValues[4], const Mpp32f * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_32f_C4IR(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                 const Mpp32f * pValues[4], const Mpp32f * pLevels[4], int nLevels[4]);

/**
 * 4 channel 32-bit floating point look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_32f_AC4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                     const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_32f_AC4R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                 const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3]);

/**
 * 4 channel 32-bit floating point look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_32f_AC4IR_Ctx(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                      const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_32f_AC4IR(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                  const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3]);

/** @} image_color_LUT */

/** @defgroup image_color_LUT_Linear ColorLUTLinear
 * Perform image color processing using linear interpolation between members of various types of color look up tables.
 * 
 * @{
 */

/**
 * 8-bit unsigned linear interpolated look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points through linear interpolation. 
 *
 * >>>>>>> ATTENTION ATTENTION <<<<<<<
 *
 * NOTE: As of the 5.0 release of MPP, the pValues and pLevels pointers need to be device memory pointers.
 *
 * >>>>>>>                     <<<<<<<
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is now a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is now a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Linear_8u_C1R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                          const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 8-bit unsigned linear interpolated look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points through linear interpolation. 
 *
 * >>>>>>> ATTENTION ATTENTION <<<<<<<
 *
 * NOTE: As of the 5.0 release of MPP, the pValues and pLevels pointers need to be device memory pointers.
 *
 * >>>>>>>                     <<<<<<<
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is now a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is now a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Linear_8u_C1R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels);

/**
 * 8-bit unsigned linear interpolated look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points through linear interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Linear_8u_C1IR_Ctx(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                           const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 8-bit unsigned linear interpolated look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points through linear interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Linear_8u_C1IR(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                       const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels);

/**
 * 3 channel 8-bit unsigned linear interpolated look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through linear interpolation. 
 *
 * >>>>>>> ATTENTION ATTENTION <<<<<<<
 *
 * NOTE: As of the 5.0 release of MPP, the pValues and pLevels pointers need to be host memory pointers to arrays of device memory pointers.
 *
 * >>>>>>>                     <<<<<<<
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Linear_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                          const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned linear interpolated look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through linear interpolation. 
 *
 * >>>>>>> ATTENTION ATTENTION <<<<<<<
 *
 * NOTE: As of the 5.0 release of MPP, the pValues and pLevels pointers need to be host memory pointers to arrays of device memory pointers.
 *
 * >>>>>>>                     <<<<<<<
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Linear_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 3 channel 8-bit unsigned linear interpolated look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through linear interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Linear_8u_C3IR_Ctx(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                           const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned linear interpolated look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through linear interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Linear_8u_C3IR(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                       const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 4 channel 8-bit unsigned linear interpolated look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through linear interpolation. 
 *
 * >>>>>>> ATTENTION ATTENTION <<<<<<<
 *
 * NOTE: As of the 5.0 release of MPP, the pValues and pLevels pointers need to be host memory pointers to arrays of device memory pointers.
 *
 * >>>>>>>                     <<<<<<<
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Linear_8u_C4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                          const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned linear interpolated look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through linear interpolation. 
 *
 * >>>>>>> ATTENTION ATTENTION <<<<<<<
 *
 * NOTE: As of the 5.0 release of MPP, the pValues and pLevels pointers need to be host memory pointers to arrays of device memory pointers.
 *
 * >>>>>>>                     <<<<<<<
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Linear_8u_C4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4]);

/**
 * 4 channel 8-bit unsigned linear interpolated look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through linear interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Linear_8u_C4IR_Ctx(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                           const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned linear interpolated look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through linear interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Linear_8u_C4IR(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                       const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4]);

/**
 * 4 channel 8-bit unsigned linear interpolated look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points through linear interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * >>>>>>> ATTENTION ATTENTION <<<<<<<
 *
 * NOTE: As of the 5.0 release of MPP, the pValues and pLevels pointers need to be host memory pointers to arrays of device memory pointers.
 *
 * >>>>>>>                     <<<<<<<
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Linear_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                           const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned linear interpolated look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points through linear interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * >>>>>>> ATTENTION ATTENTION <<<<<<<
 *
 * NOTE: As of the 5.0 release of MPP, the pValues and pLevels pointers need to be host memory pointers to arrays of device memory pointers.
 *
 * >>>>>>>                     <<<<<<<
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Linear_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                       const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 4 channel 8-bit unsigned linear interpolated look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points through linear interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Linear_8u_AC4IR_Ctx(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                            const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);
       
/**
 * 4 channel 8-bit unsigned linear interpolated look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points through linear interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Linear_8u_AC4IR(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                        const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);
       
/**
 * 16-bit unsigned look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16u_C1R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                           const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 16-bit unsigned look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16u_C1R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                       const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels);

/**
 * 16-bit unsigned look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16u_C1IR_Ctx(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                            const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 16-bit unsigned look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16u_C1IR(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                        const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels);

/**
 * 3 channel 16-bit unsigned look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16u_C3R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                           const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit unsigned look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16u_C3R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                       const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 3 channel 16-bit unsigned look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16u_C3IR_Ctx(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                            const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit unsigned look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16u_C3IR(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                        const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 4 channel 16-bit unsigned look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16u_C4R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                           const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit unsigned look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16u_C4R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                       const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4]);

/**
 * 4 channel 16-bit unsigned look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16u_C4IR_Ctx(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                            const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit unsigned look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16u_C4IR(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                        const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4]);

/**
 * 4 channel 16-bit unsigned look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16u_AC4R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                            const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit unsigned look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16u_AC4R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                        const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 4 channel 16-bit unsigned look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16u_AC4IR_Ctx(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                             const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit unsigned look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16u_AC4IR(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                         const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 16-bit signed look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16s_C1R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                           const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 16-bit signed look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16s_C1R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                       const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels);

/**
 * 16-bit signed look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16s_C1IR_Ctx(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                            const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 16-bit signed look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16s_C1IR(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                        const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels);

/**
 * 3 channel 16-bit signed look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16s_C3R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                           const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit signed look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16s_C3R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                       const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 3 channel 16-bit signed look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16s_C3IR_Ctx(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                            const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit signed look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16s_C3IR(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                        const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 4 channel 16-bit signed look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16s_C4R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                           const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit signed look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16s_C4R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                       const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4]);

/**
 * 4 channel 16-bit signed look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16s_C4IR_Ctx(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                            const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit signed look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16s_C4IR(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                        const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4]);

/**
 * 4 channel 16-bit signed look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16s_AC4R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                            const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit signed look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16s_AC4R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                        const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 4 channel 16-bit signed look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16s_AC4IR_Ctx(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                             const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit signed look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_16s_AC4IR(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                         const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 32-bit floating point look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_32f_C1R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                           const Mpp32f * pValues, const Mpp32f * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 32-bit floating point look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_32f_C1R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                       const Mpp32f * pValues, const Mpp32f * pLevels, int nLevels);

/**
 * 32-bit floating point look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_32f_C1IR_Ctx(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                            const Mpp32f * pValues, const Mpp32f * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 32-bit floating point look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_32f_C1IR(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                        const Mpp32f * pValues, const Mpp32f * pLevels, int nLevels);

/**
 * 3 channel 32-bit floating point look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_32f_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                           const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit floating point look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_32f_C3R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                       const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3]);

/**
 * 3 channel 32-bit floating point look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_32f_C3IR_Ctx(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                            const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit floating point look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_32f_C3IR(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                        const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3]);

/**
 * 4 channel 32-bit floating point look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_32f_C4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                           const Mpp32f * pValues[4], const Mpp32f * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using linear interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_32f_C4R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                       const Mpp32f * pValues[4], const Mpp32f * pLevels[4], int nLevels[4]);

/**
 * 4 channel 32-bit floating point look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_32f_C4IR_Ctx(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                            const Mpp32f * pValues[4], const Mpp32f * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_32f_C4IR(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                        const Mpp32f * pValues[4], const Mpp32f * pLevels[4], int nLevels[4]);

/**
 * 4 channel 32-bit floating point look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_32f_AC4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                            const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);
/**
 * 4 channel 32-bit floating point look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */

MppStatus 
mppiLUT_Linear_32f_AC4R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                        const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3]);

/**
 * 4 channel 32-bit floating point look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_32f_AC4IR_Ctx(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                             const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Linear_32f_AC4IR(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                         const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3]);

/** @} image_color_LUT_linear */

/** @defgroup image_color_LUT_cubic ColorLUTCubic
 *  Perform image color processing using linear interpolation between members of various types of color look up tables.
 * 
 * @{
 */
/**
 * 8-bit unsigned cubic interpolated look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Cubic_8u_C1R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                         const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 8-bit unsigned cubic interpolated look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Cubic_8u_C1R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                     const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels);

/**
 * 8-bit unsigned cubic interpolated look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Cubic_8u_C1IR_Ctx(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                          const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 8-bit unsigned cubic interpolated look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Cubic_8u_C1IR(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                      const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels);

/**
 * 3 channel 8-bit unsigned cubic interpolated look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Cubic_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                         const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned cubic interpolated look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Cubic_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                     const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 3 channel 8-bit unsigned cubic interpolated look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Cubic_8u_C3IR_Ctx(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                          const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned cubic interpolated look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Cubic_8u_C3IR(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                      const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 4 channel 8-bit unsigned cubic interpolated look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Cubic_8u_C4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                         const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned cubic interpolated look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Cubic_8u_C4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                     const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4]);

/**
 * 4 channel 8-bit unsigned cubic interpolated look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Cubic_8u_C4IR_Ctx(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                          const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned cubic interpolated look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Cubic_8u_C4IR(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                      const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4]);

/**
 * 4 channel 8-bit unsigned cubic interpolated look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Cubic_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                          const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned cubic interpolated look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Cubic_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 4 channel 8-bit unsigned cubic interpolated look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Cubic_8u_AC4IR_Ctx(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                           const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned cubic interpolated look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Cubic_8u_AC4IR(      Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                       const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 16-bit unsigned look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16u_C1R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                          const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 16-bit unsigned look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16u_C1R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels);

/**
 * 16-bit unsigned look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16u_C1IR_Ctx(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                           const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 16-bit unsigned look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16u_C1IR(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                       const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels);

/**
 * 3 channel 16-bit unsigned look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16u_C3R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                          const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit unsigned look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16u_C3R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 3 channel 16-bit unsigned look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16u_C3IR_Ctx(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                           const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit unsigned look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16u_C3IR(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                       const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 4 channel 16-bit unsigned look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16u_C4R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                          const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit unsigned look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16u_C4R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4]);

/**
 * 4 channel 16-bit unsigned look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16u_C4IR_Ctx(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                           const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit unsigned look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16u_C4IR(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                       const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4]);

/**
 * 4 channel 16-bit unsigned look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16u_AC4R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                           const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit unsigned look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16u_AC4R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                       const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 4 channel 16-bit unsigned look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16u_AC4IR_Ctx(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                            const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit unsigned look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16u_AC4IR(      Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                        const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 16-bit signed look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16s_C1R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                          const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 16-bit signed look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16s_C1R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels);

/**
 * 16-bit signed look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16s_C1IR_Ctx(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                           const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 16-bit signed look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16s_C1IR(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                       const Mpp32s * pValues, const Mpp32s * pLevels, int nLevels);

/**
 * 3 channel 16-bit signed look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16s_C3R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                          const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit signed look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16s_C3R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 3 channel 16-bit signed look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16s_C3IR_Ctx(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                           const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit signed look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16s_C3IR(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                       const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 4 channel 16-bit signed look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16s_C4R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                          const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit signed look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16s_C4R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4]);

/**
 * 4 channel 16-bit signed look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16s_C4IR_Ctx(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                           const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit signed look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16s_C4IR(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                       const Mpp32s * pValues[4], const Mpp32s * pLevels[4], int nLevels[4]);

/**
 * 4 channel 16-bit signed look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16s_AC4R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                           const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit signed look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16s_AC4R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                       const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 4 channel 16-bit signed look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16s_AC4IR_Ctx(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                            const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit signed look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_16s_AC4IR(      Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                        const Mpp32s * pValues[3], const Mpp32s * pLevels[3], int nLevels[3]);

/**
 * 32-bit floating point look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_32f_C1R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                          const Mpp32f * pValues, const Mpp32f * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 32-bit floating point look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_32f_C1R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp32f * pValues, const Mpp32f * pLevels, int nLevels);

/**
 * 32-bit floating point look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_32f_C1IR_Ctx(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                           const Mpp32f * pValues, const Mpp32f * pLevels, int nLevels, MppStreamContext mppStreamCtx);

/**
 * 32-bit floating point look-up-table in place color conversion.
 *
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Pointer to an array of user defined OUTPUT values (this is a device memory pointer)
 * \param pLevels Pointer to an array of user defined INPUT values  (this is a device memory pointer)
 * \param nLevels Number of user defined number of input/output mapping points (levels)
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_32f_C1IR(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                       const Mpp32f * pValues, const Mpp32f * pLevels, int nLevels);

/**
 * 3 channel 32-bit floating point look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_32f_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                          const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit floating point look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_32f_C3R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3]);

/**
 * 3 channel 32-bit floating point look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_32f_C3IR_Ctx(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                           const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit floating point look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_32f_C3IR(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                       const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3]);

/**
 * 4 channel 32-bit floating point look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_32f_C4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                          const Mpp32f * pValues[4], const Mpp32f * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points through cubic interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_32f_C4R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp32f * pValues[4], const Mpp32f * pLevels[4], int nLevels[4]);

/**
 * 4 channel 32-bit floating point look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_32f_C4IR_Ctx(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                           const Mpp32f * pValues[4], const Mpp32f * pLevels[4], int nLevels[4], MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point look-up-table in place color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_32f_C4IR(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                       const Mpp32f * pValues[4], const Mpp32f * pLevels[4], int nLevels[4]);

/**
 * 4 channel 32-bit floating point look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_32f_AC4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                           const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_32f_AC4R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                       const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3]);

/**
 * 4 channel 32-bit floating point look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_32f_AC4IR_Ctx(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                            const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point look-up-table in place color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points using no interpolation. 
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.
 * \param pLevels Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.
 * \param nLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 1024 (the current size limit).
 */
MppStatus 
mppiLUT_Cubic_32f_AC4IR(      Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                        const Mpp32f * pValues[3], const Mpp32f * pLevels[3], int nLevels[3]);

/** @} image_color_LUT_cubic */

/** @defgroup image_color_LUT_Trilinear ColorLUTTrilinear
 *  Perform image color processing using 3D trilinear interpolation between members of various types of color look up tables.
 * 
 * @{
 */

/**
 * Four channel 8-bit unsigned 3D trilinear interpolated look-up-table color conversion, with alpha copy.
 * Alpha channel is the last channel and is copied to the destination unmodified.
 *
 * The LUT is derived from a set of user defined mapping points through trilinear interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Device pointer to aLevels[2] number of contiguous 2D x,y planes of 4-byte packed RGBX values
 *        containing the user defined base OUTPUT values at that x,y, and z (R,G,B) level location. Each level must contain x * y 4-byte
 *        packed pixel values (4th byte is used for alignement only and is ignored) in row (x) order.
 * \param pLevels Host pointer to an array of 3 host pointers, one per cube edge, pointing to user defined INPUT level values.
 * \param aLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per 3D cube edge.
 *        aLevels[0] represents the number of x axis levels (Red), aLevels[1] represents the number of y axis levels (Green), 
 *        and aLevels[2] represets the number of z axis levels (Blue).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Trilinear_8u_C4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                                   Mpp32u * pValues, Mpp8u * pLevels[3], int aLevels[3], MppStreamContext mppStreamCtx);

/**
 * Four channel 8-bit unsigned 3D trilinear interpolated look-up-table color conversion, with alpha copy.
 * Alpha channel is the last channel and is copied to the destination unmodified.
 *
 * The LUT is derived from a set of user defined mapping points through trilinear interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Device pointer to aLevels[2] number of contiguous 2D x,y planes of 4-byte packed RGBX values
 *        containing the user defined base OUTPUT values at that x,y, and z (R,G,B) level location. Each level must contain x * y 4-byte
 *        packed pixel values (4th byte is used for alignement only and is ignored) in row (x) order.
 * \param pLevels Host pointer to an array of 3 host pointers, one per cube edge, pointing to user defined INPUT level values.
 * \param aLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per 3D cube edge.
 *        aLevels[0] represents the number of x axis levels (Red), aLevels[1] represents the number of y axis levels (Green), 
 *        and aLevels[2] represets the number of z axis levels (Blue).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Trilinear_8u_C4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                               Mpp32u * pValues, Mpp8u * pLevels[3], int aLevels[3]);

/**
 * Four channel 8-bit unsigned 3D trilinear interpolated look-up-table color conversion, not affecting alpha.
 * Alpha channel is the last channel and is not processed.
 *
 * The LUT is derived from a set of user defined mapping points through trilinear interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Device pointer to aLevels[2] number of contiguous 2D x,y planes of 4-byte packed RGBX values
 *        containing the user defined base OUTPUT values at that x,y, and z (R,G,B) level location. Each level must contain x * y 4-byte
 *        packed pixel values (4th byte is used for alignement only and is ignored) in row (x) order.
 * \param pLevels Host pointer to an array of 3 host pointers, one per cube edge, pointing to user defined INPUT level values.
 * \param aLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per 3D cube edge.
 *        aLevels[0] represents the number of x axis levels (Red), aLevels[1] represents the number of y axis levels (Green), 
 *        and aLevels[2] represets the number of z axis levels (Blue).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Trilinear_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                                    Mpp32u * pValues, Mpp8u * pLevels[3], int aLevels[3], MppStreamContext mppStreamCtx);

/**
 * Four channel 8-bit unsigned 3D trilinear interpolated look-up-table color conversion, not affecting alpha.
 * Alpha channel is the last channel and is not processed.
 *
 * The LUT is derived from a set of user defined mapping points through trilinear interpolation. 
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Device pointer to aLevels[2] number of contiguous 2D x,y planes of 4-byte packed RGBX values
 *        containing the user defined base OUTPUT values at that x,y, and z (R,G,B) level location. Each level must contain x * y 4-byte
 *        packed pixel values (4th byte is used for alignement only and is ignored) in row (x) order.
 * \param pLevels Host pointer to an array of 3 host pointers, one per cube edge, pointing to user defined INPUT level values.
 * \param aLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per 3D cube edge.
 *        aLevels[0] represents the number of x axis levels (Red), aLevels[1] represents the number of y axis levels (Green), 
 *        and aLevels[2] represets the number of z axis levels (Blue).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Trilinear_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                                Mpp32u * pValues, Mpp8u * pLevels[3], int aLevels[3]);

/**
 * Four channel 8-bit unsigned 3D trilinear interpolated look-up-table in place color conversion, not affecting alpha.
 * Alpha channel is the last channel and is not processed.
 *
 * The LUT is derived from a set of user defined mapping points through trilinear interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Device pointer aLevels[2] number of contiguous 2D x,y planes of 4-byte packed RGBX values
 *        containing the user defined base OUTPUT values at that x,y, and z (R,G,B) level location. Each level must contain x * y 4-byte
 *        packed pixel values (4th byte is used for alignement only and is ignored) in row (x) order.
 * \param pLevels Host pointer to an array of 3 host pointers, one per cube edge, pointing to user defined INPUT level values.
 * \param aLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per 3D cube edge.
 *        aLevels[0] represents the number of x axis levels (Red), aLevels[1] represents the number of y axis levels (Green), 
 *        and aLevels[2] represets the number of z axis levels (Blue).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Trilinear_8u_AC4IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                               Mpp32u * pValues, Mpp8u * pLevels[3], int aLevels[3], MppStreamContext mppStreamCtx);

/**
 * Four channel 8-bit unsigned 3D trilinear interpolated look-up-table in place color conversion, not affecting alpha.
 * Alpha channel is the last channel and is not processed.
 *
 * The LUT is derived from a set of user defined mapping points through trilinear interpolation. 
 *
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pValues Device pointer aLevels[2] number of contiguous 2D x,y planes of 4-byte packed RGBX values
 *        containing the user defined base OUTPUT values at that x,y, and z (R,G,B) level location. Each level must contain x * y 4-byte
 *        packed pixel values (4th byte is used for alignement only and is ignored) in row (x) order.
 * \param pLevels Host pointer to an array of 3 host pointers, one per cube edge, pointing to user defined INPUT level values.
 * \param aLevels Host pointer to an array of 3 user defined number of input/output mapping points, one per 3D cube edge.
 *        aLevels[0] represents the number of x axis levels (Red), aLevels[1] represents the number of y axis levels (Green), 
 *        and aLevels[2] represets the number of z axis levels (Blue).
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_NUMBER_OF_LEVELS_ERROR if the number of levels is less than 2 or greater than 256.
 */
MppStatus 
mppiLUT_Trilinear_8u_AC4IR(Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, 
                           Mpp32u * pValues, Mpp8u * pLevels[3], int aLevels[3]);

/** @} image_color_LUT_Trilinear */

/** 
 *  @defgroup image_color_LUT_palette ColorLUTPalette
 *  Perform image color processing using various types of bit range restricted palette color look up tables.
 * 
 * @{
 */

/**
 * One channel 8-bit unsigned bit range restricted palette look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pTable Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)
 * \param nBitSize Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 8.
 */
MppStatus 
mppiLUTPalette_8u_C1R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                          const Mpp8u * pTable, int nBitSize, MppStreamContext mppStreamCtx);

/**
 * One channel 8-bit unsigned bit range restricted palette look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pTable Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)
 * \param nBitSize Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 8.
 */
MppStatus 
mppiLUTPalette_8u_C1R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp8u * pTable, int nBitSize);

/**
 * One channel 8-bit unsigned bit range restricted 24-bit palette look-up-table color conversion with 24-bit destination output per pixel.
 *
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step (3 bytes per pixel).
 * \param oSizeROI \ref roi_specification.
 * \param pTable Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)
 * \param nBitSize Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 8.
 */
MppStatus 
mppiLUTPalette_8u24u_C1R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                             const Mpp8u * pTable, int nBitSize, MppStreamContext mppStreamCtx);

/**
 * One channel 8-bit unsigned bit range restricted 24-bit palette look-up-table color conversion with 24-bit destination output per pixel.
 *
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step (3 bytes per pixel).
 * \param oSizeROI \ref roi_specification.
 * \param pTable Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)
 * \param nBitSize Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 8.
 */
MppStatus 
mppiLUTPalette_8u24u_C1R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                         const Mpp8u * pTable, int nBitSize);

/**
 * One channel 8-bit unsigned bit range restricted 32-bit palette look-up-table color conversion with 32-bit destination output per pixel.
 *
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step (4 bytes per pixel).
 * \param oSizeROI \ref roi_specification.
 * \param pTable Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)
 * \param nBitSize Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 8.
 */
MppStatus 
mppiLUTPalette_8u32u_C1R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp32u * pDst, int nDstStep, MppiSize oSizeROI, 
                             const Mpp32u * pTable, int nBitSize, MppStreamContext mppStreamCtx);

/**
 * One channel 8-bit unsigned bit range restricted 32-bit palette look-up-table color conversion with 32-bit destination output per pixel.
 *
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step (4 bytes per pixel).
 * \param oSizeROI \ref roi_specification.
 * \param pTable Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)
 * \param nBitSize Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 8.
 */
MppStatus 
mppiLUTPalette_8u32u_C1R(const Mpp8u * pSrc, int nSrcStep, Mpp32u * pDst, int nDstStep, MppiSize oSizeROI, 
                         const Mpp32u * pTable, int nBitSize);

/**
 * Three channel 8-bit unsigned bit range restricted palette look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pTables Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
 * \param nBitSize Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 8.
 */
MppStatus 
mppiLUTPalette_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                          const Mpp8u * pTables[3], int nBitSize, MppStreamContext mppStreamCtx);

/**
 * Three channel 8-bit unsigned bit range restricted palette look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pTables Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
 * \param nBitSize Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 8.
 */
MppStatus 
mppiLUTPalette_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp8u * pTables[3], int nBitSize);

/**
 * Four channel 8-bit unsigned bit range restricted palette look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pTables Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
 * \param nBitSize Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 8.
 */
MppStatus 
mppiLUTPalette_8u_C4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                          const Mpp8u * pTables[4], int nBitSize, MppStreamContext mppStreamCtx);

/**
 * Four channel 8-bit unsigned bit range restricted palette look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pTables Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
 * \param nBitSize Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 8.
 */
MppStatus 
mppiLUTPalette_8u_C4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp8u * pTables[4], int nBitSize);

/**
 * Four channel 8-bit unsigned bit range restricted palette look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pTables Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
 * \param nBitSize Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 8.
 */
MppStatus 
mppiLUTPalette_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                           const Mpp8u * pTables[3], int nBitSize, MppStreamContext mppStreamCtx);

/**
 * Four channel 8-bit unsigned bit range restricted palette look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pTables Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
 * \param nBitSize Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 8.
 */
MppStatus 
mppiLUTPalette_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                       const Mpp8u * pTables[3], int nBitSize);

/**
 * One channel 16-bit unsigned bit range restricted palette look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pTable Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)
 * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
 */
MppStatus 
mppiLUTPalette_16u_C1R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                           const Mpp16u * pTable, int nBitSize, MppStreamContext mppStreamCtx);

/**
 * One channel 16-bit unsigned bit range restricted palette look-up-table color conversion.
 *
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pTable Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)
 * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
 */
MppStatus 
mppiLUTPalette_16u_C1R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                       const Mpp16u * pTable, int nBitSize);

/**
 * One channel 16-bit unsigned bit range restricted 8-bit unsigned palette look-up-table color conversion with 8-bit unsigned destination output per pixel.
 *
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step (1 unsigned byte per pixel).
 * \param oSizeROI \ref roi_specification.
 * \param pTable Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)
 * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
 */
MppStatus 
mppiLUTPalette_16u8u_C1R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                             const Mpp8u * pTable, int nBitSize, MppStreamContext mppStreamCtx);

/**
 * One channel 16-bit unsigned bit range restricted 8-bit unsigned palette look-up-table color conversion with 8-bit unsigned destination output per pixel.
 *
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step (1 unsigned byte per pixel).
 * \param oSizeROI \ref roi_specification.
 * \param pTable Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)
 * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
 */
MppStatus 
mppiLUTPalette_16u8u_C1R(const Mpp16u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                         const Mpp8u * pTable, int nBitSize);

/**
 * One channel 16-bit unsigned bit range restricted 24-bit unsigned palette look-up-table color conversion with 24-bit unsigned destination output per pixel.
 *
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step (3 unsigned bytes per pixel).
 * \param oSizeROI \ref roi_specification.
 * \param pTable Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)
 * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
 */
MppStatus 
mppiLUTPalette_16u24u_C1R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                              const Mpp8u * pTable, int nBitSize, MppStreamContext mppStreamCtx);

/**
 * One channel 16-bit unsigned bit range restricted 24-bit unsigned palette look-up-table color conversion with 24-bit unsigned destination output per pixel.
 *
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step (3 unsigned bytes per pixel).
 * \param oSizeROI \ref roi_specification.
 * \param pTable Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)
 * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
 */
MppStatus 
mppiLUTPalette_16u24u_C1R(const Mpp16u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                          const Mpp8u * pTable, int nBitSize);

/**
 * One channel 16-bit unsigned bit range restricted 32-bit palette look-up-table color conversion with 32-bit unsigned destination output per pixel.
 *
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step (4 bytes per pixel).
 * \param oSizeROI \ref roi_specification.
 * \param pTable Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)
 * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
 */
MppStatus 
mppiLUTPalette_16u32u_C1R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp32u * pDst, int nDstStep, MppiSize oSizeROI, 
                              const Mpp32u * pTable, int nBitSize, MppStreamContext mppStreamCtx);

/**
 * One channel 16-bit unsigned bit range restricted 32-bit palette look-up-table color conversion with 32-bit unsigned destination output per pixel.
 *
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step (4 bytes per pixel).
 * \param oSizeROI \ref roi_specification.
 * \param pTable Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)
 * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
 */
MppStatus 
mppiLUTPalette_16u32u_C1R(const Mpp16u * pSrc, int nSrcStep, Mpp32u * pDst, int nDstStep, MppiSize oSizeROI, 
                          const Mpp32u * pTable, int nBitSize);

/**
 * Three channel 16-bit unsigned bit range restricted palette look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pTables Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
 * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
 */
MppStatus 
mppiLUTPalette_16u_C3R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                           const Mpp16u * pTables[3], int nBitSize, MppStreamContext mppStreamCtx);

/**
 * Three channel 16-bit unsigned bit range restricted palette look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pTables Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
 * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
 */
MppStatus 
mppiLUTPalette_16u_C3R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                       const Mpp16u * pTables[3], int nBitSize);

/**
 * Four channel 16-bit unsigned bit range restricted palette look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pTables Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
 * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
 */
MppStatus 
mppiLUTPalette_16u_C4R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                           const Mpp16u * pTables[4], int nBitSize, MppStreamContext mppStreamCtx);

/**
 * Four channel 16-bit unsigned bit range restricted palette look-up-table color conversion.
 * 
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pTables Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
 * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
 */
MppStatus 
mppiLUTPalette_16u_C4R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                       const Mpp16u * pTables[4], int nBitSize);

/**
 * Four channel 16-bit unsigned bit range restricted palette look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pTables Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
 * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
 */
MppStatus 
mppiLUTPalette_16u_AC4R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                            const Mpp16u * pTables[3], int nBitSize, MppStreamContext mppStreamCtx);

/**
 * Four channel 16-bit unsigned bit range restricted palette look-up-table color conversion, not affecting Alpha.
 *
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 * Alpha channel is the last channel and is not processed.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pTables Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
 * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
 */
MppStatus 
mppiLUTPalette_16u_AC4R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                        const Mpp16u * pTables[3], int nBitSize);

/**
 * Three channel 8-bit unsigned source bit range restricted palette look-up-table color conversion to four channel 8-bit unsigned destination output with alpha.
 *
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 * This function also reverses the source pixel channel order in the destination so the Alpha channel is the first channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step (3 bytes per pixel).
 * \param nAlphaValue Signed alpha value that will be used to initialize the pixel alpha channel position in all modified destination pixels.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step (4 bytes per pixel with alpha).
 * \param oSizeROI \ref roi_specification.
 * \param pTables Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
 * Alpha values < 0 or > 255 will cause destination pixel alpha channel values to be unmodified.
 * \param nBitSize Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 8.
 */
MppStatus 
mppiLUTPaletteSwap_8u_C3A0C4R_Ctx(const Mpp8u * pSrc, int nSrcStep, int nAlphaValue, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                                  const Mpp8u * pTables[3], int nBitSize, MppStreamContext mppStreamCtx);

/**
 * Three channel 8-bit unsigned source bit range restricted palette look-up-table color conversion to four channel 8-bit unsigned destination output with alpha.
 *
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 * This function also reverses the source pixel channel order in the destination so the Alpha channel is the first channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step (3 bytes per pixel).
 * \param nAlphaValue Signed alpha value that will be used to initialize the pixel alpha channel position in all modified destination pixels.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step (4 bytes per pixel with alpha).
 * \param oSizeROI \ref roi_specification.
 * \param pTables Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
 * Alpha values < 0 or > 255 will cause destination pixel alpha channel values to be unmodified.
 * \param nBitSize Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 8.
 */
MppStatus 
mppiLUTPaletteSwap_8u_C3A0C4R(const Mpp8u * pSrc, int nSrcStep, int nAlphaValue, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                              const Mpp8u * pTables[3], int nBitSize);

/**
 * Three channel 16-bit unsigned source bit range restricted palette look-up-table color conversion to four channel 16-bit unsigned destination output with alpha.
 *
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 * This function also reverses the source pixel channel order in the destination so the Alpha channel is the first channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step (3 unsigned short integers per pixel).
 * \param nAlphaValue Signed alpha value that will be used to initialize the pixel alpha channel position in all modified destination pixels.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step (4 unsigned short integers per pixel with alpha).
 * \param oSizeROI \ref roi_specification.
 * \param pTables Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
 * Alpha values < 0 or > 65535 will cause destination pixel alpha channel values to be unmodified.
 * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
 */
MppStatus 
mppiLUTPaletteSwap_16u_C3A0C4R_Ctx(const Mpp16u * pSrc, int nSrcStep, int nAlphaValue, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                                   const Mpp16u * pTables[3], int nBitSize, MppStreamContext mppStreamCtx);

/**
 * Three channel 16-bit unsigned source bit range restricted palette look-up-table color conversion to four channel 16-bit unsigned destination output with alpha.
 *
 * The LUT is derived from a set of user defined mapping points in a palette and 
 * source pixels are then processed using a restricted bit range when looking up palette values.
 * This function also reverses the source pixel channel order in the destination so the Alpha channel is the first channel.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step (3 unsigned short integers per pixel).
 * \param nAlphaValue Signed alpha value that will be used to initialize the pixel alpha channel position in all modified destination pixels.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step (4 unsigned short integers per pixel with alpha).
 * \param oSizeROI \ref roi_specification.
 * \param pTables Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.
 * Alpha values < 0 or > 65535 will cause destination pixel alpha channel values to be unmodified.
 * \param nBitSize Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *        - MPP_LUT_PALETTE_BITSIZE_ERROR if nBitSize is < 1 or > 16.
 */
MppStatus 
mppiLUTPaletteSwap_16u_C3A0C4R(const Mpp16u * pSrc, int nSrcStep, int nAlphaValue, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                               const Mpp16u * pTables[3], int nBitSize);

/** @} image_color_LUT_palette */

/** @name RGBToYUVColorTwist
 *
 *  Normally, color twist is a 3-channel operation that takes place on RGB data.  The following functions also
 *  perform the color twist operation, but while converting between image formats: RGBToYUV420, RGBToYUV422,
 *  and RGBToNV12.
 *
 *  To recap color twist pixel processing:  Color twist consists of applying the following formula to each
 *  image pixel using coefficients from the user supplied color twist host matrix array as follows where
 *  dst[x] and src[x] represent destination pixel and source pixel channel or plane x. The full sized
 *  coefficient matrix should be sent for all pixel channel sizes, the function will process the appropriate
 *  coefficients and channels for the corresponding pixel size.
 *
 *  This is how the matrix works for a RGB->YUV420/YUV422/NV12 forward transform:
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0] + aTwist[0][1] * src[1] + aTwist[0][2] * src[2] + aTwist[0][3]
 *      dst[1] = aTwist[1][0] * src[0] + aTwist[1][1] * src[1] + aTwist[1][2] * src[2] + aTwist[1][3]
 *      dst[2] = aTwist[2][0] * src[0] + aTwist[2][1] * src[1] + aTwist[2][2] * src[2] + aTwist[2][3]
 *  \endcode
 *
 * @{
 */

 /**
 * Three channel 8-bit unsigned planar RGB convertion to three channel 8-bit unsigned planar YUV 4:2:0, using a Color Twist to compute the exact
 * color space arithmetic.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiRGBToYUV420_8u_ColorTwist32f_P3R_Ctx(const Mpp8u * const pSrc[3], int aSrcStep[3],
    Mpp8u * pDst[3], int aDstStep[3],
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);


/**
* Three channel 16-bit unsigned planar RGB convertion to three channel 16-bit unsigned planar YUV 4:2:0, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param aSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param aDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiRGBToYUV420_16u_ColorTwist32f_P3R_Ctx(const Mpp16u * const pSrc[3], int aSrcStep[3],
    Mpp16u * pDst[3], int aDstStep[3],
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Three channel 8-bit unsigned packed RGB convertion to three channel 8-bit unsigned planar YUV 4:2:0, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param nSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param aDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiRGBToYUV420_8u_ColorTwist32f_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep,
    Mpp8u * pDst[3], int aDstStep[3],
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Three channel 16-bit unsigned packed RGB convertion to three channel 16-bit unsigned planar YUV 4:2:0, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param nSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param aDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiRGBToYUV420_16u_ColorTwist32f_C3P3R_Ctx(const Mpp16u * pSrc, int nSrcStep,
    Mpp16u * pDst[3], int aDstStep[3],
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Three channel 8-bit unsigned planar RGB convertion to three channel 8-bit unsigned planar YUV 4:2:2, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param aSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param aDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiRGBToYUV422_8u_ColorTwist32f_P3R_Ctx(const Mpp8u * const pSrc[3], int aSrcStep[3],
    Mpp8u * pDst[3], int aDstStep[3],
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Three channel 16-bit unsigned planar RGB convertion to three channel 16-bit unsigned planar YUV 4:2:2, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param aSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param aDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiRGBToYUV422_16u_ColorTwist32f_P3R_Ctx(const Mpp16u * const pSrc[3], int aSrcStep[3],
    Mpp16u * pDst[3], int aDstStep[3],
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Three channel 8-bit unsigned packed RGB convertion to two channel 8-bit unsigned packed YUV 4:2:2, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param nSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param nDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiRGBToYUV422_8u_ColorTwist32f_C3C2R_Ctx(const Mpp8u * pSrc, int nSrcStep,
    Mpp8u * pDst, int nDstStep,
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Three channel 16-bit unsigned packed RGB convertion to two channel 16-bit unsigned packed YUV 4:2:2, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param nSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param nDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiRGBToYUV422_16u_ColorTwist32f_C3C2R_Ctx(const Mpp16u * pSrc, int nSrcStep,
    Mpp16u * pDst, int nDstStep,
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Three channel 8-bit unsigned packed RGB convertion to three channel 8-bit unsigned planar YUV 4:2:2, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param nSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param aDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiRGBToYUV422_8u_ColorTwist32f_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep,
    Mpp8u * pDst[3], int aDstStep[3],
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Three channel 16-bit unsigned packed RGB convertion to three channel 16-bit unsigned planar YUV 4:2:2, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param nSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param aDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiRGBToYUV422_16u_ColorTwist32f_C3P3R_Ctx(const Mpp16u * pSrc, int nSrcStep,
    Mpp16u * pDst[3], int aDstStep[3],
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Three channel 8-bit unsigned packed RGB convertion to two channel 8-bit unsigned planar NV12, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param nSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param aDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx(const Mpp8u * pSrc, int nSrcStep,
    Mpp8u * pDst[2], int aDstStep[2],
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Three channel 16-bit unsigned packed RGB convertion to two channel 16-bit unsigned planar NV12, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param nSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param aDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiRGBToNV12_16u_ColorTwist32f_C3P2R_Ctx(const Mpp16u * pSrc, int nSrcStep,
    Mpp16u * pDst[2], int aDstStep[2],
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Three channel 8-bit unsigned planar RGB convertion to two channel 8-bit unsigned planar NV12, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param aSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param aDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiRGBToNV12_8u_ColorTwist32f_P3P2R_Ctx(const Mpp8u * const pSrc[3], int aSrcStep[3],
    Mpp8u * pDst[2], int aDstStep[2],
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Three channel 16-bit unsigned planar RGB convertion to two channel 16-bit unsigned planar NV12, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param aSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param aDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiRGBToNV12_16u_ColorTwist32f_P3P2R_Ctx(const Mpp16u * const pSrc[3], int aSrcStep[3],
    Mpp16u * pDst[2], int aDstStep[2],
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/** @} RGBToYUVColorTwist*/

/** @name YUVToRGBColorTwist
 *
 *  This is the inverse color twist, a 3-channel operation that takes place on YUV/NV12 data to produce RGB data,
 *  supporting the following conversions: YUV420ToRGB, YUV422ToRGB, and NV12ToRGB.
 *
 *  The INVERSE Color twist consists of applying the following formula to each image pixel using
 *  coefficients from the user supplied color twist host matrix array as follows where dst[x] and src[x]
 *  represent destination pixel and source pixel channel or plane x. The full sized coefficient matrix
 *  should be sent for all pixel channel sizes, the function will process the appropriate coefficients
 *  and channels for the corresponding pixel size.
 *
 *  This is how the matrix works for the YUV420/YUV/422/NV12->RGB INVERSE transform:
 *
 * (note- do the offsets first):
 *
 *  \code
 *      src[0]' = src[0] + aTwist[0][3]
 *      src[1]' = src[1] + aTwist[1][3]
 *      src[2]' = src[2] + aTwist[2][3]
 *  \endcode
 *
 *  And then the remaining 3x3 twist matrix is applied using those modified values:
 *
 *  \code
 *      dst[0] = aTwist[0][0] * src[0]' + aTwist[0][1] * src[1]' + aTwist[0][2] * src[2]'
 *      dst[1] = aTwist[1][0] * src[0]' + aTwist[1][1] * src[1]' + aTwist[1][2] * src[2]'
 *      dst[2] = aTwist[2][0] * src[0]' + aTwist[2][1] * src[1]' + aTwist[2][2] * src[2]'
 *  \endcode
 *
 *  Since the 4th column of the matrix is the offsets (either 0 or half-max to shift the values to
 *  be positive), as they're applied last in the forward transform (to YUV), they need to be applied
 *  FIRST in the inverse transform (back to RGB).  This does mean that +-16384 has to be used for
 *  16u images where there was a +-128 for 8u images in both forward and inverse cases.
 *
 * @{
 */

 /**
 * Three channel 8-bit unsigned planar YUV4:2:0 convertion to three channel 8-bit unsigned planar RGB, using a Color Twist to compute the exact
 * color space arithmetic.
 *
 * \param pSrc \ref source_image_pointer.
 * \param aSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param aDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aTwist The color twist matrix with floating-point coefficient values.
 * \param mppStreamCtx \ref application_managed_stream_context.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiYUV420ToRGB_8u_ColorTwist32f_P3R_Ctx(const Mpp8u * const pSrc[3], int aSrcStep[3],
    Mpp8u * pDst[3], int aDstStep[3],
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Three channel 16-bit unsigned planar YUV4:2:0 convertion to three channel 16-bit unsigned planar RGB, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param aSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param aDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiYUV420ToRGB_16u_ColorTwist32f_P3R_Ctx(const Mpp16u * const pSrc[3], int aSrcStep[3],
    Mpp16u * pDst[3], int aDstStep[3],
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Three channel 8-bit unsigned planar YUV4:2:0 convertion to three channel 8-bit unsigned packed RGB, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param aSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param nDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiYUV420ToRGB_8u_ColorTwist32f_P3C3R_Ctx(const Mpp8u * const pSrc[3], int aSrcStep[3],
    Mpp8u * pDst, int nDstStep,
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Three channel 16-bit unsigned planar YUV4:2:0 convertion to three channel 16-bit unsigned packed RGB, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param aSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param nDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiYUV420ToRGB_16u_ColorTwist32f_P3C3R_Ctx(const Mpp16u * const pSrc[3], int aSrcStep[3],
    Mpp16u * pDst, int nDstStep,
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Three channel 8-bit unsigned planar YUV4:2:0 convertion to four channel 8-bit unsigned packed RGB, using a Color Twist to compute the exact
* color space arithmetic, with constant alpha (0xFF).
*
* \param pSrc \ref source_image_pointer.
* \param aSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param nDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiYUV420ToRGB_8u_ColorTwist32f_P3C4R_Ctx(const Mpp8u * const pSrc[3], int aSrcStep[3],
    Mpp8u * pDst, int nDstStep,
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Three channel 16-bit unsigned planar YUV4:2:0 convertion to four channel 16-bit unsigned packed RGB, using a Color Twist to compute the exact
* color space arithmetic, with constant alpha (0xFFFF).
*
* \param pSrc \ref source_image_pointer.
* \param aSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param nDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiYUV420ToRGB_16u_ColorTwist32f_P3C4R_Ctx(const Mpp16u * const pSrc[3], int aSrcStep[3],
    Mpp16u * pDst, int nDstStep,
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Three channel 8-bit unsigned planar YUV4:2:0 convertion to four channel 8-bit unsigned packed RGB, using a Color Twist to compute the exact
* color space arithmetic, with user set alpha.
*
* \param pSrc \ref source_image_pointer.
* \param aSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param nDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param nAlpha 8-bit unsigned alpha constant.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiYUV420ToRGB_8u_ColorTwist32f_P3AC4R_Ctx(const Mpp8u * const pSrc[3], int aSrcStep[3],
    Mpp8u * pDst, int nDstStep,
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], const Mpp8u nAlpha, MppStreamContext mppStreamCtx);

/**
* Three channel 16-bit unsigned planar YUV4:2:0 convertion to four channel 16-bit unsigned packed RGB, using a Color Twist to compute the exact
* color space arithmetic, with user set alpha.
*
* \param pSrc \ref source_image_pointer.
* \param aSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param nDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param nAlpha 16-bit unsigned alpha constant.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiYUV420ToRGB_16u_ColorTwist32f_P3AC4R_Ctx(const Mpp16u * const pSrc[3], int aSrcStep[3],
    Mpp16u * pDst, int nDstStep,
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], const Mpp16u nAlpha, MppStreamContext mppStreamCtx);

/**
* Two channel 8-bit unsigned packed YUV4:2:2 convertion to three channel 8-bit unsigned packed RGB, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param nSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param nDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiYUV422ToRGB_8u_ColorTwist32f_C2C3R_Ctx(const Mpp8u * pSrc, int nSrcStep,
    Mpp8u * pDst, int nDstStep,
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Two channel 16-bit unsigned packed YUV4:2:2 convertion to three channel 16-bit unsigned packed RGB, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param nSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param nDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiYUV422ToRGB_16u_ColorTwist32f_C2C3R_Ctx(const Mpp16u * pSrc, int nSrcStep,
    Mpp16u * pDst, int nDstStep,
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Three channel 8-bit unsigned planar YUV4:2:2 convertion to three channel 8-bit unsigned planar RGB, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param aSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param aDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiYUV422ToRGB_8u_ColorTwist32f_P3R_Ctx(const Mpp8u * const pSrc[3], int aSrcStep[3],
    Mpp8u * pDst[3], int aDstStep[3],
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Three channel 16-bit unsigned planar YUV4:2:2 convertion to three channel 16-bit unsigned planar RGB, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param aSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param aDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiYUV422ToRGB_16u_ColorTwist32f_P3R_Ctx(const Mpp16u * const pSrc[3], int aSrcStep[3],
    Mpp16u * pDst[3], int aDstStep[3],
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Three channel 8-bit unsigned planar YUV4:2:2 convertion to three channel 8-bit unsigned packed RGB, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param aSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param nDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiYUV422ToRGB_8u_ColorTwist32f_P3C3R_Ctx(const Mpp8u * const pSrc[3], int aSrcStep[3],
    Mpp8u * pDst, int nDstStep,
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Three channel 8-bit unsigned planar YUV4:2:2 convertion to three channel 8-bit unsigned packed RGB, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param aSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param nDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiYUV422ToRGB_16u_ColorTwist32f_P3C3R_Ctx(const Mpp16u * const pSrc[3], int aSrcStep[3],
    Mpp16u * pDst, int nDstStep,
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Three channel 8-bit unsigned planar YUV4:2:2 convertion to three channel 8-bit unsigned packed RGB, using a Color Twist to compute the exact
* color space arithmetic, with user set alpha.
*
* \param pSrc \ref source_image_pointer.
* \param aSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param nDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param nAlpha 8-bit unsigned alpha constant.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiYUV422ToRGB_8u_ColorTwist32f_P3AC4R_Ctx(const Mpp8u * const pSrc[3], int aSrcStep[3],
    Mpp8u * pDst, int nDstStep,
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], const Mpp8u nAlpha, MppStreamContext mppStreamCtx);

/**
* Three channel 16-bit unsigned planar YUV4:2:2 convertion to three channel 16-bit unsigned packed RGB, using a Color Twist to compute the exact
* color space arithmetic, with user set alpha.
*
* \param pSrc \ref source_image_pointer.
* \param aSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param nDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param nAlpha 16-bit unsigned alpha constant.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiYUV422ToRGB_16u_ColorTwist32f_P3AC4R_Ctx(const Mpp16u * const pSrc[3], int aSrcStep[3],
    Mpp16u * pDst, int nDstStep,
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], const Mpp16u nAlpha, MppStreamContext mppStreamCtx);

/**
* Two channel 8-bit unsigned planar NV12 convertion to three channel 8-bit unsigned packed RGB, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param aSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param nDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx(const Mpp8u * const pSrc[2], int aSrcStep[2],
    Mpp8u * pDst, int nDstStep,
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/**
* Two channel 16-bit unsigned planar NV12 convertion to three channel 16-bit unsigned packed RGB, using a Color Twist to compute the exact
* color space arithmetic.
*
* \param pSrc \ref source_image_pointer.
* \param aSrcStep \ref source_image_line_step.
* \param pDst \ref destination_image_pointer.
* \param nDstStep \ref destination_image_line_step.
* \param oSizeROI \ref roi_specification.
* \param aTwist The color twist matrix with floating-point coefficient values.
* \param mppStreamCtx \ref application_managed_stream_context.
* \return \ref image_data_error_codes, \ref roi_error_codes
*/
MppStatus mppiNV12ToRGB_16u_ColorTwist32f_P2C3R_Ctx(const Mpp16u * const pSrc[2], int aSrcStep[2],
    Mpp16u * pDst, int nDstStep,
    MppiSize oSizeROI, const Mpp32f aTwist[3][4], MppStreamContext mppStreamCtx);

/** @} YUVToRGBColorTwist*/

/** @} image_color_processing */ 

/** @} image_color_conversion */ 

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MC_MPPI_COLOR_CONVERSION_H */
