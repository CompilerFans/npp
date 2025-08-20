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
#ifndef MC_MPPI_DATA_EXCHANGE_AND_INITIALIZATION_H
#define MC_MPPI_DATA_EXCHANGE_AND_INITIALIZATION_H
 
/**
 * \file mppi_data_exchange_and_initialization.h
 * MPP Image Processing Functionality.
 */
 
#include "mppdefs.h"

#ifdef __cplusplus
extern "C" {
#endif

/** 
 *  \page image_data_exchange_and_initialization Data Exchange and Initialization
 *  @defgroup image_data_exchange_and_initialization Data Exchange and Initialization
 *  @ingroup mppi
 *
 * Functions for initializing, copying and converting image data.
 *
 * @{
 *
 * These functions can be found in the nppidei library. Linking to only the sub-libraries that you use can significantly
 * save link time, application load time, and MACA runtime startup time when using dynamic libraries.
 *
 */

/** 
 * \section image_set_operations Image Set Operations
 * @defgroup image_set_operations Image
 * Functions for setting all pixels within the ROI to a specific value.
 *
 * @{
 *
 */

/** 
 * \section image_set Set
 * @defgroup image_set Set
 * Functions for setting all pixels within the ROI to a specific value.
 *
 * \subsection CommonImageSetParameters Common parameters for mppiSet functions:
 *
 * \param nValue The pixel value to be set.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */


/** 
 * 8-bit image set.
 * 
 * For common parameter descriptions see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_8s_C1R_Ctx(const Mpp8s nValue, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 8-bit image set.
 * 
 * For common parameter descriptions see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_8s_C1R(const Mpp8s nValue, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 8-bit two-channel image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_8s_C2R_Ctx(const Mpp8s aValue[2], Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 8-bit two-channel image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_8s_C2R(const Mpp8s aValue[2], Mpp8s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 8-bit three-channel image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_8s_C3R_Ctx(const Mpp8s aValue[3], Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 8-bit three-channel image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_8s_C3R(const Mpp8s aValue[3], Mpp8s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 8-bit four-channel image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_8s_C4R_Ctx(const Mpp8s aValue[4], Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 8-bit four-channel image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_8s_C4R(const Mpp8s aValue[4], Mpp8s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 8-bit four-channel image set ignoring alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_8s_AC4R_Ctx(const Mpp8s aValue[3], Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 8-bit four-channel image set ignoring alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_8s_AC4R(const Mpp8s aValue[3], Mpp8s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 8-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_8u_C1R_Ctx(const Mpp8u nValue, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 8-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_8u_C1R(const Mpp8u nValue, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 2 channel 8-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_8u_C2R_Ctx(const Mpp8u aValue[2], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 2 channel 8-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_8u_C2R(const Mpp8u aValue[2], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 3 channel 8-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_8u_C3R_Ctx(const Mpp8u aValue[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 3 channel 8-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_8u_C3R(const Mpp8u aValue[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 8-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_8u_C4R_Ctx(const Mpp8u aValue[4], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 8-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_8u_C4R(const Mpp8u aValue[4], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 8-bit unsigned image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_8u_AC4R_Ctx(const Mpp8u aValue[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 8-bit unsigned image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_8u_AC4R(const Mpp8u aValue[3], Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 16-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16u_C1R_Ctx(const Mpp16u nValue, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 16-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16u_C1R(const Mpp16u nValue, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 2 channel 16-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16u_C2R_Ctx(const Mpp16u aValue[2], Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 2 channel 16-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16u_C2R(const Mpp16u aValue[2], Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 3 channel 16-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16u_C3R_Ctx(const Mpp16u aValue[3], Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 3 channel 16-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16u_C3R(const Mpp16u aValue[3], Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 16-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16u_C4R_Ctx(const Mpp16u aValue[4], Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 16-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16u_C4R(const Mpp16u aValue[4], Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 16-bit unsigned image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16u_AC4R_Ctx(const Mpp16u aValue[3], Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 16-bit unsigned image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16u_AC4R(const Mpp16u aValue[3], Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 16-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16s_C1R_Ctx(const Mpp16s nValue, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 16-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16s_C1R(const Mpp16s nValue, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 2 channel 16-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16s_C2R_Ctx(const Mpp16s aValue[2], Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 2 channel 16-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16s_C2R(const Mpp16s aValue[2], Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 3 channel 16-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16s_C3R_Ctx(const Mpp16s aValue[3], Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 3 channel 16-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16s_C3R(const Mpp16s aValue[3], Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 16-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16s_C4R_Ctx(const Mpp16s aValue[4], Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 16-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16s_C4R(const Mpp16s aValue[4], Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 16-bit image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16s_AC4R_Ctx(const Mpp16s aValue[3], Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 16-bit image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16s_AC4R(const Mpp16s aValue[3], Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 16-bit complex integer image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16sc_C1R_Ctx(const Mpp16sc oValue, Mpp16sc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 16-bit complex integer image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16sc_C1R(const Mpp16sc oValue, Mpp16sc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 16-bit complex integer two-channel image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16sc_C2R_Ctx(const Mpp16sc aValue[2], Mpp16sc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 16-bit complex integer two-channel image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16sc_C2R(const Mpp16sc aValue[2], Mpp16sc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 16-bit complex integer three-channel image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16sc_C3R_Ctx(const Mpp16sc aValue[3], Mpp16sc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 16-bit complex integer three-channel image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16sc_C3R(const Mpp16sc aValue[3], Mpp16sc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 16-bit complex integer four-channel image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16sc_C4R_Ctx(const Mpp16sc aValue[4], Mpp16sc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 16-bit complex integer four-channel image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16sc_C4R(const Mpp16sc aValue[4], Mpp16sc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 16-bit complex integer four-channel image set ignoring alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16sc_AC4R_Ctx(const Mpp16sc aValue[3], Mpp16sc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 16-bit complex integer four-channel image set ignoring alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16sc_AC4R(const Mpp16sc aValue[3], Mpp16sc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 32-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32s_C1R_Ctx(const Mpp32s nValue, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 32-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32s_C1R(const Mpp32s nValue, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 2 channel 32-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32s_C2R_Ctx(const Mpp32s aValue[2], Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 2 channel 32-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32s_C2R(const Mpp32s aValue[2], Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 3 channel 32-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32s_C3R_Ctx(const Mpp32s aValue[3], Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 3 channel 32-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32s_C3R(const Mpp32s aValue[3], Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 32-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32s_C4R_Ctx(const Mpp32s aValue[4], Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 32-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32s_C4R(const Mpp32s aValue[4], Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 32-bit image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32s_AC4R_Ctx(const Mpp32s aValue[3], Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 32-bit image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32s_AC4R(const Mpp32s aValue[3], Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 32-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32u_C1R_Ctx(const Mpp32u nValue, Mpp32u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 32-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32u_C1R(const Mpp32u nValue, Mpp32u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 2 channel 32-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32u_C2R_Ctx(const Mpp32u aValue[2], Mpp32u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 2 channel 32-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32u_C2R(const Mpp32u aValue[2], Mpp32u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 3 channel 32-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32u_C3R_Ctx(const Mpp32u aValue[3], Mpp32u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 3 channel 32-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32u_C3R(const Mpp32u aValue[3], Mpp32u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 32-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32u_C4R_Ctx(const Mpp32u aValue[4], Mpp32u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 32-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32u_C4R(const Mpp32u aValue[4], Mpp32u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 32-bit unsigned image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32u_AC4R_Ctx(const Mpp32u aValue[3], Mpp32u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 32-bit unsigned image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32u_AC4R(const Mpp32u aValue[3], Mpp32u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 32-bit complex integer image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32sc_C1R_Ctx(const Mpp32sc oValue, Mpp32sc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit complex integer image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32sc_C1R(const Mpp32sc oValue, Mpp32sc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Two channel 32-bit complex integer image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32sc_C2R_Ctx(const Mpp32sc aValue[2], Mpp32sc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Two channel 32-bit complex integer image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32sc_C2R(const Mpp32sc aValue[2], Mpp32sc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three channel 32-bit complex integer image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32sc_C3R_Ctx(const Mpp32sc aValue[3], Mpp32sc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 32-bit complex integer image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32sc_C3R(const Mpp32sc aValue[3], Mpp32sc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 32-bit complex integer image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32sc_C4R_Ctx(const Mpp32sc aValue[4], Mpp32sc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 32-bit complex integer image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32sc_C4R(const Mpp32sc aValue[4], Mpp32sc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 32-bit complex integer four-channel image set ignoring alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32sc_AC4R_Ctx(const Mpp32sc aValue[3], Mpp32sc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 32-bit complex integer four-channel image set ignoring alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32sc_AC4R(const Mpp32sc aValue[3], Mpp32sc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 16-bit floating point image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16f_C1R_Ctx(const Mpp32f nValue, Mpp16f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 16-bit floating point image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16f_C1R(const Mpp32f nValue, Mpp16f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 2 channel 16-bit floating point image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16f_C2R_Ctx(const Mpp32f aValues[2], Mpp16f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 2 channel 16-bit floating point image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16f_C2R(const Mpp32f aValues[2], Mpp16f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 3 channel 16-bit floating point image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16f_C3R_Ctx(const Mpp32f aValues[3], Mpp16f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 3 channel 16-bit floating point image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16f_C3R(const Mpp32f aValues[3], Mpp16f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 16-bit floating point image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16f_C4R_Ctx(const Mpp32f aValues[4], Mpp16f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 16-bit floating point image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_16f_C4R(const Mpp32f aValues[4], Mpp16f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 32-bit floating point image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32f_C1R_Ctx(const Mpp32f nValue, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 32-bit floating point image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32f_C1R(const Mpp32f nValue, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 2 channel 32-bit floating point image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32f_C2R_Ctx(const Mpp32f aValue[2], Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 2 channel 32-bit floating point image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32f_C2R(const Mpp32f aValue[2], Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 3 channel 32-bit floating point image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32f_C3R_Ctx(const Mpp32f aValue[3], Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 3 channel 32-bit floating point image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32f_C3R(const Mpp32f aValue[3], Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 32-bit floating point image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32f_C4R_Ctx(const Mpp32f aValue[4], Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 32-bit floating point image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32f_C4R(const Mpp32f aValue[4], Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 32-bit floating point image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32f_AC4R_Ctx(const Mpp32f aValue[3], Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 32-bit floating point image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32f_AC4R(const Mpp32f aValue[3], Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 32-bit complex image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32fc_C1R_Ctx(const Mpp32fc oValue, Mpp32fc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit complex image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32fc_C1R(const Mpp32fc oValue, Mpp32fc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Two channel 32-bit complex image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32fc_C2R_Ctx(const Mpp32fc aValue[2], Mpp32fc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Two channel 32-bit complex image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32fc_C2R(const Mpp32fc aValue[2], Mpp32fc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three channel 32-bit complex image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32fc_C3R_Ctx(const Mpp32fc aValue[3], Mpp32fc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 32-bit complex image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32fc_C3R(const Mpp32fc aValue[3], Mpp32fc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 32-bit complex image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32fc_C4R_Ctx(const Mpp32fc aValue[4], Mpp32fc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 32-bit complex image set.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32fc_C4R(const Mpp32fc aValue[4], Mpp32fc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 32-bit complex four-channel image set ignoring alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32fc_AC4R_Ctx(const Mpp32fc aValue[3], Mpp32fc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 32-bit complex four-channel image set ignoring alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageSetParameters.
 *
 */
MppStatus 
mppiSet_32fc_AC4R(const Mpp32fc aValue[3], Mpp32fc * pDst, int nDstStep, MppiSize oSizeROI);

/** @} image_set */

/** 
 * \section image_masked_set Masked Set
 * @defgroup image_masked_set Masked Set
 * The masked set primitives have an additional "mask image" input. The mask  
 * controls which pixels within the ROI are set. For details see \ref masked_operation.
 *
 * \subsection CommonImageMaskedSetParameters Common parameters for mppiSet_CXM functions:
 *
 * \param nValue The pixel value to be set for single channel functions.
 * \param aValue The pixel-value to be set for multi-channel functions.
 * \param pDst Pointer \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * Masked 8-bit unsigned image set. 
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_8u_C1MR_Ctx(Mpp8u nValue, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * Masked 8-bit unsigned image set. 
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_8u_C1MR(Mpp8u nValue, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, const Mpp8u * pMask, int nMaskStep);

/** 
 * Masked 3 channel 8-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_8u_C3MR_Ctx(const Mpp8u aValue[3], Mpp8u* pDst, int nDstStep, MppiSize oSizeROI,
                    const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * Masked 3 channel 8-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_8u_C3MR(const Mpp8u aValue[3], Mpp8u* pDst, int nDstStep, MppiSize oSizeROI,
                const Mpp8u * pMask, int nMaskStep);

/** 
 * Masked 4 channel 8-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_8u_C4MR_Ctx(const Mpp8u aValue[4], Mpp8u* pDst, int nDstStep, MppiSize oSizeROI,
                    const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * Masked 4 channel 8-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_8u_C4MR(const Mpp8u aValue[4], Mpp8u* pDst, int nDstStep, MppiSize oSizeROI,
                const Mpp8u * pMask, int nMaskStep);

/** 
 * Masked 4 channel 8-bit unsigned image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_8u_AC4MR_Ctx(const Mpp8u aValue[3], Mpp8u * pDst, int nDstStep, 
                     MppiSize oSizeROI,
                     const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * Masked 4 channel 8-bit unsigned image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_8u_AC4MR(const Mpp8u aValue[3], Mpp8u * pDst, int nDstStep, 
                 MppiSize oSizeROI,
                 const Mpp8u * pMask, int nMaskStep);

/** 
 * Masked 16-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_16u_C1MR_Ctx(Mpp16u nValue, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * Masked 16-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_16u_C1MR(Mpp16u nValue, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, const Mpp8u * pMask, int nMaskStep);

/** 
 * Masked 3 channel 16-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_16u_C3MR_Ctx(const Mpp16u aValue[3], Mpp16u * pDst, int nDstStep, 
                     MppiSize oSizeROI,
                     const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * Masked 3 channel 16-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_16u_C3MR(const Mpp16u aValue[3], Mpp16u * pDst, int nDstStep, 
                 MppiSize oSizeROI,
                 const Mpp8u * pMask, int nMaskStep);

/** 
 * Masked 4 channel 16-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_16u_C4MR_Ctx(const Mpp16u aValue[4], Mpp16u * pDst, int nDstStep, 
                     MppiSize oSizeROI,
                     const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * Masked 4 channel 16-bit unsigned image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_16u_C4MR(const Mpp16u aValue[4], Mpp16u * pDst, int nDstStep, 
                 MppiSize oSizeROI,
                 const Mpp8u * pMask, int nMaskStep);

/** 
 * Masked 4 channel 16-bit unsigned image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_16u_AC4MR_Ctx(const Mpp16u aValue[3], Mpp16u * pDst, int nDstStep, 
                      MppiSize oSizeROI,
                      const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * Masked 4 channel 16-bit unsigned image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_16u_AC4MR(const Mpp16u aValue[3], Mpp16u * pDst, int nDstStep, 
                  MppiSize oSizeROI,
                  const Mpp8u * pMask, int nMaskStep);

/** 
 * Masked 16-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_16s_C1MR_Ctx(Mpp16s nValue, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * Masked 16-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_16s_C1MR(Mpp16s nValue, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, const Mpp8u * pMask, int nMaskStep);

/** 
 * Masked 3 channel 16-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_16s_C3MR_Ctx(const Mpp16s aValue[3], Mpp16s * pDst, int nDstStep, 
                     MppiSize oSizeROI,
                     const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);
                          
/** 
 * Masked 3 channel 16-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_16s_C3MR(const Mpp16s aValue[3], Mpp16s * pDst, int nDstStep, 
                 MppiSize oSizeROI,
                 const Mpp8u * pMask, int nMaskStep);
                          
/** 
 * Masked 4 channel 16-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_16s_C4MR_Ctx(const Mpp16s aValue[4], Mpp16s * pDst, int nDstStep, 
                     MppiSize oSizeROI,
                     const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);
                          
/** 
 * Masked 4 channel 16-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_16s_C4MR(const Mpp16s aValue[4], Mpp16s * pDst, int nDstStep, 
                 MppiSize oSizeROI,
                 const Mpp8u * pMask, int nMaskStep);
                          
/** 
 * Masked 4 channel 16-bit image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_16s_AC4MR_Ctx(const Mpp16s aValue[3], Mpp16s * pDst, int nDstStep, 
                      MppiSize oSizeROI, 
                      const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * Masked 4 channel 16-bit image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_16s_AC4MR(const Mpp16s aValue[3], Mpp16s * pDst, int nDstStep, 
                  MppiSize oSizeROI, 
                  const Mpp8u * pMask, int nMaskStep);

/** 
 * Masked 32-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_32s_C1MR_Ctx(Mpp32s nValue, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * Masked 32-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_32s_C1MR(Mpp32s nValue, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, const Mpp8u * pMask, int nMaskStep);

/** 
 * Masked 3 channel 32-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_32s_C3MR_Ctx(const Mpp32s aValue[3], Mpp32s * pDst, int nDstStep, 
                     MppiSize oSizeROI,
                     const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);
                          
/** 
 * Masked 3 channel 32-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_32s_C3MR(const Mpp32s aValue[3], Mpp32s * pDst, int nDstStep, 
                 MppiSize oSizeROI,
                 const Mpp8u * pMask, int nMaskStep);
                          
/** 
 * Masked 4 channel 32-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_32s_C4MR_Ctx(const Mpp32s aValue[4], Mpp32s * pDst, int nDstStep, 
                     MppiSize oSizeROI,
                     const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);
                          
/** 
 * Masked 4 channel 32-bit image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_32s_C4MR(const Mpp32s aValue[4], Mpp32s * pDst, int nDstStep, 
                 MppiSize oSizeROI,
                 const Mpp8u * pMask, int nMaskStep);
                          
/** 
 * Masked 4 channel 16-bit image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_32s_AC4MR_Ctx(const Mpp32s aValue[3], Mpp32s * pDst, int nDstStep, 
                      MppiSize oSizeROI, 
                      const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * Masked 4 channel 16-bit image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_32s_AC4MR(const Mpp32s aValue[3], Mpp32s * pDst, int nDstStep, 
                  MppiSize oSizeROI, 
                  const Mpp8u * pMask, int nMaskStep);

/** 
 * Masked 32-bit floating point image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_32f_C1MR_Ctx(Mpp32f nValue, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * Masked 32-bit floating point image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_32f_C1MR(Mpp32f nValue, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, const Mpp8u * pMask, int nMaskStep);

/** 
 * Masked 3 channel 32-bit floating point image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_32f_C3MR_Ctx(const Mpp32f aValue[3], Mpp32f * pDst, int nDstStep, 
                     MppiSize oSizeROI,
                     const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * Masked 3 channel 32-bit floating point image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_32f_C3MR(const Mpp32f aValue[3], Mpp32f * pDst, int nDstStep, 
                 MppiSize oSizeROI,
                 const Mpp8u * pMask, int nMaskStep);

/** 
 * Masked 4 channel 32-bit floating point image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_32f_C4MR_Ctx(const Mpp32f aValue[4], Mpp32f * pDst, int nDstStep, 
                     MppiSize oSizeROI,
                     const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);
                          
/** 
 * Masked 4 channel 32-bit floating point image set.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_32f_C4MR(const Mpp32f aValue[4], Mpp32f * pDst, int nDstStep, 
                 MppiSize oSizeROI,
                 const Mpp8u * pMask, int nMaskStep);
                          
/** 
 * Masked 4 channel 32-bit floating point image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_32f_AC4MR_Ctx(const Mpp32f aValue[3], Mpp32f * pDst, int nDstStep, 
                      MppiSize oSizeROI,
                      const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * Masked 4 channel 32-bit floating point image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedSetParameters.
 *
 */
MppStatus 
mppiSet_32f_AC4MR(const Mpp32f aValue[3], Mpp32f * pDst, int nDstStep, 
                  MppiSize oSizeROI,
                  const Mpp8u * pMask, int nMaskStep);

/** @} image_masked_set */

/** 
 * \section image_channel_set Channel Set
 * @defgroup image_channel_set Channel Set
 * The selected-channel set primitives set a single color channel in multi-channel images
 * to a given value. The channel is selected by adjusting the pDst pointer to point to 
 * the desired color channel (see \ref channel_of_interest).
 *
 * \subsection CommonImageChannelSetParameters Common parameters for mppiSet_CXC functions:
 *
 * \param nValue The pixel-value to be set.
 * \param pDst \ref select_destination_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * 3 channel 8-bit unsigned image set affecting only single channel.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelSetParameters.
 *
 */
MppStatus 
mppiSet_8u_C3CR_Ctx(Mpp8u nValue, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 3 channel 8-bit unsigned image set affecting only single channel.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelSetParameters.
 *
 */
MppStatus 
mppiSet_8u_C3CR(Mpp8u nValue, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 8-bit unsigned image set affecting only single channel.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelSetParameters.
 *
 */
MppStatus 
mppiSet_8u_C4CR_Ctx(Mpp8u nValue, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 8-bit unsigned image set affecting only single channel.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelSetParameters.
 *
 */
MppStatus 
mppiSet_8u_C4CR(Mpp8u nValue, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 3 channel 16-bit unsigned image set affecting only single channel.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelSetParameters.
 *
 */
MppStatus 
mppiSet_16u_C3CR_Ctx(Mpp16u nValue, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 3 channel 16-bit unsigned image set affecting only single channel.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelSetParameters.
 *
 */
MppStatus 
mppiSet_16u_C3CR(Mpp16u nValue, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 16-bit unsigned image set affecting only single channel.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelSetParameters.
 *
 */
MppStatus 
mppiSet_16u_C4CR_Ctx(Mpp16u nValue, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 16-bit unsigned image set affecting only single channel.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelSetParameters.
 *
 */
MppStatus 
mppiSet_16u_C4CR(Mpp16u nValue, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 3 channel 16-bit signed image set affecting only single channel.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelSetParameters.
 *
 */
MppStatus 
mppiSet_16s_C3CR_Ctx(Mpp16s nValue, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 3 channel 16-bit signed image set affecting only single channel.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelSetParameters.
 *
 */
MppStatus 
mppiSet_16s_C3CR(Mpp16s nValue, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 16-bit signed image set affecting only single channel.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelSetParameters.
 *
 */
MppStatus 
mppiSet_16s_C4CR_Ctx(Mpp16s nValue, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 16-bit signed image set affecting only single channel.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelSetParameters.
 *
 */
MppStatus 
mppiSet_16s_C4CR(Mpp16s nValue, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 3 channel 32-bit unsigned image set affecting only single channel.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelSetParameters.
 *
 */
MppStatus 
mppiSet_32s_C3CR_Ctx(Mpp32s nValue, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 3 channel 32-bit unsigned image set affecting only single channel.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelSetParameters.
 *
 */
MppStatus 
mppiSet_32s_C3CR(Mpp32s nValue, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 32-bit unsigned image set affecting only single channel.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelSetParameters.
 *
 */
MppStatus 
mppiSet_32s_C4CR_Ctx(Mpp32s nValue, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 32-bit unsigned image set affecting only single channel.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelSetParameters.
 *
 */
MppStatus 
mppiSet_32s_C4CR(Mpp32s nValue, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 3 channel 32-bit floating point image set affecting only single channel.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelSetParameters.
 *
 */
MppStatus 
mppiSet_32f_C3CR_Ctx(Mpp32f nValue, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 3 channel 32-bit floating point image set affecting only single channel.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelSetParameters.
 *
 */
MppStatus 
mppiSet_32f_C3CR(Mpp32f nValue, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 32-bit floating point image set affecting only single channel.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelSetParameters.
 *
 */
MppStatus 
mppiSet_32f_C4CR_Ctx(Mpp32f nValue, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 32-bit floating point image set affecting only single channel.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelSetParameters.
 *
 */
MppStatus 
mppiSet_32f_C4CR(Mpp32f nValue, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** @} image_channel_set */

/** @} image_set_operations */

/** 
 * @defgroup image_copy_operations Image Copy Operations
 * Functions for copying image pixels.
 *
 * @{
 *
 */

/** 
 * \section image_copy Copy
 * @defgroup image_copy Copy
 * Copy pixels from one image to another.
 *
 * \subsection CommonImageCopyParameters Common parameters for mppiCopy functions:
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * 8-bit image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_8s_C1R_Ctx(const Mpp8s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 8-bit image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_8s_C1R(const Mpp8s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Two-channel 8-bit image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_8s_C2R_Ctx(const Mpp8s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Two-channel 8-bit image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_8s_C2R(const Mpp8s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three-channel 8-bit image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_8s_C3R_Ctx(const Mpp8s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three-channel 8-bit image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_8s_C3R(const Mpp8s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four-channel 8-bit image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_8s_C4R_Ctx(const Mpp8s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four-channel 8-bit image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_8s_C4R(const Mpp8s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four-channel 8-bit image copy, ignoring alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_8s_AC4R_Ctx(const Mpp8s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four-channel 8-bit image copy, ignoring alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_8s_AC4R(const Mpp8s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_8u_C1R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_8u_C1R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_8u_C4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_8u_C4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 8-bit unsigned image copy, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 8-bit unsigned image copy, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16u_C1R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16u_C1R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16u_C3R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16u_C3R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16u_C4R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16u_C4R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 16-bit unsigned image copy, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16u_AC4R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 16-bit unsigned image copy, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16u_AC4R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 16-bit image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16s_C1R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 16-bit image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16s_C1R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three channel 16-bit image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16s_C3R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 16-bit image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16s_C3R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 16-bit image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16s_C4R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 16-bit image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16s_C4R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 16-bit image copy, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16s_AC4R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 16-bit image copy, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16s_AC4R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 16-bit complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16sc_C1R_Ctx(const Mpp16sc * pSrc, int nSrcStep, Mpp16sc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 16-bit complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16sc_C1R(const Mpp16sc * pSrc, int nSrcStep, Mpp16sc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Two-channel 16-bit complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16sc_C2R_Ctx(const Mpp16sc * pSrc, int nSrcStep, Mpp16sc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Two-channel 16-bit complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16sc_C2R(const Mpp16sc * pSrc, int nSrcStep, Mpp16sc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three-channel 16-bit complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16sc_C3R_Ctx(const Mpp16sc * pSrc, int nSrcStep, Mpp16sc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three-channel 16-bit complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16sc_C3R(const Mpp16sc * pSrc, int nSrcStep, Mpp16sc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four-channel 16-bit complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16sc_C4R_Ctx(const Mpp16sc * pSrc, int nSrcStep, Mpp16sc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four-channel 16-bit complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16sc_C4R(const Mpp16sc * pSrc, int nSrcStep, Mpp16sc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four-channel 16-bit complex image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16sc_AC4R_Ctx(const Mpp16sc * pSrc, int nSrcStep, Mpp16sc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four-channel 16-bit complex image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16sc_AC4R(const Mpp16sc * pSrc, int nSrcStep, Mpp16sc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 32-bit image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32s_C1R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 32-bit image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32s_C1R(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three channel 32-bit image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32s_C3R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 32-bit image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32s_C3R(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 32-bit image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32s_C4R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 32-bit image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32s_C4R(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 32-bit image copy, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32s_AC4R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 32-bit image copy, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32s_AC4R(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 32-bit complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32sc_C1R_Ctx(const Mpp32sc * pSrc, int nSrcStep, Mpp32sc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 32-bit complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32sc_C1R(const Mpp32sc * pSrc, int nSrcStep, Mpp32sc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Two-channel 32-bit complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32sc_C2R_Ctx(const Mpp32sc * pSrc, int nSrcStep, Mpp32sc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Two-channel 32-bit complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32sc_C2R(const Mpp32sc * pSrc, int nSrcStep, Mpp32sc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three-channel 32-bit complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32sc_C3R_Ctx(const Mpp32sc * pSrc, int nSrcStep, Mpp32sc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three-channel 32-bit complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32sc_C3R(const Mpp32sc * pSrc, int nSrcStep, Mpp32sc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four-channel 32-bit complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32sc_C4R_Ctx(const Mpp32sc * pSrc, int nSrcStep, Mpp32sc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four-channel 32-bit complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32sc_C4R(const Mpp32sc * pSrc, int nSrcStep, Mpp32sc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four-channel 32-bit complex image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32sc_AC4R_Ctx(const Mpp32sc * pSrc, int nSrcStep, Mpp32sc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four-channel 32-bit complex image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32sc_AC4R(const Mpp32sc * pSrc, int nSrcStep, Mpp32sc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 16-bit floating point image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16f_C1R_Ctx(const Mpp16f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 16-bit floating point image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16f_C1R(const Mpp16f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three channel 16-bit floating point image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16f_C3R_Ctx(const Mpp16f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 16-bit floating point image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16f_C3R(const Mpp16f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 16-bit floating point image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16f_C4R_Ctx(const Mpp16f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 16-bit floating point image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_16f_C4R(const Mpp16f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 32-bit floating point image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32f_C1R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 32-bit floating point image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32f_C1R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three channel 32-bit floating point image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32f_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 32-bit floating point image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32f_C3R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 32-bit floating point image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32f_C4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 32-bit floating point image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32f_C4R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 4 channel 32-bit floating point image copy, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32f_AC4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 4 channel 32-bit floating point image copy, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32f_AC4R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * 32-bit floating-point complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32fc_C1R_Ctx(const Mpp32fc * pSrc, int nSrcStep, Mpp32fc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 32-bit floating-point complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32fc_C1R(const Mpp32fc * pSrc, int nSrcStep, Mpp32fc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Two-channel 32-bit floating-point complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32fc_C2R_Ctx(const Mpp32fc * pSrc, int nSrcStep, Mpp32fc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Two-channel 32-bit floating-point complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32fc_C2R(const Mpp32fc * pSrc, int nSrcStep, Mpp32fc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three-channel 32-bit floating-point complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32fc_C3R_Ctx(const Mpp32fc * pSrc, int nSrcStep, Mpp32fc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three-channel 32-bit floating-point complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32fc_C3R(const Mpp32fc * pSrc, int nSrcStep, Mpp32fc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four-channel 32-bit floating-point complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32fc_C4R_Ctx(const Mpp32fc * pSrc, int nSrcStep, Mpp32fc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four-channel 32-bit floating-point complex image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32fc_C4R(const Mpp32fc * pSrc, int nSrcStep, Mpp32fc * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four-channel 32-bit floating-point complex image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32fc_AC4R_Ctx(const Mpp32fc * pSrc, int nSrcStep, Mpp32fc * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four-channel 32-bit floating-point complex image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyParameters.
 *
 */
MppStatus 
mppiCopy_32fc_AC4R(const Mpp32fc * pSrc, int nSrcStep, Mpp32fc * pDst, int nDstStep, MppiSize oSizeROI);

/** @} image_copy */

/** 
 * \section image_masked_copy Masked Copy
 * @defgroup image_masked_copy Masked Copy
 * The masked copy primitives have an additional "mask image" input. The mask  
 * controls which pixels within the ROI are copied. For details see \ref masked_operation.
 *
 * \subsection CommonImageMaskedCopyParameters Common parameters for mppiCopy_CXM functions:
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * \ref masked_operation 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_8u_C1MR_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                     const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * \ref masked_operation 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_8u_C1MR(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                 const Mpp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation three channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_8u_C3MR_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                     const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * \ref masked_operation three channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_8u_C3MR(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                 const Mpp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_8u_C4MR_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                     const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * \ref masked_operation four channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_8u_C4MR(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                 const Mpp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 8-bit unsigned image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_8u_AC4MR_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * \ref masked_operation four channel 8-bit unsigned image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_8u_AC4MR(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, 
                  const Mpp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_16u_C1MR_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * \ref masked_operation 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_16u_C1MR(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                  const Mpp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation three channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_16u_C3MR_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * \ref masked_operation three channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_16u_C3MR(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                  const Mpp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_16u_C4MR_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * \ref masked_operation four channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_16u_C4MR(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                  const Mpp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 16-bit unsigned image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_16u_AC4MR_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                       const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * \ref masked_operation four channel 16-bit unsigned image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_16u_AC4MR(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, 
                   const Mpp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation 16-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_16s_C1MR_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * \ref masked_operation 16-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_16s_C1MR(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                  const Mpp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation three channel 16-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_16s_C3MR_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * \ref masked_operation three channel 16-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_16s_C3MR(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                  const Mpp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 16-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_16s_C4MR_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * \ref masked_operation four channel 16-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_16s_C4MR(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                  const Mpp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 16-bit signed image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_16s_AC4MR_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                       const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * \ref masked_operation four channel 16-bit signed image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_16s_AC4MR(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, 
                   const Mpp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation 32-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_32s_C1MR_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * \ref masked_operation 32-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_32s_C1MR(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, 
                  const Mpp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation three channel 32-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_32s_C3MR_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * \ref masked_operation three channel 32-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_32s_C3MR(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, 
                  const Mpp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 32-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_32s_C4MR_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * \ref masked_operation four channel 32-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_32s_C4MR(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, 
                  const Mpp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 32-bit signed image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_32s_AC4MR_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, 
                       const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * \ref masked_operation four channel 32-bit signed image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_32s_AC4MR(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, 
                   const Mpp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation 32-bit float image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_32f_C1MR_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * \ref masked_operation 32-bit float image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_32f_C1MR(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                  const Mpp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation three channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_32f_C3MR_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * \ref masked_operation three channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_32f_C3MR(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                  const Mpp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_32f_C4MR_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                      const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * \ref masked_operation four channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_32f_C4MR(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                  const Mpp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 32-bit float image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_32f_AC4MR_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                       const Mpp8u * pMask, int nMaskStep, MppStreamContext mppStreamCtx);

/** 
 * \ref masked_operation four channel 32-bit float image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageMaskedCopyParameters.
 *
 */
MppStatus 
mppiCopy_32f_AC4MR(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, 
                   const Mpp8u * pMask, int nMaskStep);

/** @} image_masked_copy */


/** 
 * \section image_channel_copy Channel Copy
 * @defgroup image_channel_copy Channel Copy
 * The channel copy primitives copy a single color channel from a multi-channel source image
 * to any other color channel in a multi-channel destination image. The channel is selected 
 * by adjusting the respective image pointers to point to the desired color channel 
 * (see \ref channel_of_interest).
 *
 * \subsection CommonImageChannelCopyParameters Common parameters for mppiCopy_CXC functions:
 *
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref select_destination_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * Selected channel 8-bit unsigned image copy for three-channel images.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_8u_C3CR_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Selected channel 8-bit unsigned image copy for three-channel images.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_8u_C3CR(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Selected channel 8-bit unsigned image copy for four-channel images.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_8u_C4CR_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Selected channel 8-bit unsigned image copy for four-channel images.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_8u_C4CR(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Selected channel 16-bit signed image copy for three-channel images.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16s_C3CR_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Selected channel 16-bit signed image copy for three-channel images.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16s_C3CR(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Selected channel 16-bit signed image copy for four-channel images.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16s_C4CR_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Selected channel 16-bit signed image copy for four-channel images.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16s_C4CR(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Selected channel 16-bit unsigned image copy for three-channel images.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16u_C3CR_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Selected channel 16-bit unsigned image copy for three-channel images.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16u_C3CR(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Selected channel 16-bit unsigned image copy for four-channel images.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16u_C4CR_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Selected channel 16-bit unsigned image copy for four-channel images.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16u_C4CR(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Selected channel 32-bit signed image copy for three-channel images.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32s_C3CR_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Selected channel 32-bit signed image copy for three-channel images.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32s_C3CR(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Selected channel 32-bit signed image copy for four-channel images.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32s_C4CR_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Selected channel 32-bit signed image copy for four-channel images.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32s_C4CR(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Selected channel 32-bit float image copy for three-channel images.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_C3CR_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Selected channel 32-bit float image copy for three-channel images.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_C3CR(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Selected channel 32-bit float image copy for four-channel images.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_C4CR_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Selected channel 32-bit float image copy for four-channel images.
 * 
 * For common parameter descriptions, see \ref CommonImageChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_C4CR(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** @} image_channel_copy */

/** 
 * \section image_extract_channel_copy Extract Channel Copy
 * @defgroup image_extract_channel_copy Extract Channel Copy
 * The channel extract primitives copy a single color channel from a multi-channel source image
 * to singl-channel destination image. The channel is selected by adjusting the source image pointer
 * to point to the desired color channel (see \ref channel_of_interest).
 *
 * \subsection CommonImageExtractChannelCopyParameters Common parameters for mppiCopy_CXC1 functions:
 *
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * Three-channel to single-channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageExtractChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_8u_C3C1R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three-channel to single-channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageExtractChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_8u_C3C1R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four-channel to single-channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageExtractChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_8u_C4C1R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four-channel to single-channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageExtractChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_8u_C4C1R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three-channel to single-channel 16-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageExtractChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16s_C3C1R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three-channel to single-channel 16-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageExtractChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16s_C3C1R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four-channel to single-channel 16-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageExtractChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16s_C4C1R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four-channel to single-channel 16-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageExtractChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16s_C4C1R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three-channel to single-channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageExtractChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16u_C3C1R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three-channel to single-channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageExtractChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16u_C3C1R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four-channel to single-channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageExtractChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16u_C4C1R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four-channel to single-channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageExtractChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16u_C4C1R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three-channel to single-channel 32-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageExtractChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32s_C3C1R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three-channel to single-channel 32-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageExtractChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32s_C3C1R(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four-channel to single-channel 32-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageExtractChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32s_C4C1R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four-channel to single-channel 32-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageExtractChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32s_C4C1R(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Two-channel to single-channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageExtractChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_C2C1R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Two-channel to single-channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageExtractChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_C2C1R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three-channel to single-channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageExtractChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_C3C1R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three-channel to single-channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageExtractChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_C3C1R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four-channel to single-channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageExtractChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_C4C1R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four-channel to single-channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageExtractChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_C4C1R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** @} image_extract_channel_copy */

/** 
 * \section image_insert_channel_copy Insert Channel Copy
 * @defgroup image_insert_channel_copy Insert Channel Copy
 * The channel insert primitives copy a single-channel source image into one of the color channels
 * in a multi-channel destination image. The channel is selected by adjusting the destination image pointer
 * to point to the desired color channel (see \ref channel_of_interest).
 *
 * \subsection CommonImageInsertChannelCopyParameters Common parameters for mppiCopy_C1CX functions:
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref select_destination_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * Single-channel to three-channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageInsertChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_8u_C1C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single-channel to three-channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageInsertChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_8u_C1C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single-channel to four-channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageInsertChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_8u_C1C4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single-channel to four-channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageInsertChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_8u_C1C4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single-channel to three-channel 16-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageInsertChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16s_C1C3R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single-channel to three-channel 16-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageInsertChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16s_C1C3R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single-channel to four-channel 16-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageInsertChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16s_C1C4R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single-channel to four-channel 16-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageInsertChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16s_C1C4R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single-channel to three-channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageInsertChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16u_C1C3R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single-channel to three-channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageInsertChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16u_C1C3R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single-channel to four-channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageInsertChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16u_C1C4R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single-channel to four-channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageInsertChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16u_C1C4R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single-channel to three-channel 32-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageInsertChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32s_C1C3R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single-channel to three-channel 32-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageInsertChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32s_C1C3R(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single-channel to four-channel 32-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageInsertChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32s_C1C4R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single-channel to four-channel 32-bit signed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageInsertChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32s_C1C4R(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single-channel to two-channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageInsertChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_C1C2R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single-channel to two-channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageInsertChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_C1C2R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single-channel to three-channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageInsertChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_C1C3R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single-channel to three-channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageInsertChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_C1C3R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single-channel to four-channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageInsertChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_C1C4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single-channel to four-channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see \ref CommonImageInsertChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_C1C4R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** @} image_insert_channel_copy */


/** 
 * \section image_packed_to_planar_channel_copy Packed To Planar Channel Copy
 * @defgroup image_packed_to_planar_channel_copy Packed To Planar Channel Copy
 * Split a packed multi-channel image into multiple single channel planes.
 *
 * E.g. copy the three channels of an RGB image into three separate single-channel
 * images.
 *
 * \subsection CommonImagePackedToPlanarChannelCopyParameters Common parameters for mppiCopy_CXPX functions:
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param aDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * Three-channel 8-bit unsigned packed to planar image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePackedToPlanarChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_8u_C3P3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * const aDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three-channel 8-bit unsigned packed to planar image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePackedToPlanarChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_8u_C3P3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * const aDst[3], int nDstStep, MppiSize oSizeROI);

/** 
 * Four-channel 8-bit unsigned packed to planar image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePackedToPlanarChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_8u_C4P4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * const aDst[4], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four-channel 8-bit unsigned packed to planar image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePackedToPlanarChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_8u_C4P4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * const aDst[4], int nDstStep, MppiSize oSizeROI);

/** 
 * Three-channel 16-bit signed packed to planar image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePackedToPlanarChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16s_C3P3R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * const aDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three-channel 16-bit signed packed to planar image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePackedToPlanarChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16s_C3P3R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * const aDst[3], int nDstStep, MppiSize oSizeROI);

/** 
 * Four-channel 16-bit signed packed to planar image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePackedToPlanarChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16s_C4P4R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * const aDst[4], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four-channel 16-bit signed packed to planar image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePackedToPlanarChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16s_C4P4R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * const aDst[4], int nDstStep, MppiSize oSizeROI);

/** 
 * Three-channel 16-bit unsigned packed to planar image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePackedToPlanarChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16u_C3P3R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * const aDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three-channel 16-bit unsigned packed to planar image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePackedToPlanarChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16u_C3P3R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * const aDst[3], int nDstStep, MppiSize oSizeROI);

/** 
 * Four-channel 16-bit unsigned packed to planar image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePackedToPlanarChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16u_C4P4R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * const aDst[4], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four-channel 16-bit unsigned packed to planar image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePackedToPlanarChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16u_C4P4R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * const aDst[4], int nDstStep, MppiSize oSizeROI);

/** 
 * Three-channel 32-bit signed packed to planar image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePackedToPlanarChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32s_C3P3R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp32s * const aDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three-channel 32-bit signed packed to planar image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePackedToPlanarChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32s_C3P3R(const Mpp32s * pSrc, int nSrcStep, Mpp32s * const aDst[3], int nDstStep, MppiSize oSizeROI);

/** 
 * Four-channel 32-bit signed packed to planar image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePackedToPlanarChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32s_C4P4R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp32s * const aDst[4], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four-channel 32-bit signed packed to planar image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePackedToPlanarChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32s_C4P4R(const Mpp32s * pSrc, int nSrcStep, Mpp32s * const aDst[4], int nDstStep, MppiSize oSizeROI);

/** 
 * Three-channel 32-bit float packed to planar image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePackedToPlanarChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_C3P3R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * const aDst[3], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three-channel 32-bit float packed to planar image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePackedToPlanarChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_C3P3R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * const aDst[3], int nDstStep, MppiSize oSizeROI);

/** 
 * Four-channel 32-bit float packed to planar image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePackedToPlanarChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_C4P4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * const aDst[4], int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four-channel 32-bit float packed to planar image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePackedToPlanarChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_C4P4R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * const aDst[4], int nDstStep, MppiSize oSizeROI);

/** @} image_packed_to_planar_channel_copy */


/** 
 * \section image_planar_to_packed_channel_copy Planar To Packed Channel Copy
 * @defgroup image_planar_to_packed_channel_copy Planar To Packed Channel Copy
 * Combine multiple image planes into a packed multi-channel image.
 *
 * E.g. copy three single-channel images into a single 3-channel image.
 *
 * \subsection CommonImagePlanarToPackedChannelCopyParameters Common parameters for mppiCopy_PXCX functions:
 *
 * \param aSrc Planar \ref source_image_pointer.
 * \param nSrcStep \ref source_planar_image_pointer_array.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * Three-channel 8-bit unsigned planar to packed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePlanarToPackedChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_8u_P3C3R_Ctx(const Mpp8u * const aSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three-channel 8-bit unsigned planar to packed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePlanarToPackedChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_8u_P3C3R(const Mpp8u * const aSrc[3], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four-channel 8-bit unsigned planar to packed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePlanarToPackedChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_8u_P4C4R_Ctx(const Mpp8u * const aSrc[4], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four-channel 8-bit unsigned planar to packed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePlanarToPackedChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_8u_P4C4R(const Mpp8u * const aSrc[4], int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three-channel 16-bit unsigned planar to packed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePlanarToPackedChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16u_P3C3R_Ctx(const Mpp16u * const aSrc[3], int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three-channel 16-bit unsigned planar to packed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePlanarToPackedChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16u_P3C3R(const Mpp16u * const aSrc[3], int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four-channel 16-bit unsigned planar to packed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePlanarToPackedChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16u_P4C4R_Ctx(const Mpp16u * const aSrc[4], int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four-channel 16-bit unsigned planar to packed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePlanarToPackedChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16u_P4C4R(const Mpp16u * const aSrc[4], int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three-channel 16-bit signed planar to packed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePlanarToPackedChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16s_P3C3R_Ctx(const Mpp16s * const aSrc[3], int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three-channel 16-bit signed planar to packed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePlanarToPackedChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16s_P3C3R(const Mpp16s * const aSrc[3], int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four-channel 16-bit signed planar to packed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePlanarToPackedChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16s_P4C4R_Ctx(const Mpp16s * const aSrc[4], int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four-channel 16-bit signed planar to packed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePlanarToPackedChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_16s_P4C4R(const Mpp16s * const aSrc[4], int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three-channel 32-bit signed planar to packed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePlanarToPackedChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32s_P3C3R_Ctx(const Mpp32s * const aSrc[3], int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three-channel 32-bit signed planar to packed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePlanarToPackedChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32s_P3C3R(const Mpp32s * const aSrc[3], int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four-channel 32-bit signed planar to packed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePlanarToPackedChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32s_P4C4R_Ctx(const Mpp32s * const aSrc[4], int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four-channel 32-bit signed planar to packed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePlanarToPackedChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32s_P4C4R(const Mpp32s * const aSrc[4], int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three-channel 32-bit float planar to packed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePlanarToPackedChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_P3C3R_Ctx(const Mpp32f * const aSrc[3], int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three-channel 32-bit float planar to packed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePlanarToPackedChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_P3C3R(const Mpp32f * const aSrc[3], int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four-channel 32-bit float planar to packed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePlanarToPackedChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_P4C4R_Ctx(const Mpp32f * const aSrc[4], int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four-channel 32-bit float planar to packed image copy.
 * 
 * For common parameter descriptions, see \ref CommonImagePlanarToPackedChannelCopyParameters.
 *
 */
MppStatus
mppiCopy_32f_P4C4R(const Mpp32f * const aSrc[4], int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** @} image_planar_to_packed_channel_copy */

/** 
 * \section image_copy_constant_border Copy Constant Border
 * @defgroup image_copy_constant_border Copy Constant Border
 * Methods for copying images and padding borders with a constant, user-specifiable color.
 * 
 * \subsection CommonImageCopyConstantBorderParameters Common parameters for mppiCopyConstBorder functions:
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSizeROI Size of the source region of pixels.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
 *      data from the source image (inner part) and constant border color (outer part).
 * \param nTopBorderHeight Height (in pixels) of the top border. The number of pixel rows at the top of the
 *      destination ROI that will be filled with the constant border color.
 *      nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.
 * \param nLeftBorderWidth Width (in pixels) of the left border. The width of the border at the right side of the
 *      destination ROI is implicitly defined by the size of the source ROI:
 *      nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.
 * \param nValue The pixel value to be set for border pixels for single channel functions.
 * \param aValue Vector of the RGBA values of the border pixels to be set for multi-channel functions.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_8u_C1R_Ctx(const Mpp8u * pSrc,   int nSrcStep, MppiSize oSrcSizeROI,
                                               Mpp8u * pDst,   int nDstStep, MppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth,
                                         Mpp8u nValue, MppStreamContext mppStreamCtx);

/** 
 * 1 channel 8-bit unsigned integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_8u_C1R(const Mpp8u * pSrc,   int nSrcStep, MppiSize oSrcSizeROI,
                                           Mpp8u * pDst,   int nDstStep, MppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth,
                                     Mpp8u nValue);

/**
 * 3 channel 8-bit unsigned integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_8u_C3R_Ctx(const Mpp8u * pSrc,   int nSrcStep, MppiSize oSrcSizeROI,
                                               Mpp8u * pDst,   int nDstStep, MppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth,
                                         const Mpp8u aValue[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_8u_C3R(const Mpp8u * pSrc,   int nSrcStep, MppiSize oSrcSizeROI,
                                           Mpp8u * pDst,   int nDstStep, MppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth,
                                     const Mpp8u aValue[3]);

/**
 * 4 channel 8-bit unsigned integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_8u_C4R_Ctx(const Mpp8u * pSrc,   int nSrcStep, MppiSize oSrcSizeROI,
                                               Mpp8u * pDst,   int nDstStep, MppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth,
                                         const Mpp8u aValue[4], MppStreamContext mppStreamCtx);
                                       
/**
 * 4 channel 8-bit unsigned integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_8u_C4R(const Mpp8u * pSrc,   int nSrcStep, MppiSize oSrcSizeROI,
                                           Mpp8u * pDst,   int nDstStep, MppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth,
                                     const Mpp8u aValue[4]);
                                       
/**
 * 4 channel 8-bit unsigned integer image copy with constant border color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_8u_AC4R_Ctx(const Mpp8u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp8u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          const Mpp8u aValue[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned integer image copy with constant border color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_8u_AC4R(const Mpp8u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                            Mpp8u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      const Mpp8u aValue[3]);

/** 
 * 1 channel 16-bit unsigned integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_16u_C1R_Ctx(const Mpp16u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp16u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          Mpp16u nValue, MppStreamContext mppStreamCtx);

/** 
 * 1 channel 16-bit unsigned integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_16u_C1R(const Mpp16u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                            Mpp16u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      Mpp16u nValue);

/**
 * 3 channel 16-bit unsigned integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_16u_C3R_Ctx(const Mpp16u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp16u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          const Mpp16u aValue[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit unsigned integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_16u_C3R(const Mpp16u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                            Mpp16u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      const Mpp16u aValue[3]);

/**
 * 4 channel 16-bit unsigned integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_16u_C4R_Ctx(const Mpp16u * pSrc, int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp16u * pDst, int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          const Mpp16u aValue[4], MppStreamContext mppStreamCtx);
                                       
/**
 * 4 channel 16-bit unsigned integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_16u_C4R(const Mpp16u * pSrc, int nSrcStep, MppiSize oSrcSizeROI,
                                            Mpp16u * pDst, int nDstStep, MppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      const Mpp16u aValue[4]);
                                       
/**
 * 4 channel 16-bit unsigned integer image copy with constant border color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_16u_AC4R_Ctx(const Mpp16u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                 Mpp16u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                           int nTopBorderHeight, int nLeftBorderWidth,
                                           const Mpp16u aValue[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit unsigned integer image copy with constant border color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_16u_AC4R(const Mpp16u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                             Mpp16u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                       int nTopBorderHeight, int nLeftBorderWidth,
                                       const Mpp16u aValue[3]);

/** 
 * 1 channel 16-bit signed integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_16s_C1R_Ctx(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          Mpp16s nValue, MppStreamContext mppStreamCtx);

/** 
 * 1 channel 16-bit signed integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_16s_C1R(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                            Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      Mpp16s nValue);

/**
 * 3 channel 16-bit signed integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_16s_C3R_Ctx(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          const Mpp16s aValue[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit signed integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_16s_C3R(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                            Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      const Mpp16s aValue[3]);

/**
 * 4 channel 16-bit signed integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_16s_C4R_Ctx(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          const Mpp16s aValue[4], MppStreamContext mppStreamCtx);
                                       
/**
 * 4 channel 16-bit signed integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_16s_C4R(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                            Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      const Mpp16s aValue[4]);
                                       
/**
 * 4 channel 16-bit signed integer image copy with constant border color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_16s_AC4R_Ctx(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                 Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                           int nTopBorderHeight, int nLeftBorderWidth,
                                           const Mpp16s aValue[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit signed integer image copy with constant border color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_16s_AC4R(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                             Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                       int nTopBorderHeight, int nLeftBorderWidth,
                                       const Mpp16s aValue[3]);

/** 
 * 1 channel 32-bit signed integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_32s_C1R_Ctx(const Mpp32s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp32s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          Mpp32s nValue, MppStreamContext mppStreamCtx);

/** 
 * 1 channel 32-bit signed integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_32s_C1R(const Mpp32s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                            Mpp32s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      Mpp32s nValue);

/**
 * 3 channel 32-bit signed integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_32s_C3R_Ctx(const Mpp32s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp32s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          const Mpp32s aValue[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit signed integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_32s_C3R(const Mpp32s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                            Mpp32s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      const Mpp32s aValue[3]);

/**
 * 4 channel 32-bit signed integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_32s_C4R_Ctx(const Mpp32s * pSrc, int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp32s * pDst, int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          const Mpp32s aValue[4], MppStreamContext mppStreamCtx);
                                       
/**
 * 4 channel 32-bit signed integer image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_32s_C4R(const Mpp32s * pSrc, int nSrcStep, MppiSize oSrcSizeROI,
                                            Mpp32s * pDst, int nDstStep, MppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      const Mpp32s aValue[4]);
                                       
/**
 * 4 channel 32-bit signed integer image copy with constant border color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_32s_AC4R_Ctx(const Mpp32s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                 Mpp32s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                           int nTopBorderHeight, int nLeftBorderWidth,
                                           const Mpp32s aValue[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit signed integer image copy with constant border color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_32s_AC4R(const Mpp32s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                             Mpp32s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                       int nTopBorderHeight, int nLeftBorderWidth,
                                       const Mpp32s aValue[3]);

/** 
 * 1 channel 32-bit floating point image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_32f_C1R_Ctx(const Mpp32f * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp32f * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          Mpp32f nValue, MppStreamContext mppStreamCtx);

/** 
 * 1 channel 32-bit floating point image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_32f_C1R(const Mpp32f * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                            Mpp32f * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      Mpp32f nValue);

/**
 * 3 channel 32-bit floating point image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_32f_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp32f * pDst, int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          const Mpp32f aValue[3], MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit floating point image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_32f_C3R(const Mpp32f * pSrc, int nSrcStep, MppiSize oSrcSizeROI,
                                            Mpp32f * pDst, int nDstStep, MppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      const Mpp32f aValue[3]);

/**
 * 4 channel 32-bit floating point image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_32f_C4R_Ctx(const Mpp32f * pSrc, int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp32f * pDst, int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          const Mpp32f aValue[4], MppStreamContext mppStreamCtx);
                                       
/**
 * 4 channel 32-bit floating point image copy with constant border color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_32f_C4R(const Mpp32f * pSrc, int nSrcStep, MppiSize oSrcSizeROI,
                                            Mpp32f * pDst, int nDstStep, MppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      const Mpp32f aValue[4]);
                                       
/**
 * 4 channel 32-bit floating point image copy with constant border color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_32f_AC4R_Ctx(const Mpp32f * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                 Mpp32f * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                           int nTopBorderHeight, int nLeftBorderWidth,
                                           const Mpp32f aValue[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point image copy with constant border color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyConstantBorderParameters.
 *
 */
MppStatus mppiCopyConstBorder_32f_AC4R(const Mpp32f * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                             Mpp32f * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                       int nTopBorderHeight, int nLeftBorderWidth,
                                       const Mpp32f aValue[3]);

/** @} image_copy_constant_border */

/** 
 * \section image_copy_replicate_border Copy Replicate Border
 * @defgroup image_copy_replicate_border Copy Replicate Border
 * Methods for copying images and padding borders with a replicates of the nearest source image pixel color.
 *
 * \subsection CommonImageCopyReplicateBorderParameters Common parameters for mppiCopyReplicateBorder functions:
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSizeROI Size of the source region of pixels.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
 *      data from the source image (inner part) and nearest source image pixel color (outer part).
 * \param nTopBorderHeight Height (in pixels) of the top border. The number of pixel rows at the top of the
 *      destination ROI that will be filled with the nearest source image pixel color.
 *      nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.
 * \param nLeftBorderWidth Width (in pixels) of the left border. The width of the border at the right side of the
 *      destination ROI is implicitly defined by the size of the source ROI:
 *      nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_8u_C1R_Ctx(const Mpp8u * pSrc,   int nSrcStep, MppiSize oSrcSizeROI,
                                                   Mpp8u * pDst,   int nDstStep, MppiSize oDstSizeROI,
                                             int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/** 
 * 1 channel 8-bit unsigned integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_8u_C1R(const Mpp8u * pSrc,   int nSrcStep, MppiSize oSrcSizeROI,
                                               Mpp8u * pDst,   int nDstStep, MppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 3 channel 8-bit unsigned integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_8u_C3R_Ctx(const Mpp8u * pSrc,   int nSrcStep, MppiSize oSrcSizeROI,
                                                   Mpp8u * pDst,   int nDstStep, MppiSize oDstSizeROI,
                                             int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_8u_C3R(const Mpp8u * pSrc,   int nSrcStep, MppiSize oSrcSizeROI,
                                               Mpp8u * pDst,   int nDstStep, MppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 4 channel 8-bit unsigned integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_8u_C4R_Ctx(const Mpp8u * pSrc,   int nSrcStep, MppiSize oSrcSizeROI,
                                                   Mpp8u * pDst,   int nDstStep, MppiSize oDstSizeROI,
                                             int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);
                                       
/**
 * 4 channel 8-bit unsigned integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_8u_C4R(const Mpp8u * pSrc,   int nSrcStep, MppiSize oSrcSizeROI,
                                               Mpp8u * pDst,   int nDstStep, MppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth);
                                       
/**
 * 4 channel 8-bit unsigned integer image copy with nearest source image pixel color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_8u_AC4R_Ctx(const Mpp8u * pSrc,   int nSrcStep, MppiSize oSrcSizeROI,
                                                    Mpp8u * pDst,   int nDstStep, MppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned integer image copy with nearest source image pixel color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_8u_AC4R(const Mpp8u * pSrc,   int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp8u * pDst,   int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);

/** 
 * 1 channel 16-bit unsigned integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_16u_C1R_Ctx(const Mpp16u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                    Mpp16u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/** 
 * 1 channel 16-bit unsigned integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_16u_C1R(const Mpp16u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp16u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 3 channel 16-bit unsigned integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_16u_C3R_Ctx(const Mpp16u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                    Mpp16u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit unsigned integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_16u_C3R(const Mpp16u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp16u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 4 channel 16-bit unsigned integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_16u_C4R_Ctx(const Mpp16u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                    Mpp16u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);
                                       
/**
 * 4 channel 16-bit unsigned integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_16u_C4R(const Mpp16u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp16u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);
                                       
/**
 * 4 channel 16-bit unsigned image copy with nearest source image pixel color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_16u_AC4R_Ctx(const Mpp16u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                     Mpp16u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                               int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit unsigned image copy with nearest source image pixel color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_16u_AC4R(const Mpp16u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                 Mpp16u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                           int nTopBorderHeight, int nLeftBorderWidth);

/** 
 * 1 channel 16-bit signed integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_16s_C1R_Ctx(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                    Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/** 
 * 1 channel 16-bit signed integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_16s_C1R(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 3 channel 16-bit signed integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_16s_C3R_Ctx(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                    Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit signed integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_16s_C3R(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 4 channel 16-bit signed integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_16s_C4R_Ctx(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                    Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);
                                       
/**
 * 4 channel 16-bit signed integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_16s_C4R(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);
                                       
/**
 * 4 channel 16-bit signed integer image copy with nearest source image pixel color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_16s_AC4R_Ctx(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                     Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                               int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit signed integer image copy with nearest source image pixel color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_16s_AC4R(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                 Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                           int nTopBorderHeight, int nLeftBorderWidth);

/** 
 * 1 channel 32-bit signed integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_32s_C1R_Ctx(const Mpp32s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                    Mpp32s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/** 
 * 1 channel 32-bit signed integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_32s_C1R(const Mpp32s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp32s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 3 channel 32-bit signed image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_32s_C3R_Ctx(const Mpp32s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                    Mpp32s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit signed image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_32s_C3R(const Mpp32s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp32s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 4 channel 32-bit signed integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_32s_C4R_Ctx(const Mpp32s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                    Mpp32s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);
                                       
/**
 * 4 channel 32-bit signed integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_32s_C4R(const Mpp32s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp32s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);
                                       
/**
 * 4 channel 32-bit signed integer image copy with nearest source image pixel color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_32s_AC4R_Ctx(const Mpp32s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                     Mpp32s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                               int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit signed integer image copy with nearest source image pixel color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_32s_AC4R(const Mpp32s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                 Mpp32s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                           int nTopBorderHeight, int nLeftBorderWidth);

/** 
 * 1 channel 32-bit floating point image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_32f_C1R_Ctx(const Mpp32f * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                    Mpp32f * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/** 
 * 1 channel 32-bit floating point image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_32f_C1R(const Mpp32f * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp32f * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 3 channel 32-bit floating point image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_32f_C3R_Ctx(const Mpp32f * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                    Mpp32f * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit floating point image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_32f_C3R(const Mpp32f * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp32f * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 4 channel 32-bit floating point image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_32f_C4R_Ctx(const Mpp32f * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                    Mpp32f * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);
                                       
/**
 * 4 channel 32-bit floating point image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_32f_C4R(const Mpp32f * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp32f * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);
                                       
/**
 * 4 channel 32-bit floating point image copy with nearest source image pixel color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_32f_AC4R_Ctx(const Mpp32f * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                     Mpp32f * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                               int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point image copy with nearest source image pixel color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyReplicateBorderParameters.
 *
 */
MppStatus mppiCopyReplicateBorder_32f_AC4R(const Mpp32f * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                 Mpp32f * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                           int nTopBorderHeight, int nLeftBorderWidth);

/** @} image_copy_replicate_border */

/** 
 * \section image_copy_wrap_border Copy Wrap Border
 * @defgroup image_copy_wrap_border Copy Wrap Border
 * Methods for copying images and padding borders with wrapped replications of the source image pixel colors.
 *
 * \subsection CommonImageCopyWrapBorderParameters Common parameters for mppiCopyWrapBorder functions:
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSizeROI Size of the source region of pixels.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
 *      data from the source image (inner part) and a border consisting of wrapped replication of the source image pixel colors (outer part).
 * \param nTopBorderHeight Height (in pixels) of the top border. The number of pixel rows at the top of the
 *      destination ROI that will be filled with the wrapped replication of the corresponding column of source image pixels colors.
 *      nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.
 * \param nLeftBorderWidth Width (in pixels) of the left border. The width of the border at the right side of the
 *      destination ROI is implicitly defined by the size of the source ROI:
 *      nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_8u_C1R_Ctx(const Mpp8u * pSrc,   int nSrcStep, MppiSize oSrcSizeROI,
                                              Mpp8u * pDst,   int nDstStep, MppiSize oDstSizeROI,
                                        int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/** 
 * 1 channel 8-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_8u_C1R(const Mpp8u * pSrc,   int nSrcStep, MppiSize oSrcSizeROI,
                                          Mpp8u * pDst,   int nDstStep, MppiSize oDstSizeROI,
                                    int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 3 channel 8-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_8u_C3R_Ctx(const Mpp8u * pSrc,   int nSrcStep, MppiSize oSrcSizeROI,
                                              Mpp8u * pDst,   int nDstStep, MppiSize oDstSizeROI,
                                        int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_8u_C3R(const Mpp8u * pSrc,   int nSrcStep, MppiSize oSrcSizeROI,
                                          Mpp8u * pDst,   int nDstStep, MppiSize oDstSizeROI,
                                    int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 4 channel 8-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_8u_C4R_Ctx(const Mpp8u * pSrc,   int nSrcStep, MppiSize oSrcSizeROI,
                                              Mpp8u * pDst,   int nDstStep, MppiSize oDstSizeROI,
                                        int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);
                                       
/**
 * 4 channel 8-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_8u_C4R(const Mpp8u * pSrc,   int nSrcStep, MppiSize oSrcSizeROI,
                                          Mpp8u * pDst,   int nDstStep, MppiSize oDstSizeROI,
                                    int nTopBorderHeight, int nLeftBorderWidth);
                                       
/**
 * 4 channel 8-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_8u_AC4R_Ctx(const Mpp8u * pSrc,   int nSrcStep, MppiSize oSrcSizeROI,
                                               Mpp8u * pDst,   int nDstStep, MppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_8u_AC4R(const Mpp8u * pSrc,   int nSrcStep, MppiSize oSrcSizeROI,
                                           Mpp8u * pDst,   int nDstStep, MppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);

/** 
 * 1 channel 16-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_16u_C1R_Ctx(const Mpp16u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                               Mpp16u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/** 
 * 1 channel 16-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_16u_C1R(const Mpp16u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                           Mpp16u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 3 channel 16-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_16u_C3R_Ctx(const Mpp16u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                               Mpp16u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_16u_C3R(const Mpp16u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                           Mpp16u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 4 channel 16-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_16u_C4R_Ctx(const Mpp16u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                               Mpp16u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);
                                       
/**
 * 4 channel 16-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_16u_C4R(const Mpp16u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                           Mpp16u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);
                                       
/**
 * 4 channel 16-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_16u_AC4R_Ctx(const Mpp16u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp16u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_16u_AC4R(const Mpp16u * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                            Mpp16u * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth);

/** 
 * 1 channel 16-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_16s_C1R_Ctx(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                               Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/** 
 * 1 channel 16-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_16s_C1R(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                           Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 3 channel 16-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_16s_C3R_Ctx(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                               Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_16s_C3R(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                           Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 4 channel 16-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_16s_C4R_Ctx(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                               Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);
                                       
/**
 * 4 channel 16-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_16s_C4R(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                           Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);
                                       
/**
 * 4 channel 16-bit signed integer image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_16s_AC4R_Ctx(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit signed integer image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_16s_AC4R(const Mpp16s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                            Mpp16s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth);

/** 
 * 1 channel 32-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_32s_C1R_Ctx(const Mpp32s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                               Mpp32s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/** 
 * 1 channel 32-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_32s_C1R(const Mpp32s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                           Mpp32s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 3 channel 32-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_32s_C3R_Ctx(const Mpp32s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                               Mpp32s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_32s_C3R(const Mpp32s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                           Mpp32s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 4 channel 32-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_32s_C4R_Ctx(const Mpp32s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                               Mpp32s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);
                                       
/**
 * 4 channel 32-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_32s_C4R(const Mpp32s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                           Mpp32s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);
                                       
/**
 * 4 channel 32-bit signed integer image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_32s_AC4R_Ctx(const Mpp32s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp32s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit signed integer image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_32s_AC4R(const Mpp32s * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                            Mpp32s * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth);

/** 
 * 1 channel 32-bit floating point image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_32f_C1R_Ctx(const Mpp32f * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                               Mpp32f * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/** 
 * 1 channel 32-bit floating point image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_32f_C1R(const Mpp32f * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                           Mpp32f * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 3 channel 32-bit floating point image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_32f_C3R_Ctx(const Mpp32f * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                               Mpp32f * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit floating point image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_32f_C3R(const Mpp32f * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                           Mpp32f * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 4 channel 32-bit floating point image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_32f_C4R_Ctx(const Mpp32f * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                               Mpp32f * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);
                                       
/**
 * 4 channel 32-bit floating point image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_32f_C4R(const Mpp32f * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                           Mpp32f * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);
                                       
/**
 * 1 channel 32-bit floating point image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_32f_AC4R_Ctx(const Mpp32f * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                                Mpp32f * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth, MppStreamContext mppStreamCtx);

/**
 * 1 channel 32-bit floating point image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopyWrapBorderParameters.
 *
 */
MppStatus mppiCopyWrapBorder_32f_AC4R(const Mpp32f * pSrc,  int nSrcStep, MppiSize oSrcSizeROI,
                                            Mpp32f * pDst,  int nDstStep, MppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth);

/** @} image_copy_wrap_border */

/** 
 * \section image_copy_sub_pixel Copy Sub-Pixel
 * @defgroup image_copy_sub_pixel Copy Sub-Pixel
 * Functions for copying linearly interpolated images using source image subpixel coordinates.
 *
 * \subsection CommonImageCopySubPixelParameters Common parameters for mppiCopySubPix functions:
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
 *      data from the source image, source image ROI is assumed to be same as destination image ROI.
 * \param nDx Fractional part of source image X coordinate.
 * \param nDy Fractional part of source image Y coordinate.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_8u_C1R_Ctx(const Mpp8u * pSrc, int nSrcStep, 
                                          Mpp8u * pDst, int nDstStep, MppiSize oDstSizeROI, 
                                    Mpp32f nDx, Mpp32f nDy, MppStreamContext mppStreamCtx);

/** 
 * 1 channel 8-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_8u_C1R(const Mpp8u * pSrc, int nSrcStep, 
                                      Mpp8u * pDst, int nDstStep, MppiSize oDstSizeROI, 
                                Mpp32f nDx, Mpp32f nDy);

/**
 * 3 channel 8-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, 
                                          Mpp8u * pDst, int nDstStep, MppiSize oDstSizeROI, 
                                    Mpp32f nDx, Mpp32f nDy, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_8u_C3R(const Mpp8u * pSrc, int nSrcStep, 
                                      Mpp8u * pDst, int nDstStep, MppiSize oDstSizeROI, 
                                Mpp32f nDx, Mpp32f nDy);

/**
 * 4 channel 8-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_8u_C4R_Ctx(const Mpp8u * pSrc, int nSrcStep, 
                                          Mpp8u * pDst, int nDstStep, MppiSize oDstSizeROI, 
                                    Mpp32f nDx, Mpp32f nDy, MppStreamContext mppStreamCtx);
                                       
/**
 * 4 channel 8-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_8u_C4R(const Mpp8u * pSrc, int nSrcStep, 
                                      Mpp8u * pDst, int nDstStep, MppiSize oDstSizeROI, 
                                Mpp32f nDx, Mpp32f nDy);
                                       
/**
 * 4 channel 8-bit unsigned integer linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, 
                                           Mpp8u * pDst, int nDstStep, MppiSize oDstSizeROI, 
                                     Mpp32f nDx, Mpp32f nDy, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned integer linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, 
                                       Mpp8u * pDst, int nDstStep, MppiSize oDstSizeROI, 
                                 Mpp32f nDx, Mpp32f nDy);

/** 
 * 1 channel 16-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_16u_C1R_Ctx(const Mpp16u * pSrc, int nSrcStep, 
                                           Mpp16u * pDst, int nDstStep, MppiSize oDstSizeROI, 
                                     Mpp32f nDx, Mpp32f nDy, MppStreamContext mppStreamCtx);

/** 
 * 1 channel 16-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_16u_C1R(const Mpp16u * pSrc, int nSrcStep, 
                                       Mpp16u * pDst, int nDstStep, MppiSize oDstSizeROI, 
                                 Mpp32f nDx, Mpp32f nDy);

/**
 * 3 channel 16-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_16u_C3R_Ctx(const Mpp16u * pSrc, int nSrcStep, 
                                           Mpp16u * pDst, int nDstStep, MppiSize oDstSizeROI, 
                                     Mpp32f nDx, Mpp32f nDy, MppStreamContext mppStreamCtx);
 
/**
 * 3 channel 16-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_16u_C3R(const Mpp16u * pSrc, int nSrcStep, 
                                       Mpp16u * pDst, int nDstStep, MppiSize oDstSizeROI, 
                                 Mpp32f nDx, Mpp32f nDy);

/**
 * 4 channel 16-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_16u_C4R_Ctx(const Mpp16u * pSrc, int nSrcStep, 
                                           Mpp16u * pDst, int nDstStep, MppiSize oDstSizeROI, 
                                     Mpp32f nDx, Mpp32f nDy, MppStreamContext mppStreamCtx);
                                       
/**
 * 4 channel 16-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_16u_C4R(const Mpp16u * pSrc, int nSrcStep, 
                                       Mpp16u * pDst, int nDstStep, MppiSize oDstSizeROI, 
                                 Mpp32f nDx, Mpp32f nDy);
                                       
/**
 * 4 channel 16-bit unsigned linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_16u_AC4R_Ctx(const Mpp16u * pSrc, int nSrcStep, 
                                            Mpp16u * pDst, int nDstStep, MppiSize oDstSizeROI, 
                                      Mpp32f nDx, Mpp32f nDy, MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit unsigned linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_16u_AC4R(const Mpp16u * pSrc, int nSrcStep, 
                                        Mpp16u * pDst, int nDstStep, MppiSize oDstSizeROI, 
                                  Mpp32f nDx, Mpp32f nDy);

/** 
 * 1 channel 16-bit signed integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_16s_C1R_Ctx(const Mpp16s * pSrc, int nSrcStep, 
                                           Mpp16s * pDst, int nDstStep, MppiSize oDstSizeROI, 
                                     Mpp32f nDx, Mpp32f nDy, MppStreamContext mppStreamCtx);

/** 
 * 1 channel 16-bit signed integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_16s_C1R(const Mpp16s * pSrc, int nSrcStep, 
                                       Mpp16s * pDst, int nDstStep, MppiSize oDstSizeROI, 
                                 Mpp32f nDx, Mpp32f nDy);

/**
 * 3 channel 16-bit signed integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_16s_C3R_Ctx(const Mpp16s * pSrc, int nSrcStep, 
                                           Mpp16s * pDst, int nDstStep, MppiSize oDstSizeROI, 
                                     Mpp32f nDx, Mpp32f nDy, MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit signed integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_16s_C3R(const Mpp16s * pSrc, int nSrcStep, 
                                       Mpp16s * pDst, int nDstStep, MppiSize oDstSizeROI, 
                                 Mpp32f nDx, Mpp32f nDy);

/**
 * 4 channel 16-bit signed integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_16s_C4R_Ctx(const Mpp16s * pSrc, int nSrcStep, 
                                           Mpp16s * pDst, int nDstStep, MppiSize oDstSizeROI, 
                                     Mpp32f nDx, Mpp32f nDy, MppStreamContext mppStreamCtx);
                                       
/**
 * 4 channel 16-bit signed integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_16s_C4R(const Mpp16s * pSrc, int nSrcStep, 
                                       Mpp16s * pDst, int nDstStep, MppiSize oDstSizeROI, 
                                 Mpp32f nDx, Mpp32f nDy);
                                       
/**
 * 4 channel 16-bit signed integer linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_16s_AC4R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                            Mpp16s * pDst, int nDstStep, MppiSize oDstSizeROI,
                                      Mpp32f nDx, Mpp32f nDy, MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit signed integer linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_16s_AC4R(const Mpp16s * pSrc, int nSrcStep,
                                        Mpp16s * pDst, int nDstStep, MppiSize oDstSizeROI,
                                  Mpp32f nDx, Mpp32f nDy);

/** 
 * 1 channel 32-bit signed integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_32s_C1R_Ctx(const Mpp32s * pSrc, int nSrcStep,
                                           Mpp32s * pDst, int nDstStep, MppiSize oDstSizeROI,
                                     Mpp32f nDx, Mpp32f nDy, MppStreamContext mppStreamCtx);

/** 
 * 1 channel 32-bit signed integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_32s_C1R(const Mpp32s * pSrc, int nSrcStep,
                                       Mpp32s * pDst, int nDstStep, MppiSize oDstSizeROI,
                                 Mpp32f nDx, Mpp32f nDy);

/**
 * 3 channel 32-bit signed linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_32s_C3R_Ctx(const Mpp32s * pSrc, int nSrcStep,
                                           Mpp32s * pDst, int nDstStep, MppiSize oDstSizeROI,
                                     Mpp32f nDx, Mpp32f nDy, MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit signed linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_32s_C3R(const Mpp32s * pSrc, int nSrcStep,
                                       Mpp32s * pDst, int nDstStep, MppiSize oDstSizeROI,
                                 Mpp32f nDx, Mpp32f nDy);

/**
 * 4 channel 32-bit signed integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_32s_C4R_Ctx(const Mpp32s * pSrc, int nSrcStep,
                                           Mpp32s * pDst, int nDstStep, MppiSize oDstSizeROI,
                                     Mpp32f nDx, Mpp32f nDy, MppStreamContext mppStreamCtx);
                                       
/**
 * 4 channel 32-bit signed integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_32s_C4R(const Mpp32s * pSrc, int nSrcStep,
                                       Mpp32s * pDst, int nDstStep, MppiSize oDstSizeROI,
                                 Mpp32f nDx, Mpp32f nDy);
                                       
/**
 * 4 channel 32-bit signed integer linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_32s_AC4R_Ctx(const Mpp32s * pSrc, int nSrcStep,
                                            Mpp32s * pDst, int nDstStep, MppiSize oDstSizeROI,
                                      Mpp32f nDx, Mpp32f nDy, MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit signed integer linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_32s_AC4R(const Mpp32s * pSrc, int nSrcStep,
                                        Mpp32s * pDst, int nDstStep, MppiSize oDstSizeROI,
                                  Mpp32f nDx, Mpp32f nDy);

/** 
 * 1 channel 32-bit floating point linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_32f_C1R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                           Mpp32f * pDst, int nDstStep, MppiSize oDstSizeROI,
                                     Mpp32f nDx, Mpp32f nDy, MppStreamContext mppStreamCtx);

/** 
 * 1 channel 32-bit floating point linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_32f_C1R(const Mpp32f * pSrc, int nSrcStep,
                                       Mpp32f * pDst, int nDstStep, MppiSize oDstSizeROI,
                                 Mpp32f nDx, Mpp32f nDy);

/**
 * 3 channel 32-bit floating point linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_32f_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                           Mpp32f * pDst, int nDstStep, MppiSize oDstSizeROI,
                                     Mpp32f nDx, Mpp32f nDy, MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit floating point linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_32f_C3R(const Mpp32f * pSrc, int nSrcStep,
                                       Mpp32f * pDst, int nDstStep, MppiSize oDstSizeROI,
                                 Mpp32f nDx, Mpp32f nDy);

/**
 * 4 channel 32-bit floating point linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_32f_C4R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                           Mpp32f * pDst, int nDstStep, MppiSize oDstSizeROI,
                                     Mpp32f nDx, Mpp32f nDy, MppStreamContext mppStreamCtx);
                                       
/**
 * 4 channel 32-bit floating point linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_32f_C4R(const Mpp32f * pSrc, int nSrcStep,
                                       Mpp32f * pDst, int nDstStep, MppiSize oDstSizeROI,
                                 Mpp32f nDx, Mpp32f nDy);
                                       
/**
 * 4 channel 32-bit floating point linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_32f_AC4R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                            Mpp32f * pDst, int nDstStep, MppiSize oDstSizeROI,
                                      Mpp32f nDx, Mpp32f nDy, MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected.
 * 
 * For common parameter descriptions, see \ref CommonImageCopySubPixelParameters.
 *
 */
MppStatus mppiCopySubpix_32f_AC4R(const Mpp32f * pSrc, int nSrcStep,
                                        Mpp32f * pDst, int nDstStep, MppiSize oDstSizeROI,
                                  Mpp32f nDx, Mpp32f nDy);

/** @} image_copy_sub_pixel */

/** @} image_copy_operations */

/** 
 * \section image_convert Convert Bit Depth
 * @defgroup image_convert Convert Bit Depth
 * Functions for converting bit depth without scaling.
 *
 * @{
 *
 */

/** 
 * \section image_convert_increase Convert To Increased Bit Depth
 * @defgroup image_convert_increase Convert To Increased Bit Depth
 * The integer conversion methods do not involve any scaling. Also, even when increasing the bit-depth
 * loss of information may occur:
 * - When converting integers (e.g. Mpp32u) to float (e.g. Mpp32f) integer value not accurately representable 
 *   by the float are rounded to the closest floating-point value.
 * - When converting signed integers to unsigned integers all negative values are lost (saturated to 0).
 *  
 * Note that all pointers and step sizes for images with 16f (Mpp16f) data types perform best when they are at least 16 byte aligned. 
 *  
 * \subsection CommonImageConvertToIncreasedBitDepthParameters Common parameters for mppiConvert to increased bit depth functions:
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * Single channel 8-bit unsigned to 16-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u16u_C1R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 8-bit unsigned to 16-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u16u_C1R(const Mpp8u  * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three channel 8-bit unsigned to 16-bit unsigned  conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u16u_C3R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 8-bit unsigned to 16-bit unsigned  conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u16u_C3R(const Mpp8u  * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 16-bit unsigned  conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u16u_C4R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 8-bit unsigned to 16-bit unsigned  conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u16u_C4R(const Mpp8u  * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 16-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u16u_AC4R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 8-bit unsigned to 16-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u16u_AC4R(const Mpp8u  * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 8-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u16s_C1R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 8-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u16s_C1R(const Mpp8u  * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three channel 8-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u16s_C3R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 8-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u16s_C3R(const Mpp8u  * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u16s_C4R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 8-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u16s_C4R(const Mpp8u  * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 16-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u16s_AC4R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 8-bit unsigned to 16-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u16s_AC4R(const Mpp8u  * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 8-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u32s_C1R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 8-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u32s_C1R(const Mpp8u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three channel 8-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u32s_C3R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 8-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u32s_C3R(const Mpp8u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u32s_C4R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 8-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u32s_C4R(const Mpp8u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 32-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u32s_AC4R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 8-bit unsigned to 32-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u32s_AC4R(const Mpp8u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 8-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u32f_C1R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 8-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u32f_C1R(const Mpp8u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three channel 8-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u32f_C3R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 8-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u32f_C3R(const Mpp8u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u32f_C4R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 8-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u32f_C4R(const Mpp8u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 32-bit floating-point conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u32f_AC4R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 8-bit unsigned to 32-bit floating-point conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u32f_AC4R(const Mpp8u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 8-bit signed to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s32s_C1R_Ctx(const Mpp8s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 8-bit signed to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s32s_C1R(const Mpp8s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three channel 8-bit signed to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s32s_C3R_Ctx(const Mpp8s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 8-bit signed to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s32s_C3R(const Mpp8s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 8-bit signed to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s32s_C4R_Ctx(const Mpp8s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 8-bit signed to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s32s_C4R(const Mpp8s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 8-bit signed to 32-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s32s_AC4R_Ctx(const Mpp8s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 8-bit signed to 32-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s32s_AC4R(const Mpp8s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 8-bit signed to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s32f_C1R_Ctx(const Mpp8s * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 8-bit signed to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s32f_C1R(const Mpp8s * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three channel 8-bit signed to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s32f_C3R_Ctx(const Mpp8s * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 8-bit signed to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s32f_C3R(const Mpp8s * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 8-bit signed to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s32f_C4R_Ctx(const Mpp8s * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 8-bit signed to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s32f_C4R(const Mpp8s * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 8-bit signed to 32-bit floating-point conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s32f_AC4R_Ctx(const Mpp8s * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 8-bit signed to 32-bit floating-point conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s32f_AC4R(const Mpp8s * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 16-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u32s_C1R_Ctx(const Mpp16u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 16-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u32s_C1R(const Mpp16u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three channel 16-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u32s_C3R_Ctx(const Mpp16u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 16-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u32s_C3R(const Mpp16u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 16-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u32s_C4R_Ctx(const Mpp16u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 16-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u32s_C4R(const Mpp16u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 16-bit unsigned to 32-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u32s_AC4R_Ctx(const Mpp16u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 16-bit unsigned to 32-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u32s_AC4R(const Mpp16u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 16-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u32f_C1R_Ctx(const Mpp16u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 16-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u32f_C1R(const Mpp16u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three channel 16-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u32f_C3R_Ctx(const Mpp16u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 16-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u32f_C3R(const Mpp16u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 16-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u32f_C4R_Ctx(const Mpp16u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 16-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u32f_C4R(const Mpp16u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 16-bit unsigned to 32-bit floating-point conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u32f_AC4R_Ctx(const Mpp16u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 16-bit unsigned to 32-bit floating-point conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u32f_AC4R(const Mpp16u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 16-bit signed to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s32s_C1R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 16-bit signed to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s32s_C1R(const Mpp16s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three channel 16-bit signed to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s32s_C3R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 16-bit signed to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s32s_C3R(const Mpp16s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 16-bit signed to 32-bit signed conversion.
 * 
 */
MppStatus 
mppiConvert_16s32s_C4R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 16-bit signed to 32-bit signed conversion.
 * 
 */
MppStatus 
mppiConvert_16s32s_C4R(const Mpp16s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 16-bit signed to 32-bit signed conversion, not affecting Alpha.
 * 
  * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
*/
MppStatus 
mppiConvert_16s32s_AC4R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 16-bit signed to 32-bit signed conversion.
 * 
 */
MppStatus 
mppiConvert_16s32s_AC4R(const Mpp16s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 16-bit signed to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s32f_C1R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 16-bit signed to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s32f_C1R(const Mpp16s * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three channel 16-bit signed to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s32f_C3R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 16-bit signed to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s32f_C3R(const Mpp16s * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 16-bit signed to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s32f_C4R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 16-bit signed to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s32f_C4R(const Mpp16s * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 16-bit signed to 32-bit floating-point conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s32f_AC4R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 16-bit signed to 32-bit floating-point conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s32f_AC4R(const Mpp16s * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 16-bit floating-point to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16f32f_C1R_Ctx(const Mpp16f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 16-bit floating-point to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16f32f_C1R(const Mpp16f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three channel 16-bit floating-point to 32-bit floating-point conversion. 
 *  
 * Note that all pointers and step sizes for images with 16f (Mpp16f) data types perform best when they are at least 16 byte aligned 
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16f32f_C3R_Ctx(const Mpp16f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 16-bit floating-point to 32-bit floating-point conversion. 
 *  
 * Note that all pointers and step sizes for images with 16f (Mpp16f) data types perform best when they are at least 16 byte aligned 
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16f32f_C3R(const Mpp16f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 16-bit floating-point to 32-bit floating-point conversion.
 *  
 * Note that all pointers and step sizes for images with 16f (Mpp16f) data types perform best when they are at least 16 byte aligned. 
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16f32f_C4R_Ctx(const Mpp16f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 16-bit floating-point to 32-bit floating-point conversion.
 *  
 * Note that all pointers and step sizes for images with 16f (Mpp16f) data types perform best when they are at least 16 byte aligned. 
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16f32f_C4R(const Mpp16f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 16-bit floating-point to 32-bit floating-point conversion, not affecting Alpha.
 *  
 * Note that all pointers and step sizes for images with 16f (Mpp16f) data types perform best when they are at least 16 byte aligned. 
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16f32f_AC4R_Ctx(const Mpp16f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 16-bit floating-point to 32-bit floating-point conversion, not affecting Alpha.
 *  
 * Note that all pointers and step sizes for images with 16f (Mpp16f) data types perform best when they are at least 16 byte aligned. 
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16f32f_AC4R(const Mpp16f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 8-bit signed to 8-bit unsigned conversion with saturation.
 *  
 * Note that all pointers and step sizes for images with 16f (Mpp16f) data types perform best when they are at least 16 byte aligned. 
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s8u_C1Rs_Ctx(const Mpp8s * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 8-bit signed to 8-bit unsigned conversion with saturation.
 *  
 * Note that all pointers and step sizes for images with 16f (Mpp16f) data types perform best when they are at least 16 byte aligned. 
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s8u_C1Rs(const Mpp8s * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 8-bit signed to 16-bit unsigned conversion with saturation.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s16u_C1Rs_Ctx(const Mpp8s * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 8-bit signed to 16-bit unsigned conversion with saturation.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s16u_C1Rs(const Mpp8s * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 8-bit signed to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s16s_C1R_Ctx(const Mpp8s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 8-bit signed to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s16s_C1R(const Mpp8s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 8-bit signed to 32-bit unsigned conversion with saturation.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s32u_C1Rs_Ctx(const Mpp8s * pSrc, int nSrcStep, Mpp32u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 8-bit signed to 32-bit unsigned conversion with saturation.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8s32u_C1Rs(const Mpp8s * pSrc, int nSrcStep, Mpp32u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 16-bit signed to 16-bit unsigned conversion with saturation.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s16u_C1Rs_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 16-bit signed to 16-bit unsigned conversion with saturation.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s16u_C1Rs(const Mpp16s * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 16-bit signed to 32-bit unsigned conversion with saturation.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s32u_C1Rs_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp32u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 16-bit signed to 32-bit unsigned conversion with saturation.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s32u_C1Rs(const Mpp16s * pSrc, int nSrcStep, Mpp32u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 16-bit unsigned to 32-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u32u_C1R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp32u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 16-bit unsigned to 32-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u32u_C1R(const Mpp16u * pSrc, int nSrcStep, Mpp32u * pDst, int nDstStep, MppiSize oSizeROI);


/** 
 * Single channel 32-bit signed to 32-bit unsigned conversion with saturation.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32s32u_C1Rs_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp32u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit signed to 32-bit unsigned conversion with saturation.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32s32u_C1Rs(const Mpp32s * pSrc, int nSrcStep, Mpp32u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 32-bit signed to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32s32f_C1R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit signed to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32s32f_C1R(const Mpp32s * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 32-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32u32f_C1R_Ctx(const Mpp32u * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToIncreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32u32f_C1R(const Mpp32u * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI);

/** @} image_convert_increase */

/** 
 * \section image_convert_decrease Convert To Decreased Bit Depth
 * @defgroup image_convert_decrease Convert To Decreased Bit Depth
 * The integer conversion methods do not involve any scaling. When converting floating-point values
 * to integers the user may choose the most appropriate rounding-mode. Typically information is lost when
 * converting to lower bit depth:
 * - All converted values are saturated to the destination type's range. E.g. any values larger than
 *   the largest value of the destination type are clamped to the destination's maximum.
 * - Converting floating-point values to integer also involves rounding, effectively loosing all
 *   fractional value information in the process.
 *  
 * Note that all pointers and step sizes for images with 16f (Mpp16f) data types perform best when they are at least 16 byte aligned. 
 *
 * \subsection CommonImageConvertToDecreasedBitDepthParameters Common parameters for mppiConvert to decreased bit depth functions:
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode \ref rounding_mode_parameter.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * Single channel 16-bit unsigned to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u8u_C1R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 16-bit unsigned to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u8u_C1R(const Mpp16u * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three channel 16-bit unsigned to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u8u_C3R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 16-bit unsigned to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u8u_C3R(const Mpp16u * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 16-bit unsigned to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u8u_C4R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 16-bit unsigned to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u8u_C4R(const Mpp16u * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 16-bit unsigned to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u8u_AC4R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);
          
/** 
 * Four channel 16-bit unsigned to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u8u_AC4R(const Mpp16u * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI);
          
/** 
 * Single channel 16-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s8u_C1R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);
          
/** 
 * Single channel 16-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s8u_C1R(const Mpp16s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI);
          
/** 
 * Three channel 16-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s8u_C3R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 16-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s8u_C3R(const Mpp16s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 16-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s8u_C4R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 16-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s8u_C4R(const Mpp16s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 16-bit signed to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus
mppiConvert_16s8u_AC4R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);
          
/** 
 * Four channel 16-bit signed to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus
mppiConvert_16s8u_AC4R(const Mpp16s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI);
          
/** 
 * Single channel 32-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32s8u_C1R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);
          
/** 
 * Single channel 32-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32s8u_C1R(const Mpp32s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI);
          
/** 
 * Three channel 32-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32s8u_C3R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 32-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32s8u_C3R(const Mpp32s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 32-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32s8u_C4R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 32-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32s8u_C4R(const Mpp32s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 32-bit signed to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus
mppiConvert_32s8u_AC4R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);
          
/** 
 * Four channel 32-bit signed to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus
mppiConvert_32s8u_AC4R(const Mpp32s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI);
          
/** 
 * Single channel 32-bit signed to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32s8s_C1R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);
          
/** 
 * Single channel 32-bit signed to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32s8s_C1R(const Mpp32s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI);
          
/** 
 * Three channel 32-bit signed to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32s8s_C3R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 32-bit signed to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32s8s_C3R(const Mpp32s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 32-bit signed to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32s8s_C4R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 32-bit signed to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32s8s_C4R(const Mpp32s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 32-bit signed to 8-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus
mppiConvert_32s8s_AC4R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);
          
/** 
 * Four channel 32-bit signed to 8-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus
mppiConvert_32s8s_AC4R(const Mpp32s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI);
          
/** 
 * Single channel 8-bit unsigned to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u8s_C1RSfs_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);

/** 
 * Single channel 8-bit unsigned to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_8u8s_C1RSfs(const Mpp8u * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 16-bit unsigned to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u8s_C1RSfs_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);

/** 
 * Single channel 16-bit unsigned to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u8s_C1RSfs(const Mpp16u * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 16-bit signed to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s8s_C1RSfs_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);

/** 
 * Single channel 16-bit signed to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16s8s_C1RSfs(const Mpp16s * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 16-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u16s_C1RSfs_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);

/** 
 * Single channel 16-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_16u16s_C1RSfs(const Mpp16u * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit unsigned to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32u8u_C1RSfs_Ctx(const Mpp32u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit unsigned to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32u8u_C1RSfs(const Mpp32u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit unsigned to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32u8s_C1RSfs_Ctx(const Mpp32u * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit unsigned to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32u8s_C1RSfs(const Mpp32u * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit unsigned to 16-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32u16u_C1RSfs_Ctx(const Mpp32u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit unsigned to 16-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32u16u_C1RSfs(const Mpp32u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32u16s_C1RSfs_Ctx(const Mpp32u * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32u16s_C1RSfs(const Mpp32u * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32u32s_C1RSfs_Ctx(const Mpp32u * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32u32s_C1RSfs(const Mpp32u * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit unsigned to 16-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32s16u_C1RSfs_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit unsigned to 16-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32s16u_C1RSfs(const Mpp32s * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32s16s_C1RSfs_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32s16s_C1RSfs(const Mpp32s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit floating point to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f8u_C1R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit floating point to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f8u_C1R(const Mpp32f * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode);

/** 
 * Three channel 32-bit floating point to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f8u_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, MppStreamContext mppStreamCtx);

/** 
 * Three channel 32-bit floating point to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f8u_C3R(const Mpp32f * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f8u_C4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, MppStreamContext mppStreamCtx);

/** 
 * Four channel 32-bit floating point to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f8u_C4R(const Mpp32f * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f8u_AC4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, MppStreamContext mppStreamCtx);

/** 
 * Four channel 32-bit floating point to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f8u_AC4R(const Mpp32f * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode);

/** 
 * Single channel 32-bit floating point to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f8s_C1R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit floating point to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f8s_C1R(const Mpp32f * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode);

/** 
 * Three channel 32-bit floating point to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f8s_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, MppStreamContext mppStreamCtx);

/** 
 * Three channel 32-bit floating point to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f8s_C3R(const Mpp32f * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f8s_C4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, MppStreamContext mppStreamCtx);

/** 
 * Four channel 32-bit floating point to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f8s_C4R(const Mpp32f * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 8-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f8s_AC4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, MppStreamContext mppStreamCtx);

/** 
 * Four channel 32-bit floating point to 8-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f8s_AC4R(const Mpp32f * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode);

/** 
 * Single channel 32-bit floating point to 16-bit unsigned conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16u_C1R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit floating point to 16-bit unsigned conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16u_C1R(const Mpp32f * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode);

/** 
 * Three channel 32-bit floating point to 16-bit unsigned conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16u_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, MppStreamContext mppStreamCtx);

/** 
 * Three channel 32-bit floating point to 16-bit unsigned conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16u_C3R(const Mpp32f * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 16-bit unsigned conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16u_C4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, MppStreamContext mppStreamCtx);

/** 
 * Four channel 32-bit floating point to 16-bit unsigned conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16u_C4R(const Mpp32f * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 16-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16u_AC4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, MppStreamContext mppStreamCtx);

/** 
 * Four channel 32-bit floating point to 16-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16u_AC4R(const Mpp32f * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode);

/** 
 * Single channel 32-bit floating point to 16-bit signed conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16s_C1R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit floating point to 16-bit signed conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16s_C1R(const Mpp32f * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode);

/** 
 * Three channel 32-bit floating point to 16-bit signed conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16s_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, MppStreamContext mppStreamCtx);

/** 
 * Three channel 32-bit floating point to 16-bit signed conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16s_C3R(const Mpp32f * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 16-bit signed conversion.
 *
  * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
*/
MppStatus 
mppiConvert_32f16s_C4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, MppStreamContext mppStreamCtx);

/** 
 * Three channel 32-bit floating point to 16-bit signed conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16s_C4R(const Mpp32f * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 16-bit floating-point conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16s_AC4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, MppStreamContext mppStreamCtx);

/** 
 * Four channel 32-bit floating point to 16-bit floating-point conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16s_AC4R(const Mpp32f * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode);

/** 
 * Single channel 32-bit floating point to 16-bit floating-point conversion.
 *  
 * Note that all pointers and step sizes for images with 16f (Mpp16f) data types perform best when they are at least 16 byte aligned. 
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16f_C1R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit floating point to 16-bit floating-point conversion.
 *  
 * Note that all pointers and step sizes for images with 16f (Mpp16f) data types perform best when they are at least 16 byte aligned. 
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16f_C1R(const Mpp32f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode);

/** 
 * Three channel 32-bit floating point to 16-bit floating-point conversion.
 *  
 * Note that all pointers and step sizes for images with 16f (Mpp16f) data types perform best when they are at least 16 byte aligned. 
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16f_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, MppStreamContext mppStreamCtx);

/** 
 * Three channel 32-bit floating point to 16-bit floating-point conversion.
 *  
 * Note that all pointers and step sizes for images with 16f (Mpp16f) data types perform best when they are at least 16 byte aligned. 
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16f_C3R(const Mpp32f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 16-bit floating-point conversion.
 *  
 * Note that all pointers and step sizes for images with 16f (Mpp16f) data types perform best when they are at least 16 byte aligned. 
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
*/
MppStatus 
mppiConvert_32f16f_C4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, MppStreamContext mppStreamCtx);

/** 
 * Three channel 32-bit floating point to 16-bit floating-point conversion.
 *  
 * Note that all pointers and step sizes for images with 16f (Mpp16f) data types perform best when they are at least 16 byte aligned. 
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16f_C4R(const Mpp32f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 16-bit floating-point conversion.
 *  
 * Note that all pointers and step sizes for images with 16f (Mpp16f) data types perform best when they are at least 16 byte aligned. 
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16f_AC4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, MppStreamContext mppStreamCtx);

/** 
 * Four channel 32-bit floating point to 16-bit floating-point conversion.
 *  
 * Note that all pointers and step sizes for images with 16f (Mpp16f) data types perform best when they are at least 16 byte aligned. 
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16f_AC4R(const Mpp32f * pSrc, int nSrcStep, Mpp16f * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode);

/** 
 * Single channel 32-bit floating point to 8-bit unsigned conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f8u_C1RSfs_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit floating point to 8-bit unsigned conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f8u_C1RSfs(const Mpp32f * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit floating point to 8-bit signed conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f8s_C1RSfs_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit floating point to 8-bit signed conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f8s_C1RSfs(const Mpp32f * pSrc, int nSrcStep, Mpp8s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit floating point to 16-bit unsigned conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16u_C1RSfs_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit floating point to 16-bit unsigned conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16u_C1RSfs(const Mpp32f * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit floating point to 16-bit signed conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16s_C1RSfs_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit floating point to 16-bit signed conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f16s_C1RSfs(const Mpp32f * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit floating point to 32-bit unsigned conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f32u_C1RSfs_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit floating point to 32-bit unsigned conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f32u_C1RSfs(const Mpp32f * pSrc, int nSrcStep, Mpp32u * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit floating point to 32-bit signed conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f32s_C1RSfs_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit floating point to 32-bit signed conversion.
 *
 * For common parameter descriptions, see \ref CommonImageConvertToDecreasedBitDepthParameters.
 *
 */
MppStatus 
mppiConvert_32f32s_C1RSfs(const Mpp32f * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppRoundMode eRoundMode, int nScaleFactor);

/** @} image_convert_decrease */

/** @} image_convert */

/** 
 * \section image_scale Scale Bit Depth
 * @defgroup image_scale Scale
 * Functions for scaling bit depth up or down.
 *
 * @{
 *
 */

/** 
 * \section image_scale_to_higher_bit_depth Scale To Higher Bit Depth
 * @defgroup image_scale_to_higher_bit_depth Scale To Higher Bit Depth
 * Functions for scaling images to higher bit depth.
 *
 * To map source pixel srcPixelValue to destination pixel dstPixelValue the following equation is used:
 * 
 *      dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)
 *
 * where scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).
 *
 * For conversions between integer data types, the entire integer numeric range of the input data type is mapped onto 
 * the entire integer numeric range of the output data type.
 *
 * For conversions to floating point data types the floating point data range is defined by the user supplied floating point values
 * of nMax and nMin which are used as the dstMaxRangeValue and dstMinRangeValue respectively in the scaleFactor and dstPixelValue 
 * calculations and also as the saturation values to which output data is clamped.
 *
 * When converting from floating-point values to integer values, nMax and nMin are used as the srcMaxRangeValue and srcMinRangeValue
 * respectively in the scaleFactor and dstPixelValue calculations. Output values are saturated and clamped to the full output integer
 * pixel value range.
 *
 * \subsection CommonImageScaleToHigherBitDepthParameters Common parameters for mppiScale to higher bit depth functions:
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nMin specifies the minimum saturation value to which every output value will be clamped.
 * \param nMax specifies the maximum saturation value to which every output value will be clamped.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, MPP_SCALE_RANGE_ERROR indicates an error condition if nMax <= nMin.
 *
 * @{
 *
 */

/** 
 * Single channel 8-bit unsigned to 16-bit unsigned conversion.
 *
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u16u_C1R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 8-bit unsigned to 16-bit unsigned conversion.
 *
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u16u_C1R(const Mpp8u  * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three channel 8-bit unsigned to 16-bit unsigned  conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u16u_C3R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 8-bit unsigned to 16-bit unsigned  conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u16u_C3R(const Mpp8u  * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 16-bit unsigned  conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u16u_C4R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 8-bit unsigned to 16-bit unsigned  conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u16u_C4R(const Mpp8u  * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 16-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u16u_AC4R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 8-bit unsigned to 16-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u16u_AC4R(const Mpp8u  * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 8-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u16s_C1R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 8-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u16s_C1R(const Mpp8u  * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three channel 8-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u16s_C3R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 8-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u16s_C3R(const Mpp8u  * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u16s_C4R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 8-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u16s_C4R(const Mpp8u  * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 16-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u16s_AC4R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 8-bit unsigned to 16-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u16s_AC4R(const Mpp8u  * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 8-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u32s_C1R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Single channel 8-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u32s_C1R(const Mpp8u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Three channel 8-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u32s_C3R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Three channel 8-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u32s_C3R(const Mpp8u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u32s_C4R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 8-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u32s_C4R(const Mpp8u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 32-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u32s_AC4R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, MppStreamContext mppStreamCtx);

/** 
 * Four channel 8-bit unsigned to 32-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u32s_AC4R(const Mpp8u  * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSizeROI);

/** 
 * Single channel 8-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u32f_C1R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, Mpp32f nMin, Mpp32f nMax, MppStreamContext mppStreamCtx);

/** 
 * Single channel 8-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u32f_C1R(const Mpp8u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, Mpp32f nMin, Mpp32f nMax);

/** 
 * Three channel 8-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u32f_C3R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, Mpp32f nMin, Mpp32f nMax, MppStreamContext mppStreamCtx);

/** 
 * Three channel 8-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u32f_C3R(const Mpp8u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, Mpp32f nMin, Mpp32f nMax);

/** 
 * Four channel 8-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u32f_C4R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, Mpp32f nMin, Mpp32f nMax, MppStreamContext mppStreamCtx);

/** 
 * Four channel 8-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u32f_C4R(const Mpp8u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, Mpp32f nMin, Mpp32f nMax);

/** 
 * Four channel 8-bit unsigned to 32-bit floating-point conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u32f_AC4R_Ctx(const Mpp8u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, Mpp32f nMin, Mpp32f nMax, MppStreamContext mppStreamCtx);

/** 
 * Four channel 8-bit unsigned to 32-bit floating-point conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToHigherBitDepthParameters.
 *
 */
MppStatus 
mppiScale_8u32f_AC4R(const Mpp8u  * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, Mpp32f nMin, Mpp32f nMax);

/** @} image_scale_to_higher_bit_depth */

/** 
 * \section image_scale_to_lower_bit_depth Scale To Lower Bit Depth
 * @defgroup image_scale_to_lower_bit_depth Scale To Lower Bit Depth
 * Functions for scaling images to lower bit depth.
 *
 * To map source pixel srcPixelValue to destination pixel dstPixelValue the following equation is used:
 * 
 *      dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)
 *
 * where scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).
 *
 * For conversions between integer data types, the entire integer numeric range of the input data type is mapped onto 
 * the entire integer numeric range of the output data type.
 *
 * For conversions to floating point data types the floating point data range is defined by the user supplied floating point values
 * of nMax and nMin which are used as the dstMaxRangeValue and dstMinRangeValue respectively in the scaleFactor and dstPixelValue 
 * calculations and also as the saturation values to which output data is clamped.
 *
 * When converting from floating-point values to integer values, nMax and nMin are used as the srcMaxRangeValue and srcMinRangeValue
 * respectively in the scaleFactor and dstPixelValue calculations. Output values are saturated and clamped to the full output integer
 * pixel value range.
 *
 * \subsection CommonImageScaleToLowerBitDepthParameters Common parameters for mppiScale to lower bit depth functions:
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param hint algorithm performance or accuracy selector, currently ignored
 * \param nMin specifies the minimum saturation value to which every output value will be clamped.
 * \param nMax specifies the maximum saturation value to which every output value will be clamped.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, MPP_SCALE_RANGE_ERROR indicates an error condition if nMax <= nMin.
 *
 * @{
 *
 */

/** 
 * Single channel 16-bit unsigned to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_16u8u_C1R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint, MppStreamContext mppStreamCtx);

/** 
 * Single channel 16-bit unsigned to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_16u8u_C1R(const Mpp16u * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint);

/** 
 * Three channel 16-bit unsigned to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_16u8u_C3R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint, MppStreamContext mppStreamCtx);

/** 
 * Three channel 16-bit unsigned to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_16u8u_C3R(const Mpp16u * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint);

/** 
 * Four channel 16-bit unsigned to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_16u8u_C4R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint, MppStreamContext mppStreamCtx);

/** 
 * Four channel 16-bit unsigned to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_16u8u_C4R(const Mpp16u * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint);

/** 
 * Four channel 16-bit unsigned to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_16u8u_AC4R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint, MppStreamContext mppStreamCtx);
          
/** 
 * Four channel 16-bit unsigned to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_16u8u_AC4R(const Mpp16u * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint);
          
/** 
 * Single channel 16-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_16s8u_C1R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint, MppStreamContext mppStreamCtx);
          
/** 
 * Single channel 16-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_16s8u_C1R(const Mpp16s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint);
          
/** 
 * Three channel 16-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_16s8u_C3R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint, MppStreamContext mppStreamCtx);

/** 
 * Three channel 16-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_16s8u_C3R(const Mpp16s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint);

/** 
 * Four channel 16-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_16s8u_C4R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint, MppStreamContext mppStreamCtx);

/** 
 * Four channel 16-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_16s8u_C4R(const Mpp16s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint);

/** 
 * Four channel 16-bit signed to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus
mppiScale_16s8u_AC4R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint, MppStreamContext mppStreamCtx);
          
/** 
 * Four channel 16-bit signed to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus
mppiScale_16s8u_AC4R(const Mpp16s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint);
          
          
/** 
 * Single channel 32-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_32s8u_C1R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint, MppStreamContext mppStreamCtx);
          
/** 
 * Single channel 32-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_32s8u_C1R(const Mpp32s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint);
          
/** 
 * Three channel 32-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_32s8u_C3R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint, MppStreamContext mppStreamCtx);

/** 
 * Three channel 32-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_32s8u_C3R(const Mpp32s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint);

/** 
 * Four channel 32-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_32s8u_C4R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint, MppStreamContext mppStreamCtx);

/** 
 * Four channel 32-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_32s8u_C4R(const Mpp32s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint);

/** 
 * Four channel 32-bit signed to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus
mppiScale_32s8u_AC4R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint, MppStreamContext mppStreamCtx);
          
/** 
 * Four channel 32-bit signed to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus
mppiScale_32s8u_AC4R(const Mpp32s * pSrc, int nSrcStep, Mpp8u  * pDst, int nDstStep, MppiSize oSizeROI, MppHintAlgorithm hint);
          
/** 
 * Single channel 32-bit floating point to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_32f8u_C1R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp32f nMin, Mpp32f nMax, MppStreamContext mppStreamCtx);

/** 
 * Single channel 32-bit floating point to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_32f8u_C1R(const Mpp32f * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp32f nMin, Mpp32f nMax);

/** 
 * Three channel 32-bit floating point to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_32f8u_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp32f nMin, Mpp32f nMax, MppStreamContext mppStreamCtx);

/** 
 * Three channel 32-bit floating point to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_32f8u_C3R(const Mpp32f * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp32f nMin, Mpp32f nMax);

/** 
 * Four channel 32-bit floating point to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_32f8u_C4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp32f nMin, Mpp32f nMax, MppStreamContext mppStreamCtx);

/** 
 * Four channel 32-bit floating point to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_32f8u_C4R(const Mpp32f * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp32f nMin, Mpp32f nMax);

/** 
 * Four channel 32-bit floating point to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_32f8u_AC4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp32f nMin, Mpp32f nMax, MppStreamContext mppStreamCtx);

/** 
 * Four channel 32-bit floating point to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see \ref CommonImageScaleToLowerBitDepthParameters.
 *
 */
MppStatus 
mppiScale_32f8u_AC4R(const Mpp32f * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, Mpp32f nMin, Mpp32f nMax);

/** @} image_scale_to_lower_bit_depth */

/** @} image_scale */

/** 
 * \section image_duplicate_channel Duplicate Channel
 * @defgroup image_duplicate_channel Duplicate Channel
 * Functions for duplicating a single channel image in a multiple channel image.
 *
 * \subsection CommonImageDuplicateChannelParameters Common parameters for mppiDup functions:
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
 *      data from the source image, source image ROI is assumed to be same as destination image ROI.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned integer source image duplicated in all 3 channels of destination image.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_8u_C1C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, 
                                     Mpp8u * pDst, int nDstStep, MppiSize oDstSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 1 channel 8-bit unsigned integer source image duplicated in all 3 channels of destination image.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_8u_C1C3R(const Mpp8u * pSrc, int nSrcStep, 
                                 Mpp8u * pDst, int nDstStep, MppiSize oDstSizeROI);

/**
 * 1 channel 8-bit unsigned integer source image duplicated in all 4 channels of destination image.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_8u_C1C4R_Ctx(const Mpp8u * pSrc, int nSrcStep, 
                                     Mpp8u * pDst, int nDstStep, MppiSize oDstSizeROI, MppStreamContext mppStreamCtx); 
                                       
/**
 * 1 channel 8-bit unsigned integer source image duplicated in all 4 channels of destination image.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_8u_C1C4R(const Mpp8u * pSrc, int nSrcStep, 
                                 Mpp8u * pDst, int nDstStep, MppiSize oDstSizeROI); 
                                       
/**
 * 1 channel 8-bit unsigned integer source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_8u_C1AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, 
                                      Mpp8u * pDst, int nDstStep, MppiSize oDstSizeROI, MppStreamContext mppStreamCtx); 

/**
 * 1 channel 8-bit unsigned integer source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_8u_C1AC4R(const Mpp8u * pSrc, int nSrcStep, 
                                  Mpp8u * pDst, int nDstStep, MppiSize oDstSizeROI); 

/** 
 * 1 channel 16-bit unsigned integer source image duplicated in all 3 channels of destination image.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_16u_C1C3R_Ctx(const Mpp16u * pSrc, int nSrcStep, 
                                      Mpp16u * pDst, int nDstStep, MppiSize oDstSizeROI, MppStreamContext mppStreamCtx); 

/** 
 * 1 channel 16-bit unsigned integer source image duplicated in all 3 channels of destination image.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_16u_C1C3R(const Mpp16u * pSrc, int nSrcStep, 
                                  Mpp16u * pDst, int nDstStep, MppiSize oDstSizeROI); 

/**
 * 1 channel 16-bit unsigned integer source image duplicated in all 4 channels of destination image.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_16u_C1C4R_Ctx(const Mpp16u * pSrc, int nSrcStep, 
                                      Mpp16u * pDst, int nDstStep, MppiSize oDstSizeROI, MppStreamContext mppStreamCtx); 
                                       
/**
 * 1 channel 16-bit unsigned integer source image duplicated in all 4 channels of destination image.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_16u_C1C4R(const Mpp16u * pSrc, int nSrcStep, 
                                  Mpp16u * pDst, int nDstStep, MppiSize oDstSizeROI); 
                                       
/**
 * 1 channel 16-bit unsigned integer source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_16u_C1AC4R_Ctx(const Mpp16u * pSrc, int nSrcStep, 
                                       Mpp16u * pDst, int nDstStep, MppiSize oDstSizeROI, MppStreamContext mppStreamCtx); 

/**
 * 1 channel 16-bit unsigned integer source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_16u_C1AC4R(const Mpp16u * pSrc, int nSrcStep, 
                                   Mpp16u * pDst, int nDstStep, MppiSize oDstSizeROI); 

/** 
 * 1 channel 16-bit signed integer source image duplicated in all 3 channels of destination image.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_16s_C1C3R_Ctx(const Mpp16s * pSrc, int nSrcStep, 
                                      Mpp16s * pDst, int nDstStep, MppiSize oDstSizeROI, MppStreamContext mppStreamCtx); 

/** 
 * 1 channel 16-bit signed integer source image duplicated in all 3 channels of destination image.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_16s_C1C3R(const Mpp16s * pSrc, int nSrcStep, 
                                  Mpp16s * pDst, int nDstStep, MppiSize oDstSizeROI); 

/**
 * 1 channel 16-bit signed integer source image duplicated in all 4 channels of destination image.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_16s_C1C4R_Ctx(const Mpp16s * pSrc, int nSrcStep, 
                                      Mpp16s * pDst, int nDstStep, MppiSize oDstSizeROI, MppStreamContext mppStreamCtx); 
                                       
/**
 * 1 channel 16-bit signed integer source image duplicated in all 4 channels of destination image.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_16s_C1C4R(const Mpp16s * pSrc, int nSrcStep, 
                                  Mpp16s * pDst, int nDstStep, MppiSize oDstSizeROI); 
                                       
/**
 * 1 channel 16-bit signed integer source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_16s_C1AC4R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                       Mpp16s * pDst, int nDstStep, MppiSize oDstSizeROI, MppStreamContext mppStreamCtx);

/**
 * 1 channel 16-bit signed integer source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_16s_C1AC4R(const Mpp16s * pSrc, int nSrcStep,
                                   Mpp16s * pDst, int nDstStep, MppiSize oDstSizeROI);

/** 
 * 1 channel 32-bit signed integer source image duplicated in all 3 channels of destination image.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_32s_C1C3R_Ctx(const Mpp32s * pSrc, int nSrcStep,
                                      Mpp32s * pDst, int nDstStep, MppiSize oDstSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 1 channel 32-bit signed integer source image duplicated in all 3 channels of destination image.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_32s_C1C3R(const Mpp32s * pSrc, int nSrcStep,
                                  Mpp32s * pDst, int nDstStep, MppiSize oDstSizeROI);

/**
 * 1 channel 32-bit signed integer source image duplicated in all 4 channels of destination image.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_32s_C1C4R_Ctx(const Mpp32s * pSrc, int nSrcStep,
                                      Mpp32s * pDst, int nDstStep, MppiSize oDstSizeROI, MppStreamContext mppStreamCtx);
                                       
/**
 * 1 channel 32-bit signed integer source image duplicated in all 4 channels of destination image.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_32s_C1C4R(const Mpp32s * pSrc, int nSrcStep,
                                  Mpp32s * pDst, int nDstStep, MppiSize oDstSizeROI);
                                       
/**
 * 1 channel 32-bit signed integer source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_32s_C1AC4R_Ctx(const Mpp32s * pSrc, int nSrcStep,
                                       Mpp32s * pDst, int nDstStep, MppiSize oDstSizeROI, MppStreamContext mppStreamCtx);

/**
 * 1 channel 32-bit signed integer source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_32s_C1AC4R(const Mpp32s * pSrc, int nSrcStep,
                                   Mpp32s * pDst, int nDstStep, MppiSize oDstSizeROI);

/** 
 * 1 channel 32-bit floating point source image duplicated in all 3 channels of destination image.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_32f_C1C3R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                      Mpp32f * pDst, int nDstStep, MppiSize oDstSizeROI, MppStreamContext mppStreamCtx);

/** 
 * 1 channel 32-bit floating point source image duplicated in all 3 channels of destination image.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_32f_C1C3R(const Mpp32f * pSrc, int nSrcStep,
                                  Mpp32f * pDst, int nDstStep, MppiSize oDstSizeROI);

/**
 * 1 channel 32-bit floating point source image duplicated in all 4 channels of destination image.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_32f_C1C4R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                      Mpp32f * pDst, int nDstStep, MppiSize oDstSizeROI, MppStreamContext mppStreamCtx);
                                       
/**
 * 1 channel 32-bit floating point source image duplicated in all 4 channels of destination image.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_32f_C1C4R(const Mpp32f * pSrc, int nSrcStep,
                                  Mpp32f * pDst, int nDstStep, MppiSize oDstSizeROI);
                                       
/**
 * 1 channel 32-bit floating point source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_32f_C1AC4R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                       Mpp32f * pDst, int nDstStep, MppiSize oDstSizeROI, MppStreamContext mppStreamCtx);

/**
 * 1 channel 32-bit floating point source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
 *
 * For common parameter descriptions, see \ref CommonImageDuplicateChannelParameters.
 *
 */
MppStatus mppiDup_32f_C1AC4R(const Mpp32f * pSrc, int nSrcStep,
                                   Mpp32f * pDst, int nDstStep, MppiSize oDstSizeROI);

/** @} image_duplicate_channel */


/** 
 * \section image_transpose Transpose 
 * @defgroup image_transpose Transpose 
 * Functions for transposing images of various types. Like matrix transpose, image transpose is a mirror along the image's
 * diagonal (upper-left to lower-right corner).
 * 
 * \subsection CommonImageTransposeParameters Common parameters for mppiTranspose functions:
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst Pointer to the destination ROI.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSrcROI \ref roi_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/**
 * 1 channel 8-bit unsigned int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_8u_C1R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSrcROI, MppStreamContext mppStreamCtx);

/**
 * 1 channel 8-bit unsigned int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_8u_C1R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSrcROI);

/**
 * 3 channel 8-bit unsigned int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSrcROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 8-bit unsigned int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_8u_C3R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSrcROI);

/**
 * 4 channel 8-bit unsigned int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_8u_C4R_Ctx(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSrcROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 8-bit unsigned int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_8u_C4R(const Mpp8u * pSrc, int nSrcStep, Mpp8u * pDst, int nDstStep, MppiSize oSrcROI);

/**
 * 1 channel 16-bit unsigned int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_16u_C1R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSrcROI, MppStreamContext mppStreamCtx);

/**
 * 1 channel 16-bit unsigned int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_16u_C1R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSrcROI);

/**
 * 3 channel 16-bit unsigned int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_16u_C3R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSrcROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit unsigned int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_16u_C3R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSrcROI);

/**
 * 4 channel 16-bit unsigned int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_16u_C4R_Ctx(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSrcROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit unsigned int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_16u_C4R(const Mpp16u * pSrc, int nSrcStep, Mpp16u * pDst, int nDstStep, MppiSize oSrcROI);

/**
 * 1 channel 16-bit signed int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_16s_C1R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSrcROI, MppStreamContext mppStreamCtx);

/**
 * 1 channel 16-bit signed int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_16s_C1R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSrcROI);

/**
 * 3 channel 16-bit signed int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_16s_C3R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSrcROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 16-bit signed int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_16s_C3R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSrcROI);

/**
 * 4 channel 16-bit signed int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_16s_C4R_Ctx(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSrcROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit signed int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_16s_C4R(const Mpp16s * pSrc, int nSrcStep, Mpp16s * pDst, int nDstStep, MppiSize oSrcROI);

/**
 * 1 channel 32-bit signed int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_32s_C1R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSrcROI, MppStreamContext mppStreamCtx);

/**
 * 1 channel 32-bit signed int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_32s_C1R(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSrcROI);

/**
 * 3 channel 32-bit signed int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_32s_C3R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSrcROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit signed int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_32s_C3R(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSrcROI);

/**
 * 4 channel 32-bit signed int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_32s_C4R_Ctx(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSrcROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit signed int image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_32s_C4R(const Mpp32s * pSrc, int nSrcStep, Mpp32s * pDst, int nDstStep, MppiSize oSrcROI);

/**
 * 1 channel 32-bit floating point image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_32f_C1R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSrcROI, MppStreamContext mppStreamCtx);

/**
 * 1 channel 32-bit floating point image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_32f_C1R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSrcROI);

/**
 * 3 channel 32-bit floating point image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_32f_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSrcROI, MppStreamContext mppStreamCtx);

/**
 * 3 channel 32-bit floating point image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_32f_C3R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSrcROI);

/**
 * 4 channel 32-bit floating point image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_32f_C4R_Ctx(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSrcROI, MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point image transpose.
 *
 * For common parameter descriptions, see \ref CommonImageTransposeParameters.
 *
 */
MppStatus 
mppiTranspose_32f_C4R(const Mpp32f * pSrc, int nSrcStep, Mpp32f * pDst, int nDstStep, MppiSize oSrcROI);

/** @} image_transpose */

/** 
 * \section image_swap_channels Swap Channels
 * @defgroup image_swap_channels Swap Channels
 * Functions for swapping and duplicating channels in multiple channel images. 
 * The methods support arbitrary permutations of the original channels, including replication and
 * setting one or more channels to a constant value.
 *
 * @{
 *
 */

/** 
 * 3 channel 8-bit unsigned integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_8u_C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, 
                                            Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3], MppStreamContext mppStreamCtx);

/** 
 * 3 channel 8-bit unsigned integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_8u_C3R(const Mpp8u * pSrc, int nSrcStep, 
                                        Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3]);

/** 
 * 3 channel 8-bit unsigned integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_8u_C3IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, const int aDstOrder[3], MppStreamContext mppStreamCtx);

/** 
 * 3 channel 8-bit unsigned integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_8u_C3IR(Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, const int aDstOrder[3]);

/** 
 * 4 channel 8-bit unsigned integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_8u_C4C3R_Ctx(const Mpp8u * pSrc, int nSrcStep, 
                                              Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3], MppStreamContext mppStreamCtx);

/** 
 * 4 channel 8-bit unsigned integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_8u_C4C3R(const Mpp8u * pSrc, int nSrcStep, 
                                          Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3]);

/**
 * 4 channel 8-bit unsigned integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_8u_C4R_Ctx(const Mpp8u * pSrc, int nSrcStep, 
                                            Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[4], MppStreamContext mppStreamCtx); 
                                       
/**
 * 4 channel 8-bit unsigned integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_8u_C4R(const Mpp8u * pSrc, int nSrcStep, 
                                        Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[4]); 
                                       
/** 
 * 4 channel 8-bit unsigned integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_8u_C4IR_Ctx(Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, const int aDstOrder[4], MppStreamContext mppStreamCtx);

/** 
 * 4 channel 8-bit unsigned integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_8u_C4IR(Mpp8u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, const int aDstOrder[4]);

/** 
 * 3 channel 8-bit unsigned integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
 *      channel order.
 * \param nValue (V) Single channel constant value that can be replicated in one or more of the 4 destination channels.
 *      nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination
 *      channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that
 *      particular destination channel value unmodified.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_8u_C3C4R_Ctx(const Mpp8u * pSrc, int nSrcStep, 
                                              Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[4], const Mpp8u nValue, MppStreamContext mppStreamCtx);

/** 
 * 3 channel 8-bit unsigned integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
 *      channel order.
 * \param nValue (V) Single channel constant value that can be replicated in one or more of the 4 destination channels.
 *      nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination
 *      channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that
 *      particular destination channel value unmodified.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_8u_C3C4R(const Mpp8u * pSrc, int nSrcStep, 
                                          Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[4], const Mpp8u nValue);

/**
 * 4 channel 8-bit unsigned integer source image to 4 channel destination image with destination alpha channel unaffected.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
 *      channel order.
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA
 *      channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_8u_AC4R_Ctx(const Mpp8u * pSrc, int nSrcStep, 
                                             Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3], MppStreamContext mppStreamCtx); 

/**
 * 4 channel 8-bit unsigned integer source image to 4 channel destination image with destination alpha channel unaffected.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
 *      channel order.
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA
 *      channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_8u_AC4R(const Mpp8u * pSrc, int nSrcStep, 
                                         Mpp8u * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3]); 

/** 
 * 3 channel 16-bit unsigned integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16u_C3R_Ctx(const Mpp16u * pSrc, int nSrcStep, 
                                             Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3], MppStreamContext mppStreamCtx); 

/** 
 * 3 channel 16-bit unsigned integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16u_C3R(const Mpp16u * pSrc, int nSrcStep, 
                                         Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3]); 

/** 
 * 3 channel 16-bit unsigned integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16u_C3IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, const int aDstOrder[3], MppStreamContext mppStreamCtx);

/** 
 * 3 channel 16-bit unsigned integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16u_C3IR(Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, const int aDstOrder[3]);

/** 
 * 4 channel 16-bit unsigned integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16u_C4C3R_Ctx(const Mpp16u * pSrc, int nSrcStep, 
                                               Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3], MppStreamContext mppStreamCtx);

/** 
 * 4 channel 16-bit unsigned integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16u_C4C3R(const Mpp16u * pSrc, int nSrcStep, 
                                           Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3]);

/**
 * 4 channel 16-bit unsigned integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16u_C4R_Ctx(const Mpp16u * pSrc, int nSrcStep, 
                                             Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[4], MppStreamContext mppStreamCtx); 
                                       
/**
 * 4 channel 16-bit unsigned integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16u_C4R(const Mpp16u * pSrc, int nSrcStep, 
                                         Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[4]); 
                                       
/** 
 * 4 channel 16-bit unsigned integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16u_C4IR_Ctx(Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, const int aDstOrder[4], MppStreamContext mppStreamCtx);

/** 
 * 4 channel 16-bit unsigned integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16u_C4IR(Mpp16u * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, const int aDstOrder[4]);

/** 
 * 3 channel 16-bit unsigned integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
 *      channel order.
 * \param nValue (V) Single channel constant value that can be replicated in one or more of the 4 destination channels.
 *      nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination
 *      channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that
 *      particular destination channel value unmodified.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16u_C3C4R_Ctx(const Mpp16u * pSrc, int nSrcStep, 
                                               Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[4], const Mpp16u nValue, MppStreamContext mppStreamCtx);

/** 
 * 3 channel 16-bit unsigned integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
 *      channel order.
 * \param nValue (V) Single channel constant value that can be replicated in one or more of the 4 destination channels.
 *      nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination
 *      channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that
 *      particular destination channel value unmodified.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16u_C3C4R(const Mpp16u * pSrc, int nSrcStep, 
                                           Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[4], const Mpp16u nValue);

/**
 * 4 channel 16-bit unsigned integer source image to 4 channel destination image with destination alpha channel unaffected.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA
 *      channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16u_AC4R_Ctx(const Mpp16u * pSrc, int nSrcStep, 
                                              Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3], MppStreamContext mppStreamCtx); 

/**
 * 4 channel 16-bit unsigned integer source image to 4 channel destination image with destination alpha channel unaffected.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA
 *      channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16u_AC4R(const Mpp16u * pSrc, int nSrcStep, 
                                          Mpp16u * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3]); 

/** 
 * 3 channel 16-bit signed integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16s_C3R_Ctx(const Mpp16s * pSrc, int nSrcStep, 
                                             Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3], MppStreamContext mppStreamCtx); 

/** 
 * 3 channel 16-bit signed integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16s_C3R(const Mpp16s * pSrc, int nSrcStep, 
                                         Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3]); 

/** 
 * 3 channel 16-bit signed integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16s_C3IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, const int aDstOrder[3], MppStreamContext mppStreamCtx);

/** 
 * 3 channel 16-bit signed integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16s_C3IR(Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, const int aDstOrder[3]);

/** 
 * 4 channel 16-bit signed integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16s_C4C3R_Ctx(const Mpp16s * pSrc, int nSrcStep, 
                                               Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3], MppStreamContext mppStreamCtx);

/** 
 * 4 channel 16-bit signed integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16s_C4C3R(const Mpp16s * pSrc, int nSrcStep, 
                                           Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3]);

/**
 * 4 channel 16-bit signed integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16s_C4R_Ctx(const Mpp16s * pSrc, int nSrcStep, 
                                             Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[4], MppStreamContext mppStreamCtx); 
                                       
/**
 * 4 channel 16-bit signed integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16s_C4R(const Mpp16s * pSrc, int nSrcStep, 
                                         Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[4]); 
                                       
/** 
 * 4 channel 16-bit signed integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16s_C4IR_Ctx(Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, const int aDstOrder[4], MppStreamContext mppStreamCtx);

/** 
 * 4 channel 16-bit signed integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16s_C4IR(Mpp16s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, const int aDstOrder[4]);

/** 
 * 3 channel 16-bit signed integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
 *      channel order.
 * \param nValue (V) Single channel constant value that can be replicated in one or more of the 4 destination channels.
 *      nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination
 *      channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that
 *      particular destination channel value unmodified.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16s_C3C4R_Ctx(const Mpp16s * pSrc, int nSrcStep, 
                                               Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[4], const Mpp16s nValue, MppStreamContext mppStreamCtx);

/** 
 * 3 channel 16-bit signed integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
 *      channel order.
 * \param nValue (V) Single channel constant value that can be replicated in one or more of the 4 destination channels.
 *      nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination
 *      channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that
 *      particular destination channel value unmodified.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16s_C3C4R(const Mpp16s * pSrc, int nSrcStep, 
                                           Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[4], const Mpp16s nValue);

/**
 * 4 channel 16-bit signed integer source image to 4 channel destination image with destination alpha channel unaffected.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA
 *      channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16s_AC4R_Ctx(const Mpp16s * pSrc, int nSrcStep,
                                              Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 16-bit signed integer source image to 4 channel destination image with destination alpha channel unaffected.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA
 *      channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_16s_AC4R(const Mpp16s * pSrc, int nSrcStep,
                                          Mpp16s * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3]);

/** 
 * 3 channel 32-bit signed integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32s_C3R_Ctx(const Mpp32s * pSrc, int nSrcStep,
                                             Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3], MppStreamContext mppStreamCtx);

/** 
 * 3 channel 32-bit signed integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32s_C3R(const Mpp32s * pSrc, int nSrcStep,
                                         Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3]);

/** 
 * 3 channel 32-bit signed integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32s_C3IR_Ctx(Mpp32s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, const int aDstOrder[3], MppStreamContext mppStreamCtx);

/** 
 * 3 channel 32-bit signed integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32s_C3IR(Mpp32s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, const int aDstOrder[3]);

/** 
 * 4 channel 32-bit signed integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32s_C4C3R_Ctx(const Mpp32s * pSrc, int nSrcStep, 
                                               Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3], MppStreamContext mppStreamCtx);

/** 
 * 4 channel 32-bit signed integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32s_C4C3R(const Mpp32s * pSrc, int nSrcStep, 
                                           Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3]);

/**
 * 4 channel 32-bit signed integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32s_C4R_Ctx(const Mpp32s * pSrc, int nSrcStep,
                                             Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[4], MppStreamContext mppStreamCtx);
                                       
/**
 * 4 channel 32-bit signed integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32s_C4R(const Mpp32s * pSrc, int nSrcStep,
                                         Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[4]);
                                       
/** 
 * 4 channel 32-bit signed integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32s_C4IR_Ctx(Mpp32s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, const int aDstOrder[4], MppStreamContext mppStreamCtx);

/** 
 * 4 channel 32-bit signed integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32s_C4IR(Mpp32s * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, const int aDstOrder[4]);

/** 
 * 3 channel 32-bit signed integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
 *      channel order.
 * \param nValue (V) Single channel constant value that can be replicated in one or more of the 4 destination channels.
 *      nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination
 *      channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that
 *      particular destination channel value unmodified.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32s_C3C4R_Ctx(const Mpp32s * pSrc, int nSrcStep, 
                                               Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[4], const Mpp32s nValue, MppStreamContext mppStreamCtx);

/** 
 * 3 channel 32-bit signed integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
 *      channel order.
 * \param nValue (V) Single channel constant value that can be replicated in one or more of the 4 destination channels.
 *      nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination
 *      channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that
 *      particular destination channel value unmodified.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32s_C3C4R(const Mpp32s * pSrc, int nSrcStep, 
                                           Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[4], const Mpp32s nValue);

/**
 * 4 channel 32-bit signed integer source image to 4 channel destination image with destination alpha channel unaffected.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA
 *      channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32s_AC4R_Ctx(const Mpp32s * pSrc, int nSrcStep,
                                              Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit signed integer source image to 4 channel destination image with destination alpha channel unaffected.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA
 *      channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32s_AC4R(const Mpp32s * pSrc, int nSrcStep,
                                          Mpp32s * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3]);

/** 
 * 3 channel 32-bit floating point source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32f_C3R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                             Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3], MppStreamContext mppStreamCtx);

/** 
 * 3 channel 32-bit floating point source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32f_C3R(const Mpp32f * pSrc, int nSrcStep,
                                         Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3]);

/** 
 * 3 channel 32-bit floating point in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32f_C3IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, const int aDstOrder[3], MppStreamContext mppStreamCtx);

/** 
 * 3 channel 32-bit floating point in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32f_C3IR(Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, const int aDstOrder[3]);

/** 
 * 4 channel 32-bit floating point source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32f_C4C3R_Ctx(const Mpp32f * pSrc, int nSrcStep, 
                                               Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3], MppStreamContext mppStreamCtx);

/** 
 * 4 channel 32-bit floating point source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32f_C4C3R(const Mpp32f * pSrc, int nSrcStep, 
                                           Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3]);

/**
 * 4 channel 32-bit floating point source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32f_C4R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                             Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[4], MppStreamContext mppStreamCtx);
                                       
/**
 * 4 channel 32-bit floating point source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32f_C4R(const Mpp32f * pSrc, int nSrcStep,
                                         Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[4]);
                                       
/** 
 * 4 channel 32-bit floating point in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32f_C4IR_Ctx(Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, const int aDstOrder[4], MppStreamContext mppStreamCtx);

/** 
 * 4 channel 32-bit floating point in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32f_C4IR(Mpp32f * pSrcDst, int nSrcDstStep, MppiSize oSizeROI, const int aDstOrder[4]);

/** 
 * 3 channel 32-bit floating point source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
 *      channel order.
 * \param nValue (V) Single channel constant value that can be replicated in one or more of the 4 destination channels.
 *      nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination
 *      channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that
 *      particular destination channel value unmodified.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32f_C3C4R_Ctx(const Mpp32f * pSrc, int nSrcStep, 
                                               Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[4], const Mpp32f nValue, MppStreamContext mppStreamCtx);

/** 
 * 3 channel 32-bit floating point source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
 *      channel order.
 * \param nValue (V) Single channel constant value that can be replicated in one or more of the 4 destination channels.
 *      nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination
 *      channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that
 *      particular destination channel value unmodified.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32f_C3C4R(const Mpp32f * pSrc, int nSrcStep, 
                                           Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[4], const Mpp32f nValue);

/**
 * 4 channel 32-bit floating point source image to 4 channel destination image with destination alpha channel unaffected.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA
 *      channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32f_AC4R_Ctx(const Mpp32f * pSrc, int nSrcStep,
                                              Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3], MppStreamContext mppStreamCtx);

/**
 * 4 channel 32-bit floating point source image to 4 channel destination image with destination alpha channel unaffected.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA
 *      channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
MppStatus mppiSwapChannels_32f_AC4R(const Mpp32f * pSrc, int nSrcStep,
                                          Mpp32f * pDst, int nDstStep, MppiSize oSizeROI, const int aDstOrder[3]);

/** @} image_swap_channels */

/** @} image_data_exchange_and_initialization */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MC_MPPI_DATA_EXCHANGE_AND_INITIALIZATION_H */
