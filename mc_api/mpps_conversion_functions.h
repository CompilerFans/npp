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
#ifndef MC_MPPS_CONVERSION_FUNCTIONS_H
#define MC_MPPS_CONVERSION_FUNCTIONS_H
 
/**
 * \file mpps_conversion_functions.h
 * MPP Signal Processing Functionality.
 */
 
#include "mppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif


/** 
 *  \page signal_conversion_functions Conversion Functions
 *  @defgroup signal_conversion_functions Conversion Functions
 *  @ingroup mpps
 * Functions that provide conversion and threshold operations
 * @{
 *
 */

/** 
 * \section signal_convert Convert
 * @defgroup signal_convert Convert
 * The set of conversion operations available in the library
 * @{
 *
 */

/** @name Convert
 * Routines for converting the sample-data type of signals.
 *
 * @{
 *
 */

/** 
 * 8-bit signed byte to 16-bit signed short conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_8s16s_Ctx(const Mpp8s * pSrc, Mpp16s * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit signed byte to 16-bit signed short conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_8s16s(const Mpp8s * pSrc, Mpp16s * pDst, size_t nLength);

/** 
 * 8-bit signed byte to 32-bit floating point number conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_8s32f_Ctx(const Mpp8s * pSrc, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit signed byte to 32-bit floating point number conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_8s32f(const Mpp8s * pSrc, Mpp32f * pDst, size_t nLength);

/** 
 * 8-bit unsigned byte to 32-bit floating point number conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_8u32f_Ctx(const Mpp8u * pSrc, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);

/** 
 * 8-bit unsigned byte to 32-bit floating point number conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_8u32f(const Mpp8u * pSrc, Mpp32f * pDst, size_t nLength);

/** 
 * 16-bit signed short to 8-bit signed byte conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param eRoundMode \ref rounding_mode_parameter.
 *  \param nScaleFactor  nScaleFactor
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_16s8s_Sfs_Ctx(const Mpp16s * pSrc, Mpp8s * pDst, Mpp32u nLength, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short to 8-bit signed byte conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param eRoundMode \ref rounding_mode_parameter.
 * \param nScaleFactor  nScaleFactor
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_16s8s_Sfs(const Mpp16s * pSrc, Mpp8s * pDst, Mpp32u nLength, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * 16-bit signed short to 32-bit signed integer conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_16s32s_Ctx(const Mpp16s * pSrc, Mpp32s * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short to 32-bit signed integer conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_16s32s(const Mpp16s * pSrc, Mpp32s * pDst, size_t nLength);

/** 
 * 16-bit signed short to 32-bit floating point number conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_16s32f_Ctx(const Mpp16s * pSrc, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);

/** 
 * 16-bit signed short to 32-bit floating point number conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_16s32f(const Mpp16s * pSrc, Mpp32f * pDst, size_t nLength);

/** 
 * 16-bit unsigned short to 32-bit floating point number conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_16u32f_Ctx(const Mpp16u * pSrc, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short to 32-bit floating point number conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_16u32f(const Mpp16u * pSrc, Mpp32f * pDst, size_t nLength);

/** 
 * 32-bit signed integer to 16-bit signed short conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32s16s_Ctx(const Mpp32s * pSrc, Mpp16s * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer to 16-bit signed short conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32s16s(const Mpp32s * pSrc, Mpp16s * pDst, size_t nLength);

/** 
 * 32-bit signed integer to 32-bit floating point number conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32s32f_Ctx(const Mpp32s * pSrc, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer to 32-bit floating point number conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32s32f(const Mpp32s * pSrc, Mpp32f * pDst, size_t nLength);

/** 
 * 32-bit signed integer to 64-bit floating point number conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32s64f_Ctx(const Mpp32s * pSrc, Mpp64f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer to 64-bit floating point number conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32s64f(const Mpp32s * pSrc, Mpp64f * pDst, size_t nLength);

/** 
 * 32-bit floating point number to 64-bit floating point number conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32f64f_Ctx(const Mpp32f * pSrc, Mpp64f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point number to 64-bit floating point number conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32f64f(const Mpp32f * pSrc, Mpp64f * pDst, size_t nLength);

/** 
 * 64-bit signed integer to 64-bit floating point number conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_64s64f_Ctx(const Mpp64s * pSrc, Mpp64f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit signed integer to 64-bit floating point number conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_64s64f(const Mpp64s * pSrc, Mpp64f * pDst, size_t nLength);

/** 
 * 64-bit floating point number to 32-bit floating point number conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_64f32f_Ctx(const Mpp64f * pSrc, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point number to 32-bit floating point number conversion
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_64f32f(const Mpp64f * pSrc, Mpp32f * pDst, size_t nLength);


/** 
 * 16-bit signed short to 32-bit floating point number conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_16s32f_Sfs_Ctx(const Mpp16s * pSrc, Mpp32f * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short to 32-bit floating point number conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_16s32f_Sfs(const Mpp16s * pSrc, Mpp32f * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short to 64-bit floating point number conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_16s64f_Sfs_Ctx(const Mpp16s * pSrc, Mpp64f * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short to 64-bit floating point number conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_16s64f_Sfs(const Mpp16s * pSrc, Mpp64f * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer to 16-bit signed short conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32s16s_Sfs_Ctx(const Mpp32s * pSrc, Mpp16s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer to 16-bit signed short conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32s16s_Sfs(const Mpp32s * pSrc, Mpp16s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer to 32-bit floating point number conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32s32f_Sfs_Ctx(const Mpp32s * pSrc, Mpp32f * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer to 32-bit floating point number conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32s32f_Sfs(const Mpp32s * pSrc, Mpp32f * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer to 64-bit floating point number conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32s64f_Sfs_Ctx(const Mpp32s * pSrc, Mpp64f * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer to 64-bit floating point number conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32s64f_Sfs(const Mpp32s * pSrc, Mpp64f * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer to 8-bit signed byte conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param eRoundMode \ref rounding_mode_parameter.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32f8s_Sfs_Ctx(const Mpp32f * pSrc, Mpp8s * pDst, size_t nLength, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point number to 8-bit signed byte conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param eRoundMode \ref rounding_mode_parameter.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32f8s_Sfs(const Mpp32f * pSrc, Mpp8s * pDst, size_t nLength, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * 32-bit floating point number to 8-bit unsigned byte conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param eRoundMode \ref rounding_mode_parameter.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32f8u_Sfs_Ctx(const Mpp32f * pSrc, Mpp8u * pDst, size_t nLength, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point number to 8-bit unsigned byte conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param eRoundMode \ref rounding_mode_parameter.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32f8u_Sfs(const Mpp32f * pSrc, Mpp8u * pDst, size_t nLength, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * 32-bit floating point number to 16-bit signed short conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param eRoundMode \ref rounding_mode_parameter.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32f16s_Sfs_Ctx(const Mpp32f * pSrc, Mpp16s * pDst, size_t nLength, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point number to 16-bit signed short conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param eRoundMode \ref rounding_mode_parameter.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32f16s_Sfs(const Mpp32f * pSrc, Mpp16s * pDst, size_t nLength, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * 32-bit floating point number to 16-bit unsigned short conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param eRoundMode \ref rounding_mode_parameter.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32f16u_Sfs_Ctx(const Mpp32f * pSrc, Mpp16u * pDst, size_t nLength, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point number to 16-bit unsigned short conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param eRoundMode \ref rounding_mode_parameter.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32f16u_Sfs(const Mpp32f * pSrc, Mpp16u * pDst, size_t nLength, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * 32-bit floating point number to 32-bit signed integer conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param eRoundMode \ref rounding_mode_parameter.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32f32s_Sfs_Ctx(const Mpp32f * pSrc, Mpp32s * pDst, size_t nLength, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point number to 32-bit signed integer conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param eRoundMode \ref rounding_mode_parameter.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_32f32s_Sfs(const Mpp32f * pSrc, Mpp32s * pDst, size_t nLength, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * 64-bit signed integer to 32-bit signed integer conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param eRoundMode \ref rounding_mode_parameter.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_64s32s_Sfs_Ctx(const Mpp64s * pSrc, Mpp32s * pDst, size_t nLength, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 64-bit signed integer to 32-bit signed integer conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param eRoundMode \ref rounding_mode_parameter.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_64s32s_Sfs(const Mpp64s * pSrc, Mpp32s * pDst, size_t nLength, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * 64-bit floating point number to 16-bit signed short conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param eRoundMode \ref rounding_mode_parameter.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_64f16s_Sfs_Ctx(const Mpp64f * pSrc, Mpp16s * pDst, size_t nLength, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point number to 16-bit signed short conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param eRoundMode \ref rounding_mode_parameter.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_64f16s_Sfs(const Mpp64f * pSrc, Mpp16s * pDst, size_t nLength, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * 64-bit floating point number to 32-bit signed integer conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param eRoundMode \ref rounding_mode_parameter.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_64f32s_Sfs_Ctx(const Mpp64f * pSrc, Mpp32s * pDst, size_t nLength, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point number to 32-bit signed integer conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param eRoundMode \ref rounding_mode_parameter.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_64f32s_Sfs(const Mpp64f * pSrc, Mpp32s * pDst, size_t nLength, MppRoundMode eRoundMode, int nScaleFactor);

/** 
 * 64-bit floating point number to 64-bit signed integer conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param eRoundMode \ref rounding_mode_parameter.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_64f64s_Sfs_Ctx(const Mpp64f * pSrc, Mpp64s * pDst, size_t nLength, MppRoundMode eRoundMode, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point number to 64-bit signed integer conversion with scaling
 *
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param eRoundMode \ref rounding_mode_parameter.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsConvert_64f64s_Sfs(const Mpp64f * pSrc, Mpp64s * pDst, size_t nLength, MppRoundMode eRoundMode, int nScaleFactor);

/** @} end of Convert */

/** @} signal_convert */

/** 
 * \section signal_threshold Threshold
 * @defgroup signal_threshold Threshold
 * The set of threshold operations available in the library.
 * @{
 *
 */

/** @name Threshold Functions
 * Performs the threshold operation on the samples of a signal by limiting the sample values by a specified constant value.
 *
 * @{
 *
 */

/** 
 * 16-bit signed short signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_16s_Ctx(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength, Mpp16s nLevel, MppCmpOp nRelOp, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_16s(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength, Mpp16s nLevel, MppCmpOp nRelOp);

/** 
 * 16-bit in place signed short signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_16s_I_Ctx(Mpp16s * pSrcDst, size_t nLength, Mpp16s nLevel, MppCmpOp nRelOp, MppStreamContext mppStreamCtx);
/** 
 * 16-bit in place signed short signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_16s_I(Mpp16s * pSrcDst, size_t nLength, Mpp16s nLevel, MppCmpOp nRelOp);

/** 
 * 16-bit signed short complex number signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_16sc_Ctx(const Mpp16sc * pSrc, Mpp16sc * pDst, size_t nLength, Mpp16s nLevel, MppCmpOp nRelOp, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short complex number signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_16sc(const Mpp16sc * pSrc, Mpp16sc * pDst, size_t nLength, Mpp16s nLevel, MppCmpOp nRelOp);

/** 
 * 16-bit in place signed short complex number signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_16sc_I_Ctx(Mpp16sc * pSrcDst, size_t nLength, Mpp16s nLevel, MppCmpOp nRelOp, MppStreamContext mppStreamCtx);
/** 
 * 16-bit in place signed short complex number signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_16sc_I(Mpp16sc * pSrcDst, size_t nLength, Mpp16s nLevel, MppCmpOp nRelOp);

/** 
 * 32-bit floating point signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_32f_Ctx(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength, Mpp32f nLevel, MppCmpOp nRelOp, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_32f(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength, Mpp32f nLevel, MppCmpOp nRelOp);

/** 
 * 32-bit in place floating point signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_32f_I_Ctx(Mpp32f * pSrcDst, size_t nLength, Mpp32f nLevel, MppCmpOp nRelOp, MppStreamContext mppStreamCtx);
/** 
 * 32-bit in place floating point signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_32f_I(Mpp32f * pSrcDst, size_t nLength, Mpp32f nLevel, MppCmpOp nRelOp);

/** 
 * 32-bit floating point complex number signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_32fc_Ctx(const Mpp32fc * pSrc, Mpp32fc * pDst, size_t nLength, Mpp32f nLevel, MppCmpOp nRelOp, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point complex number signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_32fc(const Mpp32fc * pSrc, Mpp32fc * pDst, size_t nLength, Mpp32f nLevel, MppCmpOp nRelOp);

/** 
 * 32-bit in place floating point complex number signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_32fc_I_Ctx(Mpp32fc * pSrcDst, size_t nLength, Mpp32f nLevel, MppCmpOp nRelOp, MppStreamContext mppStreamCtx);
/** 
 * 32-bit in place floating point complex number signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_32fc_I(Mpp32fc * pSrcDst, size_t nLength, Mpp32f nLevel, MppCmpOp nRelOp);

/** 
 * 64-bit floating point signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_64f_Ctx(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength, Mpp64f nLevel, MppCmpOp nRelOp, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_64f(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength, Mpp64f nLevel, MppCmpOp nRelOp);

/** 
 * 64-bit in place floating point signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_64f_I_Ctx(Mpp64f * pSrcDst, size_t nLength, Mpp64f nLevel, MppCmpOp nRelOp, MppStreamContext mppStreamCtx);
/** 
 * 64-bit in place floating point signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_64f_I(Mpp64f * pSrcDst, size_t nLength, Mpp64f nLevel, MppCmpOp nRelOp);

/** 
 * 64-bit floating point complex number signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_64fc_Ctx(const Mpp64fc * pSrc, Mpp64fc * pDst, size_t nLength, Mpp64f nLevel, MppCmpOp nRelOp, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point complex number signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_64fc(const Mpp64fc * pSrc, Mpp64fc * pDst, size_t nLength, Mpp64f nLevel, MppCmpOp nRelOp);

/** 
 * 64-bit in place floating point complex number signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_64fc_I_Ctx(Mpp64fc * pSrcDst, size_t nLength, Mpp64f nLevel, MppCmpOp nRelOp, MppStreamContext mppStreamCtx);
/** 
 * 64-bit in place floating point complex number signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp MppCmpOp type of thresholding operation (MPP_CMP_LESS or MPP_CMP_GREATER only).
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_64fc_I(Mpp64fc * pSrcDst, size_t nLength, Mpp64f nLevel, MppCmpOp nRelOp);

/** 
 * 16-bit signed short signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_16s_Ctx(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength, Mpp16s nLevel, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_16s(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength, Mpp16s nLevel);

/** 
 * 16-bit in place signed short signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_16s_I_Ctx(Mpp16s * pSrcDst, size_t nLength, Mpp16s nLevel, MppStreamContext mppStreamCtx);
/** 
 * 16-bit in place signed short signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_16s_I(Mpp16s * pSrcDst, size_t nLength, Mpp16s nLevel);

/** 
 * 16-bit signed short complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_16sc_Ctx(const Mpp16sc * pSrc, Mpp16sc * pDst, size_t nLength, Mpp16s nLevel, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_16sc(const Mpp16sc * pSrc, Mpp16sc * pDst, size_t nLength, Mpp16s nLevel);

/** 
 * 16-bit in place signed short complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_16sc_I_Ctx(Mpp16sc * pSrcDst, size_t nLength, Mpp16s nLevel, MppStreamContext mppStreamCtx);
/** 
 * 16-bit in place signed short complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_16sc_I(Mpp16sc * pSrcDst, size_t nLength, Mpp16s nLevel);

/** 
 * 32-bit floating point signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_32f_Ctx(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength, Mpp32f nLevel, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_32f(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength, Mpp32f nLevel);

/** 
 * 32-bit in place floating point signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_32f_I_Ctx(Mpp32f * pSrcDst, size_t nLength, Mpp32f nLevel, MppStreamContext mppStreamCtx);
/** 
 * 32-bit in place floating point signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_32f_I(Mpp32f * pSrcDst, size_t nLength, Mpp32f nLevel);

/** 
 * 32-bit floating point complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_32fc_Ctx(const Mpp32fc * pSrc, Mpp32fc * pDst, size_t nLength, Mpp32f nLevel, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_32fc(const Mpp32fc * pSrc, Mpp32fc * pDst, size_t nLength, Mpp32f nLevel);

/** 
 * 32-bit in place floating point complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_32fc_I_Ctx(Mpp32fc * pSrcDst, size_t nLength, Mpp32f nLevel, MppStreamContext mppStreamCtx);
/** 
 * 32-bit in place floating point complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_32fc_I(Mpp32fc * pSrcDst, size_t nLength, Mpp32f nLevel);

/** 
 * 64-bit floating point signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_64f_Ctx(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength, Mpp64f nLevel, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_64f(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength, Mpp64f nLevel);

/** 
 * 64-bit in place floating point signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_64f_I_Ctx(Mpp64f * pSrcDst, size_t nLength, Mpp64f nLevel, MppStreamContext mppStreamCtx);
/** 
 * 64-bit in place floating point signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_64f_I(Mpp64f * pSrcDst, size_t nLength, Mpp64f nLevel);

/** 
 * 64-bit floating point complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_64fc_Ctx(const Mpp64fc * pSrc, Mpp64fc * pDst, size_t nLength, Mpp64f nLevel, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_64fc(const Mpp64fc * pSrc, Mpp64fc * pDst, size_t nLength, Mpp64f nLevel);

/** 
 * 64-bit in place floating point complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_64fc_I_Ctx(Mpp64fc * pSrcDst, size_t nLength, Mpp64f nLevel, MppStreamContext mppStreamCtx);
/** 
 * 64-bit in place floating point complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LT_64fc_I(Mpp64fc * pSrcDst, size_t nLength, Mpp64f nLevel);

/** 
 * 16-bit signed short signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_16s_Ctx(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength, Mpp16s nLevel, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_16s(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength, Mpp16s nLevel);

/** 
 * 16-bit in place signed short signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_16s_I_Ctx(Mpp16s * pSrcDst, size_t nLength, Mpp16s nLevel, MppStreamContext mppStreamCtx);
/** 
 * 16-bit in place signed short signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_16s_I(Mpp16s * pSrcDst, size_t nLength, Mpp16s nLevel);

/** 
 * 16-bit signed short complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_16sc_Ctx(const Mpp16sc * pSrc, Mpp16sc * pDst, size_t nLength, Mpp16s nLevel, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_16sc(const Mpp16sc * pSrc, Mpp16sc * pDst, size_t nLength, Mpp16s nLevel);

/** 
 * 16-bit in place signed short complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_16sc_I_Ctx(Mpp16sc * pSrcDst, size_t nLength, Mpp16s nLevel, MppStreamContext mppStreamCtx);
/** 
 * 16-bit in place signed short complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_16sc_I(Mpp16sc * pSrcDst, size_t nLength, Mpp16s nLevel);

/** 
 * 32-bit floating point signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_32f_Ctx(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength, Mpp32f nLevel, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_32f(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength, Mpp32f nLevel);

/** 
 * 32-bit in place floating point signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_32f_I_Ctx(Mpp32f * pSrcDst, size_t nLength, Mpp32f nLevel, MppStreamContext mppStreamCtx);
/** 
 * 32-bit in place floating point signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_32f_I(Mpp32f * pSrcDst, size_t nLength, Mpp32f nLevel);

/** 
 * 32-bit floating point complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_32fc_Ctx(const Mpp32fc * pSrc, Mpp32fc * pDst, size_t nLength, Mpp32f nLevel, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_32fc(const Mpp32fc * pSrc, Mpp32fc * pDst, size_t nLength, Mpp32f nLevel);

/** 
 * 32-bit in place floating point complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_32fc_I_Ctx(Mpp32fc * pSrcDst, size_t nLength, Mpp32f nLevel, MppStreamContext mppStreamCtx);
/** 
 * 32-bit in place floating point complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_32fc_I(Mpp32fc * pSrcDst, size_t nLength, Mpp32f nLevel);

/** 
 * 64-bit floating point signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_64f_Ctx(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength, Mpp64f nLevel, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_64f(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength, Mpp64f nLevel);

/** 
 * 64-bit in place floating point signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_64f_I_Ctx(Mpp64f * pSrcDst, size_t nLength, Mpp64f nLevel, MppStreamContext mppStreamCtx);
/** 
 * 64-bit in place floating point signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_64f_I(Mpp64f * pSrcDst, size_t nLength, Mpp64f nLevel);

/** 
 * 64-bit floating point complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_64fc_Ctx(const Mpp64fc * pSrc, Mpp64fc * pDst, size_t nLength, Mpp64f nLevel, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_64fc(const Mpp64fc * pSrc, Mpp64fc * pDst, size_t nLength, Mpp64f nLevel);

/** 
 * 64-bit in place floating point complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_64fc_I_Ctx(Mpp64fc * pSrcDst, size_t nLength, Mpp64f nLevel, MppStreamContext mppStreamCtx);
/** 
 * 64-bit in place floating point complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GT_64fc_I(Mpp64fc * pSrcDst, size_t nLength, Mpp64f nLevel);

/** 
 * 16-bit signed short signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_16s_Ctx(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength, Mpp16s nLevel, Mpp16s nValue, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_16s(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength, Mpp16s nLevel, Mpp16s nValue);

/** 
 * 16-bit in place signed short signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_16s_I_Ctx(Mpp16s * pSrcDst, size_t nLength, Mpp16s nLevel, Mpp16s nValue, MppStreamContext mppStreamCtx);
/** 
 * 16-bit in place signed short signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_16s_I(Mpp16s * pSrcDst, size_t nLength, Mpp16s nLevel, Mpp16s nValue);

/** 
 * 16-bit signed short complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_16sc_Ctx(const Mpp16sc * pSrc, Mpp16sc * pDst, size_t nLength, Mpp16s nLevel, Mpp16sc nValue, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_16sc(const Mpp16sc * pSrc, Mpp16sc * pDst, size_t nLength, Mpp16s nLevel, Mpp16sc nValue);

/** 
 * 16-bit in place signed short complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_16sc_I_Ctx(Mpp16sc * pSrcDst, size_t nLength, Mpp16s nLevel, Mpp16sc nValue, MppStreamContext mppStreamCtx);
/** 
 * 16-bit in place signed short complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_16sc_I(Mpp16sc * pSrcDst, size_t nLength, Mpp16s nLevel, Mpp16sc nValue);

/** 
 * 32-bit floating point signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_32f_Ctx(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength, Mpp32f nLevel, Mpp32f nValue, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_32f(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength, Mpp32f nLevel, Mpp32f nValue);

/** 
 * 32-bit in place floating point signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_32f_I_Ctx(Mpp32f * pSrcDst, size_t nLength, Mpp32f nLevel, Mpp32f nValue, MppStreamContext mppStreamCtx);
/** 
 * 32-bit in place floating point signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_32f_I(Mpp32f * pSrcDst, size_t nLength, Mpp32f nLevel, Mpp32f nValue);

/** 
 * 32-bit floating point complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_32fc_Ctx(const Mpp32fc * pSrc, Mpp32fc * pDst, size_t nLength, Mpp32f nLevel, Mpp32fc nValue, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_32fc(const Mpp32fc * pSrc, Mpp32fc * pDst, size_t nLength, Mpp32f nLevel, Mpp32fc nValue);

/** 
 * 32-bit in place floating point complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_32fc_I_Ctx(Mpp32fc * pSrcDst, size_t nLength, Mpp32f nLevel, Mpp32fc nValue, MppStreamContext mppStreamCtx);
/** 
 * 32-bit in place floating point complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_32fc_I(Mpp32fc * pSrcDst, size_t nLength, Mpp32f nLevel, Mpp32fc nValue);

/** 
 * 64-bit floating point signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_64f_Ctx(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength, Mpp64f nLevel, Mpp64f nValue, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_64f(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength, Mpp64f nLevel, Mpp64f nValue);

/** 
 * 64-bit in place floating point signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_64f_I_Ctx(Mpp64f * pSrcDst, size_t nLength, Mpp64f nLevel, Mpp64f nValue, MppStreamContext mppStreamCtx);
/** 
 * 64-bit in place floating point signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_64f_I(Mpp64f * pSrcDst, size_t nLength, Mpp64f nLevel, Mpp64f nValue);

/** 
 * 64-bit floating point complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_64fc_Ctx(const Mpp64fc * pSrc, Mpp64fc * pDst, size_t nLength, Mpp64f nLevel, Mpp64fc nValue, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_64fc(const Mpp64fc * pSrc, Mpp64fc * pDst, size_t nLength, Mpp64f nLevel, Mpp64fc nValue);

/** 
 * 64-bit in place floating point complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_64fc_I_Ctx(Mpp64fc * pSrcDst, size_t nLength, Mpp64f nLevel, Mpp64fc nValue, MppStreamContext mppStreamCtx);
/** 
 * 64-bit in place floating point complex number signal MPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_LTVal_64fc_I(Mpp64fc * pSrcDst, size_t nLength, Mpp64f nLevel, Mpp64fc nValue);

/** 
 * 16-bit signed short signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_16s_Ctx(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength, Mpp16s nLevel, Mpp16s nValue, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_16s(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength, Mpp16s nLevel, Mpp16s nValue);

/** 
 * 16-bit in place signed short signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_16s_I_Ctx(Mpp16s * pSrcDst, size_t nLength, Mpp16s nLevel, Mpp16s nValue, MppStreamContext mppStreamCtx);
/** 
 * 16-bit in place signed short signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_16s_I(Mpp16s * pSrcDst, size_t nLength, Mpp16s nLevel, Mpp16s nValue);

/** 
 * 16-bit signed short complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_16sc_Ctx(const Mpp16sc * pSrc, Mpp16sc * pDst, size_t nLength, Mpp16s nLevel, Mpp16sc nValue, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_16sc(const Mpp16sc * pSrc, Mpp16sc * pDst, size_t nLength, Mpp16s nLevel, Mpp16sc nValue);

/** 
 * 16-bit in place signed short complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_16sc_I_Ctx(Mpp16sc * pSrcDst, size_t nLength, Mpp16s nLevel, Mpp16sc nValue, MppStreamContext mppStreamCtx);
/** 
 * 16-bit in place signed short complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_16sc_I(Mpp16sc * pSrcDst, size_t nLength, Mpp16s nLevel, Mpp16sc nValue);

/** 
 * 32-bit floating point signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_32f_Ctx(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength, Mpp32f nLevel, Mpp32f nValue, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_32f(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength, Mpp32f nLevel, Mpp32f nValue);

/** 
 * 32-bit in place floating point signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_32f_I_Ctx(Mpp32f * pSrcDst, size_t nLength, Mpp32f nLevel, Mpp32f nValue, MppStreamContext mppStreamCtx);
/** 
 * 32-bit in place floating point signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_32f_I(Mpp32f * pSrcDst, size_t nLength, Mpp32f nLevel, Mpp32f nValue);

/** 
 * 32-bit floating point complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_32fc_Ctx(const Mpp32fc * pSrc, Mpp32fc * pDst, size_t nLength, Mpp32f nLevel, Mpp32fc nValue, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_32fc(const Mpp32fc * pSrc, Mpp32fc * pDst, size_t nLength, Mpp32f nLevel, Mpp32fc nValue);

/** 
 * 32-bit in place floating point complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_32fc_I_Ctx(Mpp32fc * pSrcDst, size_t nLength, Mpp32f nLevel, Mpp32fc nValue, MppStreamContext mppStreamCtx);
/** 
 * 32-bit in place floating point complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_32fc_I(Mpp32fc * pSrcDst, size_t nLength, Mpp32f nLevel, Mpp32fc nValue);

/** 
 * 64-bit floating point signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_64f_Ctx(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength, Mpp64f nLevel, Mpp64f nValue, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_64f(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength, Mpp64f nLevel, Mpp64f nValue);

/** 
 * 64-bit in place floating point signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_64f_I_Ctx(Mpp64f * pSrcDst, size_t nLength, Mpp64f nLevel, Mpp64f nValue, MppStreamContext mppStreamCtx);
/** 
 * 64-bit in place floating point signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_64f_I(Mpp64f * pSrcDst, size_t nLength, Mpp64f nLevel, Mpp64f nValue);

/** 
 * 64-bit floating point complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_64fc_Ctx(const Mpp64fc * pSrc, Mpp64fc * pDst, size_t nLength, Mpp64f nLevel, Mpp64fc nValue, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_64fc(const Mpp64fc * pSrc, Mpp64fc * pDst, size_t nLength, Mpp64f nLevel, Mpp64fc nValue);

/** 
 * 64-bit in place floating point complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_64fc_I_Ctx(Mpp64fc * pSrcDst, size_t nLength, Mpp64f nLevel, Mpp64fc nValue, MppStreamContext mppStreamCtx);
/** 
 * 64-bit in place floating point complex number signal MPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsThreshold_GTVal_64fc_I(Mpp64fc * pSrcDst, size_t nLength, Mpp64f nLevel, Mpp64fc nValue);

/** @} end of Threshold */

/** @} signal_threshold */

/** @} signal_conversion_functions */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MC_MPPS_CONVERSION_FUNCTIONS_H */
