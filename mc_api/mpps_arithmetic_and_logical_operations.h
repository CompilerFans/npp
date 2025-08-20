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
#ifndef MC_MPPS_ARITHMETIC_AND_LOGICAL_OPERATIONS_H
#define MC_MPPS_ARITHMETIC_AND_LOGICAL_OPERATIONS_H
 
/**
 * \file mpps_arithmetic_and_logical_operations.h
 * Signal Arithmetic and Logical Operations.
 */
 
#include "mppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif


/** 
 * \page signal_arithmetic_and_logical_operations Arithmetic and Logical Operations
 * @defgroup signal_arithmetic_and_logical_operations Arithmetic and Logical Operations
 * @ingroup mpps
 * Functions that provide common arithmetic and logical operations.
 * @{
 *
 */

/** 
 * \section signal_arithmetic Arithmetic Operations
 * @defgroup signal_arithmetic Arithmetic Operations
 * The set of arithmetic operations for signal processing available in the library.
 * @{
 *
 */

/** 
 * \section signal_addc AddC
 * @defgroup signal_addc AddC
 * Adds a constant value to each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char in place signal add constant,
 * scale, then clamp to saturated value
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_8u_ISfs_Ctx(Mpp8u nValue, Mpp8u * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char in place signal add constant,
 * scale, then clamp to saturated value
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_8u_ISfs(Mpp8u nValue, Mpp8u * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 8-bit unsigned charvector add constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_8u_Sfs_Ctx(const Mpp8u * pSrc, Mpp8u nValue, Mpp8u * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned charvector add constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_8u_Sfs(const Mpp8u * pSrc, Mpp8u nValue, Mpp8u * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short in place signal add constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_16u_ISfs_Ctx(Mpp16u nValue, Mpp16u * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short in place signal add constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_16u_ISfs(Mpp16u nValue, Mpp16u * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short vector add constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_16u_Sfs_Ctx(const Mpp16u * pSrc, Mpp16u nValue, Mpp16u * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short vector add constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_16u_Sfs(const Mpp16u * pSrc, Mpp16u nValue, Mpp16u * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place  signal add constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_16s_ISfs_Ctx(Mpp16s nValue, Mpp16s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short in place  signal add constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_16s_ISfs(Mpp16s nValue, Mpp16s * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal add constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_16s_Sfs_Ctx(const Mpp16s * pSrc, Mpp16s nValue, Mpp16s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal add constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_16s_Sfs(const Mpp16s * pSrc, Mpp16s nValue, Mpp16s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary)signal add constant, 
 * scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_16sc_ISfs_Ctx(Mpp16sc nValue, Mpp16sc * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary)signal add constant, 
 * scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_16sc_ISfs(Mpp16sc nValue, Mpp16sc * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal add constant,
 * scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_16sc_Sfs_Ctx(const Mpp16sc * pSrc, Mpp16sc nValue, Mpp16sc * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal add constant,
 * scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_16sc_Sfs(const Mpp16sc * pSrc, Mpp16sc nValue, Mpp16sc * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer in place signal add constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_32s_ISfs_Ctx(Mpp32s nValue, Mpp32s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer in place signal add constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_32s_ISfs(Mpp32s nValue, Mpp32s * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integersignal add constant and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_32s_Sfs_Ctx(const Mpp32s * pSrc, Mpp32s nValue, Mpp32s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integersignal add constant and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_32s_Sfs(const Mpp32s * pSrc, Mpp32s nValue, Mpp32s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal
 * add constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_32sc_ISfs_Ctx(Mpp32sc nValue, Mpp32sc * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal
 * add constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_32sc_ISfs(Mpp32sc nValue, Mpp32sc * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary) signal add constant
 * and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_32sc_Sfs_Ctx(const Mpp32sc * pSrc, Mpp32sc nValue, Mpp32sc * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary) signal add constant
 * and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_32sc_Sfs(const Mpp32sc * pSrc, Mpp32sc nValue, Mpp32sc * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit floating point in place signal add constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_32f_I_Ctx(Mpp32f nValue, Mpp32f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point in place signal add constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_32f_I(Mpp32f nValue, Mpp32f * pSrcDst, size_t nLength);

/** 
 * 32-bit floating point signal add constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_32f_Ctx(const Mpp32f * pSrc, Mpp32f nValue, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal add constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_32f(const Mpp32f * pSrc, Mpp32f nValue, Mpp32f * pDst, size_t nLength);

/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
 * place signal add constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_32fc_I_Ctx(Mpp32fc nValue, Mpp32fc * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
 * place signal add constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_32fc_I(Mpp32fc nValue, Mpp32fc * pSrcDst, size_t nLength);

/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
 * add constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_32fc_Ctx(const Mpp32fc * pSrc, Mpp32fc nValue, Mpp32fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point, in place signal add constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength Length of the vectors, number of items.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_32fc(const Mpp32fc * pSrc, Mpp32fc nValue, Mpp32fc * pDst, size_t nLength);

/** 
 * 64-bit floating point, in place signal add constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength Length of the vectors, number of items.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_64f_I_Ctx(Mpp64f nValue, Mpp64f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point, in place signal add constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength Length of the vectors, number of items.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_64f_I(Mpp64f nValue, Mpp64f * pSrcDst, size_t nLength);

/** 
 * 64-bit floating pointsignal add constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_64f_Ctx(const Mpp64f * pSrc, Mpp64f nValue, Mpp64f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating pointsignal add constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_64f(const Mpp64f * pSrc, Mpp64f nValue, Mpp64f * pDst, size_t nLength);

/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
 * place signal add constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_64fc_I_Ctx(Mpp64fc nValue, Mpp64fc * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
 * place signal add constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_64fc_I(Mpp64fc nValue, Mpp64fc * pSrcDst, size_t nLength);

/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
 * add constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_64fc_Ctx(const Mpp64fc * pSrc, Mpp64fc nValue, Mpp64fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
 * add constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddC_64fc(const Mpp64fc * pSrc, Mpp64fc nValue, Mpp64fc * pDst, size_t nLength);

/** @} signal_addc */

/** 
 * \section signal_addproductc AddProductC
 * @defgroup signal_addproductc AddProductC
 * Adds product of a constant and each sample of a source signal to the each sample of destination signal.
 *
 * @{
 *
 */

/** 
 * 32-bit floating point signal add product of signal times constant to destination signal.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddProductC_32f_Ctx(const Mpp32f * pSrc, Mpp32f nValue, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal add product of signal times constant to destination signal.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddProductC_32f(const Mpp32f * pSrc, Mpp32f nValue, Mpp32f * pDst, size_t nLength);

/** @} signal_addproductc */

/** 
 * \section signal_mulc MulC
 * @defgroup signal_mulc MulC
 *
 * Multiplies each sample of a signal by a constant value.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char in place signal times constant,
 * scale, then clamp to saturated value
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_8u_ISfs_Ctx(Mpp8u nValue, Mpp8u * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char in place signal times constant,
 * scale, then clamp to saturated value
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_8u_ISfs(Mpp8u nValue, Mpp8u * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 8-bit unsigned char signal times constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_8u_Sfs_Ctx(const Mpp8u * pSrc, Mpp8u nValue, Mpp8u * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char signal times constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_8u_Sfs(const Mpp8u * pSrc, Mpp8u nValue, Mpp8u * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short in place signal times constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_16u_ISfs_Ctx(Mpp16u nValue, Mpp16u * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short in place signal times constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_16u_ISfs(Mpp16u nValue, Mpp16u * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal times constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_16u_Sfs_Ctx(const Mpp16u * pSrc, Mpp16u nValue, Mpp16u * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short signal times constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_16u_Sfs(const Mpp16u * pSrc, Mpp16u nValue, Mpp16u * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place signal times constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_16s_ISfs_Ctx(Mpp16s nValue, Mpp16s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short in place signal times constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_16s_ISfs(Mpp16s nValue, Mpp16s * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal times constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_16s_Sfs_Ctx(const Mpp16s * pSrc, Mpp16s nValue, Mpp16s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal times constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_16s_Sfs(const Mpp16s * pSrc, Mpp16s nValue, Mpp16s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary)signal times constant, 
 * scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_16sc_ISfs_Ctx(Mpp16sc nValue, Mpp16sc * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary)signal times constant, 
 * scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_16sc_ISfs(Mpp16sc nValue, Mpp16sc * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary)signal times constant,
 * scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_16sc_Sfs_Ctx(const Mpp16sc * pSrc, Mpp16sc nValue, Mpp16sc * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary)signal times constant,
 * scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_16sc_Sfs(const Mpp16sc * pSrc, Mpp16sc nValue, Mpp16sc * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer in place signal times constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_32s_ISfs_Ctx(Mpp32s nValue, Mpp32s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer in place signal times constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_32s_ISfs(Mpp32s nValue, Mpp32s * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal times constant and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_32s_Sfs_Ctx(const Mpp32s * pSrc, Mpp32s nValue, Mpp32s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer signal times constant and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_32s_Sfs(const Mpp32s * pSrc, Mpp32s nValue, Mpp32s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal
 * times constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_32sc_ISfs_Ctx(Mpp32sc nValue, Mpp32sc * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal
 * times constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_32sc_ISfs(Mpp32sc nValue, Mpp32sc * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary) signal times constant
 * and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_32sc_Sfs_Ctx(const Mpp32sc * pSrc, Mpp32sc nValue, Mpp32sc * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary) signal times constant
 * and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_32sc_Sfs(const Mpp32sc * pSrc, Mpp32sc nValue, Mpp32sc * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit floating point in place signal times constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_32f_I_Ctx(Mpp32f nValue, Mpp32f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point in place signal times constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_32f_I(Mpp32f nValue, Mpp32f * pSrcDst, size_t nLength);

/** 
 * 32-bit floating point signal times constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_32f_Ctx(const Mpp32f * pSrc, Mpp32f nValue, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal times constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_32f(const Mpp32f * pSrc, Mpp32f nValue, Mpp32f * pDst, size_t nLength);

/** 
 * 32-bit floating point signal times constant with output converted to 16-bit signed integer.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_Low_32f16s_Ctx(const Mpp32f * pSrc, Mpp32f nValue, Mpp16s * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal times constant with output converted to 16-bit signed integer.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_Low_32f16s(const Mpp32f * pSrc, Mpp32f nValue, Mpp16s * pDst, size_t nLength);

/** 
 * 32-bit floating point signal times constant with output converted to 16-bit signed integer
 * with scaling and saturation of output result.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_32f16s_Sfs_Ctx(const Mpp32f * pSrc, Mpp32f nValue, Mpp16s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal times constant with output converted to 16-bit signed integer
 * with scaling and saturation of output result.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_32f16s_Sfs(const Mpp32f * pSrc, Mpp32f nValue, Mpp16s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
 * place signal times constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_32fc_I_Ctx(Mpp32fc nValue, Mpp32fc * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
 * place signal times constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_32fc_I(Mpp32fc nValue, Mpp32fc * pSrcDst, size_t nLength);

/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
 * times constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_32fc_Ctx(const Mpp32fc * pSrc, Mpp32fc nValue, Mpp32fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
 * times constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_32fc(const Mpp32fc * pSrc, Mpp32fc nValue, Mpp32fc * pDst, size_t nLength);

/** 
 * 64-bit floating point, in place signal times constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength Length of the vectors, number of items.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_64f_I_Ctx(Mpp64f nValue, Mpp64f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point, in place signal times constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength Length of the vectors, number of items.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_64f_I(Mpp64f nValue, Mpp64f * pSrcDst, size_t nLength);

/** 
 * 64-bit floating point signal times constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_64f_Ctx(const Mpp64f * pSrc, Mpp64f nValue, Mpp64f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal times constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_64f(const Mpp64f * pSrc, Mpp64f nValue, Mpp64f * pDst, size_t nLength);

/** 
 * 64-bit floating point signal times constant with in place conversion to 64-bit signed integer
 * and with scaling and saturation of output result.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_64f64s_ISfs_Ctx(Mpp64f nValue, Mpp64s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal times constant with in place conversion to 64-bit signed integer
 * and with scaling and saturation of output result.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_64f64s_ISfs(Mpp64f nValue, Mpp64s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
 * place signal times constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_64fc_I_Ctx(Mpp64fc nValue, Mpp64fc * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
 * place signal times constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_64fc_I(Mpp64fc nValue, Mpp64fc * pSrcDst, size_t nLength);

/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
 * times constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_64fc_Ctx(const Mpp64fc * pSrc, Mpp64fc nValue, Mpp64fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
 * times constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMulC_64fc(const Mpp64fc * pSrc, Mpp64fc nValue, Mpp64fc * pDst, size_t nLength);

/** @} signal_mulc */

/** 
 * \section signal_subc SubC
 * @defgroup signal_subc SubC
 *
 * Subtracts a constant from each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char in place signal subtract constant,
 * scale, then clamp to saturated value
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_8u_ISfs_Ctx(Mpp8u nValue, Mpp8u * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char in place signal subtract constant,
 * scale, then clamp to saturated value
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_8u_ISfs(Mpp8u nValue, Mpp8u * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 8-bit unsigned char signal subtract constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_8u_Sfs_Ctx(const Mpp8u * pSrc, Mpp8u nValue, Mpp8u * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char signal subtract constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_8u_Sfs(const Mpp8u * pSrc, Mpp8u nValue, Mpp8u * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short in place signal subtract constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_16u_ISfs_Ctx(Mpp16u nValue, Mpp16u * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short in place signal subtract constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_16u_ISfs(Mpp16u nValue, Mpp16u * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal subtract constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_16u_Sfs_Ctx(const Mpp16u * pSrc, Mpp16u nValue, Mpp16u * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short signal subtract constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_16u_Sfs(const Mpp16u * pSrc, Mpp16u nValue, Mpp16u * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place signal subtract constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_16s_ISfs_Ctx(Mpp16s nValue, Mpp16s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short in place signal subtract constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_16s_ISfs(Mpp16s nValue, Mpp16s * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal subtract constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_16s_Sfs_Ctx(const Mpp16s * pSrc, Mpp16s nValue, Mpp16s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal subtract constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_16s_Sfs(const Mpp16s * pSrc, Mpp16s nValue, Mpp16s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract constant, 
 * scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_16sc_ISfs_Ctx(Mpp16sc nValue, Mpp16sc * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract constant, 
 * scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_16sc_ISfs(Mpp16sc nValue, Mpp16sc * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract constant,
 * scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_16sc_Sfs_Ctx(const Mpp16sc * pSrc, Mpp16sc nValue, Mpp16sc * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract constant,
 * scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_16sc_Sfs(const Mpp16sc * pSrc, Mpp16sc nValue, Mpp16sc * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer in place signal subtract constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_32s_ISfs_Ctx(Mpp32s nValue, Mpp32s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer in place signal subtract constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_32s_ISfs(Mpp32s nValue, Mpp32s * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal subtract constant and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_32s_Sfs_Ctx(const Mpp32s * pSrc, Mpp32s nValue, Mpp32s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer signal subtract constant and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_32s_Sfs(const Mpp32s * pSrc, Mpp32s nValue, Mpp32s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal
 * subtract constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_32sc_ISfs_Ctx(Mpp32sc nValue, Mpp32sc * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal
 * subtract constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_32sc_ISfs(Mpp32sc nValue, Mpp32sc * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary)signal subtract constant
 * and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_32sc_Sfs_Ctx(const Mpp32sc * pSrc, Mpp32sc nValue, Mpp32sc * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary)signal subtract constant
 * and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_32sc_Sfs(const Mpp32sc * pSrc, Mpp32sc nValue, Mpp32sc * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit floating point in place signal subtract constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_32f_I_Ctx(Mpp32f nValue, Mpp32f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point in place signal subtract constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_32f_I(Mpp32f nValue, Mpp32f * pSrcDst, size_t nLength);

/** 
 * 32-bit floating point signal subtract constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_32f_Ctx(const Mpp32f * pSrc, Mpp32f nValue, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal subtract constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_32f(const Mpp32f * pSrc, Mpp32f nValue, Mpp32f * pDst, size_t nLength);

/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
 * place signal subtract constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_32fc_I_Ctx(Mpp32fc nValue, Mpp32fc * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
 * place signal subtract constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_32fc_I(Mpp32fc nValue, Mpp32fc * pSrcDst, size_t nLength);

/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
 * subtract constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_32fc_Ctx(const Mpp32fc * pSrc, Mpp32fc nValue, Mpp32fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
 * subtract constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_32fc(const Mpp32fc * pSrc, Mpp32fc nValue, Mpp32fc * pDst, size_t nLength);

/** 
 * 64-bit floating point, in place signal subtract constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength Length of the vectors, number of items.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_64f_I_Ctx(Mpp64f nValue, Mpp64f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point, in place signal subtract constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength Length of the vectors, number of items.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_64f_I(Mpp64f nValue, Mpp64f * pSrcDst, size_t nLength);

/** 
 * 64-bit floating point signal subtract constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_64f_Ctx(const Mpp64f * pSrc, Mpp64f nValue, Mpp64f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal subtract constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_64f(const Mpp64f * pSrc, Mpp64f nValue, Mpp64f * pDst, size_t nLength);

/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
 * place signal subtract constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_64fc_I_Ctx(Mpp64fc nValue, Mpp64fc * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
 * place signal subtract constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_64fc_I(Mpp64fc nValue, Mpp64fc * pSrcDst, size_t nLength);

/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
 * subtract constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_64fc_Ctx(const Mpp64fc * pSrc, Mpp64fc nValue, Mpp64fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
 * subtract constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubC_64fc(const Mpp64fc * pSrc, Mpp64fc nValue, Mpp64fc * pDst, size_t nLength);

/** @} signal_subc */

/** 
 * \section signal_subcrev SubCRev
 * @defgroup signal_subcrev SubCRev
 *
 * Subtracts each sample of a signal from a constant.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char in place signal subtract from constant,
 * scale, then clamp to saturated value
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_8u_ISfs_Ctx(Mpp8u nValue, Mpp8u * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char in place signal subtract from constant,
 * scale, then clamp to saturated value
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_8u_ISfs(Mpp8u nValue, Mpp8u * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 8-bit unsigned char signal subtract from constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_8u_Sfs_Ctx(const Mpp8u * pSrc, Mpp8u nValue, Mpp8u * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char signal subtract from constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_8u_Sfs(const Mpp8u * pSrc, Mpp8u nValue, Mpp8u * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short in place signal subtract from constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_16u_ISfs_Ctx(Mpp16u nValue, Mpp16u * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short in place signal subtract from constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_16u_ISfs(Mpp16u nValue, Mpp16u * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal subtract from constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_16u_Sfs_Ctx(const Mpp16u * pSrc, Mpp16u nValue, Mpp16u * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short signal subtract from constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_16u_Sfs(const Mpp16u * pSrc, Mpp16u nValue, Mpp16u * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place signal subtract from constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_16s_ISfs_Ctx(Mpp16s nValue, Mpp16s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short in place signal subtract from constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_16s_ISfs(Mpp16s nValue, Mpp16s * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal subtract from constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_16s_Sfs_Ctx(const Mpp16s * pSrc, Mpp16s nValue, Mpp16s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal subtract from constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_16s_Sfs(const Mpp16s * pSrc, Mpp16s nValue, Mpp16s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract from constant, 
 * scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_16sc_ISfs_Ctx(Mpp16sc nValue, Mpp16sc * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract from constant, 
 * scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_16sc_ISfs(Mpp16sc nValue, Mpp16sc * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract from constant,
 * scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_16sc_Sfs_Ctx(const Mpp16sc * pSrc, Mpp16sc nValue, Mpp16sc * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract from constant,
 * scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_16sc_Sfs(const Mpp16sc * pSrc, Mpp16sc nValue, Mpp16sc * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer in place signal subtract from constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_32s_ISfs_Ctx(Mpp32s nValue, Mpp32s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer in place signal subtract from constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_32s_ISfs(Mpp32s nValue, Mpp32s * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integersignal subtract from constant and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_32s_Sfs_Ctx(const Mpp32s * pSrc, Mpp32s nValue, Mpp32s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integersignal subtract from constant and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_32s_Sfs(const Mpp32s * pSrc, Mpp32s nValue, Mpp32s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal
 * subtract from constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_32sc_ISfs_Ctx(Mpp32sc nValue, Mpp32sc * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal
 * subtract from constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_32sc_ISfs(Mpp32sc nValue, Mpp32sc * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary) signal subtract from constant
 * and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_32sc_Sfs_Ctx(const Mpp32sc * pSrc, Mpp32sc nValue, Mpp32sc * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary) signal subtract from constant
 * and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_32sc_Sfs(const Mpp32sc * pSrc, Mpp32sc nValue, Mpp32sc * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit floating point in place signal subtract from constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_32f_I_Ctx(Mpp32f nValue, Mpp32f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point in place signal subtract from constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_32f_I(Mpp32f nValue, Mpp32f * pSrcDst, size_t nLength);

/** 
 * 32-bit floating point signal subtract from constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_32f_Ctx(const Mpp32f * pSrc, Mpp32f nValue, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal subtract from constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_32f(const Mpp32f * pSrc, Mpp32f nValue, Mpp32f * pDst, size_t nLength);

/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
 * place signal subtract from constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_32fc_I_Ctx(Mpp32fc nValue, Mpp32fc * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
 * place signal subtract from constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_32fc_I(Mpp32fc nValue, Mpp32fc * pSrcDst, size_t nLength);

/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
 * subtract from constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_32fc_Ctx(const Mpp32fc * pSrc, Mpp32fc nValue, Mpp32fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
 * subtract from constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_32fc(const Mpp32fc * pSrc, Mpp32fc nValue, Mpp32fc * pDst, size_t nLength);

/** 
 * 64-bit floating point, in place signal subtract from constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength Length of the vectors, number of items.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_64f_I_Ctx(Mpp64f nValue, Mpp64f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point, in place signal subtract from constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength Length of the vectors, number of items.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_64f_I(Mpp64f nValue, Mpp64f * pSrcDst, size_t nLength);

/** 
 * 64-bit floating point signal subtract from constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_64f_Ctx(const Mpp64f * pSrc, Mpp64f nValue, Mpp64f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal subtract from constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_64f(const Mpp64f * pSrc, Mpp64f nValue, Mpp64f * pDst, size_t nLength);

/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
 * place signal subtract from constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_64fc_I_Ctx(Mpp64fc nValue, Mpp64fc * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
 * place signal subtract from constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_64fc_I(Mpp64fc nValue, Mpp64fc * pSrcDst, size_t nLength);

/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
 * subtract from constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_64fc_Ctx(const Mpp64fc * pSrc, Mpp64fc nValue, Mpp64fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
 * subtract from constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSubCRev_64fc(const Mpp64fc * pSrc, Mpp64fc nValue, Mpp64fc * pDst, size_t nLength);

/** @} signal_subcrev */

/** 
 * \section signal_divc DivC
 * @defgroup signal_divc DivC
 *
 * Divides each sample of a signal by a constant.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char in place signal divided by constant,
 * scale, then clamp to saturated value
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_8u_ISfs_Ctx(Mpp8u nValue, Mpp8u * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char in place signal divided by constant,
 * scale, then clamp to saturated value
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_8u_ISfs(Mpp8u nValue, Mpp8u * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 8-bit unsigned char signal divided by constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_8u_Sfs_Ctx(const Mpp8u * pSrc, Mpp8u nValue, Mpp8u * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char signal divided by constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_8u_Sfs(const Mpp8u * pSrc, Mpp8u nValue, Mpp8u * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short in place signal divided by constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_16u_ISfs_Ctx(Mpp16u nValue, Mpp16u * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short in place signal divided by constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_16u_ISfs(Mpp16u nValue, Mpp16u * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal divided by constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_16u_Sfs_Ctx(const Mpp16u * pSrc, Mpp16u nValue, Mpp16u * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short signal divided by constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_16u_Sfs(const Mpp16u * pSrc, Mpp16u nValue, Mpp16u * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place signal divided by constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_16s_ISfs_Ctx(Mpp16s nValue, Mpp16s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short in place signal divided by constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_16s_ISfs(Mpp16s nValue, Mpp16s * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal divided by constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_16s_Sfs_Ctx(const Mpp16s * pSrc, Mpp16s nValue, Mpp16s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal divided by constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_16s_Sfs(const Mpp16s * pSrc, Mpp16s nValue, Mpp16s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary)signal divided by constant, 
 * scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_16sc_ISfs_Ctx(Mpp16sc nValue, Mpp16sc * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary)signal divided by constant, 
 * scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_16sc_ISfs(Mpp16sc nValue, Mpp16sc * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal divided by constant,
 * scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_16sc_Sfs_Ctx(const Mpp16sc * pSrc, Mpp16sc nValue, Mpp16sc * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal divided by constant,
 * scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_16sc_Sfs(const Mpp16sc * pSrc, Mpp16sc nValue, Mpp16sc * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit floating point in place signal divided by constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_32f_I_Ctx(Mpp32f nValue, Mpp32f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point in place signal divided by constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_32f_I(Mpp32f nValue, Mpp32f * pSrcDst, size_t nLength);

/** 
 * 32-bit floating point signal divided by constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_32f_Ctx(const Mpp32f * pSrc, Mpp32f nValue, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal divided by constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_32f(const Mpp32f * pSrc, Mpp32f nValue, Mpp32f * pDst, size_t nLength);

/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
 * place signal divided by constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_32fc_I_Ctx(Mpp32fc nValue, Mpp32fc * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
 * place signal divided by constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_32fc_I(Mpp32fc nValue, Mpp32fc * pSrcDst, size_t nLength);

/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
 * divided by constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_32fc_Ctx(const Mpp32fc * pSrc, Mpp32fc nValue, Mpp32fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
 * divided by constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_32fc(const Mpp32fc * pSrc, Mpp32fc nValue, Mpp32fc * pDst, size_t nLength);

/** 
 * 64-bit floating point in place signal divided by constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength Length of the vectors, number of items.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_64f_I_Ctx(Mpp64f nValue, Mpp64f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point in place signal divided by constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength Length of the vectors, number of items.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_64f_I(Mpp64f nValue, Mpp64f * pSrcDst, size_t nLength);

/** 
 * 64-bit floating point signal divided by constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_64f_Ctx(const Mpp64f * pSrc, Mpp64f nValue, Mpp64f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal divided by constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_64f(const Mpp64f * pSrc, Mpp64f nValue, Mpp64f * pDst, size_t nLength);

/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
 * place signal divided by constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_64fc_I_Ctx(Mpp64fc nValue, Mpp64fc * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
 * place signal divided by constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_64fc_I(Mpp64fc nValue, Mpp64fc * pSrcDst, size_t nLength);

/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
 * divided by constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_64fc_Ctx(const Mpp64fc * pSrc, Mpp64fc nValue, Mpp64fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
 * divided by constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.

 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivC_64fc(const Mpp64fc * pSrc, Mpp64fc nValue, Mpp64fc * pDst, size_t nLength);

/** @} signal_divc */

/** 
 * \section signal_divcrev DivCRev
 * @defgroup signal_divcrev DivCRev
 *
 * Divides a constant by each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 16-bit unsigned short in place constant divided by signal, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided by each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivCRev_16u_I_Ctx(Mpp16u nValue, Mpp16u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short in place constant divided by signal, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided by each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivCRev_16u_I(Mpp16u nValue, Mpp16u * pSrcDst, size_t nLength);

/** 
 * 16-bit unsigned short signal divided by constant, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivCRev_16u_Ctx(const Mpp16u * pSrc, Mpp16u nValue, Mpp16u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short signal divided by constant, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivCRev_16u(const Mpp16u * pSrc, Mpp16u nValue, Mpp16u * pDst, size_t nLength);

/** 
 * 32-bit floating point in place constant divided by signal.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided by each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivCRev_32f_I_Ctx(Mpp32f nValue, Mpp32f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point in place constant divided by signal.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided by each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivCRev_32f_I(Mpp32f nValue, Mpp32f * pSrcDst, size_t nLength);

/** 
 * 32-bit floating point constant divided by signal.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivCRev_32f_Ctx(const Mpp32f * pSrc, Mpp32f nValue, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point constant divided by signal.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDivCRev_32f(const Mpp32f * pSrc, Mpp32f nValue, Mpp32f * pDst, size_t nLength);

/** @} signal_divcrev */

/** 
 * \section signal_add Add
 * @defgroup signal_add Add
 *
 * Sample by sample addition of two signals.
 *
 * @{
 *
 */

/** 
 * 16-bit signed short signal add signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_16s_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp16s * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal add signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_16s(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp16s * pDst, size_t nLength);

/** 
 * 16-bit unsigned short signal add signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_16u_Ctx(const Mpp16u * pSrc1, const Mpp16u * pSrc2, Mpp16u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short signal add signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_16u(const Mpp16u * pSrc1, const Mpp16u * pSrc2, Mpp16u * pDst, size_t nLength);

/** 
 * 32-bit unsigned int signal add signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_32u_Ctx(const Mpp32u * pSrc1, const Mpp32u * pSrc2, Mpp32u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned int signal add signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_32u(const Mpp32u * pSrc1, const Mpp32u * pSrc2, Mpp32u * pDst, size_t nLength);

/** 
 * 32-bit floating point signal add signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_32f_Ctx(const Mpp32f * pSrc1, const Mpp32f * pSrc2, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal add signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_32f(const Mpp32f * pSrc1, const Mpp32f * pSrc2, Mpp32f * pDst, size_t nLength);

/** 
 * 64-bit floating point signal add signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_64f_Ctx(const Mpp64f * pSrc1, const Mpp64f * pSrc2, Mpp64f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal add signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_64f(const Mpp64f * pSrc1, const Mpp64f * pSrc2, Mpp64f * pDst, size_t nLength);

/** 
 * 32-bit complex floating point signal add signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_32fc_Ctx(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, Mpp32fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit complex floating point signal add signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_32fc(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, Mpp32fc * pDst, size_t nLength);

/** 
 * 64-bit complex floating point signal add signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_64fc_Ctx(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, Mpp64fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit complex floating point signal add signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_64fc(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, Mpp64fc * pDst, size_t nLength);

/** 
 * 8-bit unsigned char signal add signal with 16-bit unsigned result,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_8u16u_Ctx(const Mpp8u * pSrc1, const Mpp8u * pSrc2, Mpp16u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char signal add signal with 16-bit unsigned result,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_8u16u(const Mpp8u * pSrc1, const Mpp8u * pSrc2, Mpp16u * pDst, size_t nLength);

/** 
 * 16-bit signed short signal add signal with 32-bit floating point result,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_16s32f_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal add signal with 32-bit floating point result,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_16s32f(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp32f * pDst, size_t nLength);

/** 
 * 8-bit unsigned char add signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_8u_Sfs_Ctx(const Mpp8u * pSrc1, const Mpp8u * pSrc2, Mpp8u * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char add signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_8u_Sfs(const Mpp8u * pSrc1, const Mpp8u * pSrc2, Mpp8u * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short add signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_16u_Sfs_Ctx(const Mpp16u * pSrc1, const Mpp16u * pSrc2, Mpp16u * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short add signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_16u_Sfs(const Mpp16u * pSrc1, const Mpp16u * pSrc2, Mpp16u * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short add signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_16s_Sfs_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp16s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short add signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_16s_Sfs(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp16s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer add signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_32s_Sfs_Ctx(const Mpp32s * pSrc1, const Mpp32s * pSrc2, Mpp32s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer add signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_32s_Sfs(const Mpp32s * pSrc1, const Mpp32s * pSrc2, Mpp32s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 64-bit signed integer add signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_64s_Sfs_Ctx(const Mpp64s * pSrc1, const Mpp64s * pSrc2, Mpp64s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 64-bit signed integer add signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_64s_Sfs(const Mpp64s * pSrc1, const Mpp64s * pSrc2, Mpp64s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed complex short add signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_16sc_Sfs_Ctx(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, Mpp16sc * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed complex short add signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_16sc_Sfs(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, Mpp16sc * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed complex integer add signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_32sc_Sfs_Ctx(const Mpp32sc * pSrc1, const Mpp32sc * pSrc2, Mpp32sc * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed complex integer add signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_32sc_Sfs(const Mpp32sc * pSrc1, const Mpp32sc * pSrc2, Mpp32sc * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place signal add signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_16s_I_Ctx(const Mpp16s * pSrc, Mpp16s * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short in place signal add signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_16s_I(const Mpp16s * pSrc, Mpp16s * pSrcDst, size_t nLength);

/** 
 * 32-bit floating point in place signal add signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_32f_I_Ctx(const Mpp32f * pSrc, Mpp32f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point in place signal add signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_32f_I(const Mpp32f * pSrc, Mpp32f * pSrcDst, size_t nLength);

/** 
 * 64-bit floating point in place signal add signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_64f_I_Ctx(const Mpp64f * pSrc, Mpp64f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point in place signal add signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_64f_I(const Mpp64f * pSrc, Mpp64f * pSrcDst, size_t nLength);

/** 
 * 32-bit complex floating point in place signal add signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_32fc_I_Ctx(const Mpp32fc * pSrc, Mpp32fc * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit complex floating point in place signal add signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_32fc_I(const Mpp32fc * pSrc, Mpp32fc * pSrcDst, size_t nLength);

/** 
 * 64-bit complex floating point in place signal add signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_64fc_I_Ctx(const Mpp64fc * pSrc, Mpp64fc * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit complex floating point in place signal add signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_64fc_I(const Mpp64fc * pSrc, Mpp64fc * pSrcDst, size_t nLength);

/** 
 * 16/32-bit signed short in place signal add signal with 32-bit signed integer results,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_16s32s_I_Ctx(const Mpp16s * pSrc, Mpp32s * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16/32-bit signed short in place signal add signal with 32-bit signed integer results,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_16s32s_I(const Mpp16s * pSrc, Mpp32s * pSrcDst, size_t nLength);

/** 
 * 8-bit unsigned char in place signal add signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_8u_ISfs_Ctx(const Mpp8u * pSrc, Mpp8u * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char in place signal add signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_8u_ISfs(const Mpp8u * pSrc, Mpp8u * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short in place signal add signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_16u_ISfs_Ctx(const Mpp16u * pSrc, Mpp16u * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short in place signal add signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_16u_ISfs(const Mpp16u * pSrc, Mpp16u * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place signal add signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_16s_ISfs_Ctx(const Mpp16s * pSrc, Mpp16s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short in place signal add signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_16s_ISfs(const Mpp16s * pSrc, Mpp16s * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer in place signal add signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_32s_ISfs_Ctx(const Mpp32s * pSrc, Mpp32s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer in place signal add signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_32s_ISfs(const Mpp32s * pSrc, Mpp32s * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit complex signed short in place signal add signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_16sc_ISfs_Ctx(const Mpp16sc * pSrc, Mpp16sc * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit complex signed short in place signal add signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_16sc_ISfs(const Mpp16sc * pSrc, Mpp16sc * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit complex signed integer in place signal add signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_32sc_ISfs_Ctx(const Mpp32sc * pSrc, Mpp32sc * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit complex signed integer in place signal add signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAdd_32sc_ISfs(const Mpp32sc * pSrc, Mpp32sc * pSrcDst, size_t nLength, int nScaleFactor);

/** @} signal_add */

/** 
 * \section signal_addproduct AddProduct
 * @defgroup signal_addproduct AddProduct
 *
 * Adds sample by sample product of two signals to the destination signal.
 *
 * @{
 *
 */

/** 
 * 32-bit floating point signal add product of source signal times destination signal to destination signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddProduct_32f_Ctx(const Mpp32f * pSrc1, const Mpp32f * pSrc2, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal add product of source signal times destination signal to destination signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddProduct_32f(const Mpp32f * pSrc1, const Mpp32f * pSrc2, Mpp32f * pDst, size_t nLength);

/** 
 * 64-bit floating point signal add product of source signal times destination signal to destination signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddProduct_64f_Ctx(const Mpp64f * pSrc1, const Mpp64f * pSrc2, Mpp64f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal add product of source signal times destination signal to destination signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddProduct_64f(const Mpp64f * pSrc1, const Mpp64f * pSrc2, Mpp64f * pDst, size_t nLength);

/** 
 * 32-bit complex floating point signal add product of source signal times destination signal to destination signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddProduct_32fc_Ctx(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, Mpp32fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit complex floating point signal add product of source signal times destination signal to destination signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddProduct_32fc(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, Mpp32fc * pDst, size_t nLength);

/** 
 * 64-bit complex floating point signal add product of source signal times destination signal to destination signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddProduct_64fc_Ctx(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, Mpp64fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit complex floating point signal add product of source signal times destination signal to destination signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddProduct_64fc(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, Mpp64fc * pDst, size_t nLength);

/** 
 * 16-bit signed short signal add product of source signal1 times source signal2 to destination signal,
 * with scaling, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddProduct_16s_Sfs_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp16s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal add product of source signal1 times source signal2 to destination signal,
 * with scaling, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddProduct_16s_Sfs(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp16s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed short signal add product of source signal1 times source signal2 to destination signal,
 * with scaling, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddProduct_32s_Sfs_Ctx(const Mpp32s * pSrc1, const Mpp32s * pSrc2, Mpp32s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed short signal add product of source signal1 times source signal2 to destination signal,
 * with scaling, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddProduct_32s_Sfs(const Mpp32s * pSrc1, const Mpp32s * pSrc2, Mpp32s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal add product of source signal1 times source signal2 to 32-bit signed integer destination signal,
 * with scaling, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddProduct_16s32s_Sfs_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp32s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal add product of source signal1 times source signal2 to 32-bit signed integer destination signal,
 * with scaling, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAddProduct_16s32s_Sfs(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp32s * pDst, size_t nLength, int nScaleFactor);

/** @} signal_addproduct */

/** 
 * \section signal_mul Mul
 * @defgroup signal_mul Mul
 *
 * Sample by sample multiplication the samples of two signals.
 *
 * @{
 *
 */

/** 
 * 16-bit signed short signal times signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_16s_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp16s * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal times signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_16s(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp16s * pDst, size_t nLength);

/** 
 * 32-bit floating point signal times signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32f_Ctx(const Mpp32f * pSrc1, const Mpp32f * pSrc2, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal times signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32f(const Mpp32f * pSrc1, const Mpp32f * pSrc2, Mpp32f * pDst, size_t nLength);

/** 
 * 64-bit floating point signal times signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_64f_Ctx(const Mpp64f * pSrc1, const Mpp64f * pSrc2, Mpp64f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal times signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_64f(const Mpp64f * pSrc1, const Mpp64f * pSrc2, Mpp64f * pDst, size_t nLength);

/** 
 * 32-bit complex floating point signal times signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32fc_Ctx(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, Mpp32fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit complex floating point signal times signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32fc(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, Mpp32fc * pDst, size_t nLength);

/** 
 * 64-bit complex floating point signal times signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_64fc_Ctx(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, Mpp64fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit complex floating point signal times signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_64fc(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, Mpp64fc * pDst, size_t nLength);

/** 
 * 8-bit unsigned char signal times signal with 16-bit unsigned result,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_8u16u_Ctx(const Mpp8u * pSrc1, const Mpp8u * pSrc2, Mpp16u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char signal times signal with 16-bit unsigned result,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_8u16u(const Mpp8u * pSrc1, const Mpp8u * pSrc2, Mpp16u * pDst, size_t nLength);

/** 
 * 16-bit signed short signal times signal with 32-bit floating point result,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_16s32f_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal times signal with 32-bit floating point result,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_16s32f(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp32f * pDst, size_t nLength);

/** 
 * 32-bit floating point signal times 32-bit complex floating point signal with complex 32-bit floating point result,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32f32fc_Ctx(const Mpp32f * pSrc1, const Mpp32fc * pSrc2, Mpp32fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal times 32-bit complex floating point signal with complex 32-bit floating point result,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32f32fc(const Mpp32f * pSrc1, const Mpp32fc * pSrc2, Mpp32fc * pDst, size_t nLength);

/** 
 * 8-bit unsigned char signal times signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_8u_Sfs_Ctx(const Mpp8u * pSrc1, const Mpp8u * pSrc2, Mpp8u * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char signal times signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_8u_Sfs(const Mpp8u * pSrc1, const Mpp8u * pSrc2, Mpp8u * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal time signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_16u_Sfs_Ctx(const Mpp16u * pSrc1, const Mpp16u * pSrc2, Mpp16u * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short signal time signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_16u_Sfs(const Mpp16u * pSrc1, const Mpp16u * pSrc2, Mpp16u * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal times signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_16s_Sfs_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp16s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal times signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_16s_Sfs(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp16s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal times signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32s_Sfs_Ctx(const Mpp32s * pSrc1, const Mpp32s * pSrc2, Mpp32s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer signal times signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32s_Sfs(const Mpp32s * pSrc1, const Mpp32s * pSrc2, Mpp32s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed complex short signal times signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_16sc_Sfs_Ctx(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, Mpp16sc * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed complex short signal times signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_16sc_Sfs(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, Mpp16sc * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed complex integer signal times signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32sc_Sfs_Ctx(const Mpp32sc * pSrc1, const Mpp32sc * pSrc2, Mpp32sc * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed complex integer signal times signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32sc_Sfs(const Mpp32sc * pSrc1, const Mpp32sc * pSrc2, Mpp32sc * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal times 16-bit signed short signal, scale, then clamp to 16-bit signed saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_16u16s_Sfs_Ctx(const Mpp16u * pSrc1, const Mpp16s * pSrc2, Mpp16s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short signal times 16-bit signed short signal, scale, then clamp to 16-bit signed saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_16u16s_Sfs(const Mpp16u * pSrc1, const Mpp16s * pSrc2, Mpp16s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal times signal, scale, then clamp to 32-bit signed saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_16s32s_Sfs_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp32s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal times signal, scale, then clamp to 32-bit signed saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_16s32s_Sfs(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp32s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal times 32-bit complex signed integer signal, scale, then clamp to 32-bit complex integer saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32s32sc_Sfs_Ctx(const Mpp32s * pSrc1, const Mpp32sc * pSrc2, Mpp32sc * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer signal times 32-bit complex signed integer signal, scale, then clamp to 32-bit complex integer saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32s32sc_Sfs(const Mpp32s * pSrc1, const Mpp32sc * pSrc2, Mpp32sc * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal times signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_Low_32s_Sfs_Ctx(const Mpp32s * pSrc1, const Mpp32s * pSrc2, Mpp32s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer signal times signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_Low_32s_Sfs(const Mpp32s * pSrc1, const Mpp32s * pSrc2, Mpp32s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place signal times signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_16s_I_Ctx(const Mpp16s * pSrc, Mpp16s * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short in place signal times signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_16s_I(const Mpp16s * pSrc, Mpp16s * pSrcDst, size_t nLength);

/** 
 * 32-bit floating point in place signal times signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32f_I_Ctx(const Mpp32f * pSrc, Mpp32f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point in place signal times signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32f_I(const Mpp32f * pSrc, Mpp32f * pSrcDst, size_t nLength);

/** 
 * 64-bit floating point in place signal times signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_64f_I_Ctx(const Mpp64f * pSrc, Mpp64f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point in place signal times signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_64f_I(const Mpp64f * pSrc, Mpp64f * pSrcDst, size_t nLength);

/** 
 * 32-bit complex floating point in place signal times signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32fc_I_Ctx(const Mpp32fc * pSrc, Mpp32fc * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit complex floating point in place signal times signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32fc_I(const Mpp32fc * pSrc, Mpp32fc * pSrcDst, size_t nLength);

/** 
 * 64-bit complex floating point in place signal times signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_64fc_I_Ctx(const Mpp64fc * pSrc, Mpp64fc * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit complex floating point in place signal times signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_64fc_I(const Mpp64fc * pSrc, Mpp64fc * pSrcDst, size_t nLength);

/** 
 * 32-bit complex floating point in place signal times 32-bit floating point signal,
 * then clamp to 32-bit complex floating point saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32f32fc_I_Ctx(const Mpp32f * pSrc, Mpp32fc * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit complex floating point in place signal times 32-bit floating point signal,
 * then clamp to 32-bit complex floating point saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32f32fc_I(const Mpp32f * pSrc, Mpp32fc * pSrcDst, size_t nLength);

/** 
 * 8-bit unsigned char in place signal times signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_8u_ISfs_Ctx(const Mpp8u * pSrc, Mpp8u * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char in place signal times signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_8u_ISfs(const Mpp8u * pSrc, Mpp8u * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short in place signal times signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_16u_ISfs_Ctx(const Mpp16u * pSrc, Mpp16u * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short in place signal times signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_16u_ISfs(const Mpp16u * pSrc, Mpp16u * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place signal times signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_16s_ISfs_Ctx(const Mpp16s * pSrc, Mpp16s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short in place signal times signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_16s_ISfs(const Mpp16s * pSrc, Mpp16s * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer in place signal times signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32s_ISfs_Ctx(const Mpp32s * pSrc, Mpp32s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer in place signal times signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32s_ISfs(const Mpp32s * pSrc, Mpp32s * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit complex signed short in place signal times signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_16sc_ISfs_Ctx(const Mpp16sc * pSrc, Mpp16sc * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit complex signed short in place signal times signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_16sc_ISfs(const Mpp16sc * pSrc, Mpp16sc * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit complex signed integer in place signal times signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32sc_ISfs_Ctx(const Mpp32sc * pSrc, Mpp32sc * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit complex signed integer in place signal times signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32sc_ISfs(const Mpp32sc * pSrc, Mpp32sc * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit complex signed integer in place signal times 32-bit signed integer signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification. 
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32s32sc_ISfs_Ctx(const Mpp32s * pSrc, Mpp32sc * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit complex signed integer in place signal times 32-bit signed integer signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification. 
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMul_32s32sc_ISfs(const Mpp32s * pSrc, Mpp32sc * pSrcDst, size_t nLength, int nScaleFactor);

/** @} signal_mul */

/** 
 * \section signal_sub Sub
 * @defgroup signal_sub Sub
 *
 * Sample by sample subtraction of the samples of two signals.
 *
 * @{
 *
 */

/** 
 * 16-bit signed short signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_16s_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp16s * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_16s(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp16s * pDst, size_t nLength);

/** 
 * 32-bit floating point signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_32f_Ctx(const Mpp32f * pSrc1, const Mpp32f * pSrc2, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_32f(const Mpp32f * pSrc1, const Mpp32f * pSrc2, Mpp32f * pDst, size_t nLength);

/** 
 * 64-bit floating point signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_64f_Ctx(const Mpp64f * pSrc1, const Mpp64f * pSrc2, Mpp64f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_64f(const Mpp64f * pSrc1, const Mpp64f * pSrc2, Mpp64f * pDst, size_t nLength);

/** 
 * 32-bit complex floating point signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_32fc_Ctx(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, Mpp32fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit complex floating point signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_32fc(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, Mpp32fc * pDst, size_t nLength);

/** 
 * 64-bit complex floating point signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_64fc_Ctx(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, Mpp64fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit complex floating point signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_64fc(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, Mpp64fc * pDst, size_t nLength);

/** 
 * 16-bit signed short signal subtract 16-bit signed short signal,
 * then clamp and convert to 32-bit floating point saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_16s32f_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal subtract 16-bit signed short signal,
 * then clamp and convert to 32-bit floating point saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_16s32f(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp32f * pDst, size_t nLength);

/** 
 * 8-bit unsigned char signal subtract signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_8u_Sfs_Ctx(const Mpp8u * pSrc1, const Mpp8u * pSrc2, Mpp8u * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char signal subtract signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_8u_Sfs(const Mpp8u * pSrc1, const Mpp8u * pSrc2, Mpp8u * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal subtract signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_16u_Sfs_Ctx(const Mpp16u * pSrc1, const Mpp16u * pSrc2, Mpp16u * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short signal subtract signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_16u_Sfs(const Mpp16u * pSrc1, const Mpp16u * pSrc2, Mpp16u * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal subtract signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_16s_Sfs_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp16s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal subtract signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_16s_Sfs(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp16s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal subtract signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_32s_Sfs_Ctx(const Mpp32s * pSrc1, const Mpp32s * pSrc2, Mpp32s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer signal subtract signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_32s_Sfs(const Mpp32s * pSrc1, const Mpp32s * pSrc2, Mpp32s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed complex short signal subtract signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_16sc_Sfs_Ctx(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, Mpp16sc * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed complex short signal subtract signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_16sc_Sfs(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, Mpp16sc * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed complex integer signal subtract signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_32sc_Sfs_Ctx(const Mpp32sc * pSrc1, const Mpp32sc * pSrc2, Mpp32sc * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed complex integer signal subtract signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_32sc_Sfs(const Mpp32sc * pSrc1, const Mpp32sc * pSrc2, Mpp32sc * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_16s_I_Ctx(const Mpp16s * pSrc, Mpp16s * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short in place signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_16s_I(const Mpp16s * pSrc, Mpp16s * pSrcDst, size_t nLength);

/** 
 * 32-bit floating point in place signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_32f_I_Ctx(const Mpp32f * pSrc, Mpp32f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point in place signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_32f_I(const Mpp32f * pSrc, Mpp32f * pSrcDst, size_t nLength);

/** 
 * 64-bit floating point in place signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_64f_I_Ctx(const Mpp64f * pSrc, Mpp64f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point in place signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_64f_I(const Mpp64f * pSrc, Mpp64f * pSrcDst, size_t nLength);

/** 
 * 32-bit complex floating point in place signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_32fc_I_Ctx(const Mpp32fc * pSrc, Mpp32fc * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit complex floating point in place signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_32fc_I(const Mpp32fc * pSrc, Mpp32fc * pSrcDst, size_t nLength);

/** 
 * 64-bit complex floating point in place signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_64fc_I_Ctx(const Mpp64fc * pSrc, Mpp64fc * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit complex floating point in place signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_64fc_I(const Mpp64fc * pSrc, Mpp64fc * pSrcDst, size_t nLength);

/** 
 * 8-bit unsigned char in place signal subtract signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_8u_ISfs_Ctx(const Mpp8u * pSrc, Mpp8u * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char in place signal subtract signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_8u_ISfs(const Mpp8u * pSrc, Mpp8u * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short in place signal subtract signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_16u_ISfs_Ctx(const Mpp16u * pSrc, Mpp16u * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short in place signal subtract signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_16u_ISfs(const Mpp16u * pSrc, Mpp16u * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place signal subtract signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_16s_ISfs_Ctx(const Mpp16s * pSrc, Mpp16s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short in place signal subtract signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_16s_ISfs(const Mpp16s * pSrc, Mpp16s * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer in place signal subtract signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_32s_ISfs_Ctx(const Mpp32s * pSrc, Mpp32s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer in place signal subtract signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_32s_ISfs(const Mpp32s * pSrc, Mpp32s * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit complex signed short in place signal subtract signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_16sc_ISfs_Ctx(const Mpp16sc * pSrc, Mpp16sc * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit complex signed short in place signal subtract signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_16sc_ISfs(const Mpp16sc * pSrc, Mpp16sc * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit complex signed integer in place signal subtract signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_32sc_ISfs_Ctx(const Mpp32sc * pSrc, Mpp32sc * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit complex signed integer in place signal subtract signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSub_32sc_ISfs(const Mpp32sc * pSrc, Mpp32sc * pSrcDst, size_t nLength, int nScaleFactor);

/** @} signal_sub */

/**
 * \section signal_div Div
 * @defgroup signal_div Div
 *
 * Sample by sample division of the samples of two signals.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char signal divide signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_8u_Sfs_Ctx(const Mpp8u * pSrc1, const Mpp8u * pSrc2, Mpp8u * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char signal divide signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_8u_Sfs(const Mpp8u * pSrc1, const Mpp8u * pSrc2, Mpp8u * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal divide signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_16u_Sfs_Ctx(const Mpp16u * pSrc1, const Mpp16u * pSrc2, Mpp16u * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short signal divide signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_16u_Sfs(const Mpp16u * pSrc1, const Mpp16u * pSrc2, Mpp16u * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal divide signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_16s_Sfs_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp16s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal divide signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_16s_Sfs(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp16s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal divide signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_32s_Sfs_Ctx(const Mpp32s * pSrc1, const Mpp32s * pSrc2, Mpp32s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer signal divide signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_32s_Sfs(const Mpp32s * pSrc1, const Mpp32s * pSrc2, Mpp32s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed complex short signal divide signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_16sc_Sfs_Ctx(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, Mpp16sc * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed complex short signal divide signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_16sc_Sfs(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, Mpp16sc * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal divided by 16-bit signed short signal, scale, then clamp to 16-bit signed short saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_32s16s_Sfs_Ctx(const Mpp16s * pSrc1, const Mpp32s * pSrc2, Mpp16s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer signal divided by 16-bit signed short signal, scale, then clamp to 16-bit signed short saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_32s16s_Sfs(const Mpp16s * pSrc1, const Mpp32s * pSrc2, Mpp16s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit floating point signal divide signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_32f_Ctx(const Mpp32f * pSrc1, const Mpp32f * pSrc2, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal divide signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_32f(const Mpp32f * pSrc1, const Mpp32f * pSrc2, Mpp32f * pDst, size_t nLength);

/** 
 * 64-bit floating point signal divide signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_64f_Ctx(const Mpp64f * pSrc1, const Mpp64f * pSrc2, Mpp64f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal divide signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_64f(const Mpp64f * pSrc1, const Mpp64f * pSrc2, Mpp64f * pDst, size_t nLength);

/** 
 * 32-bit complex floating point signal divide signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_32fc_Ctx(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, Mpp32fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit complex floating point signal divide signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_32fc(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, Mpp32fc * pDst, size_t nLength);

/** 
 * 64-bit complex floating point signal divide signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_64fc_Ctx(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, Mpp64fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit complex floating point signal divide signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_64fc(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, Mpp64fc * pDst, size_t nLength);

/** 
 * 8-bit unsigned char in place signal divide signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_8u_ISfs_Ctx(const Mpp8u * pSrc, Mpp8u * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char in place signal divide signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_8u_ISfs(const Mpp8u * pSrc, Mpp8u * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short in place signal divide signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_16u_ISfs_Ctx(const Mpp16u * pSrc, Mpp16u * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short in place signal divide signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_16u_ISfs(const Mpp16u * pSrc, Mpp16u * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place signal divide signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_16s_ISfs_Ctx(const Mpp16s * pSrc, Mpp16s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short in place signal divide signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_16s_ISfs(const Mpp16s * pSrc, Mpp16s * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit complex signed short in place signal divide signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_16sc_ISfs_Ctx(const Mpp16sc * pSrc, Mpp16sc * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit complex signed short in place signal divide signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_16sc_ISfs(const Mpp16sc * pSrc, Mpp16sc * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer in place signal divide signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_32s_ISfs_Ctx(const Mpp32s * pSrc, Mpp32s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer in place signal divide signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_32s_ISfs(const Mpp32s * pSrc, Mpp32s * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit floating point in place signal divide signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_32f_I_Ctx(const Mpp32f * pSrc, Mpp32f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point in place signal divide signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_32f_I(const Mpp32f * pSrc, Mpp32f * pSrcDst, size_t nLength);

/** 
 * 64-bit floating point in place signal divide signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_64f_I_Ctx(const Mpp64f * pSrc, Mpp64f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point in place signal divide signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_64f_I(const Mpp64f * pSrc, Mpp64f * pSrcDst, size_t nLength);

/** 
 * 32-bit complex floating point in place signal divide signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_32fc_I_Ctx(const Mpp32fc * pSrc, Mpp32fc * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit complex floating point in place signal divide signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_32fc_I(const Mpp32fc * pSrc, Mpp32fc * pSrcDst, size_t nLength);

/** 
 * 64-bit complex floating point in place signal divide signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_64fc_I_Ctx(const Mpp64fc * pSrc, Mpp64fc * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit complex floating point in place signal divide signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_64fc_I(const Mpp64fc * pSrc, Mpp64fc * pSrcDst, size_t nLength);

/** @} signal_div */

/** 
 * \section signal_divround Div_Round
 * @defgroup signal_divround Div_Round
 *
 * Sample by sample division of the samples of two signals with rounding.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char signal divide signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nRndMode various rounding modes.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_Round_8u_Sfs_Ctx(const Mpp8u * pSrc1, const Mpp8u * pSrc2, Mpp8u * pDst, size_t nLength, MppRoundMode nRndMode, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char signal divide signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nRndMode various rounding modes.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_Round_8u_Sfs(const Mpp8u * pSrc1, const Mpp8u * pSrc2, Mpp8u * pDst, size_t nLength, MppRoundMode nRndMode, int nScaleFactor);

/** 
 * 16-bit unsigned short signal divide signal, scale, round, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nRndMode various rounding modes.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_Round_16u_Sfs_Ctx(const Mpp16u * pSrc1, const Mpp16u * pSrc2, Mpp16u * pDst, size_t nLength, MppRoundMode nRndMode, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short signal divide signal, scale, round, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nRndMode various rounding modes.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_Round_16u_Sfs(const Mpp16u * pSrc1, const Mpp16u * pSrc2, Mpp16u * pDst, size_t nLength, MppRoundMode nRndMode, int nScaleFactor);

/** 
 * 16-bit signed short signal divide signal, scale, round, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nRndMode various rounding modes.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_Round_16s_Sfs_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp16s * pDst, size_t nLength, MppRoundMode nRndMode, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal divide signal, scale, round, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nRndMode various rounding modes.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_Round_16s_Sfs(const Mpp16s * pSrc1, const Mpp16s * pSrc2, Mpp16s * pDst, size_t nLength, MppRoundMode nRndMode, int nScaleFactor);

/** 
 * 8-bit unsigned char in place signal divide signal, with scaling, rounding
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nRndMode various rounding modes.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_Round_8u_ISfs_Ctx(const Mpp8u * pSrc, Mpp8u * pSrcDst, size_t nLength, MppRoundMode nRndMode, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char in place signal divide signal, with scaling, rounding
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nRndMode various rounding modes.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_Round_8u_ISfs(const Mpp8u * pSrc, Mpp8u * pSrcDst, size_t nLength, MppRoundMode nRndMode, int nScaleFactor);

/** 
 * 16-bit unsigned short in place signal divide signal, with scaling, rounding
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nRndMode various rounding modes.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_Round_16u_ISfs_Ctx(const Mpp16u * pSrc, Mpp16u * pSrcDst, size_t nLength, MppRoundMode nRndMode, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short in place signal divide signal, with scaling, rounding
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nRndMode various rounding modes.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_Round_16u_ISfs(const Mpp16u * pSrc, Mpp16u * pSrcDst, size_t nLength, MppRoundMode nRndMode, int nScaleFactor);

/** 
 * 16-bit signed short in place signal divide signal, with scaling, rounding
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nRndMode various rounding modes.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_Round_16s_ISfs_Ctx(const Mpp16s * pSrc, Mpp16s * pSrcDst, size_t nLength, MppRoundMode nRndMode, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short in place signal divide signal, with scaling, rounding
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nRndMode various rounding modes.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDiv_Round_16s_ISfs(const Mpp16s * pSrc, Mpp16s * pSrcDst, size_t nLength, MppRoundMode nRndMode, int nScaleFactor);

/** @} signal_divround */

/** 
 * \section signal_abs Abs
 * @defgroup signal_abs Abs
 *
 * Absolute value of each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 16-bit signed short signal absolute value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAbs_16s_Ctx(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal absolute value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAbs_16s(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength);

/** 
 * 32-bit signed integer signal absolute value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAbs_32s_Ctx(const Mpp32s * pSrc, Mpp32s * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer signal absolute value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAbs_32s(const Mpp32s * pSrc, Mpp32s * pDst, size_t nLength);

/** 
 * 32-bit floating point signal absolute value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAbs_32f_Ctx(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal absolute value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAbs_32f(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength);

/** 
 * 64-bit floating point signal absolute value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAbs_64f_Ctx(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal absolute value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAbs_64f(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength);

/** 
 * 16-bit signed short signal absolute value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAbs_16s_I_Ctx(Mpp16s * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal absolute value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAbs_16s_I(Mpp16s * pSrcDst, size_t nLength);

/** 
 * 32-bit signed integer signal absolute value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAbs_32s_I_Ctx(Mpp32s * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer signal absolute value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAbs_32s_I(Mpp32s * pSrcDst, size_t nLength);

/** 
 * 32-bit floating point signal absolute value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAbs_32f_I_Ctx(Mpp32f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal absolute value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAbs_32f_I(Mpp32f * pSrcDst, size_t nLength);

/** 
 * 64-bit floating point signal absolute value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAbs_64f_I_Ctx(Mpp64f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal absolute value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAbs_64f_I(Mpp64f * pSrcDst, size_t nLength);

/** @} signal_abs */

/** 
 * \section signal_square Sqr
 * @defgroup signal_square Sqr
 *
 * Squares each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 32-bit floating point signal squared.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_32f_Ctx(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal squared.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_32f(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength);

/** 
 * 64-bit floating point signal squared.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_64f_Ctx(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal squared.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_64f(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength);

/** 
 * 32-bit complex floating point signal squared.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_32fc_Ctx(const Mpp32fc * pSrc, Mpp32fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit complex floating point signal squared.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_32fc(const Mpp32fc * pSrc, Mpp32fc * pDst, size_t nLength);

/** 
 * 64-bit complex floating point signal squared.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_64fc_Ctx(const Mpp64fc * pSrc, Mpp64fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit complex floating point signal squared.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_64fc(const Mpp64fc * pSrc, Mpp64fc * pDst, size_t nLength);

/** 
 * 32-bit floating point signal squared.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_32f_I_Ctx(Mpp32f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal squared.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_32f_I(Mpp32f * pSrcDst, size_t nLength);

/** 
 * 64-bit floating point signal squared.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_64f_I_Ctx(Mpp64f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal squared.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_64f_I(Mpp64f * pSrcDst, size_t nLength);

/** 
 * 32-bit complex floating point signal squared.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_32fc_I_Ctx(Mpp32fc * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit complex floating point signal squared.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_32fc_I(Mpp32fc * pSrcDst, size_t nLength);

/** 
 * 64-bit complex floating point signal squared.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_64fc_I_Ctx(Mpp64fc * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit complex floating point signal squared.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_64fc_I(Mpp64fc * pSrcDst, size_t nLength);

/** 
 * 8-bit unsigned char signal squared, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_8u_Sfs_Ctx(const Mpp8u * pSrc, Mpp8u * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char signal squared, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_8u_Sfs(const Mpp8u * pSrc, Mpp8u * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal squared, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_16u_Sfs_Ctx(const Mpp16u * pSrc, Mpp16u * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short signal squared, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_16u_Sfs(const Mpp16u * pSrc, Mpp16u * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal squared, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_16s_Sfs_Ctx(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal squared, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_16s_Sfs(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit complex signed short signal squared, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_16sc_Sfs_Ctx(const Mpp16sc * pSrc, Mpp16sc * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit complex signed short signal squared, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_16sc_Sfs(const Mpp16sc * pSrc, Mpp16sc * pDst, size_t nLength, int nScaleFactor);

/** 
 * 8-bit unsigned char signal squared, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_8u_ISfs_Ctx(Mpp8u * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char signal squared, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_8u_ISfs(Mpp8u * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal squared, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_16u_ISfs_Ctx(Mpp16u * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short signal squared, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_16u_ISfs(Mpp16u * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal squared, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_16s_ISfs_Ctx(Mpp16s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal squared, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_16s_ISfs(Mpp16s * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit complex signed short signal squared, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_16sc_ISfs_Ctx(Mpp16sc * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit complex signed short signal squared, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqr_16sc_ISfs(Mpp16sc * pSrcDst, size_t nLength, int nScaleFactor);

/** @} signal_square */

/** 
 * \section signal_sqrt Sqrt
 * @defgroup signal_sqrt Sqrt
 *
 * Square root of each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 32-bit floating point signal square root.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_32f_Ctx(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal square root.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_32f(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength);

/** 
 * 64-bit floating point signal square root.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_64f_Ctx(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal square root.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_64f(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength);

/** 
 * 32-bit complex floating point signal square root.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_32fc_Ctx(const Mpp32fc * pSrc, Mpp32fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit complex floating point signal square root.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_32fc(const Mpp32fc * pSrc, Mpp32fc * pDst, size_t nLength);

/** 
 * 64-bit complex floating point signal square root.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_64fc_Ctx(const Mpp64fc * pSrc, Mpp64fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit complex floating point signal square root.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_64fc(const Mpp64fc * pSrc, Mpp64fc * pDst, size_t nLength);

/** 
 * 32-bit floating point signal square root.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_32f_I_Ctx(Mpp32f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal square root.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_32f_I(Mpp32f * pSrcDst, size_t nLength);

/** 
 * 64-bit floating point signal square root.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_64f_I_Ctx(Mpp64f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal square root.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_64f_I(Mpp64f * pSrcDst, size_t nLength);

/** 
 * 32-bit complex floating point signal square root.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_32fc_I_Ctx(Mpp32fc * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit complex floating point signal square root.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_32fc_I(Mpp32fc * pSrcDst, size_t nLength);

/** 
 * 64-bit complex floating point signal square root.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_64fc_I_Ctx(Mpp64fc * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit complex floating point signal square root.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_64fc_I(Mpp64fc * pSrcDst, size_t nLength);

/** 
 * 8-bit unsigned char signal square root, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_8u_Sfs_Ctx(const Mpp8u * pSrc, Mpp8u * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char signal square root, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_8u_Sfs(const Mpp8u * pSrc, Mpp8u * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal square root, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_16u_Sfs_Ctx(const Mpp16u * pSrc, Mpp16u * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short signal square root, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_16u_Sfs(const Mpp16u * pSrc, Mpp16u * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal square root, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_16s_Sfs_Ctx(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal square root, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_16s_Sfs(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit complex signed short signal square root, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_16sc_Sfs_Ctx(const Mpp16sc * pSrc, Mpp16sc * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit complex signed short signal square root, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_16sc_Sfs(const Mpp16sc * pSrc, Mpp16sc * pDst, size_t nLength, int nScaleFactor);

/** 
 * 64-bit signed integer signal square root, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_64s_Sfs_Ctx(const Mpp64s * pSrc, Mpp64s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 64-bit signed integer signal square root, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_64s_Sfs(const Mpp64s * pSrc, Mpp64s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal square root, scale, then clamp to 16-bit signed integer saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_32s16s_Sfs_Ctx(const Mpp32s * pSrc, Mpp16s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer signal square root, scale, then clamp to 16-bit signed integer saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_32s16s_Sfs(const Mpp32s * pSrc, Mpp16s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 64-bit signed integer signal square root, scale, then clamp to 16-bit signed integer saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_64s16s_Sfs_Ctx(const Mpp64s * pSrc, Mpp16s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 64-bit signed integer signal square root, scale, then clamp to 16-bit signed integer saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_64s16s_Sfs(const Mpp64s * pSrc, Mpp16s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 8-bit unsigned char signal square root, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_8u_ISfs_Ctx(Mpp8u * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char signal square root, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_8u_ISfs(Mpp8u * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal square root, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_16u_ISfs_Ctx(Mpp16u * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short signal square root, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_16u_ISfs(Mpp16u * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal square root, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_16s_ISfs_Ctx(Mpp16s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal square root, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_16s_ISfs(Mpp16s * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit complex signed short signal square root, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_16sc_ISfs_Ctx(Mpp16sc * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit complex signed short signal square root, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_16sc_ISfs(Mpp16sc * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 64-bit signed integer signal square root, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_64s_ISfs_Ctx(Mpp64s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 64-bit signed integer signal square root, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSqrt_64s_ISfs(Mpp64s * pSrcDst, size_t nLength, int nScaleFactor);

/** @} signal_sqrt */

/** 
 * \section signal_cuberoot Cubrt
 * @defgroup signal_cuberoot Cubrt
 *
 * Cube root of each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 32-bit floating point signal cube root.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCubrt_32f_Ctx(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal cube root.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCubrt_32f(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength);

/** 
 * 32-bit signed integer signal cube root, scale, then clamp to 16-bit signed integer saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCubrt_32s16s_Sfs_Ctx(const Mpp32s * pSrc, Mpp16s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer signal cube root, scale, then clamp to 16-bit signed integer saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCubrt_32s16s_Sfs(const Mpp32s * pSrc, Mpp16s * pDst, size_t nLength, int nScaleFactor);

/** @} signal_cuberoot */

/** 
 * \section signal_exp Exp
 * @defgroup signal_exp Exp
 *
 * E raised to the power of each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 32-bit floating point signal exponent.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsExp_32f_Ctx(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal exponent.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsExp_32f(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength);

/** 
 * 64-bit floating point signal exponent.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsExp_64f_Ctx(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal exponent.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsExp_64f(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength);

/** 
 * 32-bit floating point signal exponent with 64-bit floating point result.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsExp_32f64f_Ctx(const Mpp32f * pSrc, Mpp64f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal exponent with 64-bit floating point result.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsExp_32f64f(const Mpp32f * pSrc, Mpp64f * pDst, size_t nLength);

/** 
 * 32-bit floating point signal exponent.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsExp_32f_I_Ctx(Mpp32f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal exponent.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsExp_32f_I(Mpp32f * pSrcDst, size_t nLength);

/** 
 * 64-bit floating point signal exponent.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsExp_64f_I_Ctx(Mpp64f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal exponent.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsExp_64f_I(Mpp64f * pSrcDst, size_t nLength);

/** 
 * 16-bit signed short signal exponent, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsExp_16s_Sfs_Ctx(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal exponent, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsExp_16s_Sfs(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal exponent, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsExp_32s_Sfs_Ctx(const Mpp32s * pSrc, Mpp32s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer signal exponent, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsExp_32s_Sfs(const Mpp32s * pSrc, Mpp32s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 64-bit signed integer signal exponent, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsExp_64s_Sfs_Ctx(const Mpp64s * pSrc, Mpp64s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 64-bit signed integer signal exponent, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsExp_64s_Sfs(const Mpp64s * pSrc, Mpp64s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal exponent, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsExp_16s_ISfs_Ctx(Mpp16s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal exponent, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsExp_16s_ISfs(Mpp16s * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal exponent, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsExp_32s_ISfs_Ctx(Mpp32s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer signal exponent, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsExp_32s_ISfs(Mpp32s * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 64-bit signed integer signal exponent, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsExp_64s_ISfs_Ctx(Mpp64s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 64-bit signed integer signal exponent, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsExp_64s_ISfs(Mpp64s * pSrcDst, size_t nLength, int nScaleFactor);

/** @} signal_exp */

/** 
 * \section signal_ln Ln
 * @defgroup signal_ln Ln
 *
 * Natural logarithm of each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 32-bit floating point signal natural logarithm.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLn_32f_Ctx(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal natural logarithm.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLn_32f(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength);

/** 
 * 64-bit floating point signal natural logarithm.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLn_64f_Ctx(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal natural logarithm.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLn_64f(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength);

/** 
 * 64-bit floating point signal natural logarithm with 32-bit floating point result.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLn_64f32f_Ctx(const Mpp64f * pSrc, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal natural logarithm with 32-bit floating point result.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLn_64f32f(const Mpp64f * pSrc, Mpp32f * pDst, size_t nLength);

/** 
 * 32-bit floating point signal natural logarithm.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLn_32f_I_Ctx(Mpp32f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal natural logarithm.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLn_32f_I(Mpp32f * pSrcDst, size_t nLength);

/** 
 * 64-bit floating point signal natural logarithm.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLn_64f_I_Ctx(Mpp64f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal natural logarithm.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLn_64f_I(Mpp64f * pSrcDst, size_t nLength);

/** 
 * 16-bit signed short signal natural logarithm, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLn_16s_Sfs_Ctx(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal natural logarithm, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLn_16s_Sfs(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal natural logarithm, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLn_32s_Sfs_Ctx(const Mpp32s * pSrc, Mpp32s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer signal natural logarithm, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLn_32s_Sfs(const Mpp32s * pSrc, Mpp32s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal natural logarithm, scale, then clamp to 16-bit signed short saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLn_32s16s_Sfs_Ctx(const Mpp32s * pSrc, Mpp16s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer signal natural logarithm, scale, then clamp to 16-bit signed short saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLn_32s16s_Sfs(const Mpp32s * pSrc, Mpp16s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal natural logarithm, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLn_16s_ISfs_Ctx(Mpp16s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal natural logarithm, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLn_16s_ISfs(Mpp16s * pSrcDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal natural logarithm, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLn_32s_ISfs_Ctx(Mpp32s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer signal natural logarithm, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLn_32s_ISfs(Mpp32s * pSrcDst, size_t nLength, int nScaleFactor);

/** @} signal_ln */

/** 
 * \section signal_10log10 10Log10
 * @defgroup signal_10log10 10Log10
 *
 * Ten times the decimal logarithm of each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 32-bit signed integer signal 10 times base 10 logarithm, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
npps10Log10_32s_Sfs_Ctx(const Mpp32s * pSrc, Mpp32s * pDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer signal 10 times base 10 logarithm, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
npps10Log10_32s_Sfs(const Mpp32s * pSrc, Mpp32s * pDst, size_t nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal 10 times base 10 logarithm, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
npps10Log10_32s_ISfs_Ctx(Mpp32s * pSrcDst, size_t nLength, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer signal 10 times base 10 logarithm, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
npps10Log10_32s_ISfs(Mpp32s * pSrcDst, size_t nLength, int nScaleFactor);

/** @} signal_10log10 */

/** 
 * \section signal_sumln SumLn
 * @defgroup signal_sumln SumLn
 *
 * Sums up the natural logarithm of each sample of a signal.
 *
 * @{
 *
 */

/** 
 * Device scratch buffer size (in bytes) for 32f SumLn.
 * This primitive provides the correct buffer size for mppsSumLn_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumLnGetBufferSize_32f_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for 32f SumLn.
 * This primitive provides the correct buffer size for mppsSumLn_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumLnGetBufferSize_32f(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * 32-bit floating point signal sum natural logarithm.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSumLn_32f_Ctx(const Mpp32f * pSrc, size_t nLength, Mpp32f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal sum natural logarithm.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSumLn_32f(const Mpp32f * pSrc, size_t nLength, Mpp32f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * Device scratch buffer size (in bytes) for 64f SumLn.
 * This primitive provides the correct buffer size for mppsSumLn_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumLnGetBufferSize_64f_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for 64f SumLn.
 * This primitive provides the correct buffer size for mppsSumLn_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumLnGetBufferSize_64f(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * 64-bit floating point signal sum natural logarithm.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSumLn_64f_Ctx(const Mpp64f * pSrc, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal sum natural logarithm.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSumLn_64f(const Mpp64f * pSrc, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * Device scratch buffer size (in bytes) for 32f64f SumLn.
 * This primitive provides the correct buffer size for mppsSumLn_32f64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumLnGetBufferSize_32f64f_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for 32f64f SumLn.
 * This primitive provides the correct buffer size for mppsSumLn_32f64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumLnGetBufferSize_32f64f(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * 32-bit flaoting point input, 64-bit floating point output signal sum natural logarithm.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSumLn_32f64f_Ctx(const Mpp32f * pSrc, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit flaoting point input, 64-bit floating point output signal sum natural logarithm.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSumLn_32f64f(const Mpp32f * pSrc, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * Device scratch buffer size (in bytes) for 16s32f SumLn.
 * This primitive provides the correct buffer size for mppsSumLn_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumLnGetBufferSize_16s32f_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for 16s32f SumLn.
 * This primitive provides the correct buffer size for mppsSumLn_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumLnGetBufferSize_16s32f(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * 16-bit signed short integer input, 32-bit floating point output signal sum natural logarithm.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSumLn_16s32f_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp32f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer input, 32-bit floating point output signal sum natural logarithm.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSumLn_16s32f(const Mpp16s * pSrc, size_t nLength, Mpp32f * pDst, Mpp8u * pDeviceBuffer);

/** @} signal_sumln */

/** 
 * \section signal_inversetan Arctan
 * @defgroup signal_inversetan Arctan
 *
 * Inverse tangent of each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 32-bit floating point signal inverse tangent.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsArctan_32f_Ctx(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal inverse tangent.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsArctan_32f(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength);

/** 
 * 64-bit floating point signal inverse tangent.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsArctan_64f_Ctx(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal inverse tangent.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsArctan_64f(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength);

/** 
 * 32-bit floating point signal inverse tangent.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsArctan_32f_I_Ctx(Mpp32f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal inverse tangent.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsArctan_32f_I(Mpp32f * pSrcDst, size_t nLength);

/** 
 * 64-bit floating point signal inverse tangent.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsArctan_64f_I_Ctx(Mpp64f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal inverse tangent.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsArctan_64f_I(Mpp64f * pSrcDst, size_t nLength);

/** @} signal_inversetan */

/** 
 * \section signal_normalize Normalize
 * @defgroup signal_normalize Normalize
 *
 * Normalize each sample of a real or complex signal using offset and division operations.
 *
 * @{
 *
 */

/** 
 * 32-bit floating point signal normalize.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param vSub value subtracted from each signal element before division
 * \param vDiv divisor of post-subtracted signal element dividend
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormalize_32f_Ctx(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength, Mpp32f vSub, Mpp32f vDiv, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal normalize.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param vSub value subtracted from each signal element before division
 * \param vDiv divisor of post-subtracted signal element dividend
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormalize_32f(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength, Mpp32f vSub, Mpp32f vDiv);

/** 
 * 32-bit complex floating point signal normalize.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param vSub value subtracted from each signal element before division
 * \param vDiv divisor of post-subtracted signal element dividend
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormalize_32fc_Ctx(const Mpp32fc * pSrc, Mpp32fc * pDst, size_t nLength, Mpp32fc vSub, Mpp32f vDiv, MppStreamContext mppStreamCtx);
/** 
 * 32-bit complex floating point signal normalize.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param vSub value subtracted from each signal element before division
 * \param vDiv divisor of post-subtracted signal element dividend
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormalize_32fc(const Mpp32fc * pSrc, Mpp32fc * pDst, size_t nLength, Mpp32fc vSub, Mpp32f vDiv);

/** 
 * 64-bit floating point signal normalize.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param vSub value subtracted from each signal element before division
 * \param vDiv divisor of post-subtracted signal element dividend
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormalize_64f_Ctx(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength, Mpp64f vSub, Mpp64f vDiv, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point signal normalize.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param vSub value subtracted from each signal element before division
 * \param vDiv divisor of post-subtracted signal element dividend
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormalize_64f(const Mpp64f * pSrc, Mpp64f * pDst, size_t nLength, Mpp64f vSub, Mpp64f vDiv);

/** 
 * 64-bit complex floating point signal normalize.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param vSub value subtracted from each signal element before division
 * \param vDiv divisor of post-subtracted signal element dividend
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormalize_64fc_Ctx(const Mpp64fc * pSrc, Mpp64fc * pDst, size_t nLength, Mpp64fc vSub, Mpp64f vDiv, MppStreamContext mppStreamCtx);
/** 
 * 64-bit complex floating point signal normalize.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param vSub value subtracted from each signal element before division
 * \param vDiv divisor of post-subtracted signal element dividend
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormalize_64fc(const Mpp64fc * pSrc, Mpp64fc * pDst, size_t nLength, Mpp64fc vSub, Mpp64f vDiv);

/** 
 * 16-bit signed short signal normalize, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param vSub value subtracted from each signal element before division
 * \param vDiv divisor of post-subtracted signal element dividend
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormalize_16s_Sfs_Ctx(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength, Mpp16s vSub, int vDiv, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal normalize, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param vSub value subtracted from each signal element before division
 * \param vDiv divisor of post-subtracted signal element dividend
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormalize_16s_Sfs(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength, Mpp16s vSub, int vDiv, int nScaleFactor);

/** 
 * 16-bit complex signed short signal normalize, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param vSub value subtracted from each signal element before division
 * \param vDiv divisor of post-subtracted signal element dividend
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormalize_16sc_Sfs_Ctx(const Mpp16sc * pSrc, Mpp16sc * pDst, size_t nLength, Mpp16sc vSub, int vDiv, int nScaleFactor, MppStreamContext mppStreamCtx);
/** 
 * 16-bit complex signed short signal normalize, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param vSub value subtracted from each signal element before division
 * \param vDiv divisor of post-subtracted signal element dividend
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormalize_16sc_Sfs(const Mpp16sc * pSrc, Mpp16sc * pDst, size_t nLength, Mpp16sc vSub, int vDiv, int nScaleFactor);

/** @} signal_normalize */

/** 
 * \section signal_cauchy Cauchy, CauchyD, and CauchyDD2
 * @defgroup signal_cauchy Cauchy, CauchyD, and CauchyDD2
 *
 * Determine Cauchy robust error function and its first and second derivatives for each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 32-bit floating point signal Cauchy error calculation.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nParam constant used in Cauchy formula
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCauchy_32f_I_Ctx(Mpp32f * pSrcDst, size_t nLength, Mpp32f nParam, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal Cauchy error calculation.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nParam constant used in Cauchy formula
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCauchy_32f_I(Mpp32f * pSrcDst, size_t nLength, Mpp32f nParam);

/** 
 * 32-bit floating point signal Cauchy first derivative.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nParam constant used in Cauchy formula
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCauchyD_32f_I_Ctx(Mpp32f * pSrcDst, size_t nLength, Mpp32f nParam, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal Cauchy first derivative.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nParam constant used in Cauchy formula
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCauchyD_32f_I(Mpp32f * pSrcDst, size_t nLength, Mpp32f nParam);

/** 
 * 32-bit floating point signal Cauchy first and second derivatives.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param pD2FVal \ref source_signal_pointer. This signal contains the second derivative
 *      of the source signal.
 * \param nLength \ref length_specification.
 * \param nParam constant used in Cauchy formula
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCauchyDD2_32f_I_Ctx(Mpp32f * pSrcDst, Mpp32f * pD2FVal, size_t nLength, Mpp32f nParam, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point signal Cauchy first and second derivatives.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param pD2FVal \ref source_signal_pointer. This signal contains the second derivative
 *      of the source signal.
 * \param nLength \ref length_specification.
 * \param nParam constant used in Cauchy formula
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCauchyDD2_32f_I(Mpp32f * pSrcDst, Mpp32f * pD2FVal, size_t nLength, Mpp32f nParam);

/** @} signal_cauchy */

/** @} signal_arithmetic_operations */

/** 
 * \section signal_logical_and_shift_operations Logical And Shift Operations
 * @defgroup signal_logical_and_shift_operations Logical And Shift Operations
 * The set of logical and shift operations for signal processing available in the library.
 * @{
 *
 */

/** 
 * \section signal_andc AndC
 * @defgroup signal_andc AndC
 *
 * Bitwise AND of a constant and each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char signal and with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be anded with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAndC_8u_Ctx(const Mpp8u * pSrc, Mpp8u nValue, Mpp8u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char signal and with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be anded with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAndC_8u(const Mpp8u * pSrc, Mpp8u nValue, Mpp8u * pDst, size_t nLength);

/** 
 * 16-bit unsigned short signal and with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be anded with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAndC_16u_Ctx(const Mpp16u * pSrc, Mpp16u nValue, Mpp16u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short signal and with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be anded with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAndC_16u(const Mpp16u * pSrc, Mpp16u nValue, Mpp16u * pDst, size_t nLength);

/** 
 * 32-bit unsigned integer signal and with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be anded with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAndC_32u_Ctx(const Mpp32u * pSrc, Mpp32u nValue, Mpp32u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned integer signal and with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be anded with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAndC_32u(const Mpp32u * pSrc, Mpp32u nValue, Mpp32u * pDst, size_t nLength);

/** 
 * 8-bit unsigned char in place signal and with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be anded with each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAndC_8u_I_Ctx(Mpp8u nValue, Mpp8u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char in place signal and with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be anded with each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAndC_8u_I(Mpp8u nValue, Mpp8u * pSrcDst, size_t nLength);

/** 
 * 16-bit unsigned short in place signal and with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be anded with each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAndC_16u_I_Ctx(Mpp16u nValue, Mpp16u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short in place signal and with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be anded with each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAndC_16u_I(Mpp16u nValue, Mpp16u * pSrcDst, size_t nLength);

/** 
 * 32-bit unsigned signed integer in place signal and with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be anded with each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAndC_32u_I_Ctx(Mpp32u nValue, Mpp32u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned signed integer in place signal and with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be anded with each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAndC_32u_I(Mpp32u nValue, Mpp32u * pSrcDst, size_t nLength);

/** @} signal_andc */

/** 
 * \section signal_and And
 * @defgroup signal_and And
 *
 * Sample by sample bitwise AND of samples from two signals.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char signal and with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be anded with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAnd_8u_Ctx(const Mpp8u * pSrc1, const Mpp8u * pSrc2, Mpp8u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char signal and with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be anded with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAnd_8u(const Mpp8u * pSrc1, const Mpp8u * pSrc2, Mpp8u * pDst, size_t nLength);

/** 
 * 16-bit unsigned short signal and with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be anded with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAnd_16u_Ctx(const Mpp16u * pSrc1, const Mpp16u * pSrc2, Mpp16u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short signal and with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be anded with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAnd_16u(const Mpp16u * pSrc1, const Mpp16u * pSrc2, Mpp16u * pDst, size_t nLength);

/** 
 * 32-bit unsigned integer signal and with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be anded with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAnd_32u_Ctx(const Mpp32u * pSrc1, const Mpp32u * pSrc2, Mpp32u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned integer signal and with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be anded with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAnd_32u(const Mpp32u * pSrc1, const Mpp32u * pSrc2, Mpp32u * pDst, size_t nLength);

/** 
 * 8-bit unsigned char in place signal and with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be anded with signal1 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAnd_8u_I_Ctx(const Mpp8u * pSrc, Mpp8u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char in place signal and with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be anded with signal1 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAnd_8u_I(const Mpp8u * pSrc, Mpp8u * pSrcDst, size_t nLength);

/** 
 * 16-bit unsigned short in place signal and with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be anded with signal1 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAnd_16u_I_Ctx(const Mpp16u * pSrc, Mpp16u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short in place signal and with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be anded with signal1 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAnd_16u_I(const Mpp16u * pSrc, Mpp16u * pSrcDst, size_t nLength);

/** 
 * 32-bit unsigned integer in place signal and with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be anded with signal1 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAnd_32u_I_Ctx(const Mpp32u * pSrc, Mpp32u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned integer in place signal and with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be anded with signal1 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAnd_32u_I(const Mpp32u * pSrc, Mpp32u * pSrcDst, size_t nLength);

/** @} signal_and */

/** 
 * \section signal_orc OrC
 * @defgroup signal_orc OrC
 *
 * Bitwise OR of a constant and each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char signal or with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be ored with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOrC_8u_Ctx(const Mpp8u * pSrc, Mpp8u nValue, Mpp8u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char signal or with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be ored with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOrC_8u(const Mpp8u * pSrc, Mpp8u nValue, Mpp8u * pDst, size_t nLength);

/** 
 * 16-bit unsigned short signal or with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be ored with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOrC_16u_Ctx(const Mpp16u * pSrc, Mpp16u nValue, Mpp16u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short signal or with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be ored with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOrC_16u(const Mpp16u * pSrc, Mpp16u nValue, Mpp16u * pDst, size_t nLength);

/** 
 * 32-bit unsigned integer signal or with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be ored with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOrC_32u_Ctx(const Mpp32u * pSrc, Mpp32u nValue, Mpp32u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned integer signal or with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be ored with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOrC_32u(const Mpp32u * pSrc, Mpp32u nValue, Mpp32u * pDst, size_t nLength);

/** 
 * 8-bit unsigned char in place signal or with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be ored with each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOrC_8u_I_Ctx(Mpp8u nValue, Mpp8u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char in place signal or with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be ored with each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOrC_8u_I(Mpp8u nValue, Mpp8u * pSrcDst, size_t nLength);

/** 
 * 16-bit unsigned short in place signal or with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be ored with each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOrC_16u_I_Ctx(Mpp16u nValue, Mpp16u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short in place signal or with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be ored with each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOrC_16u_I(Mpp16u nValue, Mpp16u * pSrcDst, size_t nLength);

/** 
 * 32-bit unsigned signed integer in place signal or with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be ored with each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOrC_32u_I_Ctx(Mpp32u nValue, Mpp32u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned signed integer in place signal or with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be ored with each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOrC_32u_I(Mpp32u nValue, Mpp32u * pSrcDst, size_t nLength);

/** @} signal_orc */

/** 
 * \section signal_or Or
 * @defgroup signal_or Or
 *
 * Sample by sample bitwise OR of the samples from two signals.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char signal or with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be ored with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOr_8u_Ctx(const Mpp8u * pSrc1, const Mpp8u * pSrc2, Mpp8u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char signal or with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be ored with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOr_8u(const Mpp8u * pSrc1, const Mpp8u * pSrc2, Mpp8u * pDst, size_t nLength);

/** 
 * 16-bit unsigned short signal or with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be ored with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOr_16u_Ctx(const Mpp16u * pSrc1, const Mpp16u * pSrc2, Mpp16u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short signal or with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be ored with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOr_16u(const Mpp16u * pSrc1, const Mpp16u * pSrc2, Mpp16u * pDst, size_t nLength);

/** 
 * 32-bit unsigned integer signal or with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be ored with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOr_32u_Ctx(const Mpp32u * pSrc1, const Mpp32u * pSrc2, Mpp32u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned integer signal or with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be ored with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOr_32u(const Mpp32u * pSrc1, const Mpp32u * pSrc2, Mpp32u * pDst, size_t nLength);

/** 
 * 8-bit unsigned char in place signal or with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be ored with signal1 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOr_8u_I_Ctx(const Mpp8u * pSrc, Mpp8u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char in place signal or with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be ored with signal1 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOr_8u_I(const Mpp8u * pSrc, Mpp8u * pSrcDst, size_t nLength);

/** 
 * 16-bit unsigned short in place signal or with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be ored with signal1 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOr_16u_I_Ctx(const Mpp16u * pSrc, Mpp16u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short in place signal or with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be ored with signal1 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOr_16u_I(const Mpp16u * pSrc, Mpp16u * pSrcDst, size_t nLength);

/** 
 * 32-bit unsigned integer in place signal or with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be ored with signal1 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOr_32u_I_Ctx(const Mpp32u * pSrc, Mpp32u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned integer in place signal or with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be ored with signal1 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsOr_32u_I(const Mpp32u * pSrc, Mpp32u * pSrcDst, size_t nLength);

/** @} signal_or */

/** 
 * \section signal_xorc XorC
 * @defgroup signal_xorc XorC
 *
 * Bitwise XOR of a constant and each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char signal exclusive or with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be exclusive ored with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXorC_8u_Ctx(const Mpp8u * pSrc, Mpp8u nValue, Mpp8u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char signal exclusive or with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be exclusive ored with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXorC_8u(const Mpp8u * pSrc, Mpp8u nValue, Mpp8u * pDst, size_t nLength);

/** 
 * 16-bit unsigned short signal exclusive or with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be exclusive ored with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXorC_16u_Ctx(const Mpp16u * pSrc, Mpp16u nValue, Mpp16u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short signal exclusive or with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be exclusive ored with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXorC_16u(const Mpp16u * pSrc, Mpp16u nValue, Mpp16u * pDst, size_t nLength);

/** 
 * 32-bit unsigned integer signal exclusive or with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be exclusive ored with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXorC_32u_Ctx(const Mpp32u * pSrc, Mpp32u nValue, Mpp32u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned integer signal exclusive or with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be exclusive ored with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXorC_32u(const Mpp32u * pSrc, Mpp32u nValue, Mpp32u * pDst, size_t nLength);

/** 
 * 8-bit unsigned char in place signal exclusive or with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be exclusive ored with each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXorC_8u_I_Ctx(Mpp8u nValue, Mpp8u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char in place signal exclusive or with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be exclusive ored with each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXorC_8u_I(Mpp8u nValue, Mpp8u * pSrcDst, size_t nLength);

/** 
 * 16-bit unsigned short in place signal exclusive or with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be exclusive ored with each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXorC_16u_I_Ctx(Mpp16u nValue, Mpp16u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short in place signal exclusive or with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be exclusive ored with each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXorC_16u_I(Mpp16u nValue, Mpp16u * pSrcDst, size_t nLength);

/** 
 * 32-bit unsigned signed integer in place signal exclusive or with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be exclusive ored with each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXorC_32u_I_Ctx(Mpp32u nValue, Mpp32u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned signed integer in place signal exclusive or with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be exclusive ored with each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXorC_32u_I(Mpp32u nValue, Mpp32u * pSrcDst, size_t nLength);

/** @} signal_xorc */

/**
 * \section signal_xor Xor
 * @defgroup signal_xor Xor
 *
 * Sample by sample bitwise XOR of the samples from two signals.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char signal exclusive or with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXor_8u_Ctx(const Mpp8u * pSrc1, const Mpp8u * pSrc2, Mpp8u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char signal exclusive or with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXor_8u(const Mpp8u * pSrc1, const Mpp8u * pSrc2, Mpp8u * pDst, size_t nLength);

/** 
 * 16-bit unsigned short signal exclusive or with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXor_16u_Ctx(const Mpp16u * pSrc1, const Mpp16u * pSrc2, Mpp16u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short signal exclusive or with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXor_16u(const Mpp16u * pSrc1, const Mpp16u * pSrc2, Mpp16u * pDst, size_t nLength);

/** 
 * 32-bit unsigned integer signal exclusive or with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXor_32u_Ctx(const Mpp32u * pSrc1, const Mpp32u * pSrc2, Mpp32u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned integer signal exclusive or with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXor_32u(const Mpp32u * pSrc1, const Mpp32u * pSrc2, Mpp32u * pDst, size_t nLength);

/** 
 * 8-bit unsigned char in place signal exclusive or with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXor_8u_I_Ctx(const Mpp8u * pSrc, Mpp8u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char in place signal exclusive or with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXor_8u_I(const Mpp8u * pSrc, Mpp8u * pSrcDst, size_t nLength);

/** 
 * 16-bit unsigned short in place signal exclusive or with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXor_16u_I_Ctx(const Mpp16u * pSrc, Mpp16u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short in place signal exclusive or with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXor_16u_I(const Mpp16u * pSrc, Mpp16u * pSrcDst, size_t nLength);

/** 
 * 32-bit unsigned integer in place signal exclusive or with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXor_32u_I_Ctx(const Mpp32u * pSrc, Mpp32u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned integer in place signal exclusive or with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsXor_32u_I(const Mpp32u * pSrc, Mpp32u * pSrcDst, size_t nLength);

/** @} signal_xor */

/** 
 * \section signal_not Not
 * @defgroup signal_not Not
 *
 * Bitwise NOT of each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char not signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNot_8u_Ctx(const Mpp8u * pSrc, Mpp8u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char not signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNot_8u(const Mpp8u * pSrc, Mpp8u * pDst, size_t nLength);

/** 
 * 16-bit unsigned short not signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNot_16u_Ctx(const Mpp16u * pSrc, Mpp16u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short not signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNot_16u(const Mpp16u * pSrc, Mpp16u * pDst, size_t nLength);

/** 
 * 32-bit unsigned integer not signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNot_32u_Ctx(const Mpp32u * pSrc, Mpp32u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned integer not signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNot_32u(const Mpp32u * pSrc, Mpp32u * pDst, size_t nLength);

/** 
 * 8-bit unsigned char in place not signal.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNot_8u_I_Ctx(Mpp8u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char in place not signal.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNot_8u_I(Mpp8u * pSrcDst, size_t nLength);

/** 
 * 16-bit unsigned short in place not signal.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNot_16u_I_Ctx(Mpp16u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short in place not signal.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNot_16u_I(Mpp16u * pSrcDst, size_t nLength);

/** 
 * 32-bit unsigned signed integer in place not signal.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNot_32u_I_Ctx(Mpp32u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned signed integer in place not signal.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNot_32u_I(Mpp32u * pSrcDst, size_t nLength);

/** @} signal_not */

/** 
 * \section signal_lshiftc LShiftC
 * @defgroup signal_lshiftc LShiftC
 *
 * Left shifts the bits of each sample of a signal by a constant amount.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char signal left shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLShiftC_8u_Ctx(const Mpp8u * pSrc, int nValue, Mpp8u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char signal left shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLShiftC_8u(const Mpp8u * pSrc, int nValue, Mpp8u * pDst, size_t nLength);

/** 
 * 16-bit unsigned short signal left shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLShiftC_16u_Ctx(const Mpp16u * pSrc, int nValue, Mpp16u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short signal left shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLShiftC_16u(const Mpp16u * pSrc, int nValue, Mpp16u * pDst, size_t nLength);

/** 
 * 16-bit signed short signal left shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLShiftC_16s_Ctx(const Mpp16s * pSrc, int nValue, Mpp16s * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal left shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLShiftC_16s(const Mpp16s * pSrc, int nValue, Mpp16s * pDst, size_t nLength);

/** 
 * 32-bit unsigned integer signal left shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLShiftC_32u_Ctx(const Mpp32u * pSrc, int nValue, Mpp32u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned integer signal left shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLShiftC_32u(const Mpp32u * pSrc, int nValue, Mpp32u * pDst, size_t nLength);

/** 
 * 32-bit signed integer signal left shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLShiftC_32s_Ctx(const Mpp32s * pSrc, int nValue, Mpp32s * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer signal left shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLShiftC_32s(const Mpp32s * pSrc, int nValue, Mpp32s * pDst, size_t nLength);

/** 
 * 8-bit unsigned char in place signal left shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLShiftC_8u_I_Ctx(int nValue, Mpp8u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char in place signal left shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLShiftC_8u_I(int nValue, Mpp8u * pSrcDst, size_t nLength);

/** 
 * 16-bit unsigned short in place signal left shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLShiftC_16u_I_Ctx(int nValue, Mpp16u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short in place signal left shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLShiftC_16u_I(int nValue, Mpp16u * pSrcDst, size_t nLength);

/** 
 * 16-bit signed short in place signal left shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLShiftC_16s_I_Ctx(int nValue, Mpp16s * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short in place signal left shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLShiftC_16s_I(int nValue, Mpp16s * pSrcDst, size_t nLength);

/** 
 * 32-bit unsigned signed integer in place signal left shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLShiftC_32u_I_Ctx(int nValue, Mpp32u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned signed integer in place signal left shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLShiftC_32u_I(int nValue, Mpp32u * pSrcDst, size_t nLength);

/** 
 * 32-bit signed signed integer in place signal left shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLShiftC_32s_I_Ctx(int nValue, Mpp32s * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed signed integer in place signal left shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsLShiftC_32s_I(int nValue, Mpp32s * pSrcDst, size_t nLength);

/** @} signal_lshiftc */

/** 
 * \section signal_rshiftc RShiftC
 * @defgroup signal_rshiftc RShiftC
 *
 * Right shifts the bits of each sample of a signal by a constant amount.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char signal right shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsRShiftC_8u_Ctx(const Mpp8u * pSrc, int nValue, Mpp8u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char signal right shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsRShiftC_8u(const Mpp8u * pSrc, int nValue, Mpp8u * pDst, size_t nLength);

/** 
 * 16-bit unsigned short signal right shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsRShiftC_16u_Ctx(const Mpp16u * pSrc, int nValue, Mpp16u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short signal right shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsRShiftC_16u(const Mpp16u * pSrc, int nValue, Mpp16u * pDst, size_t nLength);

/** 
 * 16-bit signed short signal right shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsRShiftC_16s_Ctx(const Mpp16s * pSrc, int nValue, Mpp16s * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short signal right shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsRShiftC_16s(const Mpp16s * pSrc, int nValue, Mpp16s * pDst, size_t nLength);

/** 
 * 32-bit unsigned integer signal right shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsRShiftC_32u_Ctx(const Mpp32u * pSrc, int nValue, Mpp32u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned integer signal right shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsRShiftC_32u(const Mpp32u * pSrc, int nValue, Mpp32u * pDst, size_t nLength);

/** 
 * 32-bit signed integer signal right shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsRShiftC_32s_Ctx(const Mpp32s * pSrc, int nValue, Mpp32s * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer signal right shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsRShiftC_32s(const Mpp32s * pSrc, int nValue, Mpp32s * pDst, size_t nLength);

/** 
 * 8-bit unsigned char in place signal right shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsRShiftC_8u_I_Ctx(int nValue, Mpp8u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char in place signal right shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsRShiftC_8u_I(int nValue, Mpp8u * pSrcDst, size_t nLength);

/** 
 * 16-bit unsigned short in place signal right shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsRShiftC_16u_I_Ctx(int nValue, Mpp16u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short in place signal right shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsRShiftC_16u_I(int nValue, Mpp16u * pSrcDst, size_t nLength);

/** 
 * 16-bit signed short in place signal right shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsRShiftC_16s_I_Ctx(int nValue, Mpp16s * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short in place signal right shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsRShiftC_16s_I(int nValue, Mpp16s * pSrcDst, size_t nLength);

/** 
 * 32-bit unsigned signed integer in place signal right shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsRShiftC_32u_I_Ctx(int nValue, Mpp32u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned signed integer in place signal right shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsRShiftC_32u_I(int nValue, Mpp32u * pSrcDst, size_t nLength);

/** 
 * 32-bit signed signed integer in place signal right shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsRShiftC_32s_I_Ctx(int nValue, Mpp32s * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed signed integer in place signal right shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsRShiftC_32s_I(int nValue, Mpp32s * pSrcDst, size_t nLength);

/** @} signal_rshiftc */

/** @} signal_logical_and_shift_operations */

/** @} signal_arithmetic_and_logical_operations */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MC_MPPS_ARITHMETIC_AND_LOGICAL_OPERATIONS_H */
