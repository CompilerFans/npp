 /* Copyright 2025-2025 MetaX Corporation.  All rights reserved. 
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
#ifndef MC_MPPS_INITIALIZATION_H
#define MC_MPPS_INITIALIZATION_H
 
/**
 * \file mpps_initialization.h
 * MPP Signal Processing Functionality.
 */
 
#include "mppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif


/** 
 * \page signal_initialization Initialization
 * @defgroup signal_initialization Initialization
 * @ingroup mpps
 * Functions that provide functionality of initialization signal like: set, zero or copy other signal.
 * @{
 */

/** 
 * \section signal_set Set
 * @defgroup signal_set Set
 * The set of set initialization operations available in the library.
 * @{
 *
 */

/** @name Set 
 * Set methods for 1D vectors of various types. The copy methods operate on vector data given
 * as a pointer to the underlying data-type (e.g. 8-bit vectors would
 * be passed as pointers to Mpp8u type) and length of the vectors, i.e. the number of items.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_8u_Ctx(Mpp8u nValue, Mpp8u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_8u(Mpp8u nValue, Mpp8u * pDst, size_t nLength);

/** 
 * 8-bit signed char, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_8s_Ctx(Mpp8s nValue, Mpp8s * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit signed char, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_8s(Mpp8s nValue, Mpp8s * pDst, size_t nLength);

/** 
 * 16-bit unsigned integer, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_16u_Ctx(Mpp16u nValue, Mpp16u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned integer, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_16u(Mpp16u nValue, Mpp16u * pDst, size_t nLength);

/** 
 * 16-bit signed integer, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_16s_Ctx(Mpp16s nValue, Mpp16s * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed integer, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_16s(Mpp16s nValue, Mpp16s * pDst, size_t nLength);

/** 
 * 16-bit integer complex, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_16sc_Ctx(Mpp16sc nValue, Mpp16sc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit integer complex, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_16sc(Mpp16sc nValue, Mpp16sc * pDst, size_t nLength);

/** 
 * 32-bit unsigned integer, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_32u_Ctx(Mpp32u nValue, Mpp32u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned integer, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_32u(Mpp32u nValue, Mpp32u * pDst, size_t nLength);

/** 
 * 32-bit signed integer, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_32s_Ctx(Mpp32s nValue, Mpp32s * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_32s(Mpp32s nValue, Mpp32s * pDst, size_t nLength);

/** 
 * 32-bit integer complex, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_32sc_Ctx(Mpp32sc nValue, Mpp32sc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit integer complex, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_32sc(Mpp32sc nValue, Mpp32sc * pDst, size_t nLength);

/** 
 * 32-bit float, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_32f_Ctx(Mpp32f nValue, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_32f(Mpp32f nValue, Mpp32f * pDst, size_t nLength);

/** 
 * 32-bit float complex, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_32fc_Ctx(Mpp32fc nValue, Mpp32fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float complex, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_32fc(Mpp32fc nValue, Mpp32fc * pDst, size_t nLength);

/** 
 * 64-bit long long integer, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_64s_Ctx(Mpp64s nValue, Mpp64s * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit long long integer, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_64s(Mpp64s nValue, Mpp64s * pDst, size_t nLength);

/** 
 * 64-bit long long integer complex, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_64sc_Ctx(Mpp64sc nValue, Mpp64sc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit long long integer complex, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_64sc(Mpp64sc nValue, Mpp64sc * pDst, size_t nLength);

/** 
 * 64-bit double, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_64f_Ctx(Mpp64f nValue, Mpp64f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit double, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_64f(Mpp64f nValue, Mpp64f * pDst, size_t nLength);

/** 
 * 64-bit double complex, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_64fc_Ctx(Mpp64fc nValue, Mpp64fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit double complex, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSet_64fc(Mpp64fc nValue, Mpp64fc * pDst, size_t nLength);

/** @} end of Signal Set */
/** @} signal_set */

/** 
 * \section signal_zero Zero
 * @defgroup signal_zero Zero
 * The set of zero initialization operations available in the library.
 * @{
 *
 */

/** @name Zero
 * Set signals to zero.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsZero_8u_Ctx(Mpp8u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsZero_8u(Mpp8u * pDst, size_t nLength);

/** 
 * 16-bit integer, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsZero_16s_Ctx(Mpp16s * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit integer, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsZero_16s(Mpp16s * pDst, size_t nLength);

/** 
 * 16-bit integer complex, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsZero_16sc_Ctx(Mpp16sc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit integer complex, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsZero_16sc(Mpp16sc * pDst, size_t nLength);

/** 
 * 32-bit integer, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsZero_32s_Ctx(Mpp32s * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit integer, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsZero_32s(Mpp32s * pDst, size_t nLength);

/** 
 * 32-bit integer complex, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsZero_32sc_Ctx(Mpp32sc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit integer complex, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsZero_32sc(Mpp32sc * pDst, size_t nLength);

/** 
 * 32-bit float, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsZero_32f_Ctx(Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsZero_32f(Mpp32f * pDst, size_t nLength);

/** 
 * 32-bit float complex, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsZero_32fc_Ctx(Mpp32fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float complex, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsZero_32fc(Mpp32fc * pDst, size_t nLength);

/** 
 * 64-bit long long integer, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsZero_64s_Ctx(Mpp64s * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit long long integer, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsZero_64s(Mpp64s * pDst, size_t nLength);

/** 
 * 64-bit long long integer complex, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsZero_64sc_Ctx(Mpp64sc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit long long integer complex, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsZero_64sc(Mpp64sc * pDst, size_t nLength);

/** 
 * 64-bit double, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsZero_64f_Ctx(Mpp64f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit double, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsZero_64f(Mpp64f * pDst, size_t nLength);

/** 
 * 64-bit double complex, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsZero_64fc_Ctx(Mpp64fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit double complex, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsZero_64fc(Mpp64fc * pDst, size_t nLength);

/** @} end of Zero */

/** @} signal_zero */

/** 
 * \section signal_copy Copy
 * @defgroup signal_copy Copy
 * The set of copy initialization operations available in the library.
 * @{
 *
 */

/** @name Copy
 * Copy methods for various type signals. Copy methods operate on
 * signal data given as a pointer to the underlying data-type (e.g. 8-bit
 * vectors would be passed as pointers to Mpp8u type) and length of the
 * vectors, i.e. the number of items. 
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char, vector copy method
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCopy_8u_Ctx(const Mpp8u * pSrc, Mpp8u * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char, vector copy method
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCopy_8u(const Mpp8u * pSrc, Mpp8u * pDst, size_t nLength);

/** 
 * 16-bit signed short, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCopy_16s_Ctx(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCopy_16s(const Mpp16s * pSrc, Mpp16s * pDst, size_t nLength);

/** 
 * 32-bit signed integer, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCopy_32s_Ctx(const Mpp32s * pSrc, Mpp32s * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCopy_32s(const Mpp32s * pSrc, Mpp32s * pDst, size_t nLength);

/** 
 * 32-bit float, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCopy_32f_Ctx(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCopy_32f(const Mpp32f * pSrc, Mpp32f * pDst, size_t nLength);

/** 
 * 64-bit signed integer, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCopy_64s_Ctx(const Mpp64s * pSrc, Mpp64s * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit signed integer, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCopy_64s(const Mpp64s * pSrc, Mpp64s * pDst, size_t nLength);

/** 
 * 16-bit complex short, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCopy_16sc_Ctx(const Mpp16sc * pSrc, Mpp16sc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit complex short, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCopy_16sc(const Mpp16sc * pSrc, Mpp16sc * pDst, size_t nLength);

/** 
 * 32-bit complex signed integer, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCopy_32sc_Ctx(const Mpp32sc * pSrc, Mpp32sc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit complex signed integer, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCopy_32sc(const Mpp32sc * pSrc, Mpp32sc * pDst, size_t nLength);

/** 
 * 32-bit complex float, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCopy_32fc_Ctx(const Mpp32fc * pSrc, Mpp32fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit complex float, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCopy_32fc(const Mpp32fc * pSrc, Mpp32fc * pDst, size_t nLength);

/** 
 * 64-bit complex signed integer, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCopy_64sc_Ctx(const Mpp64sc * pSrc, Mpp64sc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit complex signed integer, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCopy_64sc(const Mpp64sc * pSrc, Mpp64sc * pDst, size_t nLength);

/** 
 * 64-bit complex double, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCopy_64fc_Ctx(const Mpp64fc * pSrc, Mpp64fc * pDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit complex double, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsCopy_64fc(const Mpp64fc * pSrc, Mpp64fc * pDst, size_t nLength);

/** @} end of Copy */

/** @} signal_copy */

/** @} signal_initialization */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MC_MPPS_INITIALIZATION_H */
