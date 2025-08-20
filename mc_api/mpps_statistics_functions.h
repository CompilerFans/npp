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
#ifndef MC_MPPS_STATISTICS_FUNCTIONS_H
#define MC_MPPS_STATISTICS_FUNCTIONS_H
 
/**
 * \file mpps_statistics_functions.h
 * MPP Signal Processing Functionality.
 */
 
#include "mppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif


/** 
 *  \page signal_statistical_functions Statistical Functions
 *  @defgroup signal_statistical_functions Statistical Functions
 *  @ingroup mpps
 * Functions that provide global signal statistics like: sum, mean, standard
 * deviation, min, max, etc.
 *
 * @{
 *
 */

/** 
 * \section signal_min_every_or_max_every MinEvery And MaxEvery Functions
 * @defgroup signal_min_every_or_max_every MinEvery And MaxEvery Functions
 * Performs the min or max operation on the samples of a signal.
 *
 * @{  
 *
 */

/** 
 * 8-bit in place min value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinEvery_8u_I_Ctx(const Mpp8u * pSrc, Mpp8u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit in place min value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinEvery_8u_I(const Mpp8u * pSrc, Mpp8u * pSrcDst, size_t nLength);

/** 
 * 16-bit unsigned short integer in place min value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinEvery_16u_I_Ctx(const Mpp16u * pSrc, Mpp16u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short integer in place min value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.

 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinEvery_16u_I(const Mpp16u * pSrc, Mpp16u * pSrcDst, size_t nLength);

/** 
 * 16-bit signed short integer in place min value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinEvery_16s_I_Ctx(const Mpp16s * pSrc, Mpp16s * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer in place min value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.

 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinEvery_16s_I(const Mpp16s * pSrc, Mpp16s * pSrcDst, size_t nLength);

/** 
 * 32-bit signed integer in place min value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinEvery_32s_I_Ctx(const Mpp32s * pSrc, Mpp32s * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer in place min value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.

 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinEvery_32s_I(const Mpp32s * pSrc, Mpp32s * pSrcDst, size_t nLength);

/** 
 * 32-bit floating point in place min value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinEvery_32f_I_Ctx(const Mpp32f * pSrc, Mpp32f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point in place min value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.

 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinEvery_32f_I(const Mpp32f * pSrc, Mpp32f * pSrcDst, size_t nLength);

/** 
 * 64-bit floating point in place min value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinEvery_64f_I_Ctx(const Mpp64f * pSrc, Mpp64f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point in place min value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.

 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinEvery_64f_I(const Mpp64f * pSrc, Mpp64f * pSrcDst, size_t nLength);

/** 
 * 8-bit in place max value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxEvery_8u_I_Ctx(const Mpp8u * pSrc, Mpp8u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 8-bit in place max value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.

 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxEvery_8u_I(const Mpp8u * pSrc, Mpp8u * pSrcDst, size_t nLength);

/** 
 * 16-bit unsigned short integer in place max value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxEvery_16u_I_Ctx(const Mpp16u * pSrc, Mpp16u * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short integer in place max value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.

 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxEvery_16u_I(const Mpp16u * pSrc, Mpp16u * pSrcDst, size_t nLength);

/** 
 * 16-bit signed short integer in place max value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxEvery_16s_I_Ctx(const Mpp16s * pSrc, Mpp16s * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer in place max value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.

 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxEvery_16s_I(const Mpp16s * pSrc, Mpp16s * pSrcDst, size_t nLength);

/** 
 * 32-bit signed integer in place max value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxEvery_32s_I_Ctx(const Mpp32s * pSrc, Mpp32s * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer in place max value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.

 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxEvery_32s_I(const Mpp32s * pSrc, Mpp32s * pSrcDst, size_t nLength);

/** 
 * 32-bit floating point in place max value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxEvery_32f_I_Ctx(const Mpp32f * pSrc, Mpp32f * pSrcDst, size_t nLength, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point in place max value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.

 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxEvery_32f_I(const Mpp32f * pSrc, Mpp32f * pSrcDst, size_t nLength);

/** 
 *
 * @} signal_min_every_or_max_every
 *
 */

/** 
 * \section signal_sum Sum
 * @defgroup signal_sum Sum
 * Performs the sum operation on the samples of a signal.
 * @{  
 *
 */
 
 /** 
 * Device scratch buffer size (in bytes) for mppsSum_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumGetBufferSize_32f_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
 /** 
 * Device scratch buffer size (in bytes) for mppsSum_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.

 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumGetBufferSize_32f(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsSum_32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumGetBufferSize_32fc_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsSum_32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.

 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumGetBufferSize_32fc(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsSum_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumGetBufferSize_64f_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsSum_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.

 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumGetBufferSize_64f(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsSum_64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumGetBufferSize_64fc_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsSum_64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.

 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumGetBufferSize_64fc(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsSum_16s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumGetBufferSize_16s_Sfs_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsSum_16s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.

 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumGetBufferSize_16s_Sfs(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsSum_16sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumGetBufferSize_16sc_Sfs_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsSum_16sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.

 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumGetBufferSize_16sc_Sfs(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsSum_16sc32sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumGetBufferSize_16sc32sc_Sfs_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsSum_16sc32sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.

 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumGetBufferSize_16sc32sc_Sfs(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsSum_32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumGetBufferSize_32s_Sfs_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsSum_32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.

 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumGetBufferSize_32s_Sfs(size_t nLength, size_t * hpBufferSize /* host pointer */);
 
/** 
 * Device scratch buffer size (in bytes) for mppsSum_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumGetBufferSize_16s32s_Sfs_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsSum_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.

 * \return MPP_SUCCESS
 */
MppStatus 
mppsSumGetBufferSize_16s32s_Sfs(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * 32-bit float vector sum method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsSumGetBufferSize_32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSum_32f_Ctx(const Mpp32f * pSrc, size_t nLength, Mpp32f * pSum, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float vector sum method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsSumGetBufferSize_32f to determine the minium number of bytes required.

 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSum_32f(const Mpp32f * pSrc, size_t nLength, Mpp32f * pSum, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit float complex vector sum method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsSumGetBufferSize_32fc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSum_32fc_Ctx(const Mpp32fc * pSrc, size_t nLength, Mpp32fc * pSum, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float complex vector sum method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsSumGetBufferSize_32fc to determine the minium number of bytes required.

 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSum_32fc(const Mpp32fc * pSrc, size_t nLength, Mpp32fc * pSum, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit double vector sum method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsSumGetBufferSize_64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSum_64f_Ctx(const Mpp64f * pSrc, size_t nLength, Mpp64f * pSum, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit double vector sum method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsSumGetBufferSize_64f to determine the minium number of bytes required.

 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSum_64f(const Mpp64f * pSrc, size_t nLength, Mpp64f * pSum, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit double complex vector sum method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsSumGetBufferSize_64fc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSum_64fc_Ctx(const Mpp64fc * pSrc, size_t nLength, Mpp64fc * pSum, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit double complex vector sum method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsSumGetBufferSize_64fc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSum_64fc(const Mpp64fc * pSrc, size_t nLength, Mpp64fc * pSum, Mpp8u * pDeviceBuffer);

/** 
 * 16-bit short vector sum with integer scaling method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsSumGetBufferSize_16s_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSum_16s_Sfs_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp16s * pSum, int nScaleFactor, 
                    Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit short vector sum with integer scaling method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsSumGetBufferSize_16s_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSum_16s_Sfs(const Mpp16s * pSrc, size_t nLength, Mpp16s * pSum, int nScaleFactor, 
                Mpp8u * pDeviceBuffer);

/** 
 * 32-bit integer vector sum with integer scaling method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsSumGetBufferSize_32s_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSum_32s_Sfs_Ctx(const Mpp32s * pSrc, size_t nLength, Mpp32s * pSum, int nScaleFactor, 
                    Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit integer vector sum with integer scaling method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsSumGetBufferSize_32s_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSum_32s_Sfs(const Mpp32s * pSrc, size_t nLength, Mpp32s * pSum, int nScaleFactor, 
                Mpp8u * pDeviceBuffer);

/** 
 * 16-bit short complex vector sum with integer scaling method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsSumGetBufferSize_16sc_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSum_16sc_Sfs_Ctx(const Mpp16sc * pSrc, size_t nLength, Mpp16sc * pSum, int nScaleFactor, 
                     Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit short complex vector sum with integer scaling method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsSumGetBufferSize_16sc_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSum_16sc_Sfs(const Mpp16sc * pSrc, size_t nLength, Mpp16sc * pSum, int nScaleFactor, 
                 Mpp8u * pDeviceBuffer);

/** 
 * 16-bit short complex vector sum (32bit int complex) with integer scaling
 * method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsSumGetBufferSize_16sc32sc_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSum_16sc32sc_Sfs_Ctx(const Mpp16sc * pSrc, size_t nLength, Mpp32sc * pSum, int nScaleFactor, 
                         Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit short complex vector sum (32bit int complex) with integer scaling
 * method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsSumGetBufferSize_16sc32sc_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSum_16sc32sc_Sfs(const Mpp16sc * pSrc, size_t nLength, Mpp32sc * pSum, int nScaleFactor, 
                     Mpp8u * pDeviceBuffer);

/** 
 * 16-bit integer vector sum (32bit) with integer scaling method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsSumGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSum_16s32s_Sfs_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp32s * pSum, int nScaleFactor,
                       Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit integer vector sum (32bit) with integer scaling method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsSumGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsSum_16s32s_Sfs(const Mpp16s * pSrc, size_t nLength, Mpp32s * pSum, int nScaleFactor,
                   Mpp8u * pDeviceBuffer);

/** @} signal_sum */


/** 
 * \section signal_max Maximum
 * @defgroup signal_max Maximum
 * Performs the maximum operation on the samples of a signal.
 * @{
 *
 */

/** 
 * Device scratch buffer size (in bytes) for mppsMax_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxGetBufferSize_16s_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMax_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxGetBufferSize_16s(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMax_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxGetBufferSize_32s_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMax_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxGetBufferSize_32s(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMax_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxGetBufferSize_32f_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMax_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxGetBufferSize_32f(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMax_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxGetBufferSize_64f_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMax_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxGetBufferSize_64f(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * 16-bit integer vector max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxGetBufferSize_16s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMax_16s_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMax, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit integer vector max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxGetBufferSize_16s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMax_16s(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMax, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit integer vector max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxGetBufferSize_32s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMax_32s_Ctx(const Mpp32s * pSrc, size_t nLength, Mpp32s * pMax, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit integer vector max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxGetBufferSize_32s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMax_32s(const Mpp32s * pSrc, size_t nLength, Mpp32s * pMax, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit float vector max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxGetBufferSize_32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMax_32f_Ctx(const Mpp32f * pSrc, size_t nLength, Mpp32f * pMax, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float vector max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxGetBufferSize_32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMax_32f(const Mpp32f * pSrc, size_t nLength, Mpp32f * pMax, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit float vector max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxGetBufferSize_64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMax_64f_Ctx(const Mpp64f * pSrc, size_t nLength, Mpp64f * pMax, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit float vector max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxGetBufferSize_64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMax_64f(const Mpp64f * pSrc, size_t nLength, Mpp64f * pMax, Mpp8u * pDeviceBuffer);

/** 
 * Device scratch buffer size (in bytes) for mppsMaxIndx_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxIndxGetBufferSize_16s_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMaxIndx_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxIndxGetBufferSize_16s(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMaxIndx_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxIndxGetBufferSize_32s_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMaxIndx_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxIndxGetBufferSize_32s(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMaxIndx_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxIndxGetBufferSize_32f_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMaxIndx_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxIndxGetBufferSize_32f(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMaxIndx_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxIndxGetBufferSize_64f_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMaxIndx_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxIndxGetBufferSize_64f(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * 16-bit integer vector max index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pIndx Pointer to the index value of the first maximum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxIndxGetBufferSize_16s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxIndx_16s_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMax, int * pIndx, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit integer vector max index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pIndx Pointer to the index value of the first maximum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxIndxGetBufferSize_16s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxIndx_16s(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMax, int * pIndx, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit integer vector max index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pIndx Pointer to the index value of the first maximum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxIndxGetBufferSize_32s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxIndx_32s_Ctx(const Mpp32s * pSrc, size_t nLength, Mpp32s * pMax, int * pIndx, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit integer vector max index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pIndx Pointer to the index value of the first maximum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxIndxGetBufferSize_32s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxIndx_32s(const Mpp32s * pSrc, size_t nLength, Mpp32s * pMax, int * pIndx, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit float vector max index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pIndx Pointer to the index value of the first maximum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxIndxGetBufferSize_32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxIndx_32f_Ctx(const Mpp32f * pSrc, size_t nLength, Mpp32f * pMax, int * pIndx, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit integer vector max index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pIndx Pointer to the index value of the first maximum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxIndxGetBufferSize_32s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxIndx_32f(const Mpp32f * pSrc, size_t nLength, Mpp32f * pMax, int * pIndx, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit float vector max index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pIndx Pointer to the index value of the first maximum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxIndxGetBufferSize_64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxIndx_64f_Ctx(const Mpp64f * pSrc, size_t nLength, Mpp64f * pMax, int * pIndx, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit float vector max index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pIndx Pointer to the index value of the first maximum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxIndxGetBufferSize_64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxIndx_64f(const Mpp64f * pSrc, size_t nLength, Mpp64f * pMax, int * pIndx, Mpp8u * pDeviceBuffer);

/** 
 * Device scratch buffer size (in bytes) for mppsMaxAbs_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxAbsGetBufferSize_16s_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMaxAbs_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxAbsGetBufferSize_16s(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMaxAbs_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxAbsGetBufferSize_32s_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMaxAbs_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxAbsGetBufferSize_32s(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * 16-bit integer vector max absolute method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMaxAbs Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxAbsGetBufferSize_16s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxAbs_16s_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMaxAbs, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit integer vector max absolute method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMaxAbs Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxAbsGetBufferSize_16s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxAbs_16s(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMaxAbs, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit integer vector max absolute method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMaxAbs Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxAbsGetBufferSize_32s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxAbs_32s_Ctx(const Mpp32s * pSrc, size_t nLength, Mpp32s * pMaxAbs, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit integer vector max absolute method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMaxAbs Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxAbsGetBufferSize_32s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxAbs_32s(const Mpp32s * pSrc, size_t nLength, Mpp32s * pMaxAbs, Mpp8u * pDeviceBuffer);

/** 
 * Device scratch buffer size (in bytes) for mppsMaxAbsIndx_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxAbsIndxGetBufferSize_16s_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMaxAbsIndx_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxAbsIndxGetBufferSize_16s(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMaxAbsIndx_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxAbsIndxGetBufferSize_32s_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMaxAbsIndx_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMaxAbsIndxGetBufferSize_32s(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * 16-bit integer vector max absolute index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMaxAbs Pointer to the output result.
 * \param pIndx Pointer to the index value of the first maximum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxAbsIndxGetBufferSize_16s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxAbsIndx_16s_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMaxAbs, int * pIndx, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit integer vector max absolute index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMaxAbs Pointer to the output result.
 * \param pIndx Pointer to the index value of the first maximum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxAbsIndxGetBufferSize_16s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxAbsIndx_16s(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMaxAbs, int * pIndx, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit integer vector max absolute index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMaxAbs Pointer to the output result.
 * \param pIndx Pointer to the index value of the first maximum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxAbsIndxGetBufferSize_32s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxAbsIndx_32s_Ctx(const Mpp32s * pSrc, size_t nLength, Mpp32s * pMaxAbs, int * pIndx, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit integer vector max absolute index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMaxAbs Pointer to the output result.
 * \param pIndx Pointer to the index value of the first maximum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaxAbsIndxGetBufferSize_32s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaxAbsIndx_32s(const Mpp32s * pSrc, size_t nLength, Mpp32s * pMaxAbs, int * pIndx, Mpp8u * pDeviceBuffer);

/** @} signal_max */

/** 
 * \section signal_min Minimum
 * @defgroup signal_min Minimum
 * Performs the minimum operation on the samples of a signal.
 * @{
 *
 */

/** 
 * Device scratch buffer size (in bytes) for mppsMin_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinGetBufferSize_16s_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMin_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinGetBufferSize_16s(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMin_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinGetBufferSize_32s_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMin_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinGetBufferSize_32s(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMin_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinGetBufferSize_32f_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMin_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinGetBufferSize_32f(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMin_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinGetBufferSize_64f_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMin_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinGetBufferSize_64f(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * 16-bit integer vector min method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinGetBufferSize_16s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMin_16s_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMin, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit integer vector min method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinGetBufferSize_16s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMin_16s(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMin, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit integer vector min method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinGetBufferSize_32s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMin_32s_Ctx(const Mpp32s * pSrc, size_t nLength, Mpp32s * pMin, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit integer vector min method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinGetBufferSize_32s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMin_32s(const Mpp32s * pSrc, size_t nLength, Mpp32s * pMin, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit integer vector min method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinGetBufferSize_32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMin_32f_Ctx(const Mpp32f * pSrc, size_t nLength, Mpp32f * pMin, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit integer vector min method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinGetBufferSize_32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMin_32f(const Mpp32f * pSrc, size_t nLength, Mpp32f * pMin, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit integer vector min method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinGetBufferSize_64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMin_64f_Ctx(const Mpp64f * pSrc, size_t nLength, Mpp64f * pMin, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit integer vector min method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinGetBufferSize_64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMin_64f(const Mpp64f * pSrc, size_t nLength, Mpp64f * pMin, Mpp8u * pDeviceBuffer);

/** 
 * Device scratch buffer size (in bytes) for mppsMinIndx_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinIndxGetBufferSize_16s_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMinIndx_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinIndxGetBufferSize_16s(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMinIndx_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinIndxGetBufferSize_32s_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMinIndx_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinIndxGetBufferSize_32s(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMinIndx_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinIndxGetBufferSize_32f_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMinIndx_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinIndxGetBufferSize_32f(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMinIndx_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinIndxGetBufferSize_64f_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMinIndx_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinIndxGetBufferSize_64f(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * 16-bit integer vector min index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pIndx Pointer to the index value of the first minimum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinIndxGetBufferSize_16s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinIndx_16s_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMin, int * pIndx, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit integer vector min index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pIndx Pointer to the index value of the first minimum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinIndxGetBufferSize_16s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinIndx_16s(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMin, int * pIndx, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit integer vector min index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pIndx Pointer to the index value of the first minimum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinIndxGetBufferSize_32s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinIndx_32s_Ctx(const Mpp32s * pSrc, size_t nLength, Mpp32s * pMin, int * pIndx, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit integer vector min index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pIndx Pointer to the index value of the first minimum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinIndxGetBufferSize_32s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinIndx_32s(const Mpp32s * pSrc, size_t nLength, Mpp32s * pMin, int * pIndx, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit float vector min index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pIndx Pointer to the index value of the first minimum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinIndxGetBufferSize_32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinIndx_32f_Ctx(const Mpp32f * pSrc, size_t nLength, Mpp32f * pMin, int * pIndx, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float vector min index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pIndx Pointer to the index value of the first minimum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinIndxGetBufferSize_32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinIndx_32f(const Mpp32f * pSrc, size_t nLength, Mpp32f * pMin, int * pIndx, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit float vector min index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pIndx Pointer to the index value of the first minimum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinIndxGetBufferSize_64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinIndx_64f_Ctx(const Mpp64f * pSrc, size_t nLength, Mpp64f * pMin, int * pIndx, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit float vector min index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pIndx Pointer to the index value of the first minimum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinIndxGetBufferSize_64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinIndx_64f(const Mpp64f * pSrc, size_t nLength, Mpp64f * pMin, int * pIndx, Mpp8u * pDeviceBuffer);

/** 
 * Device scratch buffer size (in bytes) for mppsMinAbs_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinAbsGetBufferSize_16s_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMinAbs_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinAbsGetBufferSize_16s(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMinAbs_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinAbsGetBufferSize_32s_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMinAbs_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinAbsGetBufferSize_32s(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * 16-bit integer vector min absolute method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMinAbs Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinAbsGetBufferSize_16s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinAbs_16s_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMinAbs, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit integer vector min absolute method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMinAbs Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinAbsGetBufferSize_16s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinAbs_16s(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMinAbs, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit integer vector min absolute method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMinAbs Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinAbsGetBufferSize_16s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinAbs_32s_Ctx(const Mpp32s * pSrc, size_t nLength, Mpp32s * pMinAbs, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit integer vector min absolute method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMinAbs Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinAbsGetBufferSize_16s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinAbs_32s(const Mpp32s * pSrc, size_t nLength, Mpp32s * pMinAbs, Mpp8u * pDeviceBuffer);

/** 
 * Device scratch buffer size (in bytes) for mppsMinAbsIndx_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinAbsIndxGetBufferSize_16s_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMinAbsIndx_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinAbsIndxGetBufferSize_16s(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMinAbsIndx_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinAbsIndxGetBufferSize_32s_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMinAbsIndx_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMinAbsIndxGetBufferSize_32s(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * 16-bit integer vector min absolute index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMinAbs Pointer to the output result.
 * \param pIndx Pointer to the index value of the first minimum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinAbsIndxGetBufferSize_16s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinAbsIndx_16s_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMinAbs, int * pIndx, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit integer vector min absolute index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMinAbs Pointer to the output result.
 * \param pIndx Pointer to the index value of the first minimum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinAbsIndxGetBufferSize_16s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinAbsIndx_16s(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMinAbs, int * pIndx, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit integer vector min absolute index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMinAbs Pointer to the output result.
 * \param pIndx Pointer to the index value of the first minimum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinAbsIndxGetBufferSize_32s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinAbsIndx_32s_Ctx(const Mpp32s * pSrc, size_t nLength, Mpp32s * pMinAbs, int * pIndx, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit integer vector min absolute index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMinAbs Pointer to the output result.
 * \param pIndx Pointer to the index value of the first minimum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinAbsIndxGetBufferSize_32s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinAbsIndx_32s(const Mpp32s * pSrc, size_t nLength, Mpp32s * pMinAbs, int * pIndx, Mpp8u * pDeviceBuffer);

/** @} signal_min */

/** 
 * \section signal_mean Mean
 * @defgroup signal_mean Mean
 * Performs the mean operation on the samples of a signal.
 * @{
 *
 */

/** 
 * Device scratch buffer size (in bytes) for mppsMean_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMeanGetBufferSize_32f_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMean_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMeanGetBufferSize_32f(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMean_32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMeanGetBufferSize_32fc_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMean_32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMeanGetBufferSize_32fc(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMean_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMeanGetBufferSize_64f_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMean_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMeanGetBufferSize_64f(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMean_64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMeanGetBufferSize_64fc_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMean_64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMeanGetBufferSize_64fc(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMean_16s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMeanGetBufferSize_16s_Sfs_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMean_16s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMeanGetBufferSize_16s_Sfs(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMean_32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMeanGetBufferSize_32s_Sfs_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMean_32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMeanGetBufferSize_32s_Sfs(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMean_16sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMeanGetBufferSize_16sc_Sfs_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMean_16sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMeanGetBufferSize_16sc_Sfs(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * 32-bit float vector mean method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMeanGetBufferSize_32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMean_32f_Ctx(const Mpp32f * pSrc, size_t nLength, Mpp32f * pMean, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float vector mean method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMeanGetBufferSize_32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMean_32f(const Mpp32f * pSrc, size_t nLength, Mpp32f * pMean, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit float complex vector mean method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMeanGetBufferSize_32fc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMean_32fc_Ctx(const Mpp32fc * pSrc, size_t nLength, Mpp32fc * pMean, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float complex vector mean method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMeanGetBufferSize_32fc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMean_32fc(const Mpp32fc * pSrc, size_t nLength, Mpp32fc * pMean, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit double vector mean method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMeanGetBufferSize_64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMean_64f_Ctx(const Mpp64f * pSrc, size_t nLength, Mpp64f * pMean, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit double vector mean method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMeanGetBufferSize_64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMean_64f(const Mpp64f * pSrc, size_t nLength, Mpp64f * pMean, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit double complex vector mean method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMeanGetBufferSize_64fc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMean_64fc_Ctx(const Mpp64fc * pSrc, size_t nLength, Mpp64fc * pMean, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit double complex vector mean method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMeanGetBufferSize_64fc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMean_64fc(const Mpp64fc * pSrc, size_t nLength, Mpp64fc * pMean, Mpp8u * pDeviceBuffer);

/** 
 * 16-bit short vector mean with integer scaling method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMeanGetBufferSize_16s_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMean_16s_Sfs_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMean, int nScaleFactor, 
                     Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit short vector mean with integer scaling method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMeanGetBufferSize_16s_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMean_16s_Sfs(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMean, int nScaleFactor, 
                 Mpp8u * pDeviceBuffer);

/** 
 * 32-bit integer vector mean with integer scaling method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMeanGetBufferSize_32s_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMean_32s_Sfs_Ctx(const Mpp32s * pSrc, size_t nLength, Mpp32s * pMean, int nScaleFactor, 
                     Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit integer vector mean with integer scaling method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMeanGetBufferSize_32s_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMean_32s_Sfs(const Mpp32s * pSrc, size_t nLength, Mpp32s * pMean, int nScaleFactor, 
                 Mpp8u * pDeviceBuffer);

/** 
 * 16-bit short complex vector mean with integer scaling method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMeanGetBufferSize_16sc_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMean_16sc_Sfs_Ctx(const Mpp16sc * pSrc, size_t nLength, Mpp16sc * pMean, int nScaleFactor, 
                      Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit short complex vector mean with integer scaling method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMeanGetBufferSize_16sc_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMean_16sc_Sfs(const Mpp16sc * pSrc, size_t nLength, Mpp16sc * pMean, int nScaleFactor, 
                  Mpp8u * pDeviceBuffer);

/** @} signal_mean */

/** 
 * \section signal_standard_deviation Standard Deviation
 * @defgroup signal_standard_deviation Standard Deviation
 * Calculates the standard deviation for the samples of a signal.
 * @{
 *
 */

/** 
 * Device scratch buffer size (in bytes) for mppsStdDev_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsStdDevGetBufferSize_32f_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsStdDev_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsStdDevGetBufferSize_32f(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsStdDev_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsStdDevGetBufferSize_64f_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsStdDev_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsStdDevGetBufferSize_64f(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsStdDev_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsStdDevGetBufferSize_16s32s_Sfs_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsStdDev_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsStdDevGetBufferSize_16s32s_Sfs(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsStdDev_16s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsStdDevGetBufferSize_16s_Sfs_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsStdDev_16s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsStdDevGetBufferSize_16s_Sfs(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * 32-bit float vector standard deviation method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pStdDev Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsStdDevGetBufferSize_32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsStdDev_32f_Ctx(const Mpp32f * pSrc, size_t nLength, Mpp32f * pStdDev, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float vector standard deviation method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pStdDev Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsStdDevGetBufferSize_32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsStdDev_32f(const Mpp32f * pSrc, size_t nLength, Mpp32f * pStdDev, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit float vector standard deviation method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pStdDev Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsStdDevGetBufferSize_64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsStdDev_64f_Ctx(const Mpp64f * pSrc, size_t nLength, Mpp64f * pStdDev, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit float vector standard deviation method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pStdDev Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsStdDevGetBufferSize_64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsStdDev_64f(const Mpp64f * pSrc, size_t nLength, Mpp64f * pStdDev, Mpp8u * pDeviceBuffer);

/** 
 * 16-bit float vector standard deviation method (return value is 32-bit)
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pStdDev Pointer to the output result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsStdDevGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsStdDev_16s32s_Sfs_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp32s * pStdDev, int nScaleFactor, 
                          Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit float vector standard deviation method (return value is 32-bit)
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pStdDev Pointer to the output result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsStdDevGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsStdDev_16s32s_Sfs(const Mpp16s * pSrc, size_t nLength, Mpp32s * pStdDev, int nScaleFactor, 
                      Mpp8u * pDeviceBuffer);

/** 
 * 16-bit float vector standard deviation method (return value is also 16-bit)
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pStdDev Pointer to the output result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsStdDevGetBufferSize_16s_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsStdDev_16s_Sfs_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp16s * pStdDev, int nScaleFactor, 
                       Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit float vector standard deviation method (return value is also 16-bit)
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pStdDev Pointer to the output result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsStdDevGetBufferSize_16s_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsStdDev_16s_Sfs(const Mpp16s * pSrc, size_t nLength, Mpp16s * pStdDev, int nScaleFactor, 
                   Mpp8u * pDeviceBuffer);

/** @} signal_standard_deviation */

/** 
 * \section signal_mean_and_standard_deviation Mean And Standard Deviation
 * @defgroup signal_mean_and_standard_deviation Mean And Standard Deviation
 * Performs the mean and calculates the standard deviation for the samples of a signal.
 * @{
 *
 */

/** 
 * Device scratch buffer size (in bytes) for mppsMeanStdDev_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMeanStdDevGetBufferSize_32f_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMeanStdDev_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMeanStdDevGetBufferSize_32f(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMeanStdDev_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMeanStdDevGetBufferSize_64f_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMeanStdDev_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMeanStdDevGetBufferSize_64f(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMeanStdDev_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMeanStdDevGetBufferSize_16s32s_Sfs_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMeanStdDev_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMeanStdDevGetBufferSize_16s32s_Sfs(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for mppsMeanStdDev_16s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMeanStdDevGetBufferSize_16s_Sfs_Ctx(size_t nLength, size_t * hpBufferSize /* host pointer */, MppStreamContext mppStreamCtx);
/** 
 * Device scratch buffer size (in bytes) for mppsMeanStdDev_16s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \return MPP_SUCCESS
 */
MppStatus 
mppsMeanStdDevGetBufferSize_16s_Sfs(size_t nLength, size_t * hpBufferSize /* host pointer */);

/** 
 * 32-bit float vector mean and standard deviation method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output mean value.
 * \param pStdDev Pointer to the output standard deviation value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMeanStdDevGetBufferSize_32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMeanStdDev_32f_Ctx(const Mpp32f * pSrc, size_t nLength, Mpp32f * pMean, Mpp32f * pStdDev, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float vector mean and standard deviation method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output mean value.
 * \param pStdDev Pointer to the output standard deviation value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMeanStdDevGetBufferSize_32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMeanStdDev_32f(const Mpp32f * pSrc, size_t nLength, Mpp32f * pMean, Mpp32f * pStdDev, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit float vector mean and standard deviation method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output mean value.
 * \param pStdDev Pointer to the output standard deviation value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMeanStdDevGetBufferSize_64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMeanStdDev_64f_Ctx(const Mpp64f * pSrc, size_t nLength, Mpp64f * pMean, Mpp64f * pStdDev, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit float vector mean and standard deviation method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output mean value.
 * \param pStdDev Pointer to the output standard deviation value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMeanStdDevGetBufferSize_64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMeanStdDev_64f(const Mpp64f * pSrc, size_t nLength, Mpp64f * pMean, Mpp64f * pStdDev, Mpp8u * pDeviceBuffer);

/** 
 * 16-bit float vector mean and standard deviation method (return values are 32-bit)
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output mean value.
 * \param pStdDev Pointer to the output standard deviation value.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMeanStdDevGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMeanStdDev_16s32s_Sfs_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp32s * pMean, Mpp32s * pStdDev, int nScaleFactor, 
                              Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit float vector mean and standard deviation method (return values are 32-bit)
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output mean value.
 * \param pStdDev Pointer to the output standard deviation value.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMeanStdDevGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMeanStdDev_16s32s_Sfs(const Mpp16s * pSrc, size_t nLength, Mpp32s * pMean, Mpp32s * pStdDev, int nScaleFactor, 
                          Mpp8u * pDeviceBuffer);

/** 
 * 16-bit float vector mean and standard deviation method (return values are also 16-bit)
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output mean value.
 * \param pStdDev Pointer to the output standard deviation value.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMeanStdDevGetBufferSize_16s_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMeanStdDev_16s_Sfs_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMean, Mpp16s * pStdDev, int nScaleFactor, 
                           Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit float vector mean and standard deviation method (return values are also 16-bit)
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output mean value.
 * \param pStdDev Pointer to the output standard deviation value.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMeanStdDevGetBufferSize_16s_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMeanStdDev_16s_Sfs(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMean, Mpp16s * pStdDev, int nScaleFactor, 
                       Mpp8u * pDeviceBuffer);

/** @} signal_mean_and_standard_deviation */

/** 
 * \section signal_min_max Minimum Maximum
 * @defgroup signal_min_max Minimum Maximum
 * Performs the maximum and the minimum operation on the samples of a signal.
 * @{
 *
 */

/** 
 * Device-buffer size (in bytes) for mppsMinMax_8u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxGetBufferSize_8u_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMinMax_8u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxGetBufferSize_8u(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMinMax_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxGetBufferSize_16s_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMinMax_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxGetBufferSize_16s(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMinMax_16u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxGetBufferSize_16u_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMinMax_16u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxGetBufferSize_16u(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMinMax_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxGetBufferSize_32s_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMinMax_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxGetBufferSize_32s(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMinMax_32u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxGetBufferSize_32u_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMinMax_32u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxGetBufferSize_32u(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMinMax_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxGetBufferSize_32f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMinMax_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxGetBufferSize_32f(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMinMax_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxGetBufferSize_64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMinMax_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxGetBufferSize_64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 8-bit char vector min and max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMax Pointer to the max output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxGetBufferSize_8u to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMax_8u_Ctx(const Mpp8u * pSrc, size_t nLength, Mpp8u * pMin, Mpp8u * pMax, 
                  Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 8-bit char vector min and max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMax Pointer to the max output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxGetBufferSize_8u to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMax_8u(const Mpp8u * pSrc, size_t nLength, Mpp8u * pMin, Mpp8u * pMax, 
              Mpp8u * pDeviceBuffer);

/** 
 * 16-bit signed short vector min and max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMax Pointer to the max output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxGetBufferSize_16s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMax_16s_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMin, Mpp16s * pMax, 
                   Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short vector min and max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMax Pointer to the max output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxGetBufferSize_16s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMax_16s(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMin, Mpp16s * pMax, 
               Mpp8u * pDeviceBuffer);

/** 
 * 16-bit unsigned short vector min and max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMax Pointer to the max output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxGetBufferSize_16u to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMax_16u_Ctx(const Mpp16u * pSrc, size_t nLength, Mpp16u * pMin, Mpp16u * pMax, 
                   Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short vector min and max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMax Pointer to the max output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxGetBufferSize_16u to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMax_16u(const Mpp16u * pSrc, size_t nLength, Mpp16u * pMin, Mpp16u * pMax, 
               Mpp8u * pDeviceBuffer);

/** 
 * 32-bit unsigned int vector min and max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMax Pointer to the max output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxGetBufferSize_32u to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMax_32u_Ctx(const Mpp32u * pSrc, size_t nLength, Mpp32u * pMin, Mpp32u * pMax, 
                   Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned int vector min and max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMax Pointer to the max output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxGetBufferSize_32u to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMax_32u(const Mpp32u * pSrc, size_t nLength, Mpp32u * pMin, Mpp32u * pMax, 
               Mpp8u * pDeviceBuffer);

/** 
 * 32-bit signed int vector min and max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMax Pointer to the max output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxGetBufferSize_32s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMax_32s_Ctx(const Mpp32s * pSrc, size_t nLength, Mpp32s * pMin, Mpp32s * pMax, 
                   Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed int vector min and max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMax Pointer to the max output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxGetBufferSize_32s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMax_32s(const Mpp32s * pSrc, size_t nLength, Mpp32s * pMin, Mpp32s * pMax, 
               Mpp8u * pDeviceBuffer);

/** 
 * 32-bit float vector min and max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMax Pointer to the max output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxGetBufferSize_32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMax_32f_Ctx(const Mpp32f * pSrc, size_t nLength, Mpp32f * pMin, Mpp32f * pMax, 
                   Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float vector min and max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMax Pointer to the max output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxGetBufferSize_32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMax_32f(const Mpp32f * pSrc, size_t nLength, Mpp32f * pMin, Mpp32f * pMax, 
               Mpp8u * pDeviceBuffer);

/** 
 * 64-bit double vector min and max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMax Pointer to the max output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxGetBufferSize_64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMax_64f_Ctx(const Mpp64f * pSrc, size_t nLength, Mpp64f * pMin, Mpp64f * pMax, 
                   Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit double vector min and max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMax Pointer to the max output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxGetBufferSize_64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMax_64f(const Mpp64f * pSrc, size_t nLength, Mpp64f * pMin, Mpp64f * pMax, 
               Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsMinMaxIndx_8u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxIndxGetBufferSize_8u_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMinMaxIndx_8u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxIndxGetBufferSize_8u(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMinMaxIndx_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxIndxGetBufferSize_16s_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMinMaxIndx_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxIndxGetBufferSize_16s(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMinMaxIndx_16u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxIndxGetBufferSize_16u_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMinMaxIndx_16u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxIndxGetBufferSize_16u(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMinMaxIndx_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxIndxGetBufferSize_32s_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMinMaxIndx_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxIndxGetBufferSize_32s(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMinMaxIndx_32u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxIndxGetBufferSize_32u_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMinMaxIndx_32u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxIndxGetBufferSize_32u(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMinMaxIndx_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxIndxGetBufferSize_32f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMinMaxIndx_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxIndxGetBufferSize_32f(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMinMaxIndx_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxIndxGetBufferSize_64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMinMaxIndx_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMinMaxIndxGetBufferSize_64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 8-bit char vector min and max with indices method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMinIndx Pointer to the index of the first min value.
 * \param pMax Pointer to the max output result.
 * \param pMaxIndx Pointer to the index of the first max value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxIndxGetBufferSize_8u to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMaxIndx_8u_Ctx(const Mpp8u * pSrc, size_t nLength, Mpp8u * pMin, int * pMinIndx, Mpp8u * pMax, int * pMaxIndx,
                      Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 8-bit char vector min and max with indices method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMinIndx Pointer to the index of the first min value.
 * \param pMax Pointer to the max output result.
 * \param pMaxIndx Pointer to the index of the first max value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxIndxGetBufferSize_8u to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMaxIndx_8u(const Mpp8u * pSrc, size_t nLength, Mpp8u * pMin, int * pMinIndx, Mpp8u * pMax, int * pMaxIndx,
                  Mpp8u * pDeviceBuffer);

/** 
 * 16-bit signed short vector min and max with indices method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMinIndx Pointer to the index of the first min value.
 * \param pMax Pointer to the max output result.
 * \param pMaxIndx Pointer to the index of the first max value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxIndxGetBufferSize_16s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMaxIndx_16s_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMin, int * pMinIndx, Mpp16s * pMax, int * pMaxIndx,
                       Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short vector min and max with indices method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMinIndx Pointer to the index of the first min value.
 * \param pMax Pointer to the max output result.
 * \param pMaxIndx Pointer to the index of the first max value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxIndxGetBufferSize_16s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMaxIndx_16s(const Mpp16s * pSrc, size_t nLength, Mpp16s * pMin, int * pMinIndx, Mpp16s * pMax, int * pMaxIndx,
                   Mpp8u * pDeviceBuffer);

/** 
 * 16-bit unsigned short vector min and max with indices method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMinIndx Pointer to the index of the first min value.
 * \param pMax Pointer to the max output result.
 * \param pMaxIndx Pointer to the index of the first max value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxIndxGetBufferSize_16u to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMaxIndx_16u_Ctx(const Mpp16u * pSrc, size_t nLength, Mpp16u * pMin, int * pMinIndx, Mpp16u * pMax, int * pMaxIndx,
                       Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short vector min and max with indices method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMinIndx Pointer to the index of the first min value.
 * \param pMax Pointer to the max output result.
 * \param pMaxIndx Pointer to the index of the first max value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxIndxGetBufferSize_16u to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */   
MppStatus 
mppsMinMaxIndx_16u(const Mpp16u * pSrc, size_t nLength, Mpp16u * pMin, int * pMinIndx, Mpp16u * pMax, int * pMaxIndx,
                   Mpp8u * pDeviceBuffer);

/** 
 * 32-bit signed short vector min and max with indices method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMinIndx Pointer to the index of the first min value.
 * \param pMax Pointer to the max output result.
 * \param pMaxIndx Pointer to the index of the first max value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxIndxGetBufferSize_32s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMaxIndx_32s_Ctx(const Mpp32s * pSrc, size_t nLength, Mpp32s * pMin, int * pMinIndx, Mpp32s * pMax, int * pMaxIndx,
                       Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed short vector min and max with indices method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMinIndx Pointer to the index of the first min value.
 * \param pMax Pointer to the max output result.
 * \param pMaxIndx Pointer to the index of the first max value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxIndxGetBufferSize_32s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMaxIndx_32s(const Mpp32s * pSrc, size_t nLength, Mpp32s * pMin, int * pMinIndx, Mpp32s * pMax, int * pMaxIndx,
                   Mpp8u * pDeviceBuffer);

/** 
 * 32-bit unsigned short vector min and max with indices method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMinIndx Pointer to the index of the first min value.
 * \param pMax Pointer to the max output result.
 * \param pMaxIndx Pointer to the index of the first max value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxIndxGetBufferSize_32u to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMaxIndx_32u_Ctx(const Mpp32u * pSrc, size_t nLength, Mpp32u * pMin, int * pMinIndx, Mpp32u * pMax, int * pMaxIndx,
                       Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned short vector min and max with indices method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMinIndx Pointer to the index of the first min value.
 * \param pMax Pointer to the max output result.
 * \param pMaxIndx Pointer to the index of the first max value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxIndxGetBufferSize_32u to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMaxIndx_32u(const Mpp32u * pSrc, size_t nLength, Mpp32u * pMin, int * pMinIndx, Mpp32u * pMax, int * pMaxIndx,
                   Mpp8u * pDeviceBuffer);

/** 
 * 32-bit float vector min and max with indices method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMinIndx Pointer to the index of the first min value.
 * \param pMax Pointer to the max output result.
 * \param pMaxIndx Pointer to the index of the first max value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxIndxGetBufferSize_32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMaxIndx_32f_Ctx(const Mpp32f * pSrc, size_t nLength, Mpp32f * pMin, int * pMinIndx, Mpp32f * pMax, int * pMaxIndx,
                       Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float vector min and max with indices method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMinIndx Pointer to the index of the first min value.
 * \param pMax Pointer to the max output result.
 * \param pMaxIndx Pointer to the index of the first max value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxIndxGetBufferSize_32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMaxIndx_32f(const Mpp32f * pSrc, size_t nLength, Mpp32f * pMin, int * pMinIndx, Mpp32f * pMax, int * pMaxIndx,
                   Mpp8u * pDeviceBuffer);

/** 
 * 64-bit float vector min and max with indices method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMinIndx Pointer to the index of the first min value.
 * \param pMax Pointer to the max output result.
 * \param pMaxIndx Pointer to the index of the first max value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxIndxGetBufferSize_64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMaxIndx_64f_Ctx(const Mpp64f * pSrc, size_t nLength, Mpp64f * pMin, int * pMinIndx, Mpp64f * pMax, int * pMaxIndx, 
                       Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit float vector min and max with indices method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMinIndx Pointer to the index of the first min value.
 * \param pMax Pointer to the max output result.
 * \param pMaxIndx Pointer to the index of the first max value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMinMaxIndxGetBufferSize_64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMinMaxIndx_64f(const Mpp64f * pSrc, size_t nLength, Mpp64f * pMin, int * pMinIndx, Mpp64f * pMax, int * pMaxIndx, 
                   Mpp8u * pDeviceBuffer);

/** @} signal_min_max */

/** 
 * \section signal_infinity_norm Infinity Norm
 * @defgroup signal_infinity_norm Infinity Norm
 * Performs the infinity norm on the samples of a signal.
 * @{
 *
 */

/** 
 * Device-buffer size (in bytes) for mppsNorm_Inf_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormInfGetBufferSize_32f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNorm_Inf_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormInfGetBufferSize_32f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 32-bit float vector C norm method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormInfGetBufferSize_32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_Inf_32f_Ctx(const Mpp32f * pSrc, size_t nLength, Mpp32f * pNorm,
                     Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float vector C norm method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormInfGetBufferSize_32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_Inf_32f(const Mpp32f * pSrc, size_t nLength, Mpp32f * pNorm,
                 Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNorm_Inf_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormInfGetBufferSize_64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNorm_Inf_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormInfGetBufferSize_64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 64-bit float vector C norm method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormInfGetBufferSize_64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_Inf_64f_Ctx(const Mpp64f * pSrc, size_t nLength, Mpp64f * pNorm,
                     Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit float vector C norm method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormInfGetBufferSize_64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_Inf_64f(const Mpp64f * pSrc, size_t nLength, Mpp64f * pNorm,
                 Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNorm_Inf_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormInfGetBufferSize_16s32f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNorm_Inf_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormInfGetBufferSize_16s32f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer vector C norm method, return value is 32-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormInfGetBufferSize_16s32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_Inf_16s32f_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp32f * pNorm,
                        Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer vector C norm method, return value is 32-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormInfGetBufferSize_16s32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_Inf_16s32f(const Mpp16s * pSrc, size_t nLength, Mpp32f * pNorm,
                    Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNorm_Inf_32fc32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormInfGetBufferSize_32fc32f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNorm_Inf_32fc32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormInfGetBufferSize_32fc32f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 32-bit float complex vector C norm method, return value is 32-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormInfGetBufferSize_32fc32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_Inf_32fc32f_Ctx(const Mpp32fc * pSrc, size_t nLength, Mpp32f * pNorm,
                         Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float complex vector C norm method, return value is 32-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormInfGetBufferSize_32fc32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_Inf_32fc32f(const Mpp32fc * pSrc, size_t nLength, Mpp32f * pNorm,
                     Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNorm_Inf_64fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormInfGetBufferSize_64fc64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNorm_Inf_64fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormInfGetBufferSize_64fc64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 64-bit float complex vector C norm method, return value is 64-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormInfGetBufferSize_64fc64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_Inf_64fc64f_Ctx(const Mpp64fc * pSrc, size_t nLength, Mpp64f * pNorm,
                         Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit float complex vector C norm method, return value is 64-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormInfGetBufferSize_64fc64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_Inf_64fc64f(const Mpp64fc * pSrc, size_t nLength, Mpp64f * pNorm,
                     Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNorm_Inf_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormInfGetBufferSize_16s32s_Sfs_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNorm_Inf_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormInfGetBufferSize_16s32s_Sfs(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer vector C norm method, return value is 32-bit signed integer.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormInfGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_Inf_16s32s_Sfs_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp32s * pNorm, int nScaleFactor,
                            Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer vector C norm method, return value is 32-bit signed integer.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormInfGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_Inf_16s32s_Sfs(const Mpp16s * pSrc, size_t nLength, Mpp32s * pNorm, int nScaleFactor,
                        Mpp8u * pDeviceBuffer);

/** @} signal_infinity_norm */

/** 
 * \section signal_L1_norm L1 Norm
 * @defgroup signal_L1_norm L1 Norm
 * Performs the L1 norm on the samples of a signal.
 * @{
 *
 */

/** 
 * Device-buffer size (in bytes) for mppsNorm_L1_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL1GetBufferSize_32f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNorm_L1_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL1GetBufferSize_32f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 32-bit float vector L1 norm method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL1GetBufferSize_32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L1_32f_Ctx(const Mpp32f * pSrc, size_t nLength, Mpp32f * pNorm,
                    Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float vector L1 norm method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL1GetBufferSize_32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L1_32f(const Mpp32f * pSrc, size_t nLength, Mpp32f * pNorm,
                Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNorm_L1_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL1GetBufferSize_64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNorm_L1_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL1GetBufferSize_64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 64-bit float vector L1 norm method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL1GetBufferSize_64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L1_64f_Ctx(const Mpp64f * pSrc, size_t nLength, Mpp64f * pNorm,
                    Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit float vector L1 norm method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL1GetBufferSize_64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L1_64f(const Mpp64f * pSrc, size_t nLength, Mpp64f * pNorm,
                Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNorm_L1_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL1GetBufferSize_16s32f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNorm_L1_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL1GetBufferSize_16s32f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer vector L1 norm method, return value is 32-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the L1 norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL1GetBufferSize_16s32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L1_16s32f_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp32f * pNorm,
                       Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer vector L1 norm method, return value is 32-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the L1 norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL1GetBufferSize_16s32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L1_16s32f(const Mpp16s * pSrc, size_t nLength, Mpp32f * pNorm,
                   Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNorm_L1_32fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL1GetBufferSize_32fc64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNorm_L1_32fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL1GetBufferSize_32fc64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 32-bit float complex vector L1 norm method, return value is 64-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL1GetBufferSize_32fc64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L1_32fc64f_Ctx(const Mpp32fc * pSrc, size_t nLength, Mpp64f * pNorm,
                        Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float complex vector L1 norm method, return value is 64-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL1GetBufferSize_32fc64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L1_32fc64f(const Mpp32fc * pSrc, size_t nLength, Mpp64f * pNorm,
                    Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNorm_L1_64fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL1GetBufferSize_64fc64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNorm_L1_64fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL1GetBufferSize_64fc64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 64-bit float complex vector L1 norm method, return value is 64-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL1GetBufferSize_64fc64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L1_64fc64f_Ctx(const Mpp64fc * pSrc, size_t nLength, Mpp64f * pNorm,
                        Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit float complex vector L1 norm method, return value is 64-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL1GetBufferSize_64fc64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L1_64fc64f(const Mpp64fc * pSrc, size_t nLength, Mpp64f * pNorm,
                    Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNorm_L1_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL1GetBufferSize_16s32s_Sfs_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNorm_L1_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL1GetBufferSize_16s32s_Sfs(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer vector L1 norm method, return value is 32-bit signed integer.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL1GetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L1_16s32s_Sfs_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp32s * pNorm, int nScaleFactor,
                           Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer vector L1 norm method, return value is 32-bit signed integer.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL1GetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L1_16s32s_Sfs(const Mpp16s * pSrc, size_t nLength, Mpp32s * pNorm, int nScaleFactor,
                       Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNorm_L1_16s64s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL1GetBufferSize_16s64s_Sfs_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNorm_L1_16s64s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL1GetBufferSize_16s64s_Sfs(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer vector L1 norm method, return value is 64-bit signed integer.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL1GetBufferSize_16s64s_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L1_16s64s_Sfs_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp64s * pNorm, int nScaleFactor,
                           Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer vector L1 norm method, return value is 64-bit signed integer.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL1GetBufferSize_16s64s_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L1_16s64s_Sfs(const Mpp16s * pSrc, size_t nLength, Mpp64s * pNorm, int nScaleFactor,
                       Mpp8u * pDeviceBuffer);

/** @} signal_L1_norm */

/** 
 * \section signal_L2_norm L2 Norm
 * @defgroup signal_L2_norm L2 Norm
 * Performs the L2 norm on the samples of a signal.
 * @{
 *
 */

/** 
 * Device-buffer size (in bytes) for mppsNorm_L2_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL2GetBufferSize_32f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);

/** 
 * Device-buffer size (in bytes) for mppsNorm_L2_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL2GetBufferSize_32f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 32-bit float vector L2 norm method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL2GetBufferSize_32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L2_32f_Ctx(const Mpp32f * pSrc, size_t nLength, Mpp32f * pNorm, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float vector L2 norm method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL2GetBufferSize_32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L2_32f(const Mpp32f * pSrc, size_t nLength, Mpp32f * pNorm, Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNorm_L2_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL2GetBufferSize_64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNorm_L2_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL2GetBufferSize_64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 64-bit float vector L2 norm method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL2GetBufferSize_64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L2_64f_Ctx(const Mpp64f * pSrc, size_t nLength, Mpp64f * pNorm,
                    Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit float vector L2 norm method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL2GetBufferSize_64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L2_64f(const Mpp64f * pSrc, size_t nLength, Mpp64f * pNorm,
                Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNorm_L2_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL2GetBufferSize_16s32f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNorm_L2_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL2GetBufferSize_16s32f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer vector L2 norm method, return value is 32-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL2GetBufferSize_16s32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L2_16s32f_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp32f * pNorm,
                       Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer vector L2 norm method, return value is 32-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL2GetBufferSize_16s32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L2_16s32f(const Mpp16s * pSrc, size_t nLength, Mpp32f * pNorm,
                   Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNorm_L2_32fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL2GetBufferSize_32fc64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNorm_L2_32fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL2GetBufferSize_32fc64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 32-bit float complex vector L2 norm method, return value is 64-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL2GetBufferSize_32fc64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L2_32fc64f_Ctx(const Mpp32fc * pSrc, size_t nLength, Mpp64f * pNorm,
                        Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float complex vector L2 norm method, return value is 64-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL2GetBufferSize_32fc64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L2_32fc64f(const Mpp32fc * pSrc, size_t nLength, Mpp64f * pNorm,
                    Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNorm_L2_64fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL2GetBufferSize_64fc64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNorm_L2_64fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL2GetBufferSize_64fc64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 64-bit float complex vector L2 norm method, return value is 64-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL2GetBufferSize_64fc64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L2_64fc64f_Ctx(const Mpp64fc * pSrc, size_t nLength, Mpp64f * pNorm,
                        Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit float complex vector L2 norm method, return value is 64-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL2GetBufferSize_64fc64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L2_64fc64f(const Mpp64fc * pSrc, size_t nLength, Mpp64f * pNorm,
                    Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNorm_L2_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL2GetBufferSize_16s32s_Sfs_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNorm_L2_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL2GetBufferSize_16s32s_Sfs(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer vector L2 norm method, return value is 32-bit signed integer.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL2GetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L2_16s32s_Sfs_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp32s * pNorm, int nScaleFactor,
                           Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer vector L2 norm method, return value is 32-bit signed integer.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL2GetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L2_16s32s_Sfs(const Mpp16s * pSrc, size_t nLength, Mpp32s * pNorm, int nScaleFactor,
                       Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNorm_L2Sqr_16s64s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL2SqrGetBufferSize_16s64s_Sfs_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNorm_L2Sqr_16s64s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormL2SqrGetBufferSize_16s64s_Sfs(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer vector L2 Square norm method, return value is 64-bit signed integer.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL2SqrGetBufferSize_16s64s_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L2Sqr_16s64s_Sfs_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp64s * pNorm, int nScaleFactor,
                              Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer vector L2 Square norm method, return value is 64-bit signed integer.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormL2SqrGetBufferSize_16s64s_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNorm_L2Sqr_16s64s_Sfs(const Mpp16s * pSrc, size_t nLength, Mpp64s * pNorm, int nScaleFactor,
                          Mpp8u * pDeviceBuffer);

/** @} signal_L2_norm */

/** 
 * \section signal_infinity_norm_diff Infinity Norm Diff
 * @defgroup signal_infinity_norm_diff Infinity Norm Diff
 * Performs the infinity norm on the samples of two input signals' difference.
 * @{
 *
 */

/** 
 * Device-buffer size (in bytes) for mppsNormDiff_Inf_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffInfGetBufferSize_32f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNormDiff_Inf_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffInfGetBufferSize_32f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 32-bit float C norm method on two vectors' difference
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffInfGetBufferSize_32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_Inf_32f_Ctx(const Mpp32f * pSrc1, const Mpp32f * pSrc2, size_t nLength, Mpp32f * pNorm,
                         Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float C norm method on two vectors' difference
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffInfGetBufferSize_32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_Inf_32f(const Mpp32f * pSrc1, const Mpp32f * pSrc2, size_t nLength, Mpp32f * pNorm,
                     Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNormDiff_Inf_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffInfGetBufferSize_64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNormDiff_Inf_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffInfGetBufferSize_64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 64-bit float C norm method on two vectors' difference
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffInfGetBufferSize_64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_Inf_64f_Ctx(const Mpp64f * pSrc1, const Mpp64f * pSrc2, size_t nLength, Mpp64f * pNorm,
                         Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit float C norm method on two vectors' difference
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffInfGetBufferSize_64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_Inf_64f(const Mpp64f * pSrc1, const Mpp64f * pSrc2, size_t nLength, Mpp64f * pNorm,
                     Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNormDiff_Inf_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffInfGetBufferSize_16s32f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNormDiff_Inf_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffInfGetBufferSize_16s32f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer C norm method on two vectors' difference, return value is 32-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffInfGetBufferSize_16s32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_Inf_16s32f_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp32f * pNorm,
                            Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer C norm method on two vectors' difference, return value is 32-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffInfGetBufferSize_16s32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_Inf_16s32f(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp32f * pNorm,
                        Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNormDiff_Inf_32fc32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffInfGetBufferSize_32fc32f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNormDiff_Inf_32fc32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffInfGetBufferSize_32fc32f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 32-bit float complex C norm method on two vectors' difference, return value is 32-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffInfGetBufferSize_32fc32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_Inf_32fc32f_Ctx(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, size_t nLength, Mpp32f * pNorm,
                             Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float complex C norm method on two vectors' difference, return value is 32-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffInfGetBufferSize_32fc32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_Inf_32fc32f(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, size_t nLength, Mpp32f * pNorm,
                         Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNormDiff_Inf_64fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffInfGetBufferSize_64fc64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNormDiff_Inf_64fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffInfGetBufferSize_64fc64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 64-bit float complex C norm method on two vectors' difference, return value is 64-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffInfGetBufferSize_64fc64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_Inf_64fc64f_Ctx(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, size_t nLength, Mpp64f * pNorm,
                             Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit float complex C norm method on two vectors' difference, return value is 64-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffInfGetBufferSize_64fc64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_Inf_64fc64f(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, size_t nLength, Mpp64f * pNorm,
                         Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNormDiff_Inf_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffInfGetBufferSize_16s32s_Sfs_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNormDiff_Inf_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffInfGetBufferSize_16s32s_Sfs(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer C norm method on two vectors' difference, return value is 32-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffInfGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_Inf_16s32s_Sfs_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp32s * pNorm, int nScaleFactor,
                                Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer C norm method on two vectors' difference, return value is 32-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffInfGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_Inf_16s32s_Sfs(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp32s * pNorm, int nScaleFactor,
                            Mpp8u * pDeviceBuffer);

/** @} signal_infinity_norm_diff */

/** 
 * \section signal_L1_norm_diff L1 Norm Diff
 * @defgroup signal_L1_norm_diff L1 Norm Diff
 * Performs the L1 norm on the samples of two input signals' difference.
 * @{
 *
 */

/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L1_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL1GetBufferSize_32f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L1_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL1GetBufferSize_32f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 32-bit float L1 norm method on two vectors' difference
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL1GetBufferSize_32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L1_32f_Ctx(const Mpp32f * pSrc1, const Mpp32f * pSrc2, size_t nLength, Mpp32f * pNorm,
                        Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float L1 norm method on two vectors' difference
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL1GetBufferSize_32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L1_32f(const Mpp32f * pSrc1, const Mpp32f * pSrc2, size_t nLength, Mpp32f * pNorm,
                    Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L1_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL1GetBufferSize_64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L1_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL1GetBufferSize_64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 64-bit float L1 norm method on two vectors' difference
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL1GetBufferSize_64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L1_64f_Ctx(const Mpp64f * pSrc1, const Mpp64f * pSrc2, size_t nLength, Mpp64f * pNorm,
                        Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit float L1 norm method on two vectors' difference
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL1GetBufferSize_64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L1_64f(const Mpp64f * pSrc1, const Mpp64f * pSrc2, size_t nLength, Mpp64f * pNorm,
                    Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L1_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL1GetBufferSize_16s32f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L1_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL1GetBufferSize_16s32f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer L1 norm method on two vectors' difference, return value is 32-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the L1 norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL1GetBufferSize_16s32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L1_16s32f_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp32f * pNorm,
                           Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer L1 norm method on two vectors' difference, return value is 32-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the L1 norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL1GetBufferSize_16s32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L1_16s32f(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp32f * pNorm,
                       Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L1_32fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL1GetBufferSize_32fc64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L1_32fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL1GetBufferSize_32fc64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 32-bit float complex L1 norm method on two vectors' difference, return value is 64-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL1GetBufferSize_32fc64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L1_32fc64f_Ctx(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, size_t nLength, Mpp64f * pNorm,
                            Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float complex L1 norm method on two vectors' difference, return value is 64-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL1GetBufferSize_32fc64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L1_32fc64f(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, size_t nLength, Mpp64f * pNorm,
                        Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L1_64fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL1GetBufferSize_64fc64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L1_64fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL1GetBufferSize_64fc64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 64-bit float complex L1 norm method on two vectors' difference, return value is 64-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL1GetBufferSize_64fc64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L1_64fc64f_Ctx(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, size_t nLength, Mpp64f * pNorm,
                            Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit float complex L1 norm method on two vectors' difference, return value is 64-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL1GetBufferSize_64fc64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L1_64fc64f(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, size_t nLength, Mpp64f * pNorm,
                        Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L1_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL1GetBufferSize_16s32s_Sfs_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L1_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL1GetBufferSize_16s32s_Sfs(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer L1 norm method on two vectors' difference, return value is 32-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer..
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL1GetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L1_16s32s_Sfs_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp32s * pNorm, int nScaleFactor,
                               Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer L1 norm method on two vectors' difference, return value is 32-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer..
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL1GetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L1_16s32s_Sfs(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp32s * pNorm, int nScaleFactor,
                           Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L1_16s64s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL1GetBufferSize_16s64s_Sfs_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L1_16s64s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL1GetBufferSize_16s64s_Sfs(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer L1 norm method on two vectors' difference, return value is 64-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL1GetBufferSize_16s64s_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L1_16s64s_Sfs_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp64s * pNorm, int nScaleFactor,
                               Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer L1 norm method on two vectors' difference, return value is 64-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL1GetBufferSize_16s64s_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L1_16s64s_Sfs(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp64s * pNorm, int nScaleFactor,
                           Mpp8u * pDeviceBuffer);

/** @} signal_L1_norm_diff */

/** 
 * \section signal_L2_norm_diff L2 Norm Diff
 * @defgroup signal_L2_norm_diff L2 Norm Diff
 * Performs the L2 norm on the samples of two input signals' difference.
 * @{
 *
 */

/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L2_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL2GetBufferSize_32f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L2_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL2GetBufferSize_32f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 32-bit float L2 norm method on two vectors' difference
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL2GetBufferSize_32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L2_32f_Ctx(const Mpp32f * pSrc1, const Mpp32f * pSrc2, size_t nLength, Mpp32f * pNorm,
                        Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float L2 norm method on two vectors' difference
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL2GetBufferSize_32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L2_32f(const Mpp32f * pSrc1, const Mpp32f * pSrc2, size_t nLength, Mpp32f * pNorm,
                    Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L2_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL2GetBufferSize_64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L2_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL2GetBufferSize_64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 64-bit float L2 norm method on two vectors' difference
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL2GetBufferSize_64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L2_64f_Ctx(const Mpp64f * pSrc1, const Mpp64f * pSrc2, size_t nLength, Mpp64f * pNorm,
                        Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit float L2 norm method on two vectors' difference
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL2GetBufferSize_64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L2_64f(const Mpp64f * pSrc1, const Mpp64f * pSrc2, size_t nLength, Mpp64f * pNorm,
                    Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L2_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL2GetBufferSize_16s32f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L2_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL2GetBufferSize_16s32f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer L2 norm method on two vectors' difference, return value is 32-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL2GetBufferSize_16s32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L2_16s32f_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp32f * pNorm,
                           Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer L2 norm method on two vectors' difference, return value is 32-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL2GetBufferSize_16s32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L2_16s32f(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp32f * pNorm,
                       Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L2_32fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL2GetBufferSize_32fc64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L2_32fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL2GetBufferSize_32fc64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 32-bit float complex L2 norm method on two vectors' difference, return value is 64-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL2GetBufferSize_32fc64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L2_32fc64f_Ctx(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, size_t nLength, Mpp64f * pNorm,
                            Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float complex L2 norm method on two vectors' difference, return value is 64-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL2GetBufferSize_32fc64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L2_32fc64f(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, size_t nLength, Mpp64f * pNorm,
                        Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L2_64fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL2GetBufferSize_64fc64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L2_64fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL2GetBufferSize_64fc64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 64-bit float complex L2 norm method on two vectors' difference, return value is 64-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL2GetBufferSize_64fc64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L2_64fc64f_Ctx(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, size_t nLength, Mpp64f * pNorm,
                            Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit float complex L2 norm method on two vectors' difference, return value is 64-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL2GetBufferSize_64fc64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L2_64fc64f(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, size_t nLength, Mpp64f * pNorm,
                        Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L2_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL2GetBufferSize_16s32s_Sfs_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L2_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL2GetBufferSize_16s32s_Sfs(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer L2 norm method on two vectors' difference, return value is 32-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL2GetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L2_16s32s_Sfs_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp32s * pNorm, int nScaleFactor,
                               Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer L2 norm method on two vectors' difference, return value is 32-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL2GetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L2_16s32s_Sfs(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp32s * pNorm, int nScaleFactor,
                           Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L2Sqr_16s64s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL2SqrGetBufferSize_16s64s_Sfs_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsNormDiff_L2Sqr_16s64s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsNormDiffL2SqrGetBufferSize_16s64s_Sfs(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer L2 Square norm method on two vectors' difference, return value is 64-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL2SqrGetBufferSize_16s64s_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L2Sqr_16s64s_Sfs_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp64s * pNorm, int nScaleFactor,
                                  Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer L2 Square norm method on two vectors' difference, return value is 64-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsNormDiffL2SqrGetBufferSize_16s64s_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsNormDiff_L2Sqr_16s64s_Sfs(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp64s * pNorm, int nScaleFactor,
                              Mpp8u * pDeviceBuffer);

/** @} signal_l2_norm_diff */

/** 
 * \section signal_dot_product Dot Product
 * @defgroup signal_dot_product Dot Product
 * Performs the dot product operation on the samples of two input signals.
 * @{
 *
 */

/** 
 * Device-buffer size (in bytes) for mppsDotProd_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_32f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_32f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 32-bit float dot product method, return value is 32-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_32f_Ctx(const Mpp32f * pSrc1, const Mpp32f * pSrc2, size_t nLength, Mpp32f * pDp,
                    Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float dot product method, return value is 32-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_32f(const Mpp32f * pSrc1, const Mpp32f * pSrc2, size_t nLength, Mpp32f * pDp,
                Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_32fc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_32fc(size_t nLength,  size_t * hpBufferSize);

/** 
 * 32-bit float complex dot product method, return value is 32-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_32fc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_32fc_Ctx(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, size_t nLength, Mpp32fc * pDp,
                     Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float complex dot product method, return value is 32-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_32fc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_32fc(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, size_t nLength, Mpp32fc * pDp,
                 Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_32f32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_32f32fc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_32f32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_32f32fc(size_t nLength,  size_t * hpBufferSize);

/** 
 * 32-bit float and 32-bit float complex dot product method, return value is 32-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_32f32fc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_32f32fc_Ctx(const Mpp32f * pSrc1, const Mpp32fc * pSrc2, size_t nLength, Mpp32fc * pDp,
                        Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float and 32-bit float complex dot product method, return value is 32-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_32f32fc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_32f32fc(const Mpp32f * pSrc1, const Mpp32fc * pSrc2, size_t nLength, Mpp32fc * pDp,
                    Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_32f64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_32f64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_32f64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_32f64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 32-bit float dot product method, return value is 64-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_32f64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_32f64f_Ctx(const Mpp32f * pSrc1, const Mpp32f * pSrc2, size_t nLength, Mpp64f * pDp,
                       Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float dot product method, return value is 64-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_32f64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_32f64f(const Mpp32f * pSrc1, const Mpp32f * pSrc2, size_t nLength, Mpp64f * pDp,
                   Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_32fc64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_32fc64fc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_32fc64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_32fc64fc(size_t nLength,  size_t * hpBufferSize);

/** 
 * 32-bit float complex dot product method, return value is 64-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_32fc64fc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_32fc64fc_Ctx(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, size_t nLength, Mpp64fc * pDp,
                         Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float complex dot product method, return value is 64-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_32fc64fc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_32fc64fc(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, size_t nLength, Mpp64fc * pDp,
                     Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_32f32fc64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_32f32fc64fc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_32f32fc64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_32f32fc64fc(size_t nLength,  size_t * hpBufferSize);

/** 
 * 32-bit float and 32-bit float complex dot product method, return value is 64-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_32f32fc64fc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_32f32fc64fc_Ctx(const Mpp32f * pSrc1, const Mpp32fc * pSrc2, size_t nLength, Mpp64fc * pDp,
                            Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit float and 32-bit float complex dot product method, return value is 64-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_32f32fc64fc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_32f32fc64fc(const Mpp32f * pSrc1, const Mpp32fc * pSrc2, size_t nLength, Mpp64fc * pDp,
                        Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 64-bit float dot product method, return value is 64-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_64f_Ctx(const Mpp64f * pSrc1, const Mpp64f * pSrc2, size_t nLength, Mpp64f * pDp,
                    Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit float dot product method, return value is 64-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_64f(const Mpp64f * pSrc1, const Mpp64f * pSrc2, size_t nLength, Mpp64f * pDp,
                Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_64fc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_64fc(size_t nLength,  size_t * hpBufferSize);

/** 
 * 64-bit float complex dot product method, return value is 64-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_64fc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_64fc_Ctx(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, size_t nLength, Mpp64fc * pDp,
                     Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit float complex dot product method, return value is 64-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_64fc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_64fc(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, size_t nLength, Mpp64fc * pDp,
                 Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_64f64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_64f64fc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_64f64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_64f64fc(size_t nLength,  size_t * hpBufferSize);

/** 
 * 64-bit float and 64-bit float complex dot product method, return value is 64-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_64f64fc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_64f64fc_Ctx(const Mpp64f * pSrc1, const Mpp64fc * pSrc2, size_t nLength, Mpp64fc * pDp,
                        Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit float and 64-bit float complex dot product method, return value is 64-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_64f64fc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_64f64fc(const Mpp64f * pSrc1, const Mpp64fc * pSrc2, size_t nLength, Mpp64fc * pDp,
                    Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_16s64s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16s64s_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_16s64s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16s64s(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer dot product method, return value is 64-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16s64s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16s64s_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp64s * pDp,
                       Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer dot product method, return value is 64-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16s64s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16s64s(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp64s * pDp,
                   Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_16sc64sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16sc64sc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_16sc64sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16sc64sc(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer complex dot product method, return value is 64-bit signed integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16sc64sc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16sc64sc_Ctx(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp64sc * pDp,
                         Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer complex dot product method, return value is 64-bit signed integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16sc64sc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16sc64sc(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp64sc * pDp,
                     Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_16s16sc64sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16s16sc64sc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_16s16sc64sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16s16sc64sc(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer and 16-bit signed short integer short dot product method, return value is 64-bit signed integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16s16sc64sc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16s16sc64sc_Ctx(const Mpp16s * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp64sc * pDp,
                            Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer and 16-bit signed short integer short dot product method, return value is 64-bit signed integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16s16sc64sc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16s16sc64sc(const Mpp16s * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp64sc * pDp,
                        Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16s32f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16s32f(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer dot product method, return value is 32-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16s32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16s32f_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp32f * pDp,
                       Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer dot product method, return value is 32-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16s32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16s32f(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp32f * pDp,
                   Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_16sc32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16sc32fc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_16sc32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16sc32fc(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer complex dot product method, return value is 32-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16sc32fc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16sc32fc_Ctx(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp32fc * pDp,
                         Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer complex dot product method, return value is 32-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16sc32fc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16sc32fc(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp32fc * pDp,
                     Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_16s16sc32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16s16sc32fc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_16s16sc32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16s16sc32fc(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer and 16-bit signed short integer complex dot product method, return value is 32-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16s16sc32fc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16s16sc32fc_Ctx(const Mpp16s * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp32fc * pDp,
                            Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer and 16-bit signed short integer complex dot product method, return value is 32-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16s16sc32fc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16s16sc32fc(const Mpp16s * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp32fc * pDp,
                        Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_16s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16s_Sfs_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_16s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16s_Sfs(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer dot product method, return value is 16-bit signed short integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16s_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16s_Sfs_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp16s * pDp, int nScaleFactor, 
                        Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer dot product method, return value is 16-bit signed short integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16s_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16s_Sfs(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp16s * pDp, int nScaleFactor, 
                    Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_16sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16sc_Sfs_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_16sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16sc_Sfs(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer complex dot product method, return value is 16-bit signed short integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16sc_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16sc_Sfs_Ctx(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp16sc * pDp, int nScaleFactor, 
                         Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer complex dot product method, return value is 16-bit signed short integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16sc_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16sc_Sfs(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp16sc * pDp, int nScaleFactor, 
                     Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_32s_Sfs_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_32s_Sfs(size_t nLength,  size_t * hpBufferSize);

/** 
 * 32-bit signed integer dot product method, return value is 32-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_32s_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_32s_Sfs_Ctx(const Mpp32s * pSrc1, const Mpp32s * pSrc2, size_t nLength, Mpp32s * pDp, int nScaleFactor, 
                        Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer dot product method, return value is 32-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_32s_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_32s_Sfs(const Mpp32s * pSrc1, const Mpp32s * pSrc2, size_t nLength, Mpp32s * pDp, int nScaleFactor, 
                    Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_32sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_32sc_Sfs_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_32sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_32sc_Sfs(size_t nLength,  size_t * hpBufferSize);

/** 
 * 32-bit signed integer complex dot product method, return value is 32-bit signed integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_32sc_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_32sc_Sfs_Ctx(const Mpp32sc * pSrc1, const Mpp32sc * pSrc2, size_t nLength, Mpp32sc * pDp, int nScaleFactor, 
                         Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed integer complex dot product method, return value is 32-bit signed integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_32sc_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_32sc_Sfs(const Mpp32sc * pSrc1, const Mpp32sc * pSrc2, size_t nLength, Mpp32sc * pDp, int nScaleFactor, 
                     Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16s32s_Sfs_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16s32s_Sfs(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer dot product method, return value is 32-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result. 
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16s32s_Sfs_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp32s * pDp, int nScaleFactor, 
                           Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer dot product method, return value is 32-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result. 
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16s32s_Sfs(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp32s * pDp, int nScaleFactor, 
                       Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_16s16sc32sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16s16sc32sc_Sfs_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_16s16sc32sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16s16sc32sc_Sfs(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer and 16-bit signed short integer complex dot product method, return value is 32-bit signed integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16s16sc32sc_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16s16sc32sc_Sfs_Ctx(const Mpp16s * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp32sc * pDp, int nScaleFactor, 
                                Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer and 16-bit signed short integer complex dot product method, return value is 32-bit signed integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16s16sc32sc_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16s16sc32sc_Sfs(const Mpp16s * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp32sc * pDp, int nScaleFactor, 
                            Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_16s32s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16s32s32s_Sfs_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_16s32s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16s32s32s_Sfs(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer and 32-bit signed integer dot product method, return value is 32-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16s32s32s_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16s32s32s_Sfs_Ctx(const Mpp16s * pSrc1, const Mpp32s * pSrc2, size_t nLength, Mpp32s * pDp, int nScaleFactor, 
                              Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer and 32-bit signed integer dot product method, return value is 32-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16s32s32s_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16s32s32s_Sfs(const Mpp16s * pSrc1, const Mpp32s * pSrc2, size_t nLength, Mpp32s * pDp, int nScaleFactor, 
                          Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_16s16sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16s16sc_Sfs_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_16s16sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16s16sc_Sfs(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer and 16-bit signed short integer complex dot product method, return value is 16-bit signed short integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16s16sc_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16s16sc_Sfs_Ctx(const Mpp16s * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp16sc * pDp, int nScaleFactor, 
                            Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer and 16-bit signed short integer complex dot product method, return value is 16-bit signed short integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16s16sc_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16s16sc_Sfs(const Mpp16s * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp16sc * pDp, int nScaleFactor, 
                        Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_16sc32sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16sc32sc_Sfs_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_16sc32sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_16sc32sc_Sfs(size_t nLength,  size_t * hpBufferSize);

/** 
 * 16-bit signed short integer complex dot product method, return value is 32-bit signed integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16sc32sc_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16sc32sc_Sfs_Ctx(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp32sc * pDp, int nScaleFactor, 
                             Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer complex dot product method, return value is 32-bit signed integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_16sc32sc_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_16sc32sc_Sfs(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp32sc * pDp, int nScaleFactor, 
                         Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsDotProd_32s32sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_32s32sc_Sfs_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsDotProd_32s32sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsDotProdGetBufferSize_32s32sc_Sfs(size_t nLength,  size_t * hpBufferSize);

/** 
 * 32-bit signed short integer and 32-bit signed short integer complex dot product method, return value is 32-bit signed integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_32s32sc_Sfs to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_32s32sc_Sfs_Ctx(const Mpp32s * pSrc1, const Mpp32sc * pSrc2, size_t nLength, Mpp32sc * pDp, int nScaleFactor, 
                            Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed short integer and 32-bit signed short integer complex dot product method, return value is 32-bit signed integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsDotProdGetBufferSize_32s32sc_Sfs to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsDotProd_32s32sc_Sfs(const Mpp32s * pSrc1, const Mpp32sc * pSrc2, size_t nLength, Mpp32sc * pDp, int nScaleFactor, 
                        Mpp8u * pDeviceBuffer);

/** @} signal_dot_product */

/** 
 * \section signal_count_in_range Count In Range
 * @defgroup signal_count_in_range Count In Range
 * Calculates the number of elements from specified range in the samples of a signal.
 * @{
 *
 */

/** 
 * Device-buffer size (in bytes) for mppsCountInRange_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsCountInRangeGetBufferSize_32s_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsCountInRange_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsCountInRangeGetBufferSize_32s(size_t nLength,  size_t * hpBufferSize);

/** 
 * Computes the number of elements whose values fall into the specified range on a 32-bit signed integer array.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pCounts Pointer to the number of elements.
 * \param nLowerBound Lower bound of the specified range.
 * \param nUpperBound Upper bound of the specified range.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsCountInRangeGetBufferSize_32s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus
mppsCountInRange_32s_Ctx(const Mpp32s * pSrc, size_t nLength, int * pCounts, Mpp32s nLowerBound, Mpp32s nUpperBound,
                         Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * Computes the number of elements whose values fall into the specified range on a 32-bit signed integer array.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pCounts Pointer to the number of elements.
 * \param nLowerBound Lower bound of the specified range.
 * \param nUpperBound Upper bound of the specified range.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsCountInRangeGetBufferSize_32s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus
mppsCountInRange_32s(const Mpp32s * pSrc, size_t nLength, int * pCounts, Mpp32s nLowerBound, Mpp32s nUpperBound,
                     Mpp8u * pDeviceBuffer);

/** @} signal_count_in_range */

/** 
 * \section signal_count_zero_crossings Count Zero Crossings
 * @defgroup signal_count_zero_crossings Count Zero Crossings
 *
 * @{
 * Calculates the number of zero crossings in a signal.
 */

/** 
 * Device-buffer size (in bytes) for mppsZeroCrossing_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsZeroCrossingGetBufferSize_16s32f_Ctx(size_t nLength, size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsZeroCrossing_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsZeroCrossingGetBufferSize_16s32f(size_t nLength, size_t * hpBufferSize);

/** 
 * 16-bit signed short integer zero crossing method, return value is 32-bit floating point.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pValZC Pointer to the output result.
 * \param tZCType Type of the zero crossing measure: mppZCR, mppZCXor or mppZCC.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsZeroCrossingGetBufferSize_16s32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus
mppsZeroCrossing_16s32f_Ctx(const Mpp16s * pSrc, size_t nLength, Mpp32f * pValZC, MppsZCType tZCType,
                            Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer zero crossing method, return value is 32-bit floating point.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pValZC Pointer to the output result.
 * \param tZCType Type of the zero crossing measure: mppZCR, mppZCXor or mppZCC.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsZeroCrossingGetBufferSize_16s32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus
mppsZeroCrossing_16s32f(const Mpp16s * pSrc, size_t nLength, Mpp32f * pValZC, MppsZCType tZCType,
                        Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsZeroCrossing_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsZeroCrossingGetBufferSize_32f_Ctx(size_t nLength, size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsZeroCrossing_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsZeroCrossingGetBufferSize_32f(size_t nLength, size_t * hpBufferSize);

/** 
 * 32-bit floating-point zero crossing method, return value is 32-bit floating point.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pValZC Pointer to the output result.
 * \param tZCType Type of the zero crossing measure: mppZCR, mppZCXor or mppZCC.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsZeroCrossingGetBufferSize_32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus
mppsZeroCrossing_32f_Ctx(const Mpp32f * pSrc, size_t nLength, Mpp32f * pValZC, MppsZCType tZCType,
                         Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating-point zero crossing method, return value is 32-bit floating point.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pValZC Pointer to the output result.
 * \param tZCType Type of the zero crossing measure: mppZCR, mppZCXor or mppZCC.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsZeroCrossingGetBufferSize_32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus
mppsZeroCrossing_32f(const Mpp32f * pSrc, size_t nLength, Mpp32f * pValZC, MppsZCType tZCType,
                     Mpp8u * pDeviceBuffer);

/** @} signal_count_zero_crossings */

/** 
 * \section signal_maximum_error MaximumError
 * @defgroup signal_maximum_error MaximumError
 * Primitives for computing the maximum error between two signals.
 * Given two signals \f$pSrc1\f$ and \f$pSrc2\f$ both with length \f$N\f$, 
 * the maximum error is defined as the largest absolute difference between the corresponding
 * elements of two signals.
 *
 * If the signal is in complex format, the absolute value of the complex number is used.
 * @{
 *
 */
/** 
 * 8-bit unsigned char maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_8u to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_8u_Ctx(const Mpp8u * pSrc1, const Mpp8u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_8u to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_8u(const Mpp8u * pSrc1, const Mpp8u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 8-bit signed char maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_8s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_8s_Ctx(const Mpp8s * pSrc1, const Mpp8s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 8-bit signed char maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_8s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_8s(const Mpp8s * pSrc1, const Mpp8s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 16-bit unsigned short integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_16u to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_16u_Ctx(const Mpp16u * pSrc1, const Mpp16u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_16u to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_16u(const Mpp16u * pSrc1, const Mpp16u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 16-bit signed short integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_16s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_16s_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_16s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_16s(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 16-bit unsigned short complex integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_16sc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_16sc_Ctx(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short complex integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_16sc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_16sc(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit unsigned short integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_32u to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_32u_Ctx(const Mpp32u * pSrc1, const Mpp32u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned short integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_32u to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_32u(const Mpp32u * pSrc1, const Mpp32u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit signed short integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_32s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_32s_Ctx(const Mpp32s * pSrc1, const Mpp32s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed short integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_32s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_32s(const Mpp32s * pSrc1, const Mpp32s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit unsigned short complex integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_32sc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_32sc_Ctx(const Mpp32sc * pSrc1, const Mpp32sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned short complex integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_32sc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_32sc(const Mpp32sc * pSrc1, const Mpp32sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit signed short integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_64s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_64s_Ctx(const Mpp64s * pSrc1, const Mpp64s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit signed short integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_64s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_64s(const Mpp64s * pSrc1, const Mpp64s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit unsigned short complex integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_64sc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_64sc_Ctx(const Mpp64sc * pSrc1, const Mpp64sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit unsigned short complex integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_64sc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_64sc(const Mpp64sc * pSrc1, const Mpp64sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit floating point maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_32f_Ctx(const Mpp32f * pSrc1, const Mpp32f * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_32f(const Mpp32f * pSrc1, const Mpp32f * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit floating point complex maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_32fc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_32fc_Ctx(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point complex maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_32fc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_32fc(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit floating point maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_64f_Ctx(const Mpp64f * pSrc1, const Mpp64f * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_64f(const Mpp64f * pSrc1, const Mpp64f * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit floating point complex maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_64fc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_64fc_Ctx(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point complex maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumErrorGetBufferSize_64fc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumError_64fc(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsMaximumError_8u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_8u_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumError_8u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_8u(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumError_8s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_8s_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumError_8s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_8s(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumError_16u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_16u_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumError_16u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_16u(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumError_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_16s_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumError_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_16s(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumError_16sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_16sc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumError_16sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_16sc(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumError_32u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_32u_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumError_32u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_32u(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumError_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_32s_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumError_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_32s(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumError_32sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_32sc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumError_32sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_32sc(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumError_64s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_64s_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumError_64s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_64s(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumError_64sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_64sc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumError_64sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_64sc(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumError_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_32f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumError_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_32f(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumError_32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_32fc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumError_32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_32fc(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumError_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumError_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumError_64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_64fc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumError_64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumErrorGetBufferSize_64fc(size_t nLength,  size_t * hpBufferSize);

/** @} */

/** 
 * \section signal_average_error AverageError
 * @defgroup signal_average_error AverageError
 * Primitives for computing the Average error between two signals.
 * Given two signals \f$pSrc1\f$ and \f$pSrc2\f$ both with length \f$N\f$, 
 * the average error is defined as
 * \f[Average Error = \frac{1}{N}\sum_{n=0}^{N-1}\left|pSrc1(n) - pSrc2(n)\right|\f]
 *
 * If the signal is in complex format, the absolute value of the complex number is used.
 * @{
 *
 */
/** 
 * 8-bit unsigned char Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_8u to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_8u_Ctx(const Mpp8u * pSrc1, const Mpp8u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_8u to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_8u(const Mpp8u * pSrc1, const Mpp8u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 8-bit signed char Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_8s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_8s_Ctx(const Mpp8s * pSrc1, const Mpp8s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 8-bit signed char Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_8s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_8s(const Mpp8s * pSrc1, const Mpp8s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 16-bit unsigned short integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_16u to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_16u_Ctx(const Mpp16u * pSrc1, const Mpp16u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_16u to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_16u(const Mpp16u * pSrc1, const Mpp16u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 16-bit signed short integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_16s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_16s_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_16s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_16s(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 16-bit unsigned short complex integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_16sc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_16sc_Ctx(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short complex integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_16sc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_16sc(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit unsigned short integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_32u to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_32u_Ctx(const Mpp32u * pSrc1, const Mpp32u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned short integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_32u to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_32u(const Mpp32u * pSrc1, const Mpp32u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit signed short integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_32s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_32s_Ctx(const Mpp32s * pSrc1, const Mpp32s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed short integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_32s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_32s(const Mpp32s * pSrc1, const Mpp32s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit unsigned short complex integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_32sc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_32sc_Ctx(const Mpp32sc * pSrc1, const Mpp32sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned short complex integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_32sc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_32sc(const Mpp32sc * pSrc1, const Mpp32sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit signed short integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_64s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_64s_Ctx(const Mpp64s * pSrc1, const Mpp64s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit signed short integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_64s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_64s(const Mpp64s * pSrc1, const Mpp64s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit unsigned short complex integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_64sc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_64sc_Ctx(const Mpp64sc * pSrc1, const Mpp64sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit unsigned short complex integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_64sc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_64sc(const Mpp64sc * pSrc1, const Mpp64sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit floating point Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_32f_Ctx(const Mpp32f * pSrc1, const Mpp32f * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_32f(const Mpp32f * pSrc1, const Mpp32f * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit floating point complex Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_32fc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_32fc_Ctx(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point complex Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_32fc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_32fc(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit floating point Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_64f_Ctx(const Mpp64f * pSrc1, const Mpp64f * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_64f(const Mpp64f * pSrc1, const Mpp64f * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit floating point complex Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_64fc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_64fc_Ctx(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point complex Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageErrorGetBufferSize_64fc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageError_64fc(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsAverageError_8u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_8u_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageError_8u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_8u(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageError_8s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_8s_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageError_8s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_8s(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageError_16u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_16u_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageError_16u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_16u(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageError_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_16s_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageError_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_16s(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageError_16sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_16sc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageError_16sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_16sc(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageError_32u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_32u_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageError_32u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_32u(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageError_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_32s_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageError_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_32s(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageError_32sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_32sc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageError_32sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_32sc(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageError_64s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_64s_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageError_64s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_64s(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageError_64sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_64sc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageError_64sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_64sc(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageError_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_32f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageError_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_32f(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageError_32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_32fc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageError_32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_32fc(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageError_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageError_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageError_64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_64fc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageError_64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageErrorGetBufferSize_64fc(size_t nLength,  size_t * hpBufferSize);

/** @} */

/** 
 * \section signal_maximum_relative_error MaximumRelativeError
 * @defgroup signal_maximum_relative_error MaximumRelativeError
 * Primitives for computing the MaximumRelative error between two signals.
 * Given two signals \f$pSrc1\f$ and \f$pSrc2\f$ both with length \f$N\f$, 
 * the maximum relative error is defined as
 * \f[MaximumRelativeError = max{\frac{\left|pSrc1(n) - pSrc2(n)\right|}{max(\left|pSrc1(n)\right|, \left|pSrc2(n)\right|)}}\f]
 *
 * If the signal is in complex format, the absolute value of the complex number is used.
 * @{
 *
 */
/** 
 * 8-bit unsigned char MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_8u to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_8u_Ctx(const Mpp8u * pSrc1, const Mpp8u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_8u to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_8u(const Mpp8u * pSrc1, const Mpp8u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 8-bit signed char MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_8s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_8s_Ctx(const Mpp8s * pSrc1, const Mpp8s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 8-bit signed char MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_8s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_8s(const Mpp8s * pSrc1, const Mpp8s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 16-bit unsigned short integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_16u to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_16u_Ctx(const Mpp16u * pSrc1, const Mpp16u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_16u to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_16u(const Mpp16u * pSrc1, const Mpp16u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 16-bit signed short integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_16s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_16s_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_16s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_16s(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 16-bit unsigned short complex integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_16sc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_16sc_Ctx(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short complex integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_16sc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_16sc(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit unsigned short integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_32u to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_32u_Ctx(const Mpp32u * pSrc1, const Mpp32u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned short integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_32u to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_32u(const Mpp32u * pSrc1, const Mpp32u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit signed short integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_32s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_32s_Ctx(const Mpp32s * pSrc1, const Mpp32s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed short integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_32s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_32s(const Mpp32s * pSrc1, const Mpp32s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit unsigned short complex integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_32sc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_32sc_Ctx(const Mpp32sc * pSrc1, const Mpp32sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned short complex integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_32sc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_32sc(const Mpp32sc * pSrc1, const Mpp32sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit signed short integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_64s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_64s_Ctx(const Mpp64s * pSrc1, const Mpp64s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit signed short integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_64s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_64s(const Mpp64s * pSrc1, const Mpp64s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit unsigned short complex integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_64sc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_64sc_Ctx(const Mpp64sc * pSrc1, const Mpp64sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit unsigned short complex integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_64sc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_64sc(const Mpp64sc * pSrc1, const Mpp64sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit floating point MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_32f_Ctx(const Mpp32f * pSrc1, const Mpp32f * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_32f(const Mpp32f * pSrc1, const Mpp32f * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit floating point complex MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_32fc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_32fc_Ctx(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point complex MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_32fc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_32fc(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit floating point MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_64f_Ctx(const Mpp64f * pSrc1, const Mpp64f * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_64f(const Mpp64f * pSrc1, const Mpp64f * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit floating point complex MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_64fc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_64fc_Ctx(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point complex MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsMaximumRelativeErrorGetBufferSize_64fc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsMaximumRelativeError_64fc(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_8u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_8u_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_8u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_8u(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_8s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_8s_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_8s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_8s(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_16u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_16u_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_16u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_16u(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_16s_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_16s(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_16sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_16sc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_16sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_16sc(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_32u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_32u_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_32u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_32u(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_32s_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_32s(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_32sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_32sc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_32sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_32sc(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_64s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_64s_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_64s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_64s(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_64sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_64sc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_64sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_64sc(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_32f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_32f(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_32fc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_32fc(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_64fc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsMaximumRelativeError_64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsMaximumRelativeErrorGetBufferSize_64fc(size_t nLength,  size_t * hpBufferSize);

/** @} */

/** 
 * \section signal_average_relative_error AverageRelativeError
 * @defgroup signal_average_relative_error AverageRelativeError
 * Primitives for computing the AverageRelative error between two signals.
 * Given two signals \f$pSrc1\f$ and \f$pSrc2\f$ both with length \f$N\f$, 
 * the average relative error is defined as
 * \f[AverageRelativeError = \frac{1}{N}\sum_{n=0}^{N-1}\frac{\left|pSrc1(n) - pSrc2(n)\right|}{max(\left|pSrc1(n)\right|, \left|pSrc2(n)\right|)}\f]
 *
 * If the signal is in complex format, the absolute value of the complex number is used.
 * @{
 *
 */
/** 
 * 8-bit unsigned char AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_8u to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_8u_Ctx(const Mpp8u * pSrc1, const Mpp8u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 8-bit unsigned char AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_8u to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_8u(const Mpp8u * pSrc1, const Mpp8u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 8-bit signed char AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_8s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_8s_Ctx(const Mpp8s * pSrc1, const Mpp8s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 8-bit signed char AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_8s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_8s(const Mpp8s * pSrc1, const Mpp8s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 16-bit unsigned short integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_16u to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_16u_Ctx(const Mpp16u * pSrc1, const Mpp16u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_16u to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_16u(const Mpp16u * pSrc1, const Mpp16u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 16-bit signed short integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_16s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_16s_Ctx(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit signed short integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_16s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_16s(const Mpp16s * pSrc1, const Mpp16s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 16-bit unsigned short complex integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_16sc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_16sc_Ctx(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 16-bit unsigned short complex integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_16sc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_16sc(const Mpp16sc * pSrc1, const Mpp16sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit unsigned short integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_32u to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_32u_Ctx(const Mpp32u * pSrc1, const Mpp32u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned short integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_32u to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_32u(const Mpp32u * pSrc1, const Mpp32u * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit signed short integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_32s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_32s_Ctx(const Mpp32s * pSrc1, const Mpp32s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit signed short integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_32s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_32s(const Mpp32s * pSrc1, const Mpp32s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit unsigned short complex integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_32sc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_32sc_Ctx(const Mpp32sc * pSrc1, const Mpp32sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit unsigned short complex integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_32sc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_32sc(const Mpp32sc * pSrc1, const Mpp32sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit signed short integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_64s to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_64s_Ctx(const Mpp64s * pSrc1, const Mpp64s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit signed short integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_64s to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_64s(const Mpp64s * pSrc1, const Mpp64s * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit unsigned short complex integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_64sc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_64sc_Ctx(const Mpp64sc * pSrc1, const Mpp64sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit unsigned short complex integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_64sc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_64sc(const Mpp64sc * pSrc1, const Mpp64sc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit floating point AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_32f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_32f_Ctx(const Mpp32f * pSrc1, const Mpp32f * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_32f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_32f(const Mpp32f * pSrc1, const Mpp32f * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 32-bit floating point complex AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_32fc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_32fc_Ctx(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 32-bit floating point complex AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_32fc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_32fc(const Mpp32fc * pSrc1, const Mpp32fc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit floating point AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_64f to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_64f_Ctx(const Mpp64f * pSrc1, const Mpp64f * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_64f to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_64f(const Mpp64f * pSrc1, const Mpp64f * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * 64-bit floating point complex AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_64fc to determine the minium number of bytes required.
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_64fc_Ctx(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer, MppStreamContext mppStreamCtx);
/** 
 * 64-bit floating point complex AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref mppsAverageRelativeErrorGetBufferSize_64fc to determine the minium number of bytes required.
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
MppStatus 
mppsAverageRelativeError_64fc(const Mpp64fc * pSrc1, const Mpp64fc * pSrc2, size_t nLength, Mpp64f * pDst, Mpp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_8u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_8u_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_8u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_8u(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_8s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_8s_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_8s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_8s(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_16u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_16u_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_16u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_16u(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_16s_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_16s(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_16sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_16sc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_16sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_16sc(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_32u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_32u_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_32u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_32u(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_32s_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_32s(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_32sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_32sc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_32sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_32sc(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_64s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_64s_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_64s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_64s(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_64sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_64sc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_64sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_64sc(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_32f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_32f(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_32fc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_32fc(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_64f_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_64f(size_t nLength,  size_t * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param mppStreamCtx \ref application_managed_stream_context. 
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_64fc_Ctx(size_t nLength,  size_t * hpBufferSize, MppStreamContext mppStreamCtx);
/** 
 * Device-buffer size (in bytes) for mppsAverageRelativeError_64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \return MPP_SUCCESS
 */
MppStatus
mppsAverageRelativeErrorGetBufferSize_64fc(size_t nLength,  size_t * hpBufferSize);

/** @} */

/** @} signal_statistical_functions */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MC_MPPS_STATISTICS_FUNCTIONS_H */
