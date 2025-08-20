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
#ifndef MC_MPPS_SUPPORT_FUNCTIONS_H
#define MC_MPPS_SUPPORT_FUNCTIONS_H
 
/**
 * \file mpps_support_functions.h
 * Signal Processing Support Functions.
 */
 
#include "mppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif


/** 
 *  \page signal_memory_management Memory Management
 *  @defgroup signal_memory_management Memory Management
 *  @ingroup mpps
 * Functions that provide memory management functionality like malloc and free.
 * @{
 */

/** 
 * \section signal_malloc Malloc
 * @defgroup signal_malloc Malloc
 * Signal-allocator methods for allocating 1D arrays of data in device memory.
 * All allocators have size parameters to specify the size of the signal (1D array)
 * being allocated.
 *
 * The allocator methods return a pointer to the newly allocated memory of appropriate
 * type. If device-memory allocation is not possible due to resource constaints
 * the allocators return 0 (i.e. NULL pointer). 
 *
 * All signal allocators allocate memory aligned such that it is  beneficial to the 
 * performance of the majority of the signal-processing primitives. 
 * It is no mandatory however to use these allocators. Any valid
 * MACA device-memory pointers can be passed to MPP primitives. 
 *
 * @{
 */

/**
 * 8-bit unsigned signal allocator.
 * \param nSize Number of unsigned chars in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Mpp8u * 
mppsMalloc_8u(size_t nSize);

/**
 * 8-bit signed signal allocator.
 * \param nSize Number of (signed) chars in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Mpp8s * 
mppsMalloc_8s(size_t nSize);

/**
 * 16-bit unsigned signal allocator.
 * \param nSize Number of unsigned shorts in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Mpp16u * 
mppsMalloc_16u(size_t nSize);

/**
 * 16-bit signal allocator.
 * \param nSize Number of shorts in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Mpp16s * 
mppsMalloc_16s(size_t nSize);

/**
 * 16-bit complex-value signal allocator.
 * \param nSize Number of 16-bit complex numbers in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Mpp16sc * 
mppsMalloc_16sc(size_t nSize);

/**
 * 32-bit unsigned signal allocator.
 * \param nSize Number of unsigned ints in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Mpp32u * 
mppsMalloc_32u(size_t nSize);

/**
 * 32-bit integer signal allocator.
 * \param nSize Number of ints in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Mpp32s * 
mppsMalloc_32s(size_t nSize);

/**
 * 32-bit complex integer signal allocator.
 * \param nSize Number of complex integner values in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Mpp32sc * 
mppsMalloc_32sc(size_t nSize);

/**
 * 32-bit float signal allocator.
 * \param nSize Number of floats in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Mpp32f * 
mppsMalloc_32f(size_t nSize);

/**
 * 32-bit complex float signal allocator.
 * \param nSize Number of complex float values in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Mpp32fc * 
mppsMalloc_32fc(size_t nSize);

/**
 * 64-bit long integer signal allocator.
 * \param nSize Number of long ints in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Mpp64s * 
mppsMalloc_64s(size_t nSize);

/**
 * 64-bit complex long integer signal allocator.
 * \param nSize Number of complex long int values in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Mpp64sc * 
mppsMalloc_64sc(size_t nSize);

/**
 * 64-bit float (double) signal allocator.
 * \param nSize Number of doubles in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Mpp64f * 
mppsMalloc_64f(size_t nSize);

/**
 * 64-bit complex complex signal allocator.
 * \param nSize Number of complex double valuess in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Mpp64fc * 
mppsMalloc_64fc(size_t nSize);

/** @} signal_malloc */

/** 
 * \section signal_free Free
 * @defgroup signal_free Free
 * Free  signal memory.
 *
 * @{
 */
 
/**
 * Free method for any signal memory.
 * \param pValues A pointer to memory allocated using mppiMalloc_<modifier>.
 */
void mppsFree(void * pValues);

/** @} signal_free */

/** @} signal_memory_management */


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MC_MPPS_SUPPORT_FUNCTIONS_H */
