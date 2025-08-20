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
#ifndef MC_MPPCORE_H
#define MC_MPPCORE_H

// #include <mc_runtime_api.h>  // MetaX runtime API - to be provided by implementation

/**
 * \file mppcore.h  
 * Basic MPP functionality. 
 *  This file contains functions to query the MPP version as well as 
 *  info about the MACA compute capabilities on a given computer.
 */
 
#if (!defined(_WIN32) && !defined(_WIN64))
#ifdef MPP_PLUS
#define MPP_LNX_EXTERN_C 1
/**<  Supports Linux MPP_Plus namespace linkage  */
#endif
#else // (!defined(_WIN32) && !defined(_WIN64))
#undef MPP_LNX_EXTERN_C
#endif // (!defined(_WIN32) && !defined(_WIN64))

#include "mppdefs.h"

#ifdef __cplusplus
#ifdef MPP_PLUS
using namespace mppPlusV;
#else
extern "C" {
#endif
#endif

/** 
 * \page core_npp MPP_Plus and MPP Core
 * @defgroup core_npp MPP_Plus and MPP Core
 * Basic functions for library management, in particular library version
 * and device property query functions.
 * @{
 */


#ifdef MPP_PLUS
namespace mppPlusV {
#endif
/**
 * Get the MPP library version.
 *
 * \return A struct containing separate values for major and minor revision 
 *      and build number.
 */
#ifdef MPP_LNX_EXTERN_C
extern "C"
#endif
const MppLibraryVersion * 
mppGetLibVersion(void);

// NOTE:The remainder of these functions are no longer supported for external use.

/**
 * Get the number of Streaming Multiprocessors (SM) on the active MACA device.
 *
 * \return Number of SMs of the default MACA device.
 */
#ifdef MPP_LNX_EXTERN_C
extern "C"
#endif
int 
mppGetGpuNumSMs(void);

/**
 * Get the maximum number of threads per block on the active MACA device.
 *
 * \return Maximum number of threads per block on the active MACA device.
 */
#ifdef MPP_LNX_EXTERN_C
extern "C"
#endif
int 
mppGetMaxThreadsPerBlock(void);

/**
 * Get the maximum number of threads per SM for the active GPU
 *
 * \return Maximum number of threads per SM for the active GPU
 */
#ifdef MPP_LNX_EXTERN_C
extern "C"
#endif
int 
mppGetMaxThreadsPerSM(void);

/**
 * Get the maximum number of threads per SM, maximum threads per block, and number of SMs for the active GPU
 *
 * \return mcSuccess for success, -1 for failure
 */
#ifdef MPP_LNX_EXTERN_C
extern "C"
#endif
int 
mppGetGpuDeviceProperties(int * pMaxThreadsPerSM, int * pMaxThreadsPerBlock, int * pNumberOfSMs);

/** 
 * Get the name of the active MACA device.
 *
 * \return Name string of the active graphics-card/compute device in a system.
 */
#ifdef MPP_LNX_EXTERN_C
extern "C"
#endif
const char * 
mppGetGpuName(void);

/**
 * Get the MPP MACA stream.
 * MPP enables concurrent device tasks via a global stream state varible.
 * The MPP stream by default is set to stream 0, i.e. non-concurrent mode.
 * A user can set the MPP stream to any valid MACA stream. All MACA commands
 * issued by MPP (e.g. kernels launched by the MPP library) are then
 * issed to that MPP stream.
 */
#ifdef MPP_LNX_EXTERN_C
extern "C"
#endif
macaStream_t
mppGetStream(void);

/**
 * Get the current MPP managed MACA stream context as set by calls to mppSetStream().
 * MPP enables concurrent device tasks via an MPP maintained global stream state context.
 * The MPP stream by default is set to stream 0, i.e. non-concurrent mode.
 * A user can set the MPP stream to any valid MACA stream which will update the current MPP managed stream state context 
 * or supply application initialized stream contexts to MPP calls. All MACA commands
 * issued by MPP (e.g. kernels launched by the MPP library) are then
 * issed to the current MPP managed stream or to application supplied stream contexts depending on whether 
 * the stream context is passed to the MPP function or not.  MPP managed stream context calls (those without stream context parameters) 
 * can be intermixed with application managed stream context calls but any MPP managed stream context calls will always use the most recent 
 * stream set by mppSetStream() or the NULL stream if mppSetStream() has never been called. 
 */
#ifdef MPP_LNX_EXTERN_C
extern "C"
#endif
MppStatus
mppGetStreamContext(MppStreamContext * pNppStreamContext);

/**
 * Get the number of SMs on the device associated with the current MPP MACA stream.
 * MPP enables concurrent device tasks via a global stream state varible.
 * The MPP stream by default is set to stream 0, i.e. non-concurrent mode.
 * A user can set the MPP stream to any valid MACA stream. All MACA commands
 * issued by MPP (e.g. kernels launched by the MPP library) are then
 * issed to that MPP stream.  This call avoids a mcGetDeviceProperties() call.
 */
#ifdef MPP_LNX_EXTERN_C
extern "C"
#endif
unsigned int
mppGetStreamNumSMs(void);

/**
 * Get the maximum number of threads per SM on the device associated with the current MPP MACA stream.
 * MPP enables concurrent device tasks via a global stream state varible.
 * The MPP stream by default is set to stream 0, i.e. non-concurrent mode.
 * A user can set the MPP stream to any valid MACA stream. All MACA commands
 * issued by MPP (e.g. kernels launched by the MPP library) are then
 * issed to that MPP stream.  This call avoids a mcGetDeviceProperties() call.
 */
#ifdef MPP_LNX_EXTERN_C
extern "C"
#endif
unsigned int
mppGetStreamMaxThreadsPerSM(void);

/**
 * Set the MPP MACA stream.  This function now returns an error if a problem occurs with mc stream management. 
 *   This function should only be called if a call to mppGetStream() returns a stream number which is different from
 *   the desired stream since unnecessarily flushing the current stream can significantly affect performance.
 * \see mppGetStream()
 */
#ifdef MPP_LNX_EXTERN_C
extern "C"
#endif
MppStatus
mppSetStream(macaStream_t hStream);

#ifdef MPP_PLUS
}
#endif

/** @} core_npp */ 

#ifdef __cplusplus
#ifndef MPP_PLUS
} /* extern "C" */
#endif
#endif

#endif /* MC_MPPCORE_H */
