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
#ifndef MC_MPP_H
#define MC_MPP_H

/**
 * \file mpp.h
 * Main include file for MPP library.
 *      Aggregates all other include files.
 */

/**
 * Major version
 */
#define MPP_VER_MAJOR 12
/**
 * Minor version
 */
#define MPP_VER_MINOR 3
/**
 * Patch version
 */
#define MPP_VER_PATCH 3
/**
 * Build version
 */
#define MPP_VER_BUILD 65

/**
 * Full version
 */
#define MPP_VERSION (MPP_VER_MAJOR * 1000 +     \
                     MPP_VER_MINOR *  100 +     \
                     MPP_VER_PATCH)

/**
 * Major version
 */
#define MPP_VERSION_MAJOR  MPP_VER_MAJOR
/**
 * Minor version
 */
#define MPP_VERSION_MINOR  MPP_VER_MINOR
/**
 * Patch version
 */
#define MPP_VERSION_PATCH  MPP_VER_PATCH
/**
 * Build version
 */
#define MPP_VERSION_BUILD  MPP_VER_BUILD

#include <mppdefs.h>
#include <mppcore.h>
#include <mppi.h>
#include <mpps.h>

#endif // MC_MPP_H
