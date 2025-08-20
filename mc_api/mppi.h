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
#ifndef MC_MPPI_H
#define MC_MPPI_H
 
/**
 * \file mppi.h
 * MPP Image Processing Functionality.
 */
 
#include "mppdefs.h"

#ifdef __cplusplus
#ifdef MPP_PLUS
using namespace mppPlusV;
#else
extern "C" {
#endif
#endif

/** @defgroup mppi MPP Image Processing
 * The set of image processing functions available in the library.
 * @{
 */
#ifdef MPP_PLUS

#include "mppPlus/mppiPlus_support_functions.h"
#include "mppPlus/mppiPlus_data_exchange_and_initialization.h"
#include "mppPlus/mppiPlus_arithmetic_and_logical_operations.h"
#include "mppPlus/mppiPlus_color_conversion.h"
#include "mppPlus/mppiPlus_threshold_and_compare_operations.h"
#include "mppPlus/mppiPlus_morphological_operations.h"
#include "mppPlus/mppiPlus_filtering_functions.h"
#include "mppPlus/mppiPlus_statistics_functions.h"
#include "mppPlus/mppiPlus_linear_transforms.h"
#include "mppPlus/mppiPlus_geometry_transforms.h"

#else

#include "mppi_support_functions.h"
#include "mppi_data_exchange_and_initialization.h"
#include "mppi_arithmetic_and_logical_operations.h"
#include "mppi_color_conversion.h"
#include "mppi_threshold_and_compare_operations.h"
#include "mppi_morphological_operations.h"
#include "mppi_filtering_functions.h"
#include "mppi_statistics_functions.h"
#include "mppi_linear_transforms.h"
#include "mppi_geometry_transforms.h"

#endif

/** @} end of Image Processing module */

#ifdef __cplusplus
#ifndef MPP_PLUS
} /* extern "C" */
#endif
#endif

#endif /* MC_MPPI_H */
