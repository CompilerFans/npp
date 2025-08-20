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
#ifndef MC_MPPDEFS_H
#define MC_MPPDEFS_H

#pragma once

#include <stdlib.h>
// #include <mc_runtime.h>  // MetaX runtime header - to be provided by implementation
#if !defined(macaStream_t)
typedef size_t macaStream_t;  // Temporary definition - should be provided by MACA runtime
#endif

/**
 * \file mppdefs.h
 * Typedefinitions and macros for MPP library.
 */
 
#ifdef __cplusplus
#ifdef MPP_PLUS
using namespace mppPlusV;
#else
extern "C" {
#endif
#endif

/** 
  * Workarounds for maca_fp16.h C incompatibility 
  * Mpp16f 
  */
 
typedef struct      /**< Mpp16f mc fp16 structure. */
{
   short fp16;      /**< Original mc fp16 data size and format. */
}
Mpp16f;

/** 
  * Mpp16f_2 
  */

typedef struct      /**< Mpp16f_2 mc double fp16 structure. */
{
   short fp16_0;    /**< Original mc fp16 data size and format. */
   short fp16_1;    /**< Original mc fp16 data size and format. */
}
Mpp16f_2;           

/**
 * MPP_HALF_TO_NPP16F definition
 */
#define MPP_HALF_TO_NPP16F(pHalf) (* reinterpret_cast<Mpp16f *>((void *)(pHalf)))

/** If this is a 32-bit Windows compile, don't align to 16-byte at all
  * and use a "union-trick" to create 8-byte alignment.
  */
#if defined(_WIN32) && !defined(_WIN64)

/** On 32-bit Windows platforms, do not force 8-byte alignment.
  * This is a consequence of a limitation of that platform. 
  * 8-byte data alignment
  */ 
    #define MPP_ALIGN_8
/** On 32-bit Windows platforms, do not force 16-byte alignment.
  * This is a consequence of a limitation of that platform. 
  * 16-byte data alignment
  */  
    #define MPP_ALIGN_16

#else /* _WIN32 && !_WIN64 */

    #define MPP_ALIGN_8     __align__(8)    
    /**< 8-byte data alignment */    
    #define MPP_ALIGN_16    __align__(16)
   /**< 16-byte data alignment */

#endif /* !__MACACC__ && _WIN32 && !_WIN64 */


/** \defgroup typedefs_npp MPP Type Definitions and Constants
 * Definitions of types, structures, enumerations and constants available in the library.
 * @{
 */

/** 
 * Filtering methods.
 */
typedef enum 
{
    MPPI_INTER_UNDEFINED         = 0,        /**<  Undefined filtering interpolation mode. */
    MPPI_INTER_NN                = 1,        /**<  Nearest neighbor filtering. */
    MPPI_INTER_LINEAR            = 2,        /**<  Linear interpolation. */
    MPPI_INTER_CUBIC             = 4,        /**<  Cubic interpolation. */
    MPPI_INTER_CUBIC2P_BSPLINE,              /**<  Two-parameter cubic filter (B=1, C=0) */
    MPPI_INTER_CUBIC2P_CATMULLROM,           /**<  Two-parameter cubic filter (B=0, C=1/2) */
    MPPI_INTER_CUBIC2P_B05C03,               /**<  Two-parameter cubic filter (B=1/2, C=3/10) */
    MPPI_INTER_SUPER             = 8,        /**<  Super sampling. */
    MPPI_INTER_LANCZOS           = 16,       /**<  Lanczos filtering. */
    MPPI_INTER_LANCZOS3_ADVANCED = 17,       /**<  Generic Lanczos filtering with order 3. */
    MPPI_SMOOTH_EDGE             = (int)0x8000000 /**<  Smooth edge filtering. */
} MppiInterpolationMode; 

/** 
 * Bayer Grid Position Registration.
 */
typedef enum 
{
    MPPI_BAYER_BGGR         = 0,        /**<  Default registration position BGGR. */
    MPPI_BAYER_RGGB         = 1,        /**<  Registration position RGGB. */
    MPPI_BAYER_GBRG         = 2,        /**<  Registration position GBRG. */
    MPPI_BAYER_GRBG         = 3         /**<  Registration position GRBG. */
} MppiBayerGridPosition; 

/**
 * Fixed filter-kernel sizes.
 */
typedef enum
{
    MPP_MASK_SIZE_1_X_3,            /**< 1 X 3 filter mask size. */
    MPP_MASK_SIZE_1_X_5,            /**< 1 X 5 filter mask size. */
    MPP_MASK_SIZE_3_X_1 = 100,      /**< 3 X 1 filter mask size, leaving space for more 1 X N type enum values. */ 
    MPP_MASK_SIZE_5_X_1,            /**< 5 X 1 filter mask size. */
    MPP_MASK_SIZE_3_X_3 = 200,      /**< 3 X 3 filter mask size, leaving space for more N X 1 type enum values. */
    MPP_MASK_SIZE_5_X_5,            /**< 5 X 5 filter mask size. */
    MPP_MASK_SIZE_7_X_7 = 400,      /**< 7 X 7 filter mask size. */
    MPP_MASK_SIZE_9_X_9 = 500,      /**< 9 X 9 filter mask size. */
    MPP_MASK_SIZE_11_X_11 = 600,    /**< 11 X 11 filter mask size. */
    MPP_MASK_SIZE_13_X_13 = 700,    /**< 13 X 13 filter mask size. */
    MPP_MASK_SIZE_15_X_15 = 800     /**< 15 X 15 filter mask size. */
} MppiMaskSize;

/** 
 * Differential Filter types
 */
 
typedef enum
{
    MPP_FILTER_SOBEL,   /**<  Differential kernel filter type sobel. */
    MPP_FILTER_SCHARR,  /**<  Differential kernel filter type scharr. */
} MppiDifferentialKernel;

/**
 * Error Status Codes
 *
 * Almost all MPP function return error-status information using
 * these return codes.
 * Negative return codes indicate errors, positive return codes indicate
 * warnings, a return code of 0 indicates success.
 */
typedef enum /**< MppStatus MPP return value */
{
    /* negative return-codes indicate errors */
    MPP_NOT_SUPPORTED_MODE_ERROR            = -9999,    /**< Not supported mode error.*/
    
    MPP_INVALID_HOST_POINTER_ERROR          = -1032,    /**< Invalid host memory pointer error.*/
    MPP_INVALID_DEVICE_POINTER_ERROR        = -1031,    /**< Invalid device memory pointer error.*/
    MPP_LUT_PALETTE_BITSIZE_ERROR           = -1030,    /**< Color look up table bitsize error.*/
    MPP_ZC_MODE_NOT_SUPPORTED_ERROR         = -1028,    /**< ZeroCrossing mode not supported error. */
    MPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY   = -1027,    /**< Not sufficient mc compute capability error. */
    MPP_TEXTURE_BIND_ERROR                  = -1024,    /**< Texture bind error. */
    MPP_WRONG_INTERSECTION_ROI_ERROR        = -1020,    /**< Wrong intersection region of interest error. */
    MPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR   = -1006,    /**< Haar classifier pixel match error. */
    MPP_MEMFREE_ERROR                       = -1005,    /**< Memory free request error. */
    MPP_MEMSET_ERROR                        = -1004,    /**< Memory set request error. */
    MPP_MEMCPY_ERROR                        = -1003,    /**< Memory copy request error. */
    MPP_ALIGNMENT_ERROR                     = -1002,    /**< Memory alignment error. */
    MPP_MACA_KERNEL_EXECUTION_ERROR         = -1000,    /**< mc kernel execution error, most commonly mc kernel launch error. */

    MPP_ROUND_MODE_NOT_SUPPORTED_ERROR      = -213,     /**< Unsupported round mode. */
    
    MPP_QUALITY_INDEX_ERROR                 = -210,     /**< Image pixels are constant for quality index. */

    MPP_RESIZE_NO_OPERATION_ERROR           = -201,     /**< One of the output image dimensions is less than 1 pixel. */

    MPP_OVERFLOW_ERROR                      = -109,     /**< Number overflows the upper or lower limit of the data type. */
    MPP_NOT_EVEN_STEP_ERROR                 = -108,     /**< Step value is not pixel multiple. */
    MPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR    = -107,     /**< Number of levels for histogram is less than 2. */
    MPP_LUT_NUMBER_OF_LEVELS_ERROR          = -106,     /**< Number of levels for LUT is less than 2. */

    MPP_CORRUPTED_DATA_ERROR                = -61,      /**< Processed data is corrupted, */
    MPP_CHANNEL_ORDER_ERROR                 = -60,      /**< Wrong order of the destination channels, */
    MPP_ZERO_MASK_VALUE_ERROR               = -59,      /**< All values of the mask are zero, */
    MPP_QUADRANGLE_ERROR                    = -58,      /**< The quadrangle is nonconvex or degenerates into triangle, line or point. */
    MPP_RECTANGLE_ERROR                     = -57,      /**< Size of the rectangle region is less than or equal to 1. */
    MPP_COEFFICIENT_ERROR                   = -56,      /**< Unallowable values of the transformation coefficients.   */

    MPP_NUMBER_OF_CHANNELS_ERROR            = -53,      /**< Bad or unsupported number of channels. */
    MPP_COI_ERROR                           = -52,      /**< Channel of interest is not 1, 2, or 3. */
    MPP_DIVISOR_ERROR                       = -51,      /**< Divisor is equal to zero */

    MPP_CHANNEL_ERROR                       = -47,      /**< Illegal channel index. */
    MPP_STRIDE_ERROR                        = -37,      /**< Stride is less than the row length. */
    
    MPP_ANCHOR_ERROR                        = -34,      /**< Anchor point is outside mask. */
    MPP_MASK_SIZE_ERROR                     = -33,      /**< Lower bound is larger than upper bound. */

    MPP_RESIZE_FACTOR_ERROR                 = -23,
    MPP_INTERPOLATION_ERROR                 = -22,
    MPP_MIRROR_FLIP_ERROR                   = -21,
    MPP_MOMENT_00_ZERO_ERROR                = -20,
    MPP_THRESHOLD_NEGATIVE_LEVEL_ERROR      = -19,
    MPP_THRESHOLD_ERROR                     = -18,
    MPP_CONTEXT_MATCH_ERROR                 = -17,
    MPP_FFT_FLAG_ERROR                      = -16,
    MPP_FFT_ORDER_ERROR                     = -15,
    MPP_STEP_ERROR                          = -14,       /**<  Step is less or equal zero. */
    MPP_SCALE_RANGE_ERROR                   = -13,
    MPP_DATA_TYPE_ERROR                     = -12,
    MPP_OUT_OFF_RANGE_ERROR                 = -11,
    MPP_DIVIDE_BY_ZERO_ERROR                = -10,
    MPP_MEMORY_ALLOCATION_ERR               = -9,
    MPP_NULL_POINTER_ERROR                  = -8,
    MPP_RANGE_ERROR                         = -7,
    MPP_SIZE_ERROR                          = -6,
    MPP_BAD_ARGUMENT_ERROR                  = -5,
    MPP_NO_MEMORY_ERROR                     = -4,
    MPP_NOT_IMPLEMENTED_ERROR               = -3,
    MPP_ERROR                               = -2,
    MPP_ERROR_RESERVED                      = -1,
    
    /* success */
    MPP_NO_ERROR                            = 0,        /**<  Error free operation */
    MPP_SUCCESS = MPP_NO_ERROR,                         /**<  Successful operation (same as MPP_NO_ERROR) */

    /* positive return-codes indicate warnings */
    MPP_NO_OPERATION_WARNING                = 1,        /**<  Indicates that no operation was performed */
    MPP_DIVIDE_BY_ZERO_WARNING              = 6,        /**<  Divisor is zero however does not terminate the execution */
    MPP_AFFINE_QUAD_INCORRECT_WARNING       = 28,       /**<  Indicates that the quadrangle passed to one of affine warping functions doesn't have necessary properties. First 3 vertices are used, the fourth vertex discarded. */
    MPP_WRONG_INTERSECTION_ROI_WARNING      = 29,       /**<  The given ROI has no interestion with either the source or destination ROI. Thus no operation was performed. */
    MPP_WRONG_INTERSECTION_QUAD_WARNING     = 30,       /**<  The given quadrangle has no intersection with either the source or destination ROI. Thus no operation was performed. */
    MPP_DOUBLE_SIZE_WARNING                 = 35,       /**<  Image size isn't multiple of two. Indicates that in case of 422/411/420 sampling the ROI width/height was modified for proper processing. */
    
    MPP_MISALIGNED_DST_ROI_WARNING          = 10000,    /**<  Speed reduction due to uncoalesced memory accesses warning. */
   
} MppStatus;


/**
 * MppLibraryVersion
 * This struct contains the MPP Library Version information.
 */
typedef struct 
{
    int    major;   /**<  Major version number */
    int    minor;   /**<  Minor version number */
    int    build;   /**<  Build number. This reflects the nightly build this release was made from. */
} MppLibraryVersion;  


/** @} typedefs_npp. */ 

/** 
 * \defgroup npp_basic_types Basic MPP Data Types
 * Definitions of basic types available in the library.
 * @{
 */

typedef unsigned char       Mpp8u;     /**<  8-bit unsigned chars */
typedef signed char         Mpp8s;     /**<  8-bit signed chars */
typedef unsigned short      Mpp16u;    /**<  16-bit unsigned integers */
typedef short               Mpp16s;    /**<  16-bit signed integers */
typedef unsigned int        Mpp32u;    /**<  32-bit unsigned integers */
typedef int                 Mpp32s;    /**<  32-bit signed integers */
typedef unsigned long long  Mpp64u;    /**<  64-bit unsigned integers */
typedef long long           Mpp64s;    /**<  64-bit signed integers */
typedef float               Mpp32f;    /**<  32-bit (IEEE) floating-point numbers */
typedef double              Mpp64f;    /**<  64-bit floating-point numbers */


/**
 * Complex Number
 * This struct represents an unsigned char complex number.
 */
typedef struct
{
    Mpp8u  re;     /**<  Real part */
    Mpp8u  im;     /**<  Imaginary part */
} Mpp8uc;

/**
 * Complex Number
 * This struct represents an unsigned short complex number.
 */
typedef struct
{
    Mpp16u  re;     /**<  Real part */
    Mpp16u  im;     /**<  Imaginary part */
} Mpp16uc;

/**
 * Complex Number
 * This struct represents a short complex number.
 */
typedef struct
{
    Mpp16s  re;     /**<  Real part */
    Mpp16s  im;     /**<  Imaginary part */
} Mpp16sc;

/**
 * Complex Number
 * This struct represents an unsigned int complex number.
 */
typedef struct
{
    Mpp32u  re;     /**<  Real part */
    Mpp32u  im;     /**<  Imaginary part */
} Mpp32uc;

/**
 * Complex Number
 * This struct represents a signed int complex number.
 */
typedef struct
{
    Mpp32s  re;     /**<  Real part */
    Mpp32s  im;     /**<  Imaginary part */
} Mpp32sc;

/**
 * Complex Number
 * This struct represents a single floating-point complex number.
 */
typedef struct
{
    Mpp32f  re;     /**<  Real part */
    Mpp32f  im;     /**<  Imaginary part */
} Mpp32fc;

/**
 * Complex Number
 * This struct represents a long long complex number.
 */
typedef struct
{
    Mpp64s  re;     /**<  Real part */
    Mpp64s  im;     /**<  Imaginary part */
} Mpp64sc;

/**
 * Complex Number
 * This struct represents a double floating-point complex number.
 */
typedef struct
{
    Mpp64f  re;     /**<  Real part */
    Mpp64f  im;     /**<  Imaginary part */
} Mpp64fc;


/** 
 * Data types for mppiPlus functions
 */
typedef enum
{
    MPP_8U,     /**<  8-bit unsigned integer data type */
    MPP_8S,     /**<  8-bit signed integer data type */
    MPP_16U,    /**< 16-bit unsigned integer data type */
    MPP_16S,    /**< 16-bit signed integer data type */
    MPP_32U,    /**< 32-bit unsigned integer data type */
    MPP_32S,    /**< 32-bit signed integer data type */
    MPP_64U,    /**< 64-bit unsigned integer data type */
    MPP_64S,    /**< 64-bit signed integer data type */
    MPP_16F,    /**< 16-bit original mc floating point data type */
    MPP_32F,    /**< 32-bit floating point data type */
    MPP_64F     /**< 64-bit double precision floating point data type */
} MppDataType;

/** 
 * Image channel counts for mppiPlus functions
 */
typedef enum
{
    MPP_CH_1,   /**<  Single channel per pixel data */
    MPP_CH_2,   /**<  Two channel per pixel data */
    MPP_CH_3,   /**<  Three channel per pixel data */
    MPP_CH_4,   /**<  Four channel per pixel data */
    MPP_CH_A4,  /**<  Four channel per pixel data with alpha */
    MPP_CH_P2,  /**<  Two channel single channel per plane pixel data */
    MPP_CH_P3,  /**<  Three channel single channel per plane pixel data */
    MPP_CH_P4   /**<  Four channel single channel per plane pixel data */
} MppiChannels;

/** @} npp_basic_types. */ 

#define MPP_MIN_8U      ( 0 )                        
/**<  Minimum 8-bit unsigned integer */
#define MPP_MAX_8U      ( 255 )
/**<  Maximum 8-bit unsigned integer */
#define MPP_MIN_16U     ( 0 )
/**<  Minimum 16-bit unsigned integer */
#define MPP_MAX_16U     ( 65535 )
/**<  Maximum 16-bit unsigned integer */
#define MPP_MIN_32U     ( 0 )
/**<  Minimum 32-bit unsigned integer */
#define MPP_MAX_32U     ( 4294967295U )
/**<  Maximum 32-bit unsigned integer */
#define MPP_MIN_64U     ( 0 )
/**<  Minimum 64-bit unsigned integer */
#define MPP_MAX_64U     ( 18446744073709551615ULL )  
/**<  Maximum 64-bit unsigned integer */

#define MPP_MIN_8S      (-127 - 1 )                  
/**<  Minimum 8-bit signed integer */
#define MPP_MAX_8S      ( 127 )                      
/**<  Maximum 8-bit signed integer */
#define MPP_MIN_16S     (-32767 - 1 )
/**<  Minimum 16-bit signed integer */
#define MPP_MAX_16S     ( 32767 )
/**<  Maximum 16-bit signed integer */
#define MPP_MIN_32S     (-2147483647 - 1 )           
/**<  Minimum 32-bit signed integer */
#define MPP_MAX_32S     ( 2147483647 )
/**<  Maximum 32-bit signed integer */
#define MPP_MAX_64S     ( 9223372036854775807LL )    
/**<  Minimum 64-bit signed integer */
#define MPP_MIN_64S     (-9223372036854775807LL - 1)
/**<  Minimum 64-bit signed integer */

#define MPP_MINABS_32F  ( 1.175494351e-38f )         
/**<  Smallest positive 32-bit floating point value */
#define MPP_MAXABS_32F  ( 3.402823466e+38f )         
/**<  Largest  positive 32-bit floating point value */
#define MPP_MINABS_64F  ( 2.2250738585072014e-308 )  
/**<  Smallest positive 64-bit floating point value */
#define MPP_MAXABS_64F  ( 1.7976931348623158e+308 )  
/**<  Largest  positive 64-bit floating point value */


/** 
 * 2D Point
 */
typedef struct 
{
    int x;      /**<  x-coordinate. */
    int y;      /**<  y-coordinate. */
} MppiPoint;

/** 
 * 2D Mpp32f Point
 */
typedef struct 
{
    Mpp32f x;      /**<  x-coordinate. */
    Mpp32f y;      /**<  y-coordinate. */
} MppiPoint32f;

/** 
 * 2D Mpp64f Point
 */
typedef struct 
{
    Mpp64f x;      /**<  x-coordinate. */
    Mpp64f y;      /**<  y-coordinate. */
} MppiPoint64f;

/** 
 * 2D Polar Point
 */
typedef struct {
    Mpp32f rho;     /**<  Polar rho value. */
    Mpp32f theta;   /**<  Polar theta value. */
} MppPointPolar;

/**
 * 2D Size
 * This struct typically represents the size of a a rectangular region in
 * two space.
 */
typedef struct 
{
    int width;  /**<  Rectangle width. */
    int height; /**<  Rectangle height. */
} MppiSize;

/**
 * 2D Rectangle
 * This struct contains position and size information of a rectangle in 
 * two space.
 * The rectangle's position is usually signified by the coordinate of its
 * upper-left corner.
 */
typedef struct
{
    int x;          /**<  x-coordinate of upper left corner (lowest memory address). */
    int y;          /**<  y-coordinate of upper left corner (lowest memory address). */
    int width;      /**<  Rectangle width. */
    int height;     /**<  Rectangle height. */
} MppiRect;

/**
 * mppiMirror direction controls
 */
typedef enum 
{
    MPP_HORIZONTAL_AXIS,    /**<  Flip around horizontal axis in mirror function. */
    MPP_VERTICAL_AXIS,      /**<  Flip around vertical axis in mirror function. */
    MPP_BOTH_AXIS           /**<  Flip around both axes in mirror function. */
} MppiAxis;

/**
 * Pixel comparison control values
 */
typedef enum 
{
    MPP_CMP_LESS,           /**<  Threshold test for less than. */
    MPP_CMP_LESS_EQ,        /**<  Threshold test for less than or equal. */
    MPP_CMP_EQ,             /**<  Threshold test for equal. */
    MPP_CMP_GREATER_EQ,     /**<  Threshold test for greater than or equal. */
    MPP_CMP_GREATER         /**<  Threshold test for greater than. */
} MppCmpOp;

/**
 * Rounding Modes
 *
 * The enumerated rounding modes are used by a large number of MPP primitives
 * to allow the user to specify the method by which fractional values are converted
 * to integer values. Also see \ref rounding_modes.
 *
 * For MPP release 5.5 new names for the three rounding modes are introduced that are
 * based on the naming conventions for rounding modes set forth in the IEEE-754
 * floating-point standard. Developers are encouraged to use the new, longer names
 * to be future proof as the legacy names will be deprecated in subsequent MPP releases.
 *
 */
typedef enum 
{
    /** 
     * Round to the nearest even integer.
     * All fractional numbers are rounded to their nearest integer. The ambiguous
     * cases (i.e. \<integer\>.5) are rounded to the closest even integer.
     * E.g.
     * - roundNear(0.5) = 0
     * - roundNear(0.6) = 1
     * - roundNear(1.5) = 2
     * - roundNear(-1.5) = -2
     */
    MPP_RND_NEAR,
    MPP_ROUND_NEAREST_TIES_TO_EVEN = MPP_RND_NEAR, ///< Alias name for MPP_RND_NEAR.
    /** 
     * Round according to financial rule.
     * All fractional numbers are rounded to their nearest integer. The ambiguous
     * cases (i.e. \<integer\>.5) are rounded away from zero.
     * E.g.
     * - roundFinancial(0.4)  = 0
     * - roundFinancial(0.5)  = 1
     * - roundFinancial(-1.5) = -2
     */
    MPP_RND_FINANCIAL,
    MPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO = MPP_RND_FINANCIAL, ///< Alias name for MPP_RND_FINANCIAL. 
    /**
     * Round towards zero (truncation). 
     * All fractional numbers of the form \<integer\>.\<decimals\> are truncated to
     * \<integer\>.
     * - roundZero(1.5) = 1
     * - roundZero(1.9) = 1
     * - roundZero(-2.5) = -2
     */
    MPP_RND_ZERO,
    MPP_ROUND_TOWARD_ZERO = MPP_RND_ZERO, ///< Alias name for MPP_RND_ZERO. 
    
    /**
     * Other rounding modes supported by IEEE-754 (2008) floating-point standard:
     *
     * - MPP_ROUND_TOWARD_INFINITY // ceiling
     * - MPP_ROUND_TOWARD_NEGATIVE_INFINITY // floor
     *
     */
} MppRoundMode;

/**
 * Supported image border modes
 */
typedef enum  
{
    MPP_BORDER_UNDEFINED        = 0,                    /**<  Image border type undefined. */
    MPP_BORDER_NONE             = MPP_BORDER_UNDEFINED, /**<  Image border type none. */
    MPP_BORDER_CONSTANT         = 1,                    /**<  Image border type constant value. */
    MPP_BORDER_REPLICATE        = 2,                    /**<  Image border type replicate border pixels. */
    MPP_BORDER_WRAP             = 3,                    /**<  Image border type wrap border pixels. */
    MPP_BORDER_MIRROR           = 4                     /**<  Image border type mirror border pixels. */
} MppiBorderType;


/**
 * Hints
 */
typedef enum {
    MPP_ALG_HINT_NONE,      /**<  Hint none, currently these are all ignored. */
    MPP_ALG_HINT_FAST,      /**<  Hint fast, currently these are all ignored. */
    MPP_ALG_HINT_ACCURATE   /**<  Hint accurate, currently these are all ignored. */
} MppHintAlgorithm;

/** 
 * Alpha composition mode controls. 
 */

typedef enum {
    MPPI_OP_ALPHA_OVER,         /**< Alpha composition over operation. */
    MPPI_OP_ALPHA_IN,           /**< Alpha composition in operation. */
    MPPI_OP_ALPHA_OUT,          /**< Alpha composition out operation. */
    MPPI_OP_ALPHA_ATOP,         /**< Alpha composition atop operation. */
    MPPI_OP_ALPHA_XOR,          /**< Alpha composition xor operation. */
    MPPI_OP_ALPHA_PLUS,         /**< Alpha composition plus operation. */
    MPPI_OP_ALPHA_OVER_PREMUL,  /**< Alpha composition over premultiply operation. */
    MPPI_OP_ALPHA_IN_PREMUL,    /**< Alpha composition in premultiply operation. */
    MPPI_OP_ALPHA_OUT_PREMUL,   /**< Alpha composition out premultiply operation. */
    MPPI_OP_ALPHA_ATOP_PREMUL,  /**< Alpha composition atop premultiply operation. */
    MPPI_OP_ALPHA_XOR_PREMUL,   /**< Alpha composition xor premultiply operation. */
    MPPI_OP_ALPHA_PLUS_PREMUL,  /**< Alpha composition plus premultiply operation. */
    MPPI_OP_ALPHA_PREMUL        /**< Alpha composition premultiply operation. */
} MppiAlphaOp;


/** 
 * The MppiHOGConfig structure defines the configuration parameters for the HOG descriptor: 
 */
 
typedef struct 
{
    int      cellSize;             /**<  square cell size (pixels). */
    int      histogramBlockSize;   /**<  square histogram block size (pixels). */
    int      nHistogramBins;       /**<  required number of histogram bins. */
    MppiSize detectionWindowSize;  /**<  detection window size (pixels). */
} MppiHOGConfig;

/**
 * HOG Cell controls
 */
#define MPP_HOG_MAX_CELL_SIZE                          (16) 
/**< max horizontal/vertical pixel size of cell.   */
#define MPP_HOG_MAX_BLOCK_SIZE                         (64) 
/**< max horizontal/vertical pixel size of block.  */
#define MPP_HOG_MAX_BINS_PER_CELL                      (16) 
/**< max number of histogram bins. */
#define MPP_HOG_MAX_CELLS_PER_DESCRIPTOR              (256) 
/**< max number of cells in a descriptor window.   */
#define MPP_HOG_MAX_OVERLAPPING_BLOCKS_PER_DESCRIPTOR (256) 
/**< max number of overlapping blocks in a descriptor window.   */
#define MPP_HOG_MAX_DESCRIPTOR_LOCATIONS_PER_CALL     (128) 
/**< max number of descriptor window locations per function call.   */

/**
 * Data structure for HaarClassifier_32f.
 */

typedef struct
{
    int      numClassifiers;    /**<  number of classifiers. */
    Mpp32s * classifiers;       /**<  packed classifier data 40 bytes each. */
    size_t   classifierStep;    /**<  packed classifier byte step. */
    MppiSize classifierSize;    /**<  packed classifier size. */
    Mpp32s * counterDevice;     /**<  counter device. */
} MppiHaarClassifier_32f;

/** 
  * Data structure for Haar buffer.
  */
  
typedef struct
{
    int      haarBufferSize;    /**<  size of the buffer */
    Mpp32s * haarBuffer;        /**<  buffer */
    
} MppiHaarBuffer;

/**
 * Signal sign operations
 */
typedef enum {
    mppZCR,    /**<  sign change */
    mppZCXor,  /**<  sign change XOR */
    mppZCC     /**<  sign change count_0 */
} MppsZCType;

/**
 * HuffMan Table controls
 */
typedef enum {
    mppiDCTable,    /**<  DC Table */
    mppiACTable,    /**<  AC Table */
} MppiHuffmanTableType;

/**
 * Norm controls
 */
typedef enum {
    mppiNormInf = 0, /**<  maximum */ 
    mppiNormL1 = 1,  /**<  sum */
    mppiNormL2 = 2   /**<  square root of sum of squares */
} MppiNorm;

/**
 * Data structure of connected pixel region information.
 */

typedef struct
{
	MppiRect oBoundingBox; 		 /**<  x, y, width, height == left, top, right, and bottom pixel coordinates */
	Mpp32u nConnectedPixelCount; /**< total number of pixels in connected region */
	Mpp32u aSeedPixelValue[3];	 /**< original pixel value of seed pixel, 1 or 3 channel */
} MppiConnectedRegion;

/**
 * General image descriptor. Defines the basic parameters of an image,
 * including data pointer, step, and image size.
 * This can be used by both source and destination images.
 */
typedef struct
{
    void *     pData;  /**< device memory pointer to the image */
    int        nStep;  /**< step size */
    MppiSize  oSize;   /**< width and height of the image */
} MppiImageDescriptor;


/**
 * struct MppiBufferDescriptor
 */
typedef struct
{
    void *     pData;        /**< per image device memory pointer to the corresponding buffer */
    int        nBufferSize;  /**< allocated buffer size */
} MppiBufferDescriptor;


/**
 * Provides details of uniquely labeled pixel regions of interest returned 
 * by CompressedLabelMarkersUF function. 
 */
typedef struct
{
    Mpp32u nMarkerLabelPixelCount;           /**< total number of pixels in this connected pixel region */
    Mpp32u nContourPixelCount;               /**< total number of pixels in this connected pixel region contour */
    Mpp32u nContourPixelsFound;              /**< total number of pixels in this connected pixel region contour found during geometry search */
    MppiPoint oContourFirstPixelLocation;    /**< image geometric x and y location of the first pixel in the contour */
    MppiRect oMarkerLabelBoundingBox;        /**< bounding box of this connected pixel region expressed as leftmostX, topmostY, rightmostX, and bottommostY */
} MppiCompressedMarkerLabelsInfo;


/**
 * Provides details of contour pixel grid map location and association 
 */

typedef struct
{
    Mpp32u nMarkerLabelID;                   /**< this connected pixel region contour ID */
    Mpp32u nContourPixelCount;               /**< total number of pixels in this connected pixel region contour */
    Mpp32u nContourStartingPixelOffset;      /**< base offset of starting pixel in the overall contour pixel list */
    Mpp32u nSegmentNum;                      /**< relative block segment number within this particular contour ID */
} MppiContourBlockSegment;


/**
 * Provides contour (boundary) geometry info of uniquely labeled pixel regions returned 
 * by mppiCompressedMarkerLabelsUFInfo function in host memory in counterclockwise order relative to contour interiors. 
 */
typedef struct
{
    MppiPoint oContourOrderedGeometryLocation;                 /**< image geometry X and Y location in requested output order */
    MppiPoint oContourPrevPixelLocation;                       /**< image geometry X and Y location of previous contour pixel */
    MppiPoint oContourCenterPixelLocation;                     /**< image geometry X and Y location of center contour pixel */
    MppiPoint oContourNextPixelLocation;                       /**< image geometry X and Y location of next contour pixel */
    Mpp32s nOrderIndex;                                        /**< contour pixel counterclockwise order index in geometry list */
    Mpp32s nReverseOrderIndex;                                 /**< contour pixel clockwise order index in geometry list */
    Mpp32u nFirstIndex;                                        /**< index of first ordered contour pixel in this subgroup */
    Mpp32u nLastIndex;                                         /**< index of last ordered contour pixel in this subgroup */
    Mpp32u nNextContourPixelIndex;                             /**< index of next ordered contour pixel in MppiContourPixelGeometryInfo list */
    Mpp32u nPrevContourPixelIndex;                             /**< index of previous ordered contour pixel in MppiContourPixelGeometryInfo list */
    Mpp8u nPixelAlreadyUsed;                                   /**< this test pixel is has already been used */
    Mpp8u nAlreadyLinked;                                      /**< this test pixel is already linked to center pixel */
    Mpp8u nAlreadyOutput;                                      /**< this pixel has already been output in geometry list */
    Mpp8u nContourInteriorDirection;                           /**< direction of contour region interior */
} MppiContourPixelGeometryInfo;

/**
 * Provides contour (boundary) direction info of uniquely labeled pixel regions returned 
 * by CompressedLabelMarkersUF function. 
 */

#define MPP_CONTOUR_DIRECTION_SOUTH_EAST    1       
/**< Contour direction south east */
#define MPP_CONTOUR_DIRECTION_SOUTH         2       
/**< Contour direction south */
#define MPP_CONTOUR_DIRECTION_SOUTH_WEST    4       
/**< Contour direction south west */
#define MPP_CONTOUR_DIRECTION_WEST          8       
/**< Contour direction west */
#define MPP_CONTOUR_DIRECTION_EAST         16       
/**< Contour direction east */
#define MPP_CONTOUR_DIRECTION_NORTH_EAST   32       
/**< Contour direction north east */
#define MPP_CONTOUR_DIRECTION_NORTH        64       
/**< Contour direction north */
#define MPP_CONTOUR_DIRECTION_NORTH_WEST  128       
/**< Contour direction north west */

/**
 * Provides contour (boundary) diagonal direction info of uniquely labeled pixel regions returned 
 * by CompressedLabelMarkersUF function. 
 */
#define MPP_CONTOUR_DIRECTION_ANY_NORTH  MPP_CONTOUR_DIRECTION_NORTH_EAST | MPP_CONTOUR_DIRECTION_NORTH | MPP_CONTOUR_DIRECTION_NORTH_WEST  
/**< Contour direction any north */
#define MPP_CONTOUR_DIRECTION_ANY_WEST   MPP_CONTOUR_DIRECTION_NORTH_WEST | MPP_CONTOUR_DIRECTION_WEST | MPP_CONTOUR_DIRECTION_SOUTH_WEST   
/**< Contour direction any west */
#define MPP_CONTOUR_DIRECTION_ANY_SOUTH  MPP_CONTOUR_DIRECTION_SOUTH_EAST | MPP_CONTOUR_DIRECTION_SOUTH | MPP_CONTOUR_DIRECTION_SOUTH_WEST  
/**< Contour direction any south */
#define MPP_CONTOUR_DIRECTION_ANY_EAST   MPP_CONTOUR_DIRECTION_NORTH_EAST | MPP_CONTOUR_DIRECTION_EAST | MPP_CONTOUR_DIRECTION_SOUTH_EAST   
/**< Contour direction any east */

/** 
 * Data structure for contour pixel direction information.
 */
typedef struct
{
    Mpp32u  nMarkerLabelID;                         /**< MarkerLabelID of contour interior connected region */
    Mpp8u nContourDirectionCenterPixel;             /**< provides current center contour pixel input and output direction info */
    Mpp8u nContourInteriorDirectionCenterPixel;     /**< provides current center contour pixel region interior direction info */
    Mpp8u nConnected;                               /**< direction to directly connected contour pixels, 0 if not connected */
    Mpp8u nGeometryInfoIsValid;                     /**< direction to directly connected contour pixels, 0 if not connected */
    MppiContourPixelGeometryInfo oContourPixelGeometryInfo; /**< Pixel geometry info structure */
    MppiPoint nEast1;                               /**< Pixel coordinate values in each direction */
    MppiPoint nNorthEast1;                          /**< Pixel coordinate values in each direction */
    MppiPoint nNorth1;                              /**< Pixel coordinate values in each direction */
    MppiPoint nNorthWest1;                          /**< Pixel coordinate values in each direction */
    MppiPoint nWest1;                               /**< Pixel coordinate values in each direction */
    MppiPoint nSouthWest1;                          /**< Pixel coordinate values in each direction */
    MppiPoint nSouth1;                              /**< Pixel coordinate values in each direction */
    MppiPoint nSouthEast1;                          /**< Pixel coordinate values in each direction */
    Mpp8u nTest1EastConnected;                      /**< East connected flag */
    Mpp8u nTest1NorthEastConnected;                 /**< North east connected flag */
    Mpp8u nTest1NorthConnected;                     /**< North connected flag */
    Mpp8u nTest1NorthWestConnected;                 /**< North west connected flag */
    Mpp8u nTest1WestConnected;                      /**< West connected flag */
    Mpp8u nTest1SouthWestConnected;                 /**< South west connected flag */
    Mpp8u nTest1SouthConnected;                     /**< South connected flag */
    Mpp8u nTest1SouthEastConnected;                 /**< South east connected flag */
} MppiContourPixelDirectionInfo;

/** 
 * Data structure for contour total counts.
 */
typedef struct
{
    Mpp32u nTotalImagePixelContourCount;    /**< total number of contour pixels in image */
    Mpp32u nLongestImageContourPixelCount;  /**< longest per contour pixel count in image */
} MppiContourTotalsInfo;


/**
 * Provides control of the type of segment boundaries, if any, added 
 * to the image generated by the watershed segmentation function. 
 */

typedef enum 
{
    MPP_WATERSHED_SEGMENT_BOUNDARIES_NONE,      /**< Image watershed segment boundary type none. */
    MPP_WATERSHED_SEGMENT_BOUNDARIES_BLACK,     /**< Image watershed segment boundary type black. */
    MPP_WATERSHED_SEGMENT_BOUNDARIES_WHITE,     /**< Image watershed segment boundary type white. */
    MPP_WATERSHED_SEGMENT_BOUNDARIES_CONTRAST,  /**< Image watershed segment boundary type contrasting intensiity. */
    MPP_WATERSHED_SEGMENT_BOUNDARIES_ONLY       /**< Image watershed segment boundary type render boundaries only. */
} MppiWatershedSegmentBoundaryType;

    
/** 
 * \section application_managed_stream_context Application Managed Stream Context
 * MPP stream context structure must be filled in by application. 
 * Application should not initialize or alter reserved fields. 
 * 
 */
typedef struct
{
    macaStream_t hStream;                   /**<  From current mc stream ID. */
    int nMacaDeviceId;                      /**<  From mcGetDevice(). */
    int nMultiProcessorCount;               /**<  From mcGetDeviceProperties(). */
    int nMaxThreadsPerMultiProcessor;       /**<  From mcGetDeviceProperties(). */
    int nMaxThreadsPerBlock;                /**<  From mcGetDeviceProperties(). */
    size_t nSharedMemPerBlock;              /**<  From mcGetDeviceProperties(). */
    int nMacaDevAttrComputeCapabilityMajor; /**<  From mcGetDeviceAttribute(). */
    int nMacaDevAttrComputeCapabilityMinor; /**<  From mcGetDeviceAttribute(). */
    unsigned int nStreamFlags;              /**<  From mcStreamGetFlags(). */
    int nReserved0;                         /**<  reserved, do not alter. */                         
} MppStreamContext;

/** 
 * MPP Batch Geometry Structure Definitions. 
 */
typedef struct
{
    const void * pSrc;  /**< source image device memory pointer. */
    int nSrcStep;       /**< source image byte count per row. */
    void * pDst;        /**< destination image device memory pointer. */
    int nDstStep;       /**< device image byte count per row. */
} MppiResizeBatchCXR;

/**
 * Data structure for variable ROI image batch resizing.
 * 
 */
typedef struct
{
    MppiRect oSrcRectROI;   /**< per source image rectangle parameters. */    
    MppiRect oDstRectROI;   /**< per destination image rectangle parameters. */
} MppiResizeBatchROI_Advanced; 
 
/**
 * Data structure for batched mppiMirrorBatch.
 */
typedef struct
{
    const void * pSrc;  /**< source image device memory pointer, ignored for in-place versions. */
    int nSrcStep;       /**< source image byte count per row. */
    void * pDst;        /**< destination image device memory pointer. */
    int nDstStep;       /**< device image byte count per row. */
} MppiMirrorBatchCXR;

/**
 * Data structure for batched mppiWarpAffineBatch.
 */
typedef struct
{
    const void * pSrc;  /**< source image device memory pointer. */
    int nSrcStep;       /**< source image byte count per row. */
    void * pDst;        /**< destination image device memory pointer. */
    int nDstStep;       /**< device image byte count per row. */
    Mpp64f * pCoeffs;   /**< device memory pointer to the tranformation matrix with double precision floating-point coefficient values to be used for this image. */
    Mpp64f aTransformedCoeffs[2][3]; /**< FOR INTERNAL USE, DO NOT INITIALIZE.  */
} MppiWarpAffineBatchCXR;

/**
 * Data structure for batched mppiWarpPerspectiveBatch.
 */
typedef struct
{
    const void * pSrc;  /**< source image device memory pointer. */
    int nSrcStep;       /**< source image byte count per row. */
    void * pDst;        /**< destination image device memory pointer. */
    int nDstStep;       /**< device image byte count per row. */
    Mpp64f * pCoeffs;   /**< device memory pointer to the tranformation matrix with double precision floating-point coefficient values to be used for this image. */
    Mpp64f aTransformedCoeffs[3][3]; /**< FOR INTERNAL USE, DO NOT INITIALIZE.  */
} MppiWarpPerspectiveBatchCXR;


/**
 * Data structure for batched mppiColorTwistBatch.
 */
typedef struct
{
    const void * pSrc;  /**< source image device memory pointer. */
    int nSrcStep;       /**< source image byte count per row. */
    void * pDst;        /**< destination image device memory pointer. */
    int nDstStep;       /**< device image byte count per row. */
    Mpp32f * pTwist;    /**< device memory pointer to the color twist matrix with floating-point coefficient values to be used for this image. */
} MppiColorTwistBatchCXR;

/**
 * Profile data type for radial and linear profiles
 */
typedef struct
{
    int nPixels;                /**< profile data pixel count. */    
    Mpp32f nMeanIntensity;      /**< profile data mean intensity. */
    Mpp32f nStdDevIntensity;    /**< profile data standard deviation intensity. */
} MppiProfileData;


#ifdef __cplusplus
#ifndef MPP_PLUS
} /* extern "C" */
#endif
#endif

/*@}*/
 
#endif /* MC_MPPDEFS_H */

