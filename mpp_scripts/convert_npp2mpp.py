#!/usr/bin/env python3
"""
NPP to MPP Conversion Script

This script converts all NPP-related content in the project to MPP,
including source code, directory names, file names, API headers, and build configurations.

Conversions performed:
- File/directory names: nppi* -> mppi*, npps* -> mpps*, npp* -> mpp*
- Type names: Npp* -> Mpp*, NppiSize -> MppiSize, etc.
- Function names: nppi* -> mppi*, npps* -> mpps*, npp* -> mpp*
- Constants: NPP_* -> MPP_*, NPPI_* -> MPPI_*
- API headers: API/ directory is backed up and converted

Usage:
    ./convert_npp_to_mpp.py --preview     # Preview changes without applying
    ./convert_npp_to_mpp.py --apply       # Apply all changes
    ./convert_npp_to_mpp.py --backup      # Create backup before conversion
    ./convert_npp_to_mpp.py --restore     # Restore from backup
"""

import os
import re
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Set

# Directories to process (source code, tests, etc.)
PROCESS_DIRS = ['src', 'test', 'cmake', 'examples']
# API directory to process separately (backup and convert)
API_DIR = 'API'
# Original API version directories (read-only reference, not modified)
API_VERSION_DIRS = ['API-11.4', 'API-12.2', 'API-12.8']
# Directories to skip
SKIP_DIRS = ['build', 'build-nvidia', '.git', 'third_party', 'wrapper', '.npp_backup', '.cuda_backup'] + API_VERSION_DIRS
# Root files to process and backup
ROOT_FILES = ['CMakeLists.txt', 'build.sh', 'build_maca.sh']
# File extensions to process for content replacement
CODE_EXTENSIONS = {'.cpp', '.cu', '.h', '.hpp', '.cuh', '.c'}
BUILD_EXTENSIONS = {'.cmake', '.txt', '.sh', '.md'}
ALL_EXTENSIONS = CODE_EXTENSIONS | BUILD_EXTENSIONS

# Backup directory name
BACKUP_DIR = '.npp_backup'


def get_text_replacements() -> List[Tuple[str, str]]:
    """Get all text replacement patterns (regex pattern, replacement)."""
    return [
        # Directory path patterns in CMake/build files and include paths (must be first)
        # Handle subdirectory names like ../nppi_arithmetic_operations/
        (r'\.\./nppi_([a-z_]+)', r'../mppi_\1'),
        (r'\.\./npps_([a-z_]+)', r'../mpps_\1'),
        (r'/nppi_([a-z_]+)', r'/mppi_\1'),
        (r'/npps_([a-z_]+)', r'/mpps_\1'),
        # Handle paths with trailing content (like /nppi/test_*.cpp)
        (r'/nppi/([a-zA-Z_*.])', r'/mppi/\1'),
        (r'/npps/([a-zA-Z_*.])', r'/mpps/\1'),
        (r'/nppcore/([a-zA-Z_*.])', r'/mppcore/\1'),
        # Handle directory-only paths
        (r'/nppi/', '/mppi/'),
        (r'/npps/', '/mpps/'),
        (r'/nppcore/', '/mppcore/'),

        # CMake function names (npp_xxx -> mpp_xxx)
        (r'\bnpp_set_common_standards\b', 'mpp_set_common_standards'),
        (r'\bnpp_set_common_compile_options\b', 'mpp_set_common_compile_options'),
        (r'\bnpp_add_common_includes\b', 'mpp_add_common_includes'),
        (r'\bnpp_link_cuda\b', 'mpp_link_cuda'),
        (r'\bnpp_setup_target\b', 'mpp_setup_target'),
        (r'\bnpp_setup_cuda_architectures\b', 'mpp_setup_cuda_architectures'),
        (r'\bnpp_check_cuda_environment\b', 'mpp_check_cuda_environment'),
        (r'\bnpp_configure_cuda\b', 'mpp_configure_cuda'),
        (r'\bnpp_find_cpp_sources\b', 'mpp_find_cpp_sources'),
        (r'\bnpp_find_cuda_sources\b', 'mpp_find_cuda_sources'),
        (r'\bnpp_find_all_sources\b', 'mpp_find_all_sources'),
        (r'\bnpp_find_test_sources\b', 'mpp_find_test_sources'),
        (r'\bnpp_install_headers\b', 'mpp_install_headers'),
        (r'\bnpp_install_library\b', 'mpp_install_library'),
        (r'\bnpp_install_cmake_config\b', 'mpp_install_cmake_config'),
        (r'\bnpp_install_docs_examples\b', 'mpp_install_docs_examples'),
        (r'\bnpp_install_samples\b', 'mpp_install_samples'),
        (r'\bnpp_configure_install\b', 'mpp_configure_install'),
        (r'\bnpp_setup_gtest\b', 'mpp_setup_gtest'),
        (r'\bnpp_create_test_target\b', 'mpp_create_test_target'),
        (r'\bnpp_register_test\b', 'mpp_register_test'),
        (r'\bnpp_create_test_suite\b', 'mpp_create_test_suite'),

        # Constants and macros (must be before general patterns)
        (r'\bNPP_CUDA_KERNEL_EXECUTION_ERROR\b', 'MPP_CUDA_KERNEL_EXECUTION_ERROR'),
        (r'\bNPP_SUCCESS\b', 'MPP_SUCCESS'),
        (r'\bNPP_NO_ERROR\b', 'MPP_NO_ERROR'),
        (r'\bNPP_ERROR\b', 'MPP_ERROR'),
        (r'\bNPP_NO_MEMORY_ERROR\b', 'MPP_NO_MEMORY_ERROR'),
        (r'\bNPP_INVALID_HOST_POINTER_ERROR\b', 'MPP_INVALID_HOST_POINTER_ERROR'),
        (r'\bNPP_INVALID_DEVICE_POINTER_ERROR\b', 'MPP_INVALID_DEVICE_POINTER_ERROR'),
        (r'\bNPP_LUT_PALETTE_BITSIZE_ERROR\b', 'MPP_LUT_PALETTE_BITSIZE_ERROR'),
        (r'\bNPP_ZC_MODE_NOT_SUPPORTED_ERROR\b', 'MPP_ZC_MODE_NOT_SUPPORTED_ERROR'),
        (r'\bNPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY\b', 'MPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY'),
        (r'\bNPP_TEXTURE_BIND_ERROR\b', 'MPP_TEXTURE_BIND_ERROR'),
        (r'\bNPP_WRONG_INTERSECTION_ROI_ERROR\b', 'MPP_WRONG_INTERSECTION_ROI_ERROR'),
        (r'\bNPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR\b', 'MPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR'),
        (r'\bNPP_MEMFREE_ERROR\b', 'MPP_MEMFREE_ERROR'),
        (r'\bNPP_MEMSET_ERROR\b', 'MPP_MEMSET_ERROR'),
        (r'\bNPP_MEMCPY_ERROR\b', 'MPP_MEMCPY_ERROR'),
        (r'\bNPP_ALIGNMENT_ERROR\b', 'MPP_ALIGNMENT_ERROR'),
        (r'\bNPP_CUDA_KERNEL_EXECUTION_ERROR\b', 'MPP_CUDA_KERNEL_EXECUTION_ERROR'),
        (r'\bNPP_ROUND_MODE_NOT_SUPPORTED_ERROR\b', 'MPP_ROUND_MODE_NOT_SUPPORTED_ERROR'),
        (r'\bNPP_QUALITY_INDEX_ERROR\b', 'MPP_QUALITY_INDEX_ERROR'),
        (r'\bNPP_RESIZE_NO_OPERATION_ERROR\b', 'MPP_RESIZE_NO_OPERATION_ERROR'),
        (r'\bNPP_OVERFLOW_ERROR\b', 'MPP_OVERFLOW_ERROR'),
        (r'\bNPP_NOT_EVEN_STEP_ERROR\b', 'MPP_NOT_EVEN_STEP_ERROR'),
        (r'\bNPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR\b', 'MPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR'),
        (r'\bNPP_LUT_NUMBER_OF_LEVELS_ERROR\b', 'MPP_LUT_NUMBER_OF_LEVELS_ERROR'),
        (r'\bNPP_CORRUPTED_DATA_ERROR\b', 'MPP_CORRUPTED_DATA_ERROR'),
        (r'\bNPP_CHANNEL_ORDER_ERROR\b', 'MPP_CHANNEL_ORDER_ERROR'),
        (r'\bNPP_ZERO_MASK_VALUE_ERROR\b', 'MPP_ZERO_MASK_VALUE_ERROR'),
        (r'\bNPP_QUADRANGLE_ERROR\b', 'MPP_QUADRANGLE_ERROR'),
        (r'\bNPP_RECTANGLE_ERROR\b', 'MPP_RECTANGLE_ERROR'),
        (r'\bNPP_COEFFICIENT_ERROR\b', 'MPP_COEFFICIENT_ERROR'),
        (r'\bNPP_NUMBER_OF_CHANNELS_ERROR\b', 'MPP_NUMBER_OF_CHANNELS_ERROR'),
        (r'\bNPP_COI_ERROR\b', 'MPP_COI_ERROR'),
        (r'\bNPP_DIVISOR_ERROR\b', 'MPP_DIVISOR_ERROR'),
        (r'\bNPP_CHANNEL_ERROR\b', 'MPP_CHANNEL_ERROR'),
        (r'\bNPP_STRIDE_ERROR\b', 'MPP_STRIDE_ERROR'),
        (r'\bNPP_ANCHOR_ERROR\b', 'MPP_ANCHOR_ERROR'),
        (r'\bNPP_MASK_SIZE_ERROR\b', 'MPP_MASK_SIZE_ERROR'),
        (r'\bNPP_RESIZE_FACTOR_ERROR\b', 'MPP_RESIZE_FACTOR_ERROR'),
        (r'\bNPP_INTERPOLATION_ERROR\b', 'MPP_INTERPOLATION_ERROR'),
        (r'\bNPP_MIRROR_FLIP_ERROR\b', 'MPP_MIRROR_FLIP_ERROR'),
        (r'\bNPP_MOMENT_00_ZERO_ERROR\b', 'MPP_MOMENT_00_ZERO_ERROR'),
        (r'\bNPP_THRESHOLD_NEGATIVE_LEVEL_ERROR\b', 'MPP_THRESHOLD_NEGATIVE_LEVEL_ERROR'),
        (r'\bNPP_THRESHOLD_ERROR\b', 'MPP_THRESHOLD_ERROR'),
        (r'\bNPP_CONTEXT_MATCH_ERROR\b', 'MPP_CONTEXT_MATCH_ERROR'),
        (r'\bNPP_FFT_FLAG_ERROR\b', 'MPP_FFT_FLAG_ERROR'),
        (r'\bNPP_FFT_ORDER_ERROR\b', 'MPP_FFT_ORDER_ERROR'),
        (r'\bNPP_STEP_ERROR\b', 'MPP_STEP_ERROR'),
        (r'\bNPP_SCALE_RANGE_ERROR\b', 'MPP_SCALE_RANGE_ERROR'),
        (r'\bNPP_DATA_TYPE_ERROR\b', 'MPP_DATA_TYPE_ERROR'),
        (r'\bNPP_OUT_OFF_RANGE_ERROR\b', 'MPP_OUT_OFF_RANGE_ERROR'),
        (r'\bNPP_DIVIDE_BY_ZERO_ERROR\b', 'MPP_DIVIDE_BY_ZERO_ERROR'),
        (r'\bNPP_MEMORY_ALLOCATION_ERR\b', 'MPP_MEMORY_ALLOCATION_ERR'),
        (r'\bNPP_NULL_POINTER_ERROR\b', 'MPP_NULL_POINTER_ERROR'),
        (r'\bNPP_RANGE_ERROR\b', 'MPP_RANGE_ERROR'),
        (r'\bNPP_SIZE_ERROR\b', 'MPP_SIZE_ERROR'),
        (r'\bNPP_BAD_ARGUMENT_ERROR\b', 'MPP_BAD_ARGUMENT_ERROR'),
        (r'\bNPP_NO_OPERATION_WARNING\b', 'MPP_NO_OPERATION_WARNING'),
        (r'\bNPP_DIVIDE_BY_ZERO_WARNING\b', 'MPP_DIVIDE_BY_ZERO_WARNING'),
        (r'\bNPP_AFFINE_QUAD_INCORRECT_WARNING\b', 'MPP_AFFINE_QUAD_INCORRECT_WARNING'),
        (r'\bNPP_WRONG_INTERSECTION_ROI_WARNING\b', 'MPP_WRONG_INTERSECTION_ROI_WARNING'),
        (r'\bNPP_WRONG_INTERSECTION_QUAD_WARNING\b', 'MPP_WRONG_INTERSECTION_QUAD_WARNING'),
        (r'\bNPP_DOUBLE_SIZE_WARNING\b', 'MPP_DOUBLE_SIZE_WARNING'),
        (r'\bNPP_MISALIGNED_DST_ROI_WARNING\b', 'MPP_MISALIGNED_DST_ROI_WARNING'),

        # Interpolation modes
        (r'\bNPPI_INTER_NN\b', 'MPPI_INTER_NN'),
        (r'\bNPPI_INTER_LINEAR\b', 'MPPI_INTER_LINEAR'),
        (r'\bNPPI_INTER_CUBIC\b', 'MPPI_INTER_CUBIC'),
        (r'\bNPPI_INTER_CUBIC2P_BSPLINE\b', 'MPPI_INTER_CUBIC2P_BSPLINE'),
        (r'\bNPPI_INTER_CUBIC2P_CATMULLROM\b', 'MPPI_INTER_CUBIC2P_CATMULLROM'),
        (r'\bNPPI_INTER_CUBIC2P_B05C03\b', 'MPPI_INTER_CUBIC2P_B05C03'),
        (r'\bNPPI_INTER_SUPER\b', 'MPPI_INTER_SUPER'),
        (r'\bNPPI_INTER_LANCZOS\b', 'MPPI_INTER_LANCZOS'),
        (r'\bNPPI_INTER_LANCZOS3_ADVANCED\b', 'MPPI_INTER_LANCZOS3_ADVANCED'),
        (r'\bNPPI_SMOOTH_EDGE\b', 'MPPI_SMOOTH_EDGE'),

        # Comparison operators
        (r'\bNPP_CMP_LESS\b', 'MPP_CMP_LESS'),
        (r'\bNPP_CMP_LESS_EQ\b', 'MPP_CMP_LESS_EQ'),
        (r'\bNPP_CMP_EQ\b', 'MPP_CMP_EQ'),
        (r'\bNPP_CMP_GREATER_EQ\b', 'MPP_CMP_GREATER_EQ'),
        (r'\bNPP_CMP_GREATER\b', 'MPP_CMP_GREATER'),

        # Rounding modes
        (r'\bNPP_RND_ZERO\b', 'MPP_RND_ZERO'),
        (r'\bNPP_RND_NEAR\b', 'MPP_RND_NEAR'),
        (r'\bNPP_RND_FINANCIAL\b', 'MPP_RND_FINANCIAL'),
        (r'\bNPP_RND_FLOOR\b', 'MPP_RND_FLOOR'),

        # Border types
        (r'\bNPP_BORDER_REPLICATE\b', 'MPP_BORDER_REPLICATE'),
        (r'\bNPP_BORDER_WRAP\b', 'MPP_BORDER_WRAP'),
        (r'\bNPP_BORDER_REFLECT\b', 'MPP_BORDER_REFLECT'),
        (r'\bNPP_BORDER_MIRROR\b', 'MPP_BORDER_MIRROR'),
        (r'\bNPP_BORDER_CONST\b', 'MPP_BORDER_CONST'),

        # Axis definitions
        (r'\bNPP_HORIZONTAL_AXIS\b', 'MPP_HORIZONTAL_AXIS'),
        (r'\bNPP_VERTICAL_AXIS\b', 'MPP_VERTICAL_AXIS'),
        (r'\bNPP_BOTH_AXIS\b', 'MPP_BOTH_AXIS'),

        # Type definitions - basic types
        (r'\bNpp8u\b', 'Mpp8u'),
        (r'\bNpp8s\b', 'Mpp8s'),
        (r'\bNpp16u\b', 'Mpp16u'),
        (r'\bNpp16s\b', 'Mpp16s'),
        (r'\bNpp32u\b', 'Mpp32u'),
        (r'\bNpp32s\b', 'Mpp32s'),
        (r'\bNpp64u\b', 'Mpp64u'),
        (r'\bNpp64s\b', 'Mpp64s'),
        (r'\bNpp32f\b', 'Mpp32f'),
        (r'\bNpp64f\b', 'Mpp64f'),
        (r'\bNpp16f\b', 'Mpp16f'),

        # Complex types
        (r'\bNpp8uc\b', 'Mpp8uc'),
        (r'\bNpp8sc\b', 'Mpp8sc'),
        (r'\bNpp16uc\b', 'Mpp16uc'),
        (r'\bNpp16sc\b', 'Mpp16sc'),
        (r'\bNpp32uc\b', 'Mpp32uc'),
        (r'\bNpp32sc\b', 'Mpp32sc'),
        (r'\bNpp64uc\b', 'Mpp64uc'),
        (r'\bNpp64sc\b', 'Mpp64sc'),
        (r'\bNpp32fc\b', 'Mpp32fc'),
        (r'\bNpp64fc\b', 'Mpp64fc'),

        # Struct types
        (r'\bNppiSize\b', 'MppiSize'),
        (r'\bNppiPoint\b', 'MppiPoint'),
        (r'\bNppiRect\b', 'MppiRect'),
        (r'\bNppStatus\b', 'MppStatus'),
        (r'\bNppStreamContext\b', 'MppStreamContext'),
        (r'\bNppCmpOp\b', 'MppCmpOp'),
        (r'\bNppRoundMode\b', 'MppRoundMode'),
        (r'\bNppiBorderType\b', 'MppiBorderType'),
        (r'\bNppiAxis\b', 'MppiAxis'),
        (r'\bNppiInterpolationMode\b', 'MppiInterpolationMode'),
        (r'\bNppiMaskSize\b', 'MppiMaskSize'),
        (r'\bNppiDifferentialKernel\b', 'MppiDifferentialKernel'),
        (r'\bNppiImageDescriptor\b', 'MppiImageDescriptor'),
        (r'\bNppiBufferDescriptor\b', 'MppiBufferDescriptor'),
        (r'\bNppiConnectedRegion\b', 'MppiConnectedRegion'),
        (r'\bNppiContour\b', 'MppiContour'),
        (r'\bNppiCompressedMarkerLabelsInfo\b', 'MppiCompressedMarkerLabelsInfo'),
        (r'\bNppiNorm\b', 'MppiNorm'),
        (r'\bNppiHuffmanTableType\b', 'MppiHuffmanTableType'),
        (r'\bNppiDecodeHuffmanSpec\b', 'MppiDecodeHuffmanSpec'),
        (r'\bNppiEncodeHuffmanSpec\b', 'MppiEncodeHuffmanSpec'),
        (r'\bNppiJpegFrameDescr\b', 'MppiJpegFrameDescr'),
        (r'\bNppiJpegScanDescr\b', 'MppiJpegScanDescr'),
        (r'\bNppiJpegDecodeJob\b', 'MppiJpegDecodeJob'),
        (r'\bNppiJpegDecodeJobMemory\b', 'MppiJpegDecodeJobMemory'),
        (r'\bNppiWarpTransformMatrix\b', 'MppiWarpTransformMatrix'),
        (r'\bNppiMirrorMode\b', 'MppiMirrorMode'),
        (r'\bNppiColorTwistBatchCXR\b', 'MppiColorTwistBatchCXR'),
        (r'\bNppsZCType\b', 'MppsZCType'),
        (r'\bNppHintAlgorithm\b', 'MppHintAlgorithm'),
        (r'\bNppPoint32f\b', 'MppPoint32f'),
        (r'\bNppPoint64f\b', 'MppPoint64f'),
        # Additional struct types
        (r'\bNppLibraryVersion\b', 'MppLibraryVersion'),
        (r'\bNppiContourPixelDirectionInfo\b', 'MppiContourPixelDirectionInfo'),
        (r'\bNppiContourPixelGeometryInfo\b', 'MppiContourPixelGeometryInfo'),
        (r'\bNppiContourBlockSegment\b', 'MppiContourBlockSegment'),
        (r'\bNppiContourTotalsInfo\b', 'MppiContourTotalsInfo'),
        (r'\bNppiWatershedSegmentBoundaryType\b', 'MppiWatershedSegmentBoundaryType'),
        (r'\bNppiResizeBatchCXR\b', 'MppiResizeBatchCXR'),
        (r'\bNppiResizeBatchROI_Advanced\b', 'MppiResizeBatchROI_Advanced'),
        (r'\bNppiMirrorBatchCXR\b', 'MppiMirrorBatchCXR'),
        (r'\bNppiWarpAffineBatchCXR\b', 'MppiWarpAffineBatchCXR'),
        (r'\bNppiWarpPerspectiveBatchCXR\b', 'MppiWarpPerspectiveBatchCXR'),
        (r'\bNppiHOGConfig\b', 'MppiHOGConfig'),
        (r'\bNppiHaarClassifier_32f\b', 'MppiHaarClassifier_32f'),
        (r'\bNppiHaarBuffer\b', 'MppiHaarBuffer'),
        (r'\bNppiAlphaOp\b', 'MppiAlphaOp'),
        (r'\bNppiBayerGridPosition\b', 'MppiBayerGridPosition'),
        (r'\bNppiPoint32f\b', 'MppiPoint32f'),
        (r'\bNppiPoint64f\b', 'MppiPoint64f'),
        (r'\bNppPointPolar\b', 'MppPointPolar'),
        (r'\bNppDataType\b', 'MppDataType'),
        (r'\bNppiChannels\b', 'MppiChannels'),

        # Library names in cmake/linker
        (r'\bnppial\b', 'mppial'),
        (r'\bnppicc\b', 'mppicc'),
        (r'\bnppidei\b', 'mppidei'),
        (r'\bnppif\b', 'mppif'),
        (r'\bnppig\b', 'mppig'),
        (r'\bnppim\b', 'mppim'),
        (r'\bnppist\b', 'mppist'),
        (r'\bnppisu\b', 'mppisu'),
        (r'\bnppitc\b', 'mppitc'),
        (r'\bnpps\b', 'mpps'),
        (r'\bnppc\b', 'mppc'),

        # Function names - nppi prefix (uppercase letter or digit after prefix)
        (r'\bnppi([A-Z0-9])', r'mppi\1'),
        # Function names - npps prefix
        (r'\bnpps([A-Z0-9])', r'mpps\1'),
        # Function names - npp prefix (core functions)
        (r'\bnpp([A-Z])', r'mpp\1'),

        # Namespace and identifier patterns with underscores
        (r'\bnpp_functional_test\b', 'mpp_functional_test'),
        (r'\bnpp_test_([a-z_]+)', r'mpp_test_\1'),
        (r'\bnppi_test\b', 'mppi_test'),
        (r'\bnpps_test\b', 'mpps_test'),

        # Test class names (NPPI*, Nppi*, NPP*, Npp* followed by word characters)
        (r'\bNPPI([A-Z][a-zA-Z0-9]*Test)\b', r'MPPI\1'),
        (r'\bNppi([A-Z][a-zA-Z0-9]*Test)\b', r'Mppi\1'),
        (r'\bNPP([A-Z][a-zA-Z0-9]*Test)\b', r'MPP\1'),
        (r'\bNpp([A-Z][a-zA-Z0-9]*Test)\b', r'Mpp\1'),

        # Test class names with other suffixes (Params, Base, etc.)
        (r'\bNPPI([A-Z][a-zA-Z0-9]*Params)\b', r'MPPI\1'),
        (r'\bNppi([A-Z][a-zA-Z0-9]*Params)\b', r'Mppi\1'),
        (r'\bNPP([A-Z][a-zA-Z0-9]*Base)\b', r'MPP\1'),
        (r'\bNpp([A-Z][a-zA-Z0-9]*Base)\b', r'Mpp\1'),
        (r'\bNppTestBase\b', 'MppTestBase'),
        (r'\bNppImageMemory\b', 'MppImageMemory'),

        # Include directives - with relative path support
        # Headers with any relative path (e.g., ../../npp_internal.h, ../../../common/npp_test_utils.h)
        (r'#include\s*"([^"]*/)npp_([a-z_]+)\.h"', r'#include "\1mpp_\2.h"'),
        (r'#include\s*"([^"]*/)npp([a-z_]*)\.h"', r'#include "\1mpp\2.h"'),
        # Headers without path
        (r'#include\s*"npp_([a-z_]+)\.h"', r'#include "mpp_\1.h"'),
        # Standard includes
        (r'#include\s*[<"]npp\.h[>"]', '#include "mpp.h"'),
        (r'#include\s*[<"]nppdefs\.h[>"]', '#include "mppdefs.h"'),
        (r'#include\s*[<"]nppcore\.h[>"]', '#include "mppcore.h"'),
        (r'#include\s*[<"]nppi\.h[>"]', '#include "mppi.h"'),
        (r'#include\s*[<"]npps\.h[>"]', '#include "mpps.h"'),
        (r'#include\s*[<"]nppi_([a-z_]+)\.h[>"]', r'#include "mppi_\1.h"'),
        (r'#include\s*[<"]npps_([a-z_]+)\.h[>"]', r'#include "mpps_\1.h"'),

        # Header guards
        (r'\bNV_NPP_H\b', 'MV_MPP_H'),
        (r'\bNV_NPPDEFS_H\b', 'MV_MPPDEFS_H'),
        (r'\bNV_NPPCORE_H\b', 'MV_MPPCORE_H'),
        (r'\bNV_NPPI_H\b', 'MV_MPPI_H'),
        (r'\bNV_NPPS_H\b', 'MV_MPPS_H'),
        (r'\b_NPP_H_\b', '_MPP_H_'),
        (r'\b_NPPDEFS_H_\b', '_MPPDEFS_H_'),

        # Version macros
        (r'\bNPP_VER_MAJOR\b', 'MPP_VER_MAJOR'),
        (r'\bNPP_VER_MINOR\b', 'MPP_VER_MINOR'),
        (r'\bNPP_VER_BUILD\b', 'MPP_VER_BUILD'),
        (r'\bNPP_VERSION\b', 'MPP_VERSION'),

        # Internal macros
        (r'\bNPP_ALIGN_16\b', 'MPP_ALIGN_16'),
        (r'\bNPP_ALIGN_8\b', 'MPP_ALIGN_8'),
        (r'\bNPP_PRIVATE_API\b', 'MPP_PRIVATE_API'),

        # General NPP -> MPP (at end, for any remaining cases)
        (r'\bNPP_', 'MPP_'),
        (r'\bNPPI_', 'MPPI_'),
        (r'\bNPPS_', 'MPPS_'),
    ]


def get_filename_replacements() -> List[Tuple[str, str]]:
    """Get filename replacement patterns."""
    return [
        ('nppi_', 'mppi_'),
        ('npps_', 'mpps_'),
        ('npp_', 'mpp_'),
        ('nppcore', 'mppcore'),
        ('nppdefs', 'mppdefs'),
        ('nppi.', 'mppi.'),
        ('npps.', 'mpps.'),
        ('npp.', 'mpp.'),
    ]


def get_dirname_replacements() -> List[Tuple[str, str]]:
    """Get directory name replacement patterns (order matters - longer prefixes first)."""
    return [
        ('nppi_', 'mppi_'),  # nppi_arithmetic_operations -> mppi_arithmetic_operations
        ('npps_', 'mpps_'),  # npps_statistics_functions -> mpps_statistics_functions
        ('nppcore', 'mppcore'),
        ('nppi', 'mppi'),    # nppi -> mppi (exact match)
        ('npps', 'mpps'),    # npps -> mpps (exact match)
    ]


class NppToMppConverter:
    def __init__(self, project_root: str, dry_run: bool = True, verbose: bool = False):
        self.project_root = Path(project_root)
        self.dry_run = dry_run
        self.verbose = verbose
        self.text_replacements = get_text_replacements()
        self.filename_replacements = get_filename_replacements()
        self.dirname_replacements = get_dirname_replacements()
        self.changes: List[Dict] = []

    def should_skip_dir(self, dir_path: Path) -> bool:
        """Check if directory should be skipped."""
        for skip in SKIP_DIRS:
            if skip in dir_path.parts:
                return True
        return False

    def should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed for content replacement."""
        return file_path.suffix.lower() in ALL_EXTENSIONS

    def apply_text_replacements(self, content: str) -> Tuple[str, int]:
        """Apply all text replacements to content."""
        count = 0
        for pattern, replacement in self.text_replacements:
            new_content, n = re.subn(pattern, replacement, content)
            count += n
            content = new_content
        return content, count

    def process_file_content(self, file_path: Path) -> None:
        """Process a single file's content."""
        if not self.should_process_file(file_path):
            return

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"  Warning: Could not read {file_path}: {e}")
            return

        new_content, count = self.apply_text_replacements(content)

        if count > 0:
            self.changes.append({
                'type': 'content',
                'path': str(file_path),
                'replacements': count
            })
            if self.verbose:
                print(f"  Content: {file_path} ({count} replacements)")

            if not self.dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

    def get_new_filename(self, filename: str) -> str:
        """Get new filename after NPP->MPP replacement."""
        new_name = filename
        for old, new in self.filename_replacements:
            new_name = new_name.replace(old, new)
        return new_name

    def get_new_dirname(self, dirname: str) -> str:
        """Get new directory name after NPP->MPP replacement."""
        new_name = dirname
        for old, new in self.dirname_replacements:
            if old in new_name:
                new_name = new_name.replace(old, new)
                break  # Apply only first matching rule
        return new_name

    def collect_renames(self, dir_path: Path) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
        """Collect all files and directories that need renaming."""
        file_renames = []
        dir_renames = []

        for root, dirs, files in os.walk(dir_path, topdown=False):
            root_path = Path(root)

            if self.should_skip_dir(root_path):
                continue

            # Collect file renames
            for filename in files:
                new_name = self.get_new_filename(filename)
                if new_name != filename:
                    old_path = root_path / filename
                    new_path = root_path / new_name
                    file_renames.append((old_path, new_path))

            # Collect directory renames
            for dirname in dirs:
                if dirname in SKIP_DIRS:
                    continue
                new_name = self.get_new_dirname(dirname)
                if new_name != dirname:
                    old_path = root_path / dirname
                    new_path = root_path / new_name
                    dir_renames.append((old_path, new_path))

        return file_renames, dir_renames

    def process_cmake_files(self) -> None:
        """Process CMakeLists.txt and cmake files."""
        # Process cmake files within PROCESS_DIRS
        for proc_dir in PROCESS_DIRS:
            dir_path = self.project_root / proc_dir
            if not dir_path.exists():
                continue
            for cmake_file in dir_path.glob('**/CMakeLists.txt'):
                if self.should_skip_dir(cmake_file):
                    continue
                self.process_file_content(cmake_file)

        # Process cmake/*.cmake files
        for cmake_file in self.project_root.glob('cmake/*.cmake'):
            if self.should_skip_dir(cmake_file):
                continue
            self.process_file_content(cmake_file)

    def process_root_files(self) -> None:
        """Process root level files (CMakeLists.txt, build.sh, build_maca.sh)."""
        for filename in ROOT_FILES:
            file_path = self.project_root / filename
            if file_path.exists():
                print(f"  Processing root file: {filename}")
                self.process_file_content(file_path)

    def process_api_directory(self) -> None:
        """Process API directory - convert NPP headers to MPP headers.

        Handles both regular directories and symlinks (e.g., API -> API-12.2).
        When API is a symlink, processes the target directory directly.
        """
        api_dir = self.project_root / API_DIR

        # Handle symlink: get the actual target directory
        if api_dir.is_symlink():
            link_target = os.readlink(api_dir)
            real_api_dir = self.project_root / link_target
            if not real_api_dir.exists():
                print(f"  API symlink target not found: {real_api_dir}")
                return
            print(f"  Processing API directory: {link_target}/ (via {API_DIR} symlink)")
            api_dir = real_api_dir
        elif not api_dir.exists():
            print(f"  API directory not found: {api_dir}")
            return
        else:
            print(f"  Processing API directory: {API_DIR}/")

        # Process all header files in API directory
        for header_file in api_dir.glob('*.h'):
            self.process_file_content(header_file)

        # Collect file renames for API headers
        api_file_renames = []
        for header_file in api_dir.glob('*.h'):
            new_name = self.get_new_filename(header_file.name)
            if new_name != header_file.name:
                api_file_renames.append((header_file, api_dir / new_name))

        # Apply file renames
        for old_path, new_path in api_file_renames:
            if not old_path.exists():
                continue
            self.changes.append({
                'type': 'api_file_rename',
                'old': str(old_path),
                'new': str(new_path)
            })
            if self.verbose:
                print(f"    API Header: {old_path.name} -> {new_path.name}")
            if not self.dry_run:
                if new_path.exists():
                    new_path.unlink()
                old_path.rename(new_path)

    def run(self) -> None:
        """Run the conversion process."""
        print(f"\n{'='*60}")
        print(f"NPP to MPP Conversion {'(Preview Mode)' if self.dry_run else '(Apply Mode)'}")
        print(f"{'='*60}")
        print(f"Project root: {self.project_root}")
        print()

        # Step 1: Process file contents
        print("Step 1: Processing file contents...")
        for proc_dir in PROCESS_DIRS:
            dir_path = self.project_root / proc_dir
            if not dir_path.exists():
                continue
            print(f"  Processing directory: {proc_dir}/")
            for root, dirs, files in os.walk(dir_path):
                root_path = Path(root)
                if self.should_skip_dir(root_path):
                    continue
                # Filter out skip directories
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                for filename in files:
                    self.process_file_content(root_path / filename)

        # Process cmake files and root level files
        self.process_cmake_files()
        self.process_root_files()

        # Step 2: Process API directory
        print("\nStep 2: Processing API headers...")
        self.process_api_directory()

        # Step 3: Collect renames
        print("\nStep 3: Collecting file and directory renames...")
        all_file_renames = []
        all_dir_renames = []

        for proc_dir in PROCESS_DIRS:
            dir_path = self.project_root / proc_dir
            if not dir_path.exists():
                continue
            file_renames, dir_renames = self.collect_renames(dir_path)
            all_file_renames.extend(file_renames)
            all_dir_renames.extend(dir_renames)

        # Step 4: Apply renames
        print("\nStep 4: Renaming files...")
        for old_path, new_path in all_file_renames:
            if not old_path.exists():
                continue  # Skip if source already renamed
            self.changes.append({
                'type': 'file_rename',
                'old': str(old_path),
                'new': str(new_path)
            })
            if self.verbose:
                print(f"  File: {old_path.name} -> {new_path.name}")
            if not self.dry_run:
                if new_path.exists():
                    # Target exists, overwrite it
                    new_path.unlink()
                old_path.rename(new_path)

        print("\nStep 5: Renaming directories...")
        # Sort by depth (deepest first) to avoid issues with nested renames
        all_dir_renames.sort(key=lambda x: len(x[0].parts), reverse=True)
        for old_path, new_path in all_dir_renames:
            if old_path.exists():  # Check if not already renamed
                self.changes.append({
                    'type': 'dir_rename',
                    'old': str(old_path),
                    'new': str(new_path)
                })
                if self.verbose:
                    print(f"  Dir: {old_path.name} -> {new_path.name}")
                if not self.dry_run:
                    if new_path.exists():
                        # Target exists, merge contents
                        for item in old_path.iterdir():
                            dst = new_path / item.name
                            if item.is_dir() and dst.exists():
                                # Recursively merge subdirectories
                                shutil.copytree(item, dst, dirs_exist_ok=True)
                                shutil.rmtree(item)
                            else:
                                shutil.move(str(item), str(dst))
                        # Remove empty source directory
                        if old_path.exists() and not any(old_path.iterdir()):
                            old_path.rmdir()
                    else:
                        old_path.rename(new_path)

        # Summary
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        content_changes = [c for c in self.changes if c['type'] == 'content']
        file_renames = [c for c in self.changes if c['type'] == 'file_rename']
        dir_renames = [c for c in self.changes if c['type'] == 'dir_rename']
        api_renames = [c for c in self.changes if c['type'] == 'api_file_rename']

        print(f"  Files with content changes: {len(content_changes)}")
        print(f"  Total text replacements: {sum(c['replacements'] for c in content_changes)}")
        print(f"  Files to rename: {len(file_renames)}")
        print(f"  Directories to rename: {len(dir_renames)}")
        print(f"  API headers to rename: {len(api_renames)}")

        if self.dry_run:
            print(f"\n{'='*60}")
            print("This was a preview. Use --apply to make changes.")
            print(f"{'='*60}")


def create_backup(project_root: Path) -> str:
    """Create a backup of the project."""
    backup_dir = project_root / BACKUP_DIR
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = backup_dir / timestamp

    print(f"Creating backup at: {backup_path}")
    backup_path.mkdir(parents=True, exist_ok=True)

    # Backup PROCESS_DIRS
    for proc_dir in PROCESS_DIRS:
        src_dir = project_root / proc_dir
        if src_dir.exists():
            dst_dir = backup_path / proc_dir
            if not dst_dir.exists():
                shutil.copytree(src_dir, dst_dir)
                print(f"  Backed up: {proc_dir}/")

    # Backup API directory (handle symlink specially)
    api_dir = project_root / API_DIR
    if api_dir.exists() or api_dir.is_symlink():
        if api_dir.is_symlink():
            # API is a symlink, backup the target directory
            link_target = os.readlink(api_dir)
            real_api_dir = project_root / link_target
            if real_api_dir.exists():
                dst_dir = backup_path / link_target
                if not dst_dir.exists():
                    shutil.copytree(real_api_dir, dst_dir)
                    print(f"  Backed up: {link_target}/ (API symlink target)")
                # Save symlink info
                symlink_info_file = backup_path / 'API_SYMLINK'
                with open(symlink_info_file, 'w') as f:
                    f.write(link_target)
                print(f"  Saved API symlink info: {API_DIR} -> {link_target}")
        else:
            # API is a regular directory
            dst_dir = backup_path / API_DIR
            if not dst_dir.exists():
                shutil.copytree(api_dir, dst_dir)
                print(f"  Backed up: {API_DIR}/")

    # Backup root level files
    for filename in ROOT_FILES:
        src_file = project_root / filename
        if src_file.exists():
            shutil.copy2(src_file, backup_path / filename)
            print(f"  Backed up: {filename}")

    print(f"\nBackup completed: {backup_path}")
    return str(backup_path)


def restore_backup(project_root: Path, backup_name: str = None) -> None:
    """Restore from a backup."""
    backup_dir = project_root / BACKUP_DIR

    if not backup_dir.exists():
        print("No backups found.")
        return

    backups = sorted([d for d in backup_dir.iterdir() if d.is_dir()], reverse=True)

    if not backups:
        print("No backups found.")
        return

    if backup_name:
        backup_path = backup_dir / backup_name
        if not backup_path.exists():
            print(f"Backup '{backup_name}' not found.")
            print("Available backups:")
            for b in backups:
                print(f"  {b.name}")
            return
    else:
        backup_path = backups[0]
        print(f"Restoring from latest backup: {backup_path.name}")

    # Check if API was a symlink
    symlink_info_file = backup_path / 'API_SYMLINK'
    api_symlink_target = None
    if symlink_info_file.exists():
        with open(symlink_info_file, 'r') as f:
            api_symlink_target = f.read().strip()

    # Restore directories
    for item in backup_path.iterdir():
        if item.name == 'API_SYMLINK':
            continue  # Skip symlink info file

        dst = project_root / item.name
        if item.is_dir():
            # Check if this is an API version directory that was symlinked
            if api_symlink_target and item.name == api_symlink_target:
                # Restore the symlink target directory
                if dst.exists() and not dst.is_symlink():
                    shutil.rmtree(dst)
                elif dst.is_symlink():
                    # Don't remove symlink, just restore target
                    pass
                if not dst.exists():
                    shutil.copytree(item, dst)
                else:
                    # Merge/overwrite contents
                    shutil.rmtree(dst)
                    shutil.copytree(item, dst)
                print(f"  Restored: {item.name}/ (API symlink target)")
            else:
                if dst.exists():
                    if dst.is_symlink():
                        os.unlink(dst)
                    else:
                        shutil.rmtree(dst)
                shutil.copytree(item, dst)
                print(f"  Restored: {item.name}/")
        else:
            shutil.copy2(item, dst)
            print(f"  Restored: {item.name}")

    print(f"\nRestore completed from: {backup_path}")


def list_backups(project_root: Path) -> None:
    """List available backups."""
    backup_dir = project_root / BACKUP_DIR

    if not backup_dir.exists():
        print("No backups found.")
        return

    backups = sorted([d for d in backup_dir.iterdir() if d.is_dir()], reverse=True)

    if not backups:
        print("No backups found.")
        return

    print("Available backups:")
    for b in backups:
        print(f"  {b.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert NPP project to MPP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --preview              Preview all changes without applying
  %(prog)s --preview --verbose    Preview with detailed output
  %(prog)s --backup               Create backup before conversion
  %(prog)s --apply                Apply all changes
  %(prog)s --restore              Restore from latest backup
  %(prog)s --list-backups         List available backups
        """
    )

    parser.add_argument('--preview', action='store_true',
                        help='Preview changes without applying (default behavior)')
    parser.add_argument('--apply', action='store_true',
                        help='Apply all changes')
    parser.add_argument('--backup', action='store_true',
                        help='Create backup before conversion')
    parser.add_argument('--restore', nargs='?', const='__latest__', metavar='BACKUP_NAME',
                        help='Restore from backup (latest if no name specified)')
    parser.add_argument('--list-backups', action='store_true',
                        help='List available backups')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--project-root', type=str, default='.',
                        help='Project root directory (default: current directory)')

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()

    # Validate project root
    if not (project_root / 'src').exists():
        print(f"Error: '{project_root}' does not appear to be the project root.")
        print("Please run from the project root directory or use --project-root.")
        sys.exit(1)

    # Handle list-backups
    if args.list_backups:
        list_backups(project_root)
        return

    # Handle restore
    if args.restore:
        backup_name = None if args.restore == '__latest__' else args.restore
        restore_backup(project_root, backup_name)
        return

    # Handle backup
    if args.backup:
        create_backup(project_root)
        if not args.apply:
            return

    # Run conversion
    dry_run = not args.apply
    converter = NppToMppConverter(project_root, dry_run=dry_run, verbose=args.verbose)
    converter.run()


if __name__ == '__main__':
    main()
