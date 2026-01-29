#!/usr/bin/env python3
"""
Generate NPP to MPP wrapper files

This script generates:
1. Wrapper files (*_wrapper.h) - NPP to MPP type/function mappings
2. User include files (user_include/*.h) - Top-level NPP headers

All mappings are based on actual NPP API definitions in API/nppdefs.h
"""

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent
WRAPPER_DIR = PROJECT_DIR / 'wrapper'

# =============================================================================
# Type mappings based on API/nppdefs.h
# =============================================================================

# Basic data types (typedef in nppdefs.h)
BASIC_TYPES = [
    ('Mpp8u', 'Npp8u'),
    ('Mpp8s', 'Npp8s'),
    ('Mpp16u', 'Npp16u'),
    ('Mpp16s', 'Npp16s'),
    ('Mpp32u', 'Npp32u'),
    ('Mpp32s', 'Npp32s'),
    ('Mpp64u', 'Npp64u'),
    ('Mpp64s', 'Npp64s'),
    ('Mpp32f', 'Npp32f'),
    ('Mpp64f', 'Npp64f'),
]

# FP16 types (struct in nppdefs.h)
FP16_TYPES = [
    ('Mpp16f', 'Npp16f'),
    ('Mpp16f_2', 'Npp16f_2'),
]

# Complex number types (struct in nppdefs.h)
COMPLEX_TYPES = [
    ('Mpp8uc', 'Npp8uc'),
    ('Mpp16uc', 'Npp16uc'),
    ('Mpp16sc', 'Npp16sc'),
    ('Mpp32uc', 'Npp32uc'),
    ('Mpp32sc', 'Npp32sc'),
    ('Mpp32fc', 'Npp32fc'),
    ('Mpp64sc', 'Npp64sc'),
    ('Mpp64fc', 'Npp64fc'),
]

# Geometric types (struct in nppdefs.h)
GEOMETRIC_TYPES = [
    ('MppiPoint', 'NppiPoint'),
    ('MppiPoint32f', 'NppiPoint32f'),
    ('MppiPoint64f', 'NppiPoint64f'),
    ('MppPointPolar', 'NppPointPolar'),
    ('MppiSize', 'NppiSize'),
    ('MppiRect', 'NppiRect'),
]

# Enum types
ENUM_TYPES = [
    ('MppStatus', 'NppStatus'),
    ('MppiInterpolationMode', 'NppiInterpolationMode'),
    ('MppiBayerGridPosition', 'NppiBayerGridPosition'),
    ('MppiMaskSize', 'NppiMaskSize'),
    ('MppiDifferentialKernel', 'NppiDifferentialKernel'),
    ('MppDataType', 'NppDataType'),
    ('MppiChannels', 'NppiChannels'),
    ('MppiAxis', 'NppiAxis'),
    ('MppCmpOp', 'NppCmpOp'),
    ('MppRoundMode', 'NppRoundMode'),
    ('MppiBorderType', 'NppiBorderType'),
    ('MppHintAlgorithm', 'NppHintAlgorithm'),
    ('MppiAlphaOp', 'NppiAlphaOp'),
    ('MppsZCType', 'NppsZCType'),
    ('MppiHuffmanTableType', 'NppiHuffmanTableType'),
    ('MppiNorm', 'NppiNorm'),
    ('MppiWatershedSegmentBoundaryType', 'NppiWatershedSegmentBoundaryType'),
]

# Struct types
STRUCT_TYPES = [
    ('MppLibraryVersion', 'NppLibraryVersion'),
    ('MppStreamContext', 'NppStreamContext'),
    ('MppiHOGConfig', 'NppiHOGConfig'),
    ('MppiHaarClassifier_32f', 'NppiHaarClassifier_32f'),
    ('MppiHaarBuffer', 'NppiHaarBuffer'),
    ('MppiConnectedRegion', 'NppiConnectedRegion'),
    ('MppiImageDescriptor', 'NppiImageDescriptor'),
    ('MppiBufferDescriptor', 'NppiBufferDescriptor'),
    ('MppiCompressedMarkerLabelsInfo', 'NppiCompressedMarkerLabelsInfo'),
    ('MppiContourBlockSegment', 'NppiContourBlockSegment'),
    ('MppiContourPixelGeometryInfo', 'NppiContourPixelGeometryInfo'),
    ('MppiContourPixelDirectionInfo', 'NppiContourPixelDirectionInfo'),
    ('MppiContourTotalsInfo', 'NppiContourTotalsInfo'),
    ('MppiResizeBatchCXR', 'NppiResizeBatchCXR'),
    ('MppiResizeBatchROI_Advanced', 'NppiResizeBatchROI_Advanced'),
    ('MppiMirrorBatchCXR', 'NppiMirrorBatchCXR'),
    ('MppiWarpAffineBatchCXR', 'NppiWarpAffineBatchCXR'),
    ('MppiWarpPerspectiveBatchCXR', 'NppiWarpPerspectiveBatchCXR'),
]

# =============================================================================
# Enum value mappings based on API/nppdefs.h
# =============================================================================

# NppStatus error codes (negative values)
STATUS_ERRORS = [
    ('MPP_NOT_SUPPORTED_MODE_ERROR', 'NPP_NOT_SUPPORTED_MODE_ERROR'),
    ('MPP_INVALID_HOST_POINTER_ERROR', 'NPP_INVALID_HOST_POINTER_ERROR'),
    ('MPP_INVALID_DEVICE_POINTER_ERROR', 'NPP_INVALID_DEVICE_POINTER_ERROR'),
    ('MPP_LUT_PALETTE_BITSIZE_ERROR', 'NPP_LUT_PALETTE_BITSIZE_ERROR'),
    ('MPP_ZC_MODE_NOT_SUPPORTED_ERROR', 'NPP_ZC_MODE_NOT_SUPPORTED_ERROR'),
    ('MPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY', 'NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY'),
    ('MPP_TEXTURE_BIND_ERROR', 'NPP_TEXTURE_BIND_ERROR'),
    ('MPP_WRONG_INTERSECTION_ROI_ERROR', 'NPP_WRONG_INTERSECTION_ROI_ERROR'),
    ('MPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR', 'NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR'),
    ('MPP_MEMFREE_ERROR', 'NPP_MEMFREE_ERROR'),
    ('MPP_MEMSET_ERROR', 'NPP_MEMSET_ERROR'),
    ('MPP_MEMCPY_ERROR', 'NPP_MEMCPY_ERROR'),
    ('MPP_ALIGNMENT_ERROR', 'NPP_ALIGNMENT_ERROR'),
    ('MPP_CUDA_KERNEL_EXECUTION_ERROR', 'NPP_CUDA_KERNEL_EXECUTION_ERROR'),
    ('MPP_ROUND_MODE_NOT_SUPPORTED_ERROR', 'NPP_ROUND_MODE_NOT_SUPPORTED_ERROR'),
    ('MPP_QUALITY_INDEX_ERROR', 'NPP_QUALITY_INDEX_ERROR'),
    ('MPP_RESIZE_NO_OPERATION_ERROR', 'NPP_RESIZE_NO_OPERATION_ERROR'),
    ('MPP_OVERFLOW_ERROR', 'NPP_OVERFLOW_ERROR'),
    ('MPP_NOT_EVEN_STEP_ERROR', 'NPP_NOT_EVEN_STEP_ERROR'),
    ('MPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR', 'NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR'),
    ('MPP_LUT_NUMBER_OF_LEVELS_ERROR', 'NPP_LUT_NUMBER_OF_LEVELS_ERROR'),
    ('MPP_CORRUPTED_DATA_ERROR', 'NPP_CORRUPTED_DATA_ERROR'),
    ('MPP_CHANNEL_ORDER_ERROR', 'NPP_CHANNEL_ORDER_ERROR'),
    ('MPP_ZERO_MASK_VALUE_ERROR', 'NPP_ZERO_MASK_VALUE_ERROR'),
    ('MPP_QUADRANGLE_ERROR', 'NPP_QUADRANGLE_ERROR'),
    ('MPP_RECTANGLE_ERROR', 'NPP_RECTANGLE_ERROR'),
    ('MPP_COEFFICIENT_ERROR', 'NPP_COEFFICIENT_ERROR'),
    ('MPP_NUMBER_OF_CHANNELS_ERROR', 'NPP_NUMBER_OF_CHANNELS_ERROR'),
    ('MPP_COI_ERROR', 'NPP_COI_ERROR'),
    ('MPP_DIVISOR_ERROR', 'NPP_DIVISOR_ERROR'),
    ('MPP_CHANNEL_ERROR', 'NPP_CHANNEL_ERROR'),
    ('MPP_STRIDE_ERROR', 'NPP_STRIDE_ERROR'),
    ('MPP_ANCHOR_ERROR', 'NPP_ANCHOR_ERROR'),
    ('MPP_MASK_SIZE_ERROR', 'NPP_MASK_SIZE_ERROR'),
    ('MPP_RESIZE_FACTOR_ERROR', 'NPP_RESIZE_FACTOR_ERROR'),
    ('MPP_INTERPOLATION_ERROR', 'NPP_INTERPOLATION_ERROR'),
    ('MPP_MIRROR_FLIP_ERROR', 'NPP_MIRROR_FLIP_ERROR'),
    ('MPP_MOMENT_00_ZERO_ERROR', 'NPP_MOMENT_00_ZERO_ERROR'),
    ('MPP_THRESHOLD_NEGATIVE_LEVEL_ERROR', 'NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR'),
    ('MPP_THRESHOLD_ERROR', 'NPP_THRESHOLD_ERROR'),
    ('MPP_CONTEXT_MATCH_ERROR', 'NPP_CONTEXT_MATCH_ERROR'),
    ('MPP_FFT_FLAG_ERROR', 'NPP_FFT_FLAG_ERROR'),
    ('MPP_FFT_ORDER_ERROR', 'NPP_FFT_ORDER_ERROR'),
    ('MPP_STEP_ERROR', 'NPP_STEP_ERROR'),
    ('MPP_SCALE_RANGE_ERROR', 'NPP_SCALE_RANGE_ERROR'),
    ('MPP_DATA_TYPE_ERROR', 'NPP_DATA_TYPE_ERROR'),
    ('MPP_OUT_OFF_RANGE_ERROR', 'NPP_OUT_OFF_RANGE_ERROR'),
    ('MPP_DIVIDE_BY_ZERO_ERROR', 'NPP_DIVIDE_BY_ZERO_ERROR'),
    ('MPP_MEMORY_ALLOCATION_ERR', 'NPP_MEMORY_ALLOCATION_ERR'),
    ('MPP_NULL_POINTER_ERROR', 'NPP_NULL_POINTER_ERROR'),
    ('MPP_RANGE_ERROR', 'NPP_RANGE_ERROR'),
    ('MPP_SIZE_ERROR', 'NPP_SIZE_ERROR'),
    ('MPP_BAD_ARGUMENT_ERROR', 'NPP_BAD_ARGUMENT_ERROR'),
    ('MPP_NO_MEMORY_ERROR', 'NPP_NO_MEMORY_ERROR'),
    ('MPP_NOT_IMPLEMENTED_ERROR', 'NPP_NOT_IMPLEMENTED_ERROR'),
    ('MPP_ERROR', 'NPP_ERROR'),
    ('MPP_ERROR_RESERVED', 'NPP_ERROR_RESERVED'),
    ('MPP_NO_ERROR', 'NPP_NO_ERROR'),
    ('MPP_SUCCESS', 'NPP_SUCCESS'),
]

# NppStatus warning codes (positive values)
STATUS_WARNINGS = [
    ('MPP_NO_OPERATION_WARNING', 'NPP_NO_OPERATION_WARNING'),
    ('MPP_DIVIDE_BY_ZERO_WARNING', 'NPP_DIVIDE_BY_ZERO_WARNING'),
    ('MPP_AFFINE_QUAD_INCORRECT_WARNING', 'NPP_AFFINE_QUAD_INCORRECT_WARNING'),
    ('MPP_WRONG_INTERSECTION_ROI_WARNING', 'NPP_WRONG_INTERSECTION_ROI_WARNING'),
    ('MPP_WRONG_INTERSECTION_QUAD_WARNING', 'NPP_WRONG_INTERSECTION_QUAD_WARNING'),
    ('MPP_DOUBLE_SIZE_WARNING', 'NPP_DOUBLE_SIZE_WARNING'),
    ('MPP_MISALIGNED_DST_ROI_WARNING', 'NPP_MISALIGNED_DST_ROI_WARNING'),
]

# Legacy/compat NppStatus aliases not present in API/nppdefs.h
STATUS_ALIASES = [
    ('MPP_BAD_ARGUMENT_ERROR', 'NPP_BAD_ARG_ERROR'),
    ('MPP_COEFFICIENT_ERROR', 'NPP_COEFF_ERROR'),
    ('MPP_RECTANGLE_ERROR', 'NPP_RECT_ERROR'),
    ('MPP_QUADRANGLE_ERROR', 'NPP_QUAD_ERROR'),
    ('MPP_MEMORY_ALLOCATION_ERR', 'NPP_MEM_ALLOC_ERR'),
    ('MPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR', 'NPP_HISTO_NUMBER_OF_LEVELS_ERROR'),
    ('MPP_BAD_ARGUMENT_ERROR', 'NPP_INVALID_INPUT'),
    ('MPP_NULL_POINTER_ERROR', 'NPP_POINTER_ERROR'),
    ('MPP_NO_OPERATION_WARNING', 'NPP_WARNING'),
    ('MPP_MISALIGNED_DST_ROI_WARNING', 'NPP_ODD_ROI_WARNING'),
    ('MPP_MEMFREE_ERROR', 'NPP_MEMFREE_ERR'),
    ('MPP_MEMSET_ERROR', 'NPP_MEMSET_ERR'),
    ('MPP_MEMCPY_ERROR', 'NPP_MEMCPY_ERR'),
    ('MPP_MIRROR_FLIP_ERROR', 'NPP_MIRROR_FLIP_ERR'),
]

# NppiInterpolationMode values
INTERPOLATION_MODES = [
    ('MPPI_INTER_UNDEFINED', 'NPPI_INTER_UNDEFINED'),
    ('MPPI_INTER_NN', 'NPPI_INTER_NN'),
    ('MPPI_INTER_LINEAR', 'NPPI_INTER_LINEAR'),
    ('MPPI_INTER_CUBIC', 'NPPI_INTER_CUBIC'),
    ('MPPI_INTER_CUBIC2P_BSPLINE', 'NPPI_INTER_CUBIC2P_BSPLINE'),
    ('MPPI_INTER_CUBIC2P_CATMULLROM', 'NPPI_INTER_CUBIC2P_CATMULLROM'),
    ('MPPI_INTER_CUBIC2P_B05C03', 'NPPI_INTER_CUBIC2P_B05C03'),
    ('MPPI_INTER_SUPER', 'NPPI_INTER_SUPER'),
    ('MPPI_INTER_LANCZOS', 'NPPI_INTER_LANCZOS'),
    ('MPPI_INTER_LANCZOS3_ADVANCED', 'NPPI_INTER_LANCZOS3_ADVANCED'),
    ('MPPI_SMOOTH_EDGE', 'NPPI_SMOOTH_EDGE'),
]

# NppiBayerGridPosition values
BAYER_PATTERNS = [
    ('MPPI_BAYER_BGGR', 'NPPI_BAYER_BGGR'),
    ('MPPI_BAYER_RGGB', 'NPPI_BAYER_RGGB'),
    ('MPPI_BAYER_GBRG', 'NPPI_BAYER_GBRG'),
    ('MPPI_BAYER_GRBG', 'NPPI_BAYER_GRBG'),
]

# NppiMaskSize values
MASK_SIZES = [
    ('MPP_MASK_SIZE_1_X_3', 'NPP_MASK_SIZE_1_X_3'),
    ('MPP_MASK_SIZE_1_X_5', 'NPP_MASK_SIZE_1_X_5'),
    ('MPP_MASK_SIZE_3_X_1', 'NPP_MASK_SIZE_3_X_1'),
    ('MPP_MASK_SIZE_5_X_1', 'NPP_MASK_SIZE_5_X_1'),
    ('MPP_MASK_SIZE_3_X_3', 'NPP_MASK_SIZE_3_X_3'),
    ('MPP_MASK_SIZE_5_X_5', 'NPP_MASK_SIZE_5_X_5'),
    ('MPP_MASK_SIZE_7_X_7', 'NPP_MASK_SIZE_7_X_7'),
    ('MPP_MASK_SIZE_9_X_9', 'NPP_MASK_SIZE_9_X_9'),
    ('MPP_MASK_SIZE_11_X_11', 'NPP_MASK_SIZE_11_X_11'),
    ('MPP_MASK_SIZE_13_X_13', 'NPP_MASK_SIZE_13_X_13'),
    ('MPP_MASK_SIZE_15_X_15', 'NPP_MASK_SIZE_15_X_15'),
]

# NppiDifferentialKernel values
DIFF_KERNELS = [
    ('MPP_FILTER_SOBEL', 'NPP_FILTER_SOBEL'),
    ('MPP_FILTER_SCHARR', 'NPP_FILTER_SCHARR'),
]

# NppDataType values
DATA_TYPES = [
    ('MPP_8U', 'NPP_8U'),
    ('MPP_8S', 'NPP_8S'),
    ('MPP_16U', 'NPP_16U'),
    ('MPP_16S', 'NPP_16S'),
    ('MPP_32U', 'NPP_32U'),
    ('MPP_32S', 'NPP_32S'),
    ('MPP_64U', 'NPP_64U'),
    ('MPP_64S', 'NPP_64S'),
    ('MPP_16F', 'NPP_16F'),
    ('MPP_32F', 'NPP_32F'),
    ('MPP_64F', 'NPP_64F'),
]

# NppiChannels values
CHANNELS = [
    ('MPP_CH_1', 'NPP_CH_1'),
    ('MPP_CH_2', 'NPP_CH_2'),
    ('MPP_CH_3', 'NPP_CH_3'),
    ('MPP_CH_4', 'NPP_CH_4'),
    ('MPP_CH_A4', 'NPP_CH_A4'),
    ('MPP_CH_P2', 'NPP_CH_P2'),
    ('MPP_CH_P3', 'NPP_CH_P3'),
    ('MPP_CH_P4', 'NPP_CH_P4'),
]

# NppiAxis values
AXIS_TYPES = [
    ('MPP_HORIZONTAL_AXIS', 'NPP_HORIZONTAL_AXIS'),
    ('MPP_VERTICAL_AXIS', 'NPP_VERTICAL_AXIS'),
    ('MPP_BOTH_AXIS', 'NPP_BOTH_AXIS'),
]

# NppCmpOp values
COMPARISON_OPS = [
    ('MPP_CMP_LESS', 'NPP_CMP_LESS'),
    ('MPP_CMP_LESS_EQ', 'NPP_CMP_LESS_EQ'),
    ('MPP_CMP_EQ', 'NPP_CMP_EQ'),
    ('MPP_CMP_GREATER_EQ', 'NPP_CMP_GREATER_EQ'),
    ('MPP_CMP_GREATER', 'NPP_CMP_GREATER'),
]

# NppRoundMode values
ROUNDING_MODES = [
    ('MPP_RND_NEAR', 'NPP_RND_NEAR'),
    ('MPP_ROUND_NEAREST_TIES_TO_EVEN', 'NPP_ROUND_NEAREST_TIES_TO_EVEN'),
    ('MPP_RND_FINANCIAL', 'NPP_RND_FINANCIAL'),
    ('MPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO', 'NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO'),
    ('MPP_RND_ZERO', 'NPP_RND_ZERO'),
    ('MPP_ROUND_TOWARD_ZERO', 'NPP_ROUND_TOWARD_ZERO'),
]

# NppiBorderType values
BORDER_TYPES = [
    ('MPP_BORDER_UNDEFINED', 'NPP_BORDER_UNDEFINED'),
    ('MPP_BORDER_NONE', 'NPP_BORDER_NONE'),
    ('MPP_BORDER_CONSTANT', 'NPP_BORDER_CONSTANT'),
    ('MPP_BORDER_REPLICATE', 'NPP_BORDER_REPLICATE'),
    ('MPP_BORDER_WRAP', 'NPP_BORDER_WRAP'),
    ('MPP_BORDER_MIRROR', 'NPP_BORDER_MIRROR'),
]

# NppHintAlgorithm values
HINT_ALGORITHMS = [
    ('MPP_ALG_HINT_NONE', 'NPP_ALG_HINT_NONE'),
    ('MPP_ALG_HINT_FAST', 'NPP_ALG_HINT_FAST'),
    ('MPP_ALG_HINT_ACCURATE', 'NPP_ALG_HINT_ACCURATE'),
]

# NppiAlphaOp values
ALPHA_OPS = [
    ('MPPI_OP_ALPHA_OVER', 'NPPI_OP_ALPHA_OVER'),
    ('MPPI_OP_ALPHA_IN', 'NPPI_OP_ALPHA_IN'),
    ('MPPI_OP_ALPHA_OUT', 'NPPI_OP_ALPHA_OUT'),
    ('MPPI_OP_ALPHA_ATOP', 'NPPI_OP_ALPHA_ATOP'),
    ('MPPI_OP_ALPHA_XOR', 'NPPI_OP_ALPHA_XOR'),
    ('MPPI_OP_ALPHA_PLUS', 'NPPI_OP_ALPHA_PLUS'),
    ('MPPI_OP_ALPHA_OVER_PREMUL', 'NPPI_OP_ALPHA_OVER_PREMUL'),
    ('MPPI_OP_ALPHA_IN_PREMUL', 'NPPI_OP_ALPHA_IN_PREMUL'),
    ('MPPI_OP_ALPHA_OUT_PREMUL', 'NPPI_OP_ALPHA_OUT_PREMUL'),
    ('MPPI_OP_ALPHA_ATOP_PREMUL', 'NPPI_OP_ALPHA_ATOP_PREMUL'),
    ('MPPI_OP_ALPHA_XOR_PREMUL', 'NPPI_OP_ALPHA_XOR_PREMUL'),
    ('MPPI_OP_ALPHA_PLUS_PREMUL', 'NPPI_OP_ALPHA_PLUS_PREMUL'),
    ('MPPI_OP_ALPHA_PREMUL', 'NPPI_OP_ALPHA_PREMUL'),
]

# NppsZCType values
ZC_TYPES = [
    ('mppZCR', 'nppZCR'),
    ('mppZCXor', 'nppZCXor'),
    ('mppZCC', 'nppZCC'),
]

# NppiHuffmanTableType values
HUFFMAN_TABLE_TYPES = [
    ('mppiDCTable', 'nppiDCTable'),
    ('mppiACTable', 'nppiACTable'),
]

# NppiNorm values
NORM_TYPES = [
    ('mppiNormInf', 'nppiNormInf'),
    ('mppiNormL1', 'nppiNormL1'),
    ('mppiNormL2', 'nppiNormL2'),
]

# NppiWatershedSegmentBoundaryType values
WATERSHED_TYPES = [
    ('MPP_WATERSHED_SEGMENT_BOUNDARIES_NONE', 'NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE'),
    ('MPP_WATERSHED_SEGMENT_BOUNDARIES_BLACK', 'NPP_WATERSHED_SEGMENT_BOUNDARIES_BLACK'),
    ('MPP_WATERSHED_SEGMENT_BOUNDARIES_WHITE', 'NPP_WATERSHED_SEGMENT_BOUNDARIES_WHITE'),
    ('MPP_WATERSHED_SEGMENT_BOUNDARIES_CONTRAST', 'NPP_WATERSHED_SEGMENT_BOUNDARIES_CONTRAST'),
    ('MPP_WATERSHED_SEGMENT_BOUNDARIES_ONLY', 'NPP_WATERSHED_SEGMENT_BOUNDARIES_ONLY'),
]

# Numeric constant macros (MIN/MAX values)
NUMERIC_CONSTANTS = [
    ('MPP_MIN_8U', 'NPP_MIN_8U'),
    ('MPP_MAX_8U', 'NPP_MAX_8U'),
    ('MPP_MIN_16U', 'NPP_MIN_16U'),
    ('MPP_MAX_16U', 'NPP_MAX_16U'),
    ('MPP_MIN_32U', 'NPP_MIN_32U'),
    ('MPP_MAX_32U', 'NPP_MAX_32U'),
    ('MPP_MIN_64U', 'NPP_MIN_64U'),
    ('MPP_MAX_64U', 'NPP_MAX_64U'),
    ('MPP_MIN_8S', 'NPP_MIN_8S'),
    ('MPP_MAX_8S', 'NPP_MAX_8S'),
    ('MPP_MIN_16S', 'NPP_MIN_16S'),
    ('MPP_MAX_16S', 'NPP_MAX_16S'),
    ('MPP_MIN_32S', 'NPP_MIN_32S'),
    ('MPP_MAX_32S', 'NPP_MAX_32S'),
    ('MPP_MIN_64S', 'NPP_MIN_64S'),
    ('MPP_MAX_64S', 'NPP_MAX_64S'),
    ('MPP_MINABS_32F', 'NPP_MINABS_32F'),
    ('MPP_MAXABS_32F', 'NPP_MAXABS_32F'),
    ('MPP_MINABS_64F', 'NPP_MINABS_64F'),
    ('MPP_MAXABS_64F', 'NPP_MAXABS_64F'),
]

# HOG constants
HOG_CONSTANTS = [
    ('MPP_HOG_MAX_CELL_SIZE', 'NPP_HOG_MAX_CELL_SIZE'),
    ('MPP_HOG_MAX_BLOCK_SIZE', 'NPP_HOG_MAX_BLOCK_SIZE'),
    ('MPP_HOG_MAX_BINS_PER_CELL', 'NPP_HOG_MAX_BINS_PER_CELL'),
    ('MPP_HOG_MAX_CELLS_PER_DESCRIPTOR', 'NPP_HOG_MAX_CELLS_PER_DESCRIPTOR'),
    ('MPP_HOG_MAX_OVERLAPPING_BLOCKS_PER_DESCRIPTOR', 'NPP_HOG_MAX_OVERLAPPING_BLOCKS_PER_DESCRIPTOR'),
    ('MPP_HOG_MAX_DESCRIPTOR_LOCATIONS_PER_CALL', 'NPP_HOG_MAX_DESCRIPTOR_LOCATIONS_PER_CALL'),
]

# Contour direction constants
CONTOUR_DIRECTIONS = [
    ('MPP_CONTOUR_DIRECTION_SOUTH_EAST', 'NPP_CONTOUR_DIRECTION_SOUTH_EAST'),
    ('MPP_CONTOUR_DIRECTION_SOUTH', 'NPP_CONTOUR_DIRECTION_SOUTH'),
    ('MPP_CONTOUR_DIRECTION_SOUTH_WEST', 'NPP_CONTOUR_DIRECTION_SOUTH_WEST'),
    ('MPP_CONTOUR_DIRECTION_WEST', 'NPP_CONTOUR_DIRECTION_WEST'),
    ('MPP_CONTOUR_DIRECTION_EAST', 'NPP_CONTOUR_DIRECTION_EAST'),
    ('MPP_CONTOUR_DIRECTION_NORTH_EAST', 'NPP_CONTOUR_DIRECTION_NORTH_EAST'),
    ('MPP_CONTOUR_DIRECTION_NORTH', 'NPP_CONTOUR_DIRECTION_NORTH'),
    ('MPP_CONTOUR_DIRECTION_NORTH_WEST', 'NPP_CONTOUR_DIRECTION_NORTH_WEST'),
    ('MPP_CONTOUR_DIRECTION_ANY_NORTH', 'NPP_CONTOUR_DIRECTION_ANY_NORTH'),
    ('MPP_CONTOUR_DIRECTION_ANY_WEST', 'NPP_CONTOUR_DIRECTION_ANY_WEST'),
    ('MPP_CONTOUR_DIRECTION_ANY_SOUTH', 'NPP_CONTOUR_DIRECTION_ANY_SOUTH'),
    ('MPP_CONTOUR_DIRECTION_ANY_EAST', 'NPP_CONTOUR_DIRECTION_ANY_EAST'),
]

# Utility macros
UTILITY_MACROS = [
    ('MPP_HALF_TO_MPP16F', 'NPP_HALF_TO_NPP16F'),
]

# Core function mappings
CORE_FUNCTIONS = [
    ('mppGetLibVersion', 'nppGetLibVersion'),
    ('mppGetStream', 'nppGetStream'),
    ('mppSetStream', 'nppSetStream'),
    ('mppGetStreamContext', 'nppGetStreamContext'),
    ('mppGetGpuNumSMs', 'nppGetGpuNumSMs'),
    ('mppGetMaxThreadsPerSM', 'nppGetMaxThreadsPerSM'),
    ('mppGetMaxThreadsPerBlock', 'nppGetMaxThreadsPerBlock'),
    ('mppGetGpuName', 'nppGetGpuName'),
    ('mppGetGpuDeviceProperties', 'nppGetGpuDeviceProperties'),
]

# NPPI sub-modules
NPPI_MODULES = [
    'arithmetic_and_logical_operations',
    'color_conversion',
    'data_exchange_and_initialization',
    'filtering_functions',
    'geometry_transforms',
    'linear_transforms',
    'morphological_operations',
    'statistics_functions',
    'support_functions',
    'threshold_and_compare_operations',
]

# NPPS sub-modules
NPPS_MODULES = [
    'arithmetic_and_logical_operations',
    'conversion_functions',
    'filtering_functions',
    'initialization',
    'statistics_functions',
    'support_functions',
]


def generate_typedef(mpp_type: str, npp_type: str) -> str:
    return f'typedef {mpp_type:<40} {npp_type};'


def generate_define(mpp_name: str, npp_name: str) -> str:
    return f'#define {npp_name:<50} {mpp_name}'


def generate_nppdefs_wrapper() -> str:
    lines = [
        '/*',
        ' * NPP Definitions - Compatibility Layer for MPP Library',
        ' * Maps NPP types, enums, and constants to MPP equivalents',
        ' * Based on API/nppdefs.h',
        ' */',
        '',
        '#ifndef NPP_BRIDGE_DEFS_H',
        '#define NPP_BRIDGE_DEFS_H',
        '',
        '#include <mcmpp/mppdefs.h>',
        '',
        '/* ============================================',
        ' * Basic Data Types',
        ' * ============================================ */',
    ]

    for mpp, npp in BASIC_TYPES:
        lines.append(generate_typedef(mpp, npp))

    lines.extend(['', '/* FP16 Types */'])
    for mpp, npp in FP16_TYPES:
        lines.append(generate_typedef(mpp, npp))

    lines.extend(['', '/* Complex Types */'])
    for mpp, npp in COMPLEX_TYPES:
        lines.append(generate_typedef(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * Geometric Types',
        ' * ============================================ */',
    ])
    for mpp, npp in GEOMETRIC_TYPES:
        lines.append(generate_typedef(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * Enum Types',
        ' * ============================================ */',
    ])
    for mpp, npp in ENUM_TYPES:
        lines.append(generate_typedef(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * Struct Types',
        ' * ============================================ */',
    ])
    for mpp, npp in STRUCT_TYPES:
        lines.append(generate_typedef(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * NppStatus Error Codes',
        ' * ============================================ */',
    ])
    for mpp, npp in STATUS_ERRORS:
        lines.append(generate_define(mpp, npp))

    lines.extend(['', '/* NppStatus Warning Codes */'])
    for mpp, npp in STATUS_WARNINGS:
        lines.append(generate_define(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * Legacy NppStatus Aliases',
        ' * ============================================ */',
    ])
    for mpp, npp in STATUS_ALIASES:
        lines.append(generate_define(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * NppiInterpolationMode',
        ' * ============================================ */',
    ])
    for mpp, npp in INTERPOLATION_MODES:
        lines.append(generate_define(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * NppiBayerGridPosition',
        ' * ============================================ */',
    ])
    for mpp, npp in BAYER_PATTERNS:
        lines.append(generate_define(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * NppiMaskSize',
        ' * ============================================ */',
    ])
    for mpp, npp in MASK_SIZES:
        lines.append(generate_define(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * NppiDifferentialKernel',
        ' * ============================================ */',
    ])
    for mpp, npp in DIFF_KERNELS:
        lines.append(generate_define(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * NppDataType',
        ' * ============================================ */',
    ])
    for mpp, npp in DATA_TYPES:
        lines.append(generate_define(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * NppiChannels',
        ' * ============================================ */',
    ])
    for mpp, npp in CHANNELS:
        lines.append(generate_define(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * NppiAxis',
        ' * ============================================ */',
    ])
    for mpp, npp in AXIS_TYPES:
        lines.append(generate_define(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * NppCmpOp',
        ' * ============================================ */',
    ])
    for mpp, npp in COMPARISON_OPS:
        lines.append(generate_define(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * NppRoundMode',
        ' * ============================================ */',
    ])
    for mpp, npp in ROUNDING_MODES:
        lines.append(generate_define(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * NppiBorderType',
        ' * ============================================ */',
    ])
    for mpp, npp in BORDER_TYPES:
        lines.append(generate_define(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * NppHintAlgorithm',
        ' * ============================================ */',
    ])
    for mpp, npp in HINT_ALGORITHMS:
        lines.append(generate_define(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * NppiAlphaOp',
        ' * ============================================ */',
    ])
    for mpp, npp in ALPHA_OPS:
        lines.append(generate_define(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * NppsZCType',
        ' * ============================================ */',
    ])
    for mpp, npp in ZC_TYPES:
        lines.append(generate_define(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * NppiHuffmanTableType',
        ' * ============================================ */',
    ])
    for mpp, npp in HUFFMAN_TABLE_TYPES:
        lines.append(generate_define(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * NppiNorm',
        ' * ============================================ */',
    ])
    for mpp, npp in NORM_TYPES:
        lines.append(generate_define(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * NppiWatershedSegmentBoundaryType',
        ' * ============================================ */',
    ])
    for mpp, npp in WATERSHED_TYPES:
        lines.append(generate_define(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * Numeric Constants (MIN/MAX values)',
        ' * ============================================ */',
    ])
    for mpp, npp in NUMERIC_CONSTANTS:
        lines.append(generate_define(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * HOG Constants',
        ' * ============================================ */',
    ])
    for mpp, npp in HOG_CONSTANTS:
        lines.append(generate_define(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * Contour Direction Constants',
        ' * ============================================ */',
    ])
    for mpp, npp in CONTOUR_DIRECTIONS:
        lines.append(generate_define(mpp, npp))

    lines.extend([
        '',
        '/* ============================================',
        ' * Utility Macros',
        ' * ============================================ */',
    ])
    for mpp, npp in UTILITY_MACROS:
        lines.append(generate_define(mpp, npp))

    lines.extend([
        '',
        '#endif /* NPP_BRIDGE_DEFS_H */',
        '',
    ])

    return '\n'.join(lines)


def generate_nppcore_wrapper() -> str:
    lines = [
        '/*',
        ' * NPP Core Functions - Compatibility Layer for MPP Library',
        ' */',
        '',
        '#ifndef NPP_BRIDGE_CORE_H',
        '#define NPP_BRIDGE_CORE_H',
        '',
        '#include "nppdefs_wrapper.h"',
        '#include <mcmpp/mppcore.h>',
        '',
    ]

    for mpp, npp in CORE_FUNCTIONS:
        lines.append(f'#define {npp:<40} {mpp}')

    lines.extend([
        '',
        '#endif /* NPP_BRIDGE_CORE_H */',
        '',
    ])
    return '\n'.join(lines)


def generate_nppi_wrapper() -> str:
    lines = [
        '/*',
        ' * NPPI Functions - Compatibility Layer for MPP Library',
        ' */',
        '',
        '#ifndef NPP_BRIDGE_NPPI_H',
        '#define NPP_BRIDGE_NPPI_H',
        '',
        '#include "nppdefs_wrapper.h"',
        '#include <mcmpp/mppi.h>',
        '#include "npp2mpp_functions.h"',
        '',
        '#endif /* NPP_BRIDGE_NPPI_H */',
        '',
    ]
    return '\n'.join(lines)


def generate_npps_wrapper() -> str:
    lines = [
        '/*',
        ' * NPPS Functions - Compatibility Layer for MPP Library',
        ' */',
        '',
        '#ifndef NPP_BRIDGE_NPPS_H',
        '#define NPP_BRIDGE_NPPS_H',
        '',
        '#include "nppdefs_wrapper.h"',
        '#include <mcmpp/mpps.h>',
        '#include "npp2mpp_functions.h"',
        '',
        '#endif /* NPP_BRIDGE_NPPS_H */',
        '',
    ]
    return '\n'.join(lines)


def generate_npp_wrapper() -> str:
    lines = [
        '/*',
        ' * NPP - Main Compatibility Layer for MPP Library',
        ' */',
        '',
        '#ifndef NPP_BRIDGE_MAIN_H',
        '#define NPP_BRIDGE_MAIN_H',
        '',
        '#include "../runtime/cuda_to_maca_mcr_adaptor.h"',
        '#include "../runtime/cuda_runtime_wrapper.h"',
        '',
        '#include "nppdefs_wrapper.h"',
        '#include "nppcore_wrapper.h"',
        '#include "nppi_wrapper.h"',
        '#include "npps_wrapper.h"',
        '',
        '#include <mcmpp/mpp.h>',
        '',
        '#endif /* NPP_BRIDGE_MAIN_H */',
        '',
    ]
    return '\n'.join(lines)


def generate_user_header(name: str, wrapper_name: str) -> str:
    guard = f'NV_{name.upper()}_H'
    lines = [
        '/*',
        f' * {name}.h - NPP Compatibility Header for MPP Library',
        ' */',
        '',
        f'#ifndef {guard}',
        f'#define {guard}',
        '',
        f'#include <bridge/mpp/{wrapper_name}_wrapper.h>',
        '',
        f'#endif /* {guard} */',
        '',
    ]
    return '\n'.join(lines)


def generate_submodule_header(prefix: str, module: str) -> str:
    header_name = f'{prefix}_{module}'
    guard = f'NV_{header_name.upper()}_H'
    wrapper = prefix
    lines = [
        '/*',
        f' * {header_name}.h - NPP Compatibility Header for MPP Library',
        ' */',
        '',
        f'#ifndef {guard}',
        f'#define {guard}',
        '',
        f'#include <bridge/mpp/{wrapper}_wrapper.h>',
        '',
        f'#endif /* {guard} */',
        '',
    ]
    return '\n'.join(lines)


def main():
    print('Generating NPP to MPP wrapper files...')
    print(f'  Output directory: {WRAPPER_DIR}')
    print()

    # Create directories
    WRAPPER_DIR.mkdir(parents=True, exist_ok=True)
    user_include_dir = WRAPPER_DIR / 'user_include'
    user_include_dir.mkdir(parents=True, exist_ok=True)

    # Generate wrapper files
    wrapper_files = [
        ('nppdefs_wrapper.h', generate_nppdefs_wrapper()),
        ('nppcore_wrapper.h', generate_nppcore_wrapper()),
        ('nppi_wrapper.h', generate_nppi_wrapper()),
        ('npps_wrapper.h', generate_npps_wrapper()),
        ('npp_wrapper.h', generate_npp_wrapper()),
    ]

    for filename, content in wrapper_files:
        filepath = WRAPPER_DIR / filename
        with open(filepath, 'w') as f:
            f.write(content)
        print(f'  Generated: {filename}')

    print()
    print('Generating user include files...')

    # Generate main user headers
    main_headers = [
        ('npp', 'npp'),
        ('nppdefs', 'nppdefs'),
        ('nppcore', 'nppcore'),
        ('nppi', 'nppi'),
        ('npps', 'npps'),
    ]

    for name, wrapper in main_headers:
        filepath = user_include_dir / f'{name}.h'
        with open(filepath, 'w') as f:
            f.write(generate_user_header(name, wrapper))
        print(f'  Generated: user_include/{name}.h')

    # Generate NPPI sub-module headers
    for module in NPPI_MODULES:
        header_name = f'nppi_{module}'
        filepath = user_include_dir / f'{header_name}.h'
        with open(filepath, 'w') as f:
            f.write(generate_submodule_header('nppi', module))
        print(f'  Generated: user_include/{header_name}.h')

    # Generate NPPS sub-module headers
    for module in NPPS_MODULES:
        header_name = f'npps_{module}'
        filepath = user_include_dir / f'{header_name}.h'
        with open(filepath, 'w') as f:
            f.write(generate_submodule_header('npps', module))
        print(f'  Generated: user_include/{header_name}.h')

    print()
    print('Done! Generated files:')
    print('  Wrapper files: *_wrapper.h')
    print('  User headers:  user_include/*.h')


if __name__ == '__main__':
    main()
