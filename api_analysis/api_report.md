# NPP API Analysis Report

## Version Overview

**11.4**: 12112 functions
**12.2**: 12500 functions
**12.8**: 12536 functions
**13.4**: 7752 functions

## Version Differences

### 11.4 -> 12.2

- Added: 429
- Removed: 41
- Changed: 40

**Added:**
- nppiAbsDiffDeviceC_16u_C1R_Ctx
- nppiAbsDiffDeviceC_32f_C1R_Ctx
- nppiAbsDiffDeviceC_8u_C1R_Ctx
- nppiAddDeviceC_16f_C1IR_Ctx
- nppiAddDeviceC_16f_C1R_Ctx
- nppiAddDeviceC_16f_C3IR_Ctx
- nppiAddDeviceC_16f_C3R_Ctx
- nppiAddDeviceC_16f_C4IR_Ctx
- nppiAddDeviceC_16f_C4R_Ctx
- nppiAddDeviceC_16s_AC4IRSfs_Ctx
- ... and 419 more

**Changed:**
- nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx
- nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R
- nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx
- nppiDistanceTransformPBA_16s16u_C1R_Ctx
- nppiDistanceTransformPBA_16s32f_C1R_Ctx
- ... and 35 more

### 12.2 -> 12.8

- Added: 3224
- Removed: 3188
- Changed: 3188

**Added:**
- nppiAverageErrorGetBufferHostSize_16s_C1R
- nppiAverageErrorGetBufferHostSize_16s_C1R_Ctx
- nppiAverageErrorGetBufferHostSize_16s_C2R
- nppiAverageErrorGetBufferHostSize_16s_C2R_Ctx
- nppiAverageErrorGetBufferHostSize_16s_C3R
- nppiAverageErrorGetBufferHostSize_16s_C3R_Ctx
- nppiAverageErrorGetBufferHostSize_16s_C4R
- nppiAverageErrorGetBufferHostSize_16s_C4R_Ctx
- nppiAverageErrorGetBufferHostSize_16sc_C1R
- nppiAverageErrorGetBufferHostSize_16sc_C1R_Ctx
- ... and 3214 more

**Changed:**
- nppiAverageErrorGetBufferHostSize_16s_C1R
- nppiAverageErrorGetBufferHostSize_16s_C1R_Ctx
- nppiAverageErrorGetBufferHostSize_16s_C2R
- nppiAverageErrorGetBufferHostSize_16s_C2R_Ctx
- nppiAverageErrorGetBufferHostSize_16s_C3R
- ... and 3183 more

### 12.8 -> 13.4

- Added: 67
- Removed: 4851
- Changed: 0

**Added:**
- nppiFilterCannyBorder_8u_C3C1R_Ctx
- nppiI444_8u_ToUYVP_10u_P3C3R_Ctx
- nppiNV12ToBGR_601_8u_P2C3R_Ctx
- nppiNV12ToBGR_8u_P2C4R_Ctx
- nppiNV12ToBGR_BT2020_8u_P2C3R_Ctx
- nppiNV12ToRGB_601_8u_P2C3R_Ctx
- nppiNV12ToRGB_8u_P2C4R_Ctx
- nppiNV12ToRGB_BT2020_8u_P2C3R_Ctx
- nppiNV12_8u_ToUYVP_10u_P2C3R_Ctx
- nppiNV16_8u_ToUYVP_10u_P2C3R_Ctx
- ... and 57 more

## Module Statistics (13.4)

### NPPI

Total: 6860

- nppi_arithmetic_and_logical_operations.h: 2137
- nppi_color_conversion.h: 615
- nppi_data_exchange_and_initialization.h: 484
- nppi_filtering_functions.h: 1009
- nppi_geometry_transforms.h: 358
- nppi_linear_transforms.h: 4
- nppi_morphological_operations.h: 150
- nppi_statistics_functions.h: 1654
- nppi_support_functions.h: 31
- nppi_threshold_and_compare_operations.h: 418

### NPPS

Total: 892

- npps_arithmetic_and_logical_operations.h: 370
- npps_conversion_functions.h: 87
- npps_filtering_functions.h: 2
- npps_initialization.h: 35
- npps_statistics_functions.h: 383
- npps_support_functions.h: 15
