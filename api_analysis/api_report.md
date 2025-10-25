# NPP API Analysis Report

## Version Overview

**11.4**: 12068 functions
**12.2**: 12456 functions
**12.8**: 12492 functions

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
- nppiDistanceTransformPBA_16s16u_C1R_Ctx
- nppiHistogramEvenGetBufferSize_16u_C3R_Ctx
- nppiHistogramRangeGetBufferSize_8u_AC4R_Ctx
- nppiHistogramRangeGetBufferSize_32f_C4R_Ctx
- nppiHistogramRangeGetBufferSize_8u_C1R_Ctx
- ... and 35 more

### 12.2 -> 12.8

- Added: 3210
- Removed: 3174
- Changed: 3174

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
- ... and 3200 more

**Changed:**
- nppsDivC_16u_Sfs
- nppiSameNormLevelGetBufferHostSize_8s32f_AC4R_Ctx
- nppiMinIndxGetBufferHostSize_16u_AC4R
- nppiAverageRelativeErrorGetBufferHostSize_32u_C1R
- nppsMinMaxGetBufferSize_8u_Ctx
- ... and 3169 more

## Module Statistics (12.8)

### NPPI

Total: 10738

- nppi_arithmetic_and_logical_operations.h: 2137
- nppi_color_conversion.h: 1074
- nppi_data_exchange_and_initialization.h: 966
- nppi_filtering_functions.h: 1895
- nppi_geometry_transforms.h: 705
- nppi_linear_transforms.h: 4
- nppi_morphological_operations.h: 292
- nppi_statistics_functions.h: 3246
- nppi_support_functions.h: 1
- nppi_threshold_and_compare_operations.h: 418

### NPPS

Total: 1754

- npps_arithmetic_and_logical_operations.h: 740
- npps_conversion_functions.h: 174
- npps_filtering_functions.h: 3
- npps_initialization.h: 70
- npps_statistics_functions.h: 766
- npps_support_functions.h: 1
