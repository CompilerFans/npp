# NPP API Analysis Report

## Version Overview

**11.4**: 12112 functions
**12.2**: 12500 functions
**12.8**: 12536 functions

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
- nppiDistanceTransformPBA_16u16u_C1R_Ctx
- nppiHistogramRangeGetBufferSize_16s_C1R_Ctx
- nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R
- nppiDistanceTransformPBA_8u16u_C1R_Ctx
- nppiHistogramRangeGetBufferSize_16s_AC4R_Ctx
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
- nppsMul_16s
- nppsMinIndx_64f
- nppsSubCRev_64f
- nppiHistogramRangeGetBufferSize_32f_C4R
- nppiNormRelL1GetBufferHostSize_16s_C4R_Ctx
- ... and 3183 more

## Module Statistics (12.8)

### NPPI

Total: 10768

- nppi_arithmetic_and_logical_operations.h: 2137
- nppi_color_conversion.h: 1074
- nppi_data_exchange_and_initialization.h: 966
- nppi_filtering_functions.h: 1895
- nppi_geometry_transforms.h: 705
- nppi_linear_transforms.h: 4
- nppi_morphological_operations.h: 292
- nppi_statistics_functions.h: 3246
- nppi_support_functions.h: 31
- nppi_threshold_and_compare_operations.h: 418

### NPPS

Total: 1768

- npps_arithmetic_and_logical_operations.h: 740
- npps_conversion_functions.h: 174
- npps_filtering_functions.h: 3
- npps_initialization.h: 70
- npps_statistics_functions.h: 766
- npps_support_functions.h: 15
