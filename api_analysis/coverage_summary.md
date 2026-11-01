# NPP API Catalog Summary

- Public functions (API/): 12537
- Real implementations: 11682 (93.2%)
- Stubs (NPP_NOT_IMPLEMENTED_ERROR): 84
- Missing: 771
- Tested: 8279 (66.0%)
- Real and tested: 7832

## By module

| module | total | real | stub | missing | tested |
|---|---:|---:|---:|---:|---:|
| nppi_arithmetic_and_logical_operations.h | 2137 | 2137 | 0 | 0 | 2133 |
| nppi_color_conversion.h | 1074 | 527 | 0 | 547 | 707 |
| nppi_data_exchange_and_initialization.h | 966 | 966 | 0 | 0 | 638 |
| nppi_filtering_functions.h | 1896 | 1747 | 0 | 149 | 1002 |
| nppi_geometry_transforms.h | 705 | 549 | 84 | 72 | 456 |
| nppi_linear_transforms.h | 4 | 4 | 0 | 0 | 4 |
| nppi_morphological_operations.h | 292 | 292 | 0 | 0 | 163 |
| nppi_statistics_functions.h | 3246 | 3243 | 0 | 3 | 1730 |
| nppi_support_functions.h | 31 | 31 | 0 | 0 | 16 |
| nppi_threshold_and_compare_operations.h | 418 | 418 | 0 | 0 | 226 |
| npps_arithmetic_and_logical_operations.h | 740 | 740 | 0 | 0 | 632 |
| npps_conversion_functions.h | 174 | 174 | 0 | 0 | 89 |
| npps_filtering_functions.h | 3 | 3 | 0 | 0 | 3 |
| npps_initialization.h | 70 | 70 | 0 | 0 | 70 |
| npps_statistics_functions.h | 766 | 766 | 0 | 0 | 403 |
| npps_support_functions.h | 15 | 15 | 0 | 0 | 7 |

## Largest remaining groups

| module | group | remaining | ctx |
|---|---|---:|---:|
| nppi_color_conversion.h | nppiLUT_Cubic | 64 | 32 |
| nppi_geometry_transforms.h | nppiWarpAffine | 60 | 30 |
| nppi_color_conversion.h | nppiLUT_Linear | 58 | 29 |
| nppi_color_conversion.h | nppiLUT | 56 | 28 |
| nppi_geometry_transforms.h | nppiWarpAffineQuad | 48 | 24 |
| nppi_color_conversion.h | nppiLUTPalette | 26 | 13 |
| nppi_geometry_transforms.h | nppiWarpAffineBack | 24 | 12 |
| nppi_geometry_transforms.h | nppiWarpAffineBatch | 22 | 11 |
| nppi_filtering_functions.h | nppiHistogramOfGradientsBorder | 17 | 8 |
| nppi_filtering_functions.h | nppiDistanceTransformAbsPBA | 13 | 13 |
| nppi_filtering_functions.h | nppiFloodFill | 13 | 6 |
| nppi_filtering_functions.h | nppiDistanceTransformPBA | 12 | 12 |
| nppi_filtering_functions.h | nppiFilterGaussPyramidLayerUpBorder | 12 | 6 |
| nppi_filtering_functions.h | nppiFloodFillBoundary | 12 | 6 |
| nppi_filtering_functions.h | nppiFloodFillGradientBoundary | 12 | 6 |
| nppi_filtering_functions.h | nppiFloodFillGradient | 12 | 6 |
| nppi_filtering_functions.h | nppiFloodFillRangeBoundary | 12 | 6 |
| nppi_filtering_functions.h | nppiFloodFillRange | 12 | 6 |
| nppi_color_conversion.h | nppiYCbCr411ToYCbCr422 | 8 | 4 |
| nppi_color_conversion.h | nppiYCbCr422ToYCbCr411 | 8 | 4 |
| nppi_color_conversion.h | nppiYCbCr422ToYCbCr420 | 8 | 4 |
| nppi_color_conversion.h | nppiYCbCrToBGRBatch | 8 | 4 |
| nppi_color_conversion.h | nppiYCbCrToRGBBatch | 8 | 4 |
| nppi_color_conversion.h | nppiYUV420ToRGB | 8 | 8 |
| nppi_color_conversion.h | nppiYUV422ToRGB | 8 | 8 |
| nppi_color_conversion.h | nppiYUVToBGRBatch | 8 | 4 |
| nppi_color_conversion.h | nppiYUVToRGBBatch | 8 | 4 |
| nppi_color_conversion.h | nppiCompColorKey | 6 | 3 |
| nppi_color_conversion.h | nppiLUT_Trilinear | 6 | 3 |
| nppi_color_conversion.h | nppiRGBToYUV422 | 6 | 6 |
| nppi_color_conversion.h | nppiYCbCr411ToYCbCr420 | 6 | 3 |
| nppi_color_conversion.h | nppiCFAToRGBA | 5 | 3 |
| nppi_color_conversion.h | nppiBGRToCbYCr422_709HDTV | 4 | 2 |
| nppi_color_conversion.h | nppiBGRToYCbCr420_709CSC | 4 | 2 |
| nppi_color_conversion.h | nppiBGRToYCbCr420_JPEG | 4 | 2 |
| nppi_color_conversion.h | nppiBGRToYCbCr422_JPEG | 4 | 2 |
| nppi_color_conversion.h | nppiBGRToYCbCr444_JPEG | 4 | 2 |
| nppi_color_conversion.h | nppiBGRToYCrCb420_709CSC | 4 | 2 |
| nppi_color_conversion.h | nppiBGRToYCrCb420 | 4 | 2 |
| nppi_color_conversion.h | nppiCMYKOrYCCKToBGR_JPEG | 4 | 2 |
