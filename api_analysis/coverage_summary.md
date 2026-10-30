# NPP API Catalog Summary

- Public functions (API/): 12537
- Real implementations: 9476 (75.6%)
- Stubs (NPP_NOT_IMPLEMENTED_ERROR): 118
- Missing: 2943
- Tested: 8236 (65.7%)
- Real and tested: 6676

## By module

| module | total | real | stub | missing | tested |
|---|---:|---:|---:|---:|---:|
| nppi_arithmetic_and_logical_operations.h | 2137 | 2137 | 0 | 0 | 2133 |
| nppi_color_conversion.h | 1074 | 395 | 0 | 679 | 704 |
| nppi_data_exchange_and_initialization.h | 966 | 966 | 0 | 0 | 638 |
| nppi_filtering_functions.h | 1896 | 356 | 0 | 1540 | 997 |
| nppi_geometry_transforms.h | 705 | 306 | 90 | 309 | 434 |
| nppi_linear_transforms.h | 4 | 4 | 0 | 0 | 4 |
| nppi_morphological_operations.h | 292 | 24 | 0 | 268 | 154 |
| nppi_statistics_functions.h | 3246 | 3071 | 28 | 147 | 1726 |
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
| nppi_filtering_functions.h | nppiFilterMedian | 64 | 32 |
| nppi_geometry_transforms.h | nppiWarpAffine | 60 | 30 |
| nppi_color_conversion.h | nppiLUT_Linear | 58 | 29 |
| nppi_color_conversion.h | nppiLUT | 56 | 28 |
| nppi_geometry_transforms.h | nppiResizeSqrPixel | 56 | 28 |
| nppi_geometry_transforms.h | nppiWarpAffineQuad | 48 | 24 |
| nppi_geometry_transforms.h | nppiWarpPerspectiveQuad | 48 | 24 |
| nppi_color_conversion.h | nppiColorTwist32f | 40 | 20 |
| nppi_geometry_transforms.h | nppiResizeBatch | 38 | 19 |
| nppi_statistics_functions.h | nppiQualityIndex | 36 | 18 |
| nppi_filtering_functions.h | nppiFilterColumn | 34 | 17 |
| nppi_filtering_functions.h | nppiFilterRow | 34 | 17 |
| nppi_filtering_functions.h | nppiFilterColumnBorder | 32 | 16 |
| nppi_filtering_functions.h | nppiFilterGaussAdvancedBorder | 32 | 16 |
| nppi_filtering_functions.h | nppiFilterGaussAdvanced | 32 | 16 |
| nppi_filtering_functions.h | nppiFilterGaussBorder | 32 | 16 |
| nppi_filtering_functions.h | nppiFilterHighPassBorder | 32 | 16 |
| nppi_filtering_functions.h | nppiFilterHighPass | 32 | 16 |
| nppi_filtering_functions.h | nppiFilterLowPassBorder | 32 | 16 |
| nppi_filtering_functions.h | nppiFilterLowPass | 32 | 16 |
| nppi_filtering_functions.h | nppiFilterMaxBorder | 32 | 16 |
| nppi_filtering_functions.h | nppiFilterMedianBorder | 32 | 32 |
| nppi_filtering_functions.h | nppiFilterMinBorder | 32 | 16 |
| nppi_filtering_functions.h | nppiFilterRowBorder | 32 | 16 |
| nppi_filtering_functions.h | nppiFilterSharpenBorder | 32 | 16 |
| nppi_filtering_functions.h | nppiFilterSharpen | 32 | 16 |
| nppi_filtering_functions.h | nppiFilterUnsharpBorder | 32 | 16 |
| nppi_filtering_functions.h | nppiFilter | 30 | 15 |
| nppi_geometry_transforms.h | nppiResize | 30 | 15 |
| nppi_color_conversion.h | nppiColorTwistBatch32f | 28 | 14 |
| nppi_filtering_functions.h | nppiFilterBox | 28 | 14 |
| nppi_filtering_functions.h | nppiFilterLaplaceBorder | 28 | 14 |
| nppi_filtering_functions.h | nppiFilterLaplace | 28 | 14 |
| nppi_filtering_functions.h | nppiFilterSobelHorizBorder | 28 | 14 |
| nppi_filtering_functions.h | nppiFilterSobelHoriz | 28 | 14 |
| nppi_filtering_functions.h | nppiFilterSobelVertBorder | 28 | 14 |
| nppi_filtering_functions.h | nppiFilterSobelVert | 28 | 14 |
| nppi_color_conversion.h | nppiLUTPalette | 26 | 13 |
| nppi_filtering_functions.h | nppiFilterGauss | 26 | 13 |
