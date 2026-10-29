# NPP API Catalog Summary

- Public functions (API/): 12537
- Real implementations: 6688 (53.3%)
- Stubs (NPP_NOT_IMPLEMENTED_ERROR): 166
- Missing: 5683
- Tested: 4244 (33.9%)
- Real and tested: 4235

## By module

| module | total | real | stub | missing | tested |
|---|---:|---:|---:|---:|---:|
| nppi_arithmetic_and_logical_operations.h | 2137 | 2137 | 0 | 0 | 2109 |
| nppi_color_conversion.h | 1074 | 395 | 0 | 679 | 316 |
| nppi_data_exchange_and_initialization.h | 966 | 966 | 0 | 0 | 423 |
| nppi_filtering_functions.h | 1896 | 94 | 0 | 1802 | 68 |
| nppi_geometry_transforms.h | 705 | 258 | 138 | 309 | 181 |
| nppi_linear_transforms.h | 4 | 4 | 0 | 0 | 4 |
| nppi_morphological_operations.h | 292 | 24 | 0 | 268 | 24 |
| nppi_statistics_functions.h | 3246 | 1351 | 28 | 1867 | 322 |
| nppi_support_functions.h | 31 | 31 | 0 | 0 | 16 |
| nppi_threshold_and_compare_operations.h | 418 | 418 | 0 | 0 | 91 |
| npps_arithmetic_and_logical_operations.h | 740 | 740 | 0 | 0 | 525 |
| npps_conversion_functions.h | 174 | 174 | 0 | 0 | 77 |
| npps_filtering_functions.h | 3 | 3 | 0 | 0 | 3 |
| npps_initialization.h | 70 | 70 | 0 | 0 | 70 |
| npps_statistics_functions.h | 766 | 8 | 0 | 758 | 8 |
| npps_support_functions.h | 15 | 15 | 0 | 0 | 7 |

## Largest remaining groups

| module | group | remaining | ctx |
|---|---|---:|---:|
| nppi_statistics_functions.h | nppiAverageRelativeError | 176 | 88 |
| nppi_statistics_functions.h | nppiMaximumError | 176 | 88 |
| nppi_statistics_functions.h | nppiMaximumRelativeError | 176 | 88 |
| nppi_statistics_functions.h | nppiAverageError | 172 | 86 |
| nppi_statistics_functions.h | nppiDotProd | 112 | 56 |
| npps_statistics_functions.h | nppsDotProd | 100 | 50 |
| nppi_filtering_functions.h | nppiFilter32f | 66 | 33 |
| nppi_filtering_functions.h | nppiFilterBorder32f | 66 | 33 |
| nppi_color_conversion.h | nppiLUT_Cubic | 64 | 32 |
| nppi_filtering_functions.h | nppiFilterMedian | 64 | 32 |
| nppi_geometry_transforms.h | nppiWarpAffine | 60 | 30 |
| nppi_color_conversion.h | nppiLUT_Linear | 58 | 29 |
| nppi_color_conversion.h | nppiLUT | 56 | 28 |
| nppi_geometry_transforms.h | nppiResizeSqrPixel | 56 | 28 |
| npps_statistics_functions.h | nppsAverageError | 56 | 28 |
| npps_statistics_functions.h | nppsAverageRelativeError | 56 | 28 |
| npps_statistics_functions.h | nppsMaximumError | 56 | 28 |
| npps_statistics_functions.h | nppsMaximumRelativeError | 56 | 28 |
| nppi_geometry_transforms.h | nppiWarpAffineQuad | 48 | 24 |
| nppi_geometry_transforms.h | nppiWarpPerspectiveQuad | 48 | 24 |
| nppi_statistics_functions.h | nppiCrossCorrFull_NormLevel | 48 | 24 |
| nppi_statistics_functions.h | nppiCrossCorrFull_Norm | 48 | 24 |
| nppi_statistics_functions.h | nppiCrossCorrSame_NormLevel | 48 | 24 |
| nppi_statistics_functions.h | nppiCrossCorrSame_Norm | 48 | 24 |
| nppi_statistics_functions.h | nppiCrossCorrValid_NormLevel | 48 | 24 |
| nppi_statistics_functions.h | nppiCrossCorrValid_Norm | 48 | 24 |
| nppi_statistics_functions.h | nppiFullNormLevel | 48 | 24 |
| nppi_statistics_functions.h | nppiNormRelInf | 48 | 24 |
| nppi_statistics_functions.h | nppiNormRelL1 | 48 | 24 |
| nppi_statistics_functions.h | nppiNormRelL2 | 48 | 24 |
| nppi_statistics_functions.h | nppiNormRel_Inf | 48 | 24 |
| nppi_statistics_functions.h | nppiNormRel_L1 | 48 | 24 |
| nppi_statistics_functions.h | nppiNormRel_L2 | 48 | 24 |
| nppi_statistics_functions.h | nppiSameNormLevel | 48 | 24 |
| nppi_statistics_functions.h | nppiValidNormLevel | 48 | 24 |
| nppi_color_conversion.h | nppiColorTwist32f | 40 | 20 |
| nppi_statistics_functions.h | nppiSqrDistanceFull_Norm | 40 | 20 |
| nppi_statistics_functions.h | nppiSqrDistanceSame_Norm | 40 | 20 |
| nppi_statistics_functions.h | nppiSqrDistanceValid_Norm | 40 | 20 |
| nppi_geometry_transforms.h | nppiResizeBatch | 38 | 19 |
