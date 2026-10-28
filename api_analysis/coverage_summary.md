# NPP API Catalog Summary

- Public functions (API/): 12537
- Real implementations: 3336 (26.6%)
- Stubs (NPP_NOT_IMPLEMENTED_ERROR): 166
- Missing: 9035
- Tested: 3059 (24.4%)
- Real and tested: 3029

## By module

| module | total | real | stub | missing | tested |
|---|---:|---:|---:|---:|---:|
| nppi_arithmetic_and_logical_operations.h | 2137 | 2137 | 0 | 0 | 2109 |
| nppi_color_conversion.h | 1074 | 395 | 0 | 679 | 316 |
| nppi_data_exchange_and_initialization.h | 966 | 140 | 0 | 826 | 127 |
| nppi_filtering_functions.h | 1896 | 94 | 0 | 1802 | 68 |
| nppi_geometry_transforms.h | 705 | 258 | 138 | 309 | 181 |
| nppi_linear_transforms.h | 4 | 4 | 0 | 0 | 4 |
| nppi_morphological_operations.h | 292 | 24 | 0 | 268 | 24 |
| nppi_statistics_functions.h | 3246 | 159 | 28 | 3059 | 128 |
| nppi_support_functions.h | 31 | 31 | 0 | 0 | 16 |
| nppi_threshold_and_compare_operations.h | 418 | 52 | 0 | 366 | 52 |
| npps_arithmetic_and_logical_operations.h | 740 | 6 | 0 | 734 | 6 |
| npps_conversion_functions.h | 174 | 0 | 0 | 174 | 0 |
| npps_filtering_functions.h | 3 | 3 | 0 | 0 | 3 |
| npps_initialization.h | 70 | 10 | 0 | 60 | 10 |
| npps_statistics_functions.h | 766 | 8 | 0 | 758 | 8 |
| npps_support_functions.h | 15 | 15 | 0 | 0 | 7 |

## Largest remaining groups

| module | group | remaining | ctx |
|---|---|---:|---:|
| nppi_data_exchange_and_initialization.h | nppiConvert | 200 | 100 |
| nppi_statistics_functions.h | nppiAverageRelativeError | 176 | 88 |
| nppi_statistics_functions.h | nppiMaximumError | 176 | 88 |
| nppi_statistics_functions.h | nppiMaximumRelativeError | 176 | 88 |
| nppi_statistics_functions.h | nppiAverageError | 172 | 86 |
| nppi_data_exchange_and_initialization.h | nppiCopy | 160 | 80 |
| nppi_data_exchange_and_initialization.h | nppiSet | 144 | 72 |
| nppi_statistics_functions.h | nppiDotProd | 112 | 56 |
| npps_statistics_functions.h | nppsDotProd | 100 | 50 |
| nppi_statistics_functions.h | nppiMean | 92 | 46 |
| nppi_statistics_functions.h | nppiSum | 72 | 36 |
| nppi_data_exchange_and_initialization.h | nppiSwapChannels | 70 | 35 |
| nppi_filtering_functions.h | nppiFilter32f | 66 | 33 |
| nppi_filtering_functions.h | nppiFilterBorder32f | 66 | 33 |
| nppi_color_conversion.h | nppiLUT_Cubic | 64 | 32 |
| nppi_data_exchange_and_initialization.h | nppiScale | 64 | 32 |
| nppi_filtering_functions.h | nppiFilterMedian | 64 | 32 |
| nppi_statistics_functions.h | nppiMax | 64 | 32 |
| nppi_statistics_functions.h | nppiMaxIndx | 64 | 32 |
| nppi_statistics_functions.h | nppiMin | 64 | 32 |
| nppi_statistics_functions.h | nppiMinIndx | 64 | 32 |
| nppi_statistics_functions.h | nppiMinMax | 64 | 32 |
| nppi_statistics_functions.h | nppiMinMaxIndx | 64 | 32 |
| nppi_geometry_transforms.h | nppiWarpAffine | 60 | 30 |
| npps_arithmetic_and_logical_operations.h | nppsMul | 60 | 30 |
| nppi_color_conversion.h | nppiLUT_Linear | 58 | 29 |
| nppi_color_conversion.h | nppiLUT | 56 | 28 |
| nppi_geometry_transforms.h | nppiResizeSqrPixel | 56 | 28 |
| nppi_statistics_functions.h | nppiNormInf | 50 | 25 |
| nppi_statistics_functions.h | nppiNorm_Inf | 50 | 25 |
| npps_conversion_functions.h | nppsConvert | 50 | 25 |
| nppi_geometry_transforms.h | nppiWarpAffineQuad | 48 | 24 |
| nppi_geometry_transforms.h | nppiWarpPerspectiveQuad | 48 | 24 |
| nppi_statistics_functions.h | nppiCrossCorrFull_NormLevel | 48 | 24 |
| nppi_statistics_functions.h | nppiCrossCorrFull_Norm | 48 | 24 |
| nppi_statistics_functions.h | nppiCrossCorrSame_NormLevel | 48 | 24 |
| nppi_statistics_functions.h | nppiCrossCorrSame_Norm | 48 | 24 |
| nppi_statistics_functions.h | nppiCrossCorrValid_NormLevel | 48 | 24 |
| nppi_statistics_functions.h | nppiCrossCorrValid_Norm | 48 | 24 |
| nppi_statistics_functions.h | nppiFullNormLevel | 48 | 24 |
