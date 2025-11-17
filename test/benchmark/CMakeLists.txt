# NPP 性能测试 CMake 配置

# 设置 Google Benchmark
npp_setup_benchmark()

# 包含目录
set(BENCHMARK_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/framework
    ${CMAKE_SOURCE_DIR}/test/common
)

# ============================================================
# NPPI 算术运算性能测试
# ============================================================

set(NPPI_ARITHMETIC_BENCHMARK_SOURCES
    nppi/arithmetic/benchmark_nppi_add.cpp
)

npp_create_benchmark_target(
    nppi_arithmetic_benchmark
    "${NPPI_ARITHMETIC_BENCHMARK_SOURCES}"
    npp_nppi_lib
)

target_include_directories(nppi_arithmetic_benchmark
    PRIVATE
    ${BENCHMARK_INCLUDE_DIRS}
)

# 可选：注册到 CTest（用于 CI）
if(NPP_REGISTER_BENCHMARKS_TO_CTEST)
    npp_register_benchmark(nppi_arithmetic nppi_arithmetic_benchmark)
endif()

# ============================================================
# 其他模块的性能测试（待添加）
# ============================================================

# NPPI 滤波性能测试
# set(NPPI_FILTERING_BENCHMARK_SOURCES
#     nppi/filtering/benchmark_nppi_filter.cpp
# )
# npp_create_benchmark_target(
#     nppi_filtering_benchmark
#     "${NPPI_FILTERING_BENCHMARK_SOURCES}"
#     npp_nppi_lib
# )

# NPPI 几何变换性能测试
# set(NPPI_GEOMETRY_BENCHMARK_SOURCES
#     nppi/geometry/benchmark_nppi_resize.cpp
# )
# npp_create_benchmark_target(
#     nppi_geometry_benchmark
#     "${NPPI_GEOMETRY_BENCHMARK_SOURCES}"
#     npp_nppi_lib
# )

message(STATUS "Performance benchmarks configured")
message(STATUS "  - To run: cd build/benchmark && ./nppi_arithmetic_benchmark")
message(STATUS "  - Options: --benchmark_filter=<pattern>")
message(STATUS "  - Options: --benchmark_out=<file.json>")
message(STATUS "  - Options: --benchmark_out_format=json")
