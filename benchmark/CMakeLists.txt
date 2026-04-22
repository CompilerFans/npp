#
# NPP/MPP Performance Benchmark
#
# This directory is normally added from the repo root via:
#   -DBUILD_BENCHMARK=ON
#
# It can also be configured standalone, but the recommended path is to build
# through the main project to inherit toolchain and SDK settings.
#

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.18)
    project(npp_benchmark LANGUAGES CXX CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(NPP_SDK_INCLUDE_DIR ${CUDAToolkit_INCLUDE_DIRS} CACHE PATH "CUDA include dirs")
    set(NPP_SDK_LIB_DIR ${CUDAToolkit_LIBRARY_DIR} CACHE PATH "CUDA library dir")
    set(NPP_SDK_LINK_LIBS CUDA::cudart CACHE STRING "CUDA link libraries")
endif()

# Core benchmark sources (always included)
set(BENCHMARK_CORE_SOURCES
    benchmark_main.cpp
    benchmark_runner.cpp
    benchmark_utils.cu
)

# Auto-discover benchmark source files
# nppi benchmarks (image processing)
file(GLOB_RECURSE NPPI_BENCH_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/nppi/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nppi/*.cu"
)

# npps benchmarks (signal processing)
file(GLOB_RECURSE NPPS_BENCH_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/npps/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/npps/*.cu"
)

# nppcore benchmarks (core functions)
file(GLOB_RECURSE NPPCORE_BENCH_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/nppcore/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nppcore/*.cu"
)

# Combine all sources
set(BENCHMARK_SOURCES
    ${BENCHMARK_CORE_SOURCES}
    ${NPPI_BENCH_SOURCES}
    ${NPPS_BENCH_SOURCES}
    ${NPPCORE_BENCH_SOURCES}
)

# Create benchmark executable
add_executable(npp_benchmark ${BENCHMARK_SOURCES})

# Inherit common project conventions when building from repo root.
if(COMMAND npp_setup_target)
    npp_setup_target(npp_benchmark)
endif()

# Benchmarks rely on runtime symbol lookup (dlsym). Ensure linked NPP/MPP libraries
# are kept as DT_NEEDED even if the benchmark TU doesn't reference them directly.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(npp_benchmark PRIVATE "LINKER:--no-as-needed")
endif()

# Include directories
target_include_directories(npp_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
if(DEFINED NPP_SDK_INCLUDE_DIR)
    target_include_directories(npp_benchmark PRIVATE ${NPP_SDK_INCLUDE_DIR})
endif()
if(NOT USE_NVIDIA_NPP)
    target_include_directories(npp_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/API)
endif()

# Keep output path stable for scripts: <build>/benchmark/npp_benchmark
set_target_properties(npp_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmark
)

# Link libraries based on build configuration
if(USE_NVIDIA_NPP)
    message(STATUS "Benchmark: Using NVIDIA NPP library")
    if(DEFINED NPP_SDK_LIB_DIR)
        target_link_directories(npp_benchmark PRIVATE ${NPP_SDK_LIB_DIR})
    endif()
    target_link_libraries(npp_benchmark PRIVATE
        nppc
        nppial
        nppicc
        nppidei
        nppif
        nppig
        nppim
        nppist
        nppisu
        nppitc
        npps
    )
    # Define macros for NVIDIA NPP specific code paths
    target_compile_definitions(npp_benchmark PRIVATE
        USE_NVIDIA_NPP
        USE_NVIDIA_NPP_TESTS
    )
else()
    message(STATUS "Benchmark: Using in-tree project NPP library")
    if(NOT TARGET npp)
        message(FATAL_ERROR "Benchmark requires in-tree project target 'npp' (configure from repo root without USE_NVIDIA_NPP).")
    endif()
    target_link_libraries(npp_benchmark PRIVATE npp)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(npp_benchmark PRIVATE dl)
endif()

# C++ standard
target_compile_features(npp_benchmark PRIVATE cxx_std_17)

set_target_properties(npp_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Compiler warnings (less strict for benchmarks)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(npp_benchmark PRIVATE
        -Wall
        -Wextra
        -Wno-unused-parameter
    )
endif()

# Install benchmark executable (optional)
install(TARGETS npp_benchmark
    RUNTIME DESTINATION bin
    OPTIONAL
)

# Print summary
message(STATUS "")
message(STATUS "=== Benchmark Configuration Summary ===")
if(USE_NVIDIA_NPP)
    message(STATUS "  Library: NVIDIA NPP")
else()
    message(STATUS "  Library: Project NPP (in-tree)")
endif()
message(STATUS "  Warmup/measure defaults: fixed in benchmark_base.h")
message(STATUS "  Total benchmark sources: ${BENCHMARK_SOURCES}")
message(STATUS "========================================")
