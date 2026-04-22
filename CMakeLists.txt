cmake_minimum_required(VERSION 3.15)

if(WIN32)
    message(FATAL_ERROR "This project does NOT support building on Windows!!!!!")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT WITHOUT_CU_BRIDGE)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Build options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARK "Build performance benchmarks" ON)
option(NPP_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)

option(USE_NVIDIA_NPP "Use NVIDIA NPP library instead of MPP" OFF)
option(USE_MACA_NPP "Use MACA NPP library instead of MPP" OFF)
option(WITHOUT_CU_BRIDGE "Build with MACA SDK (no CUDA Toolkit dependency)" OFF)

message("GPU_TYPE: ${GPU_TYPE}")

if(USE_NVIDIA_NPP AND USE_MACA_NPP)
    message(FATAL_ERROR "USE_NVIDIA_NPP and USE_MACA_NPP cannot both be ON")
endif()

# Include modular CMake configuration
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FindSources)
include(OptConfig)
include(TestConfig)
include(CMakePackageConfigHelpers)

if(WITHOUT_CU_BRIDGE)
    message("npp_configure_maca")
    include(MacaConfig)
    npp_configure_maca()
    set(CMAKE_CXX_COMPILER "${MACA_PATH}/mxgpu_llvm/bin/mxcc" CACHE FILEPATH "MACA compiler")
    project(MPP VERSION 1.0.0 LANGUAGES CXX)
else()
    message("npp_configure_cuda")
    include(CudaConfig)
    npp_configure_cuda()
    project(MPP VERSION 1.0.0 LANGUAGES CXX CUDA)
endif()

include(CommonConfig)
include(InstallConfig)

# build src folder --- core lib (skip when using external NPP)
if(NOT USE_NVIDIA_NPP AND NOT USE_MACA_NPP)
    add_subdirectory(src)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
    # Examples depend on the in-tree MPP library
    if(NOT USE_NVIDIA_NPP AND NOT USE_MACA_NPP)
        add_subdirectory(examples)
    endif()
endif()

# Install/packaging only for MPP build
if(NOT USE_NVIDIA_NPP AND NOT USE_MACA_NPP)
    npp_configure_install()
    set( LIBRARY_NAME mcmpp )
    include(cmake/CCPack.cmake)
endif()
if(BUILD_BENCHMARK)
    add_subdirectory(benchmark benchmark)
endif()

# Install configuration only for MPP build
npp_configure_install()

# set cpack using param
set( LIBRARY_NAME mcmpp )
include(cmake/CCPack.cmake)
