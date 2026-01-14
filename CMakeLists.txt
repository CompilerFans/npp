cmake_minimum_required(VERSION 3.15)

if(USE_MACA_SDK)
    project(MPP VERSION 1.0.0 LANGUAGES CXX)
else()
    project(MPP VERSION 1.0.0 LANGUAGES CXX CUDA)
endif()

# Build options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_SHARED_LIBS "Build shared libraries instead of static libraries" OFF)
option(NPP_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)

option(USE_NVIDIA_NPP "Use NVIDIA NPP library instead of MPP" OFF)
option(WITHOUT_CU_BRIDGE "Build with MACA SDK (no CUDA Toolkit dependency)" OFF)

# Include modular CMake configuration
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(WITHOUT_CU_BRIDGE)
    include(MacaConfig)
    npp_configure_maca()
else()
    include(CudaConfig)
    npp_configure_cuda()
endif()

include(FindSources)
include(CommonConfig)
include(OptConfig)
include(TestConfig)
include(InstallConfig)
include(CMakePackageConfigHelpers)

# build src folder --- core lib
add_subdirectory(src)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Install configuration only for MPP build
npp_configure_install()

# Packaging configuration
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Cpack.cmake")
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Cpack.cmake")
endif()
