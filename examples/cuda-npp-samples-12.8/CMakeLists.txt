cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

# CUDA NPP Samples - Standalone Build for mcMPP (MACA)
project(cuda_npp_samples LANGUAGES CXX CUDA)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# Build Options (compatible with mcMPP build system)
# =============================================================================
option(USE_NVIDIA_NPP "Use NVIDIA NPP library instead of MPP" OFF)
option(NPP_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)

# CUDA Architecture (default: 89 for Ada, mcMPP uses 88)
set(CUDA_ARCH "88" CACHE STRING "CUDA architecture")

# =============================================================================
# MPP/NPP Library Configuration
# =============================================================================
if(NOT DEFINED MPP_ROOT_DIR)
    if(DEFINED ENV{MPP_ROOT_DIR})
        set(MPP_ROOT_DIR $ENV{MPP_ROOT_DIR})
    else()
        set(MPP_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)
    endif()
endif()

if(NOT DEFINED MPP_INCLUDE_DIR)
    if(DEFINED ENV{MPP_INCLUDE_DIR})
        set(MPP_INCLUDE_DIR $ENV{MPP_INCLUDE_DIR})
    else()
        set(MPP_INCLUDE_DIR ${MPP_ROOT_DIR}/API)
    endif()
endif()

if(NOT DEFINED MPP_LIBRARY_DIR)
    if(DEFINED ENV{MPP_LIBRARY_DIR})
        set(MPP_LIBRARY_DIR $ENV{MPP_LIBRARY_DIR})
    else()
        set(MPP_LIBRARY_DIR ${MPP_ROOT_DIR}/build/src)
    endif()
endif()

# =============================================================================
# Find CUDA and Configure
# =============================================================================
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig QUIET)

# Compiler warnings
if(NPP_WARNINGS_AS_ERRORS)
    add_compile_options(-Werror)
    message(STATUS "Warnings will be treated as errors")
endif()

# =============================================================================
# Find MPP/NPP Library
# =============================================================================
if(USE_NVIDIA_NPP)
    message(STATUS "Using NVIDIA NPP library")
    set(NPP_LIBRARIES
        CUDA::nppc CUDA::nppial CUDA::nppicc CUDA::nppidei
        CUDA::nppif CUDA::nppig CUDA::nppim CUDA::nppist
        CUDA::nppisu CUDA::nppitc CUDA::npps
    )
    set(NPP_INCLUDE_DIRS "")
else()
    message(STATUS "Using MPP library")
    find_library(MPP_LIBRARY NAMES mpp_static
        PATHS ${MPP_LIBRARY_DIR} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)

    if(NOT MPP_LIBRARY)
        message(FATAL_ERROR "MPP library not found in ${MPP_LIBRARY_DIR}")
    endif()

    set(NPP_LIBRARIES ${MPP_LIBRARY})
    set(NPP_INCLUDE_DIRS ${MPP_INCLUDE_DIR})
endif()

# Add include directories
include_directories(${NPP_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Common/UtilNPP)

# Find FreeImage (optional)
find_path(FREEIMAGE_INCLUDE_DIR FreeImage.h)
find_library(FREEIMAGE_LIBRARY NAMES freeimage FreeImage)

if(FREEIMAGE_INCLUDE_DIR AND FREEIMAGE_LIBRARY)
    set(FREEIMAGE_FOUND ON)
    message(STATUS "Found FreeImage: ${FREEIMAGE_LIBRARY}")
else()
    set(FREEIMAGE_FOUND OFF)
    message(STATUS "FreeImage not found - some samples will be disabled")
endif()

# =============================================================================
# Function to add NPP sample
# =============================================================================
function(add_npp_sample SAMPLE_NAME)
    set(SAMPLE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Samples/4_CUDA_Libraries/${SAMPLE_NAME})

    if(EXISTS ${SAMPLE_DIR})
        if(NOT FREEIMAGE_FOUND AND ${SAMPLE_NAME} STREQUAL "freeImageInteropNPP")
            message(STATUS "Skipping ${SAMPLE_NAME} - requires FreeImage")
            return()
        endif()

        file(GLOB SAMPLE_SOURCES ${SAMPLE_DIR}/*.cpp ${SAMPLE_DIR}/*.cu)

        if(SAMPLE_SOURCES)
            add_executable(${SAMPLE_NAME} ${SAMPLE_SOURCES})

            set_property(TARGET ${SAMPLE_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
            set_property(TARGET ${SAMPLE_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
            set_target_properties(${SAMPLE_NAME} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")

            target_link_libraries(${SAMPLE_NAME} ${NPP_LIBRARIES} CUDA::cudart)

            if(FREEIMAGE_FOUND)
                target_include_directories(${SAMPLE_NAME} PRIVATE ${FREEIMAGE_INCLUDE_DIR})
                target_link_libraries(${SAMPLE_NAME} ${FREEIMAGE_LIBRARY})
                target_compile_definitions(${SAMPLE_NAME} PRIVATE FREEIMAGE_LIB)
            endif()

            set_target_properties(${SAMPLE_NAME} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

            message(STATUS "Added NPP sample: ${SAMPLE_NAME}")
        endif()
    endif()
endfunction()

# =============================================================================
# Add all NPP samples
# =============================================================================
add_npp_sample(batchedLabelMarkersAndLabelCompressionNPP)
add_npp_sample(FilterBorderControlNPP)
add_npp_sample(boxFilterNPP)
add_npp_sample(cannyEdgeDetectorNPP)
add_npp_sample(freeImageInteropNPP)
add_npp_sample(histEqualizationNPP)
add_npp_sample(watershedSegmentationNPP)

# Copy test data
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Common/data DESTINATION ${CMAKE_BINARY_DIR})

# =============================================================================
# Configuration Summary
# =============================================================================
message(STATUS "========================================")
message(STATUS "CUDA NPP Samples Configuration (MACA)")
message(STATUS "========================================")
if(USE_NVIDIA_NPP)
    message(STATUS "NPP Library:         NVIDIA NPP")
else()
    message(STATUS "NPP Library:         MPP")
    message(STATUS "MPP Include Dir:     ${MPP_INCLUDE_DIR}")
    message(STATUS "MPP Library:         ${MPP_LIBRARY}")
endif()

message(STATUS "CUDA Architecture:   ${CUDA_ARCH}")
message(STATUS "Warnings as Errors:  ${NPP_WARNINGS_AS_ERRORS}")
message(STATUS "FreeImage Found:     ${FREEIMAGE_FOUND}")
message(STATUS "========================================")

# Verification tool
add_executable(verify_results verify_results.cpp)
set_target_properties(verify_results PROPERTIES
    CXX_STANDARD 14
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)