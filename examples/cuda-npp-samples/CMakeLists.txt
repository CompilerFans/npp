cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# CUDA NPP Samples using OpenNPP
project(cuda_npp_samples LANGUAGES CXX CUDA)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig QUIET)

# OpenNPP library
set(OPENNPP_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(OPENNPP_INCLUDE_DIR ${OPENNPP_ROOT_DIR}/API)
set(OPENNPP_LIBRARY_DIR ${OPENNPP_ROOT_DIR}/build/src)

# Add OpenNPP includes
include_directories(${OPENNPP_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Common/UtilNPP)

# Link directories
link_directories(${OPENNPP_LIBRARY_DIR})

# Common sources
set(COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/UtilNPP/ImagesCPU.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/UtilNPP/ImagesNPP.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/UtilNPP/ImageIO.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/helper_cuda.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/helper_functions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/helper_string.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/helper_image.h
)

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

# Function to add NPP sample
function(add_npp_sample SAMPLE_NAME)
    set(SAMPLE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Samples/4_CUDA_Libraries/${SAMPLE_NAME})
    
    if(EXISTS ${SAMPLE_DIR})
        # Skip FreeImage-dependent samples if FreeImage is not available
        if(NOT FREEIMAGE_FOUND AND ${SAMPLE_NAME} STREQUAL "freeImageInteropNPP")
            message(STATUS "Skipping ${SAMPLE_NAME} - requires FreeImage")
            return()
        endif()
        
        # Find source files
        file(GLOB SAMPLE_SOURCES ${SAMPLE_DIR}/*.cpp ${SAMPLE_DIR}/*.cu)
        
        if(SAMPLE_SOURCES)
            add_executable(${SAMPLE_NAME} ${SAMPLE_SOURCES})
            
            # Set CUDA properties
            set_property(TARGET ${SAMPLE_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
            set_property(TARGET ${SAMPLE_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
            set_target_properties(${SAMPLE_NAME} PROPERTIES CUDA_ARCHITECTURES "60;61;70;75;80;86;89")
            
            # Link OpenNPP and CUDA runtime
            target_link_libraries(${SAMPLE_NAME} opennpp CUDA::cudart CUDA::cudart_static)
            
            # Add FreeImage if available and needed
            if(FREEIMAGE_FOUND)
                target_include_directories(${SAMPLE_NAME} PRIVATE ${FREEIMAGE_INCLUDE_DIR})
                target_link_libraries(${SAMPLE_NAME} ${FREEIMAGE_LIBRARY})
                target_compile_definitions(${SAMPLE_NAME} PRIVATE FREEIMAGE_LIB)
            endif()
            
            # Set runtime output directory
            set_target_properties(${SAMPLE_NAME} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
            )
            
            message(STATUS "Added NPP sample: ${SAMPLE_NAME}")
        endif()
    endif()
endfunction()

# Add all NPP samples
add_npp_sample(FilterBorderControlNPP)
add_npp_sample(boxFilterNPP)
add_npp_sample(cannyEdgeDetectorNPP)
add_npp_sample(freeImageInteropNPP)
add_npp_sample(histEqualizationNPP)
add_npp_sample(watershedSegmentationNPP)

# Copy test data to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Common/data DESTINATION ${CMAKE_BINARY_DIR})

# Information message
message(STATUS "========================================")
message(STATUS "CUDA NPP Samples Configuration Summary")
message(STATUS "========================================")
message(STATUS "OpenNPP Include Dir: ${OPENNPP_INCLUDE_DIR}")
message(STATUS "OpenNPP Library Dir: ${OPENNPP_LIBRARY_DIR}")
message(STATUS "FreeImage Found:     ${FREEIMAGE_FOUND}")
message(STATUS "CUDA Version:        ${CUDA_VERSION}")
message(STATUS "========================================")