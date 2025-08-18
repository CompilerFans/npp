cmake_minimum_required(VERSION 3.18)
project(OpenNPP VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Enable CUDA language
enable_language(CUDA)

# Set CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 80 89)

# CMake配置：输出目录设置
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/api)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test/framework)
include_directories(${CUDAToolkit_INCLUDE_DIRS})


# Source files organized by NPP module structure
set(NPP_CORE_SOURCES
    src/nppcore/nppcore.cpp
)

set(NPPI_SUPPORT_SOURCES
    src/nppi/nppi_support_functions.cpp
)

set(NPPI_ARITHMETIC_SOURCES
    src/nppi/nppi_arithmetic_operations/nppi_addc.cpp
    src/nppi/nppi_arithmetic_operations/nppi_subc.cpp
    src/nppi/nppi_arithmetic_operations/nppi_mulc.cpp
    src/nppi/nppi_arithmetic_operations/nppi_divc.cpp
    src/nppi/nppi_arithmetic_operations/nppi_arithmetic_reference.cpp
)

set(TEST_SOURCES
    test/nppi_arithmetic/test_nppi_addc.cpp
)

set(NPPI_ARITHMETIC_CUDA_SOURCES
    src/nppi/nppi_arithmetic_operations/nppi_addc.cu
    src/nppi/nppi_arithmetic_operations/nppi_subc.cu
    src/nppi/nppi_arithmetic_operations/nppi_mulc.cu
    src/nppi/nppi_arithmetic_operations/nppi_divc.cu
)

set(CPU_SOURCES
    ${NPP_CORE_SOURCES}
    ${NPPI_SUPPORT_SOURCES}
    ${NPPI_ARITHMETIC_SOURCES}
)

set(CUDA_SOURCES
    ${NPPI_ARITHMETIC_CUDA_SOURCES}
)

set(TEST_FRAMEWORK_SOURCES
    test/framework/npp_test_framework.h
    test/framework/test_nppi_addc_8u_c1rsfs.h
    test/framework/test_report.h
)

# Create static library for our NPP implementation
add_library(open_npp STATIC
    ${CPU_SOURCES}
    ${CUDA_SOURCES}
)

# Link CUDA libraries
target_link_libraries(open_npp 
    PRIVATE 
    CUDA::cudart
    CUDA::cuda_driver
)

# Set compile flags
target_compile_features(open_npp PUBLIC cxx_std_17)
target_compile_options(open_npp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>
)

# NPP通用测试运行器
add_executable(npp_test_runner test/npp_test_runner.cpp ${CPU_SOURCES} ${CUDA_SOURCES} ${TEST_SOURCES})
target_link_libraries(npp_test_runner PRIVATE CUDA::cudart)
target_compile_features(npp_test_runner PRIVATE cxx_std_17)
target_compile_options(npp_test_runner PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# Legacy NPP Core test  
add_executable(test_nppcore test/nppcore/test_nppcore.cpp ${NPP_CORE_SOURCES})
target_link_libraries(test_nppcore PRIVATE CUDA::cudart)
target_compile_features(test_nppcore PRIVATE cxx_std_17)

# Legacy NPPI Support Functions test
add_executable(test_nppi_support test/nppi/test_nppi_support_functions.cpp ${NPPI_SUPPORT_SOURCES})
target_link_libraries(test_nppi_support PRIVATE CUDA::cudart)
target_compile_features(test_nppi_support PRIVATE cxx_std_17)

# Simple SubC test
add_executable(test_simple_subc test/test_simple_subc.cpp ${CPU_SOURCES} ${CUDA_SOURCES})
target_link_libraries(test_simple_subc PRIVATE CUDA::cudart)
target_compile_features(test_simple_subc PRIVATE cxx_std_17)
target_compile_options(test_simple_subc PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# Simple MulC test
add_executable(test_simple_mulc test/test_simple_mulc.cpp ${CPU_SOURCES} ${CUDA_SOURCES})
target_link_libraries(test_simple_mulc PRIVATE CUDA::cudart)
target_compile_features(test_simple_mulc PRIVATE cxx_std_17)
target_compile_options(test_simple_mulc PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# Simple DivC test
add_executable(test_simple_divc test/test_simple_divc.cpp ${CPU_SOURCES} ${CUDA_SOURCES})
target_link_libraries(test_simple_divc PRIVATE CUDA::cudart)
target_compile_features(test_simple_divc PRIVATE cxx_std_17)
target_compile_options(test_simple_divc PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# Validation tests
add_executable(test_nppcore_validation test/validation/test_nppcore_validation.cpp ${CPU_SOURCES} ${CUDA_SOURCES})
target_link_libraries(test_nppcore_validation PRIVATE CUDA::cudart)
target_compile_features(test_nppcore_validation PRIVATE cxx_std_17)

add_executable(test_nppi_support_validation test/validation/test_nppi_support_validation.cpp ${CPU_SOURCES} ${CUDA_SOURCES})
target_link_libraries(test_nppi_support_validation PRIVATE CUDA::cudart)
target_compile_features(test_nppi_support_validation PRIVATE cxx_std_17)

add_executable(test_nppi_arithmetic_simple test/validation/test_nppi_arithmetic_simple.cpp ${CPU_SOURCES} ${CUDA_SOURCES})
target_link_libraries(test_nppi_arithmetic_simple PRIVATE CUDA::cudart)
target_compile_features(test_nppi_arithmetic_simple PRIVATE cxx_std_17)



target_link_libraries(npp_test_runner PRIVATE open_npp)


# Link against actual NPP for validation
find_library(NPP_LIBRARY_IAL nppial HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/local/cuda/lib64)
find_library(NPP_LIBRARY_CORE nppc HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/local/cuda/lib64)

if(NPP_LIBRARY_IAL AND NPP_LIBRARY_CORE)
    message(STATUS "Found NVIDIA NPP libraries: ${NPP_LIBRARY_IAL}, ${NPP_LIBRARY_CORE}")
    
    # Link NPP libraries to test runner
    target_link_libraries(npp_test_runner PRIVATE 
        ${NPP_LIBRARY_IAL} 
        ${NPP_LIBRARY_CORE}
    )
    target_compile_definitions(npp_test_runner PRIVATE HAVE_NVIDIA_NPP=1)
    
    # Link NPP libraries to validation tests
    target_link_libraries(test_nppcore_validation PRIVATE 
        ${NPP_LIBRARY_IAL} 
        ${NPP_LIBRARY_CORE}
    )
    target_compile_definitions(test_nppcore_validation PRIVATE HAVE_NVIDIA_NPP=1)
    
    target_link_libraries(test_nppi_support_validation PRIVATE 
        ${NPP_LIBRARY_IAL} 
        ${NPP_LIBRARY_CORE}
    )
    target_compile_definitions(test_nppi_support_validation PRIVATE HAVE_NVIDIA_NPP=1)
    
    target_link_libraries(test_nppi_arithmetic_simple PRIVATE 
        ${NPP_LIBRARY_IAL} 
        ${NPP_LIBRARY_CORE}
    )
    target_compile_definitions(test_nppi_arithmetic_simple PRIVATE HAVE_NVIDIA_NPP=1)

    # Simple validation test
    add_executable(test_simple_validation test/test_simple_validation.cpp ${CPU_SOURCES} ${CUDA_SOURCES})
    target_link_libraries(test_simple_validation PRIVATE 
        CUDA::cudart
        ${NPP_LIBRARY_IAL} 
        ${NPP_LIBRARY_CORE}
    )
    target_compile_definitions(test_simple_validation PRIVATE HAVE_NVIDIA_NPP=1)
    target_compile_features(test_simple_validation PRIVATE cxx_std_17)
    
    # Comprehensive arithmetic test
    add_executable(test_comprehensive_arithmetic test/test_comprehensive_arithmetic.cpp ${CPU_SOURCES} ${CUDA_SOURCES})
    target_link_libraries(test_comprehensive_arithmetic PRIVATE 
        CUDA::cudart
        ${NPP_LIBRARY_IAL} 
        ${NPP_LIBRARY_CORE}
    )
    target_compile_definitions(test_comprehensive_arithmetic PRIVATE HAVE_NVIDIA_NPP=1)
    target_compile_features(test_comprehensive_arithmetic PRIVATE cxx_std_17)
    
    # Comprehensive arithmetic test with NPPI memory allocation
    add_executable(test_comprehensive_arithmetic_nppi test/test_comprehensive_arithmetic_nppi.cpp ${CPU_SOURCES} ${CUDA_SOURCES})
    target_link_libraries(test_comprehensive_arithmetic_nppi PRIVATE 
        CUDA::cudart
        ${NPP_LIBRARY_IAL} 
        ${NPP_LIBRARY_CORE}
    )
    target_compile_definitions(test_comprehensive_arithmetic_nppi PRIVATE HAVE_NVIDIA_NPP=1)
    target_compile_features(test_comprehensive_arithmetic_nppi PRIVATE cxx_std_17)
    
    # Multichannel arithmetic test
    add_executable(test_multichannel_arithmetic test/test_multichannel_arithmetic.cpp ${CPU_SOURCES} ${CUDA_SOURCES})
    target_link_libraries(test_multichannel_arithmetic PRIVATE 
        CUDA::cudart
        ${NPP_LIBRARY_IAL} 
        ${NPP_LIBRARY_CORE}
    )
    target_compile_definitions(test_multichannel_arithmetic PRIVATE HAVE_NVIDIA_NPP=1)
    target_compile_features(test_multichannel_arithmetic PRIVATE cxx_std_17)
    
    # Three-way validation test
    add_executable(test_three_way_validation test/test_three_way_validation.cpp ${CPU_SOURCES} ${CUDA_SOURCES})
    target_link_libraries(test_three_way_validation PRIVATE 
        CUDA::cudart
        ${NPP_LIBRARY_IAL} 
        ${NPP_LIBRARY_CORE}
    )
    target_compile_definitions(test_three_way_validation PRIVATE HAVE_NVIDIA_NPP=1)
    target_compile_features(test_three_way_validation PRIVATE cxx_std_17)
    
else()
    message(WARNING "NVIDIA NPP library not found - validation tests will skip NVIDIA comparison")
    target_compile_definitions(npp_test_runner PRIVATE HAVE_NVIDIA_NPP=0)
    target_compile_definitions(test_nppcore_validation PRIVATE HAVE_NVIDIA_NPP=0)
    target_compile_definitions(test_nppi_support_validation PRIVATE HAVE_NVIDIA_NPP=0)
    target_compile_definitions(test_nppi_arithmetic_simple PRIVATE HAVE_NVIDIA_NPP=0)
    
    # Simple validation test without NVIDIA NPP
    add_executable(test_simple_validation test/test_simple_validation.cpp ${CPU_SOURCES} ${CUDA_SOURCES})
    target_link_libraries(test_simple_validation PRIVATE CUDA::cudart)
    target_compile_definitions(test_simple_validation PRIVATE HAVE_NVIDIA_NPP=0)
    target_compile_features(test_simple_validation PRIVATE cxx_std_17)
    
    # Comprehensive arithmetic test without NVIDIA NPP
    add_executable(test_comprehensive_arithmetic test/test_comprehensive_arithmetic.cpp ${CPU_SOURCES} ${CUDA_SOURCES})
    target_link_libraries(test_comprehensive_arithmetic PRIVATE CUDA::cudart)
    target_compile_definitions(test_comprehensive_arithmetic PRIVATE HAVE_NVIDIA_NPP=0)
    target_compile_features(test_comprehensive_arithmetic PRIVATE cxx_std_17)
endif()

# Compiler-specific options
if(MSVC)
    target_compile_options(open_npp PRIVATE /W4)
    # target_compile_options(test_runner PRIVATE /W4)
else()
    target_compile_options(open_npp PRIVATE -Wall -Wextra -Wpedantic)
    # target_compile_options(test_runner PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Enable testing
enable_testing()
add_test(NAME npp_test_runner COMMAND npp_test_runner)
add_test(NAME npp_test_arithmetic COMMAND npp_test_runner nppi_arithmetic_and_logical_operations)
add_test(NAME test_nppcore COMMAND test_nppcore)
add_test(NAME test_nppi_support COMMAND test_nppi_support)
