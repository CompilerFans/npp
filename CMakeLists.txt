cmake_minimum_required(VERSION 3.18)
project(OpenNPP VERSION 1.0.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找必需包
find_package(CUDAToolkit REQUIRED)

# 启用CUDA语言
enable_language(CUDA)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES 80 89)

# 输出目录设置
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/api)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test/framework)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# 源文件组织
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

# 创建OpenNPP静态库
add_library(opennpp STATIC
    ${CPU_SOURCES}
    ${CUDA_SOURCES}
)

# 链接CUDA库
target_link_libraries(opennpp 
    PRIVATE 
    CUDA::cudart
    CUDA::cuda_driver
)

# 设置编译标志
target_compile_features(opennpp PUBLIC cxx_std_17)
target_compile_options(opennpp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>
)

# 查找NVIDIA NPP库（可选，用于验证测试）
find_library(NPP_LIBRARY_IAL nppial HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/local/cuda/lib64)
find_library(NPP_LIBRARY_CORE nppc HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/local/cuda/lib64)

if(NPP_LIBRARY_IAL AND NPP_LIBRARY_CORE)
    message(STATUS "Found NVIDIA NPP libraries: ${NPP_LIBRARY_IAL}, ${NPP_LIBRARY_CORE}")
    set(HAVE_NVIDIA_NPP TRUE)
else()
    message(WARNING "NVIDIA NPP library not found - validation tests will skip NVIDIA comparison")
    set(HAVE_NVIDIA_NPP FALSE)
endif()

# 测试选项
option(BUILD_TESTS "Build tests" ON)

if(BUILD_TESTS)
    enable_testing()
    
    # 核心功能测试
    add_executable(test_nppcore 
        test/nppcore/test_nppcore.cpp 
        ${NPP_CORE_SOURCES}
    )
    target_link_libraries(test_nppcore PRIVATE CUDA::cudart)
    add_test(NAME test_nppcore COMMAND test_nppcore)
    
    # NPPI内存测试
    add_executable(test_nppi_memory 
        test/test_comprehensive_arithmetic_nppi.cpp 
        ${CPU_SOURCES} 
        ${CUDA_SOURCES}
    )
    target_link_libraries(test_nppi_memory PRIVATE CUDA::cudart)
    if(HAVE_NVIDIA_NPP)
        target_link_libraries(test_nppi_memory PRIVATE ${NPP_LIBRARY_IAL} ${NPP_LIBRARY_CORE})
        target_compile_definitions(test_nppi_memory PRIVATE HAVE_NVIDIA_NPP=1)
    endif()
    add_test(NAME test_nppi_memory COMMAND test_nppi_memory)
    
    # 算术运算综合测试
    add_executable(test_arithmetic 
        test/test_arithmetic_comprehensive.cpp 
        ${CPU_SOURCES} 
        ${CUDA_SOURCES}
    )
    target_link_libraries(test_arithmetic PRIVATE CUDA::cudart)
    if(HAVE_NVIDIA_NPP)
        target_link_libraries(test_arithmetic PRIVATE ${NPP_LIBRARY_IAL} ${NPP_LIBRARY_CORE})
        target_compile_definitions(test_arithmetic PRIVATE HAVE_NVIDIA_NPP=1)
    endif()
    add_test(NAME test_arithmetic COMMAND test_arithmetic)
    
    # 验证测试
    add_executable(test_validation 
        test/test_three_way_validation_v2.cpp 
        ${CPU_SOURCES} 
        ${CUDA_SOURCES}
    )
    target_link_libraries(test_validation PRIVATE CUDA::cudart)
    if(HAVE_NVIDIA_NPP)
        target_link_libraries(test_validation PRIVATE ${NPP_LIBRARY_IAL} ${NPP_LIBRARY_CORE})
        target_compile_definitions(test_validation PRIVATE HAVE_NVIDIA_NPP=1)
    endif()
    add_test(NAME test_validation COMMAND test_validation)
endif()

# 编译器特定选项
if(MSVC)
    target_compile_options(opennpp PRIVATE /W4)
else()
    target_compile_options(opennpp PRIVATE -Wall -Wextra -Wpedantic)
endif()