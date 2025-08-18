cmake_minimum_required(VERSION 3.18)
project(OpenNPP VERSION 1.0.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 启用测试
enable_testing()

# 添加Google Test
add_subdirectory(third_party/googletest)

# 查找必需包
find_package(CUDAToolkit REQUIRED)

# 启用CUDA语言
enable_language(CUDA)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES 80 89)

# 编译选项
# 对C++文件设置合理的警告级别
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-parameter")

# CUDA编译选项
# 注意：'style of line directive is a GCC extension'警告来自CUDA编译器
# 处理原始NVIDIA NPP头文件时的正常行为，这些警告是无害的
# 由于不能修改原始API头文件，我们选择性地抑制这些警告
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

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
    
    
    # ========== Google Test 测试 - 统一测试可执行程序 ==========
    
    # 设置所有测试源文件
    set(TEST_SOURCES
        test/gtest/nppcore/test_nppcore.cpp
        test/gtest/nppcore/test_nppcore_cuda_comparison.cpp
        test/gtest/nppi/nppi_arithmetic_operations/test_nppi_arithmetic.cpp
        test/gtest/nppi/nppi_support_functions/test_nppi_support.cpp
        test/gtest/nppi/nppi_support_functions/test_nppi_support_complete_coverage.cpp
    )
    
    # 如果有NVIDIA NPP，添加验证测试
    if(HAVE_NVIDIA_NPP)
        list(APPEND TEST_SOURCES 
            test/gtest/validation/test_consistency_simple.cpp
            test/gtest/nppi/nppi_support_functions/test_nppi_support_nvidia_comparison.cpp
        )
    endif()
    
    # 创建统一的测试可执行程序
    add_executable(opennpp_test 
        ${TEST_SOURCES}
        ${NPP_CORE_SOURCES}
        ${NPPI_SUPPORT_SOURCES}
        ${CPU_SOURCES}
        ${CUDA_SOURCES}
    )
    
    # 链接库
    target_link_libraries(opennpp_test PRIVATE 
        gtest 
        gtest_main 
        CUDA::cudart
    )
    
    # 包含目录
    target_include_directories(opennpp_test PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/googletest/include
    )
    
    # 如果有NVIDIA NPP，添加相关链接和定义
    if(HAVE_NVIDIA_NPP)
        target_link_libraries(opennpp_test PRIVATE 
            ${NPP_LIBRARY_IAL} 
            ${NPP_LIBRARY_CORE}
        )
        target_compile_definitions(opennpp_test PRIVATE HAVE_NVIDIA_NPP=1)
    endif()
    
    # 添加不同的测试过滤器
    add_test(NAME all_tests COMMAND opennpp_test)
    add_test(NAME core_tests COMMAND opennpp_test --gtest_filter=NPP*:*Properties*:*Stream*:*Error*:*Device*)
    add_test(NAME arithmetic_tests COMMAND opennpp_test --gtest_filter=*AddC*:*SubC*:*MulC*:*DivC*)
    add_test(NAME support_tests COMMAND opennpp_test --gtest_filter=*Memory*:*Support*)
    add_test(NAME quick_tests COMMAND opennpp_test --gtest_filter=-*Performance*:-*Stress*:-*Consistency*)
    
    if(HAVE_NVIDIA_NPP)
        add_test(NAME validation_tests COMMAND opennpp_test --gtest_filter=*Consistency*)
    endif()
    
    # ========== 统一测试目标 ==========
    # 设置测试目标列表
    set(ALL_TEST_TARGETS opennpp_test)
    
    # 添加test_all目标来运行所有测试
    add_custom_target(test_all
        COMMAND ${CMAKE_CTEST_COMMAND} -R all_tests --output-on-failure --verbose
        DEPENDS ${ALL_TEST_TARGETS}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running all OpenNPP tests..."
    )
    
    # 添加快速测试目标（不运行性能和压力测试）
    add_custom_target(test_quick
        COMMAND ${CMAKE_CTEST_COMMAND} -R quick_tests --output-on-failure
        DEPENDS ${ALL_TEST_TARGETS}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running quick OpenNPP tests..."
    )
    
    # 添加特定模块的测试目标
    add_custom_target(test_core
        COMMAND ${CMAKE_CTEST_COMMAND} -R core_tests --output-on-failure
        DEPENDS ${ALL_TEST_TARGETS}
        COMMENT "Running NPP Core tests..."
    )
    
    add_custom_target(test_arithmetic
        COMMAND ${CMAKE_CTEST_COMMAND} -R arithmetic_tests --output-on-failure
        DEPENDS ${ALL_TEST_TARGETS}
        COMMENT "Running NPPI Arithmetic tests..."
    )
    
    add_custom_target(test_support
        COMMAND ${CMAKE_CTEST_COMMAND} -R support_tests --output-on-failure
        DEPENDS ${ALL_TEST_TARGETS}
        COMMENT "Running NPPI Support Function tests..."
    )
    
    if(HAVE_NVIDIA_NPP)
        add_custom_target(test_validation
            COMMAND ${CMAKE_CTEST_COMMAND} -R validation_tests --output-on-failure
            DEPENDS ${ALL_TEST_TARGETS}
            COMMENT "Running OpenNPP vs NVIDIA NPP consistency tests..."
        )
    endif()
    
    # 生成测试报告的目标
    add_custom_target(test_report
        COMMAND opennpp_test --gtest_output=xml:test_results.xml
        COMMAND ${CMAKE_COMMAND} -E echo "Test report generated: test_results.xml"
        DEPENDS ${ALL_TEST_TARGETS}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating test report..."
    )
    
    # 运行特定测试的辅助目标（使用GTest过滤器）
    add_custom_target(test_filter
        COMMAND opennpp_test --gtest_filter="${GTEST_FILTER}"
        DEPENDS ${ALL_TEST_TARGETS}
        COMMENT "Running tests with filter: ${GTEST_FILTER}"
    )
endif()

# 编译器特定选项
if(MSVC)
    target_compile_options(opennpp PRIVATE /W4)
else()
    target_compile_options(opennpp PRIVATE -Wall -Wextra -Wpedantic)
endif()