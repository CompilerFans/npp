# NPP 功能单元测试 CMake配置
cmake_minimum_required(VERSION 3.18)

project(npp_functional_tests LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA设置
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70;75;80;86)

# 查找依赖
find_package(CUDAToolkit REQUIRED)

# 使用项目中的GoogleTest
if(NOT TARGET gtest)
    add_subdirectory(../../third_party/googletest googletest)
endif()

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../..  # npp根目录
    ${CMAKE_CURRENT_SOURCE_DIR}/../../api  # NPP API头文件
    ${CUDAToolkit_INCLUDE_DIRS}
)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 -lineinfo")

# 测试框架库
set(FUNCTIONAL_TEST_FRAMEWORK_SOURCES
    framework/npp_test_base.h
)

# 功能测试源文件
set(FUNCTIONAL_TEST_SOURCES
    arithmetic/test_add_functional.cpp
    arithmetic/test_arithmetic_constants_functional.cpp
    arithmetic/test_arithmetic_dual_source_functional.cpp
    core/test_npp_core_functional.cpp
    data_exchange/test_transpose_functional.cpp
)

# NPP库源文件（需要链接的实现）
set(NPP_LIB_SOURCES
    ../../src/nppcore/nppcore.cpp
    ../../src/nppi/nppi_support_functions.cpp
    # 算术运算
    ../../src/nppi/nppi_arithmetic_operations/nppi_add.cpp
    ../../src/nppi/nppi_arithmetic_operations/nppi_add.cu
    ../../src/nppi/nppi_arithmetic_operations/nppi_addc.cpp
    ../../src/nppi/nppi_arithmetic_operations/nppi_addc.cu
    ../../src/nppi/nppi_arithmetic_operations/nppi_sub.cpp
    ../../src/nppi/nppi_arithmetic_operations/nppi_sub.cu
    ../../src/nppi/nppi_arithmetic_operations/nppi_subc.cpp
    ../../src/nppi/nppi_arithmetic_operations/nppi_subc.cu
    ../../src/nppi/nppi_arithmetic_operations/nppi_mul.cpp
    ../../src/nppi/nppi_arithmetic_operations/nppi_mul.cu
    ../../src/nppi/nppi_arithmetic_operations/nppi_mulc.cpp
    ../../src/nppi/nppi_arithmetic_operations/nppi_mulc.cu
    ../../src/nppi/nppi_arithmetic_operations/nppi_div.cpp
    ../../src/nppi/nppi_arithmetic_operations/nppi_div.cu
    ../../src/nppi/nppi_arithmetic_operations/nppi_divc.cpp
    ../../src/nppi/nppi_arithmetic_operations/nppi_divc.cu
    # 数据交换
    ../../src/nppi/nppi_data_exchange_operations/nppi_transpose.cpp
    ../../src/nppi/nppi_data_exchange_operations/nppi_transpose.cu
)

# 创建NPP功能测试可执行文件
add_executable(npp_functional_tests
    ${FUNCTIONAL_TEST_SOURCES}
    ${NPP_LIB_SOURCES}
)

# 设置CUDA分离编译
set_property(TARGET npp_functional_tests PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# 链接库
target_link_libraries(npp_functional_tests
    gtest
    gtest_main
    CUDA::cudart
)

# 添加测试
enable_testing()
add_test(NAME NPP_Functional_Tests COMMAND npp_functional_tests)

# 创建各个子模块的测试目标
add_executable(npp_arithmetic_functional_tests
    arithmetic/test_add_functional.cpp
    ${NPP_LIB_SOURCES}
)

set_property(TARGET npp_arithmetic_functional_tests PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(npp_arithmetic_functional_tests
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME NPP_Arithmetic_Functional_Tests COMMAND npp_arithmetic_functional_tests)

# 安装规则（可选）
install(TARGETS npp_functional_tests
    RUNTIME DESTINATION bin
)

# 打印配置信息
message(STATUS "NPP Functional Tests Configuration:")
message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  GTest Available: Available")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")