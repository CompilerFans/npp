# NPP功能测试CMake配置

# 功能测试源文件
set(FUNCTIONAL_TEST_SOURCES
    arithmetic/test_add_functional.cpp
    arithmetic/test_arithmetic_constants_functional.cpp
    arithmetic/test_arithmetic_dual_source_functional.cpp
    core/test_npp_core_functional.cpp
    data_exchange/test_transpose_functional.cpp
)

# 创建NPP功能测试可执行文件
add_executable(npp_functional_tests ${FUNCTIONAL_TEST_SOURCES})

# 链接NPP库和测试依赖
target_link_libraries(npp_functional_tests
    PRIVATE
    npp                 # src中构建的NPP库
    gtest
    gtest_main
)

# 设置包含目录
target_include_directories(npp_functional_tests 
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..     # 项目根目录
)

# CUDA分离编译
set_property(TARGET npp_functional_tests PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# 添加测试
add_test(NAME NPP_Functional_Tests COMMAND npp_functional_tests)