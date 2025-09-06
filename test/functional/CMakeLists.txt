# NPP功能测试CMake配置

# 功能测试源文件
set(FUNCTIONAL_TEST_SOURCES
    arithmetic/test_add_functional.cpp
    arithmetic/test_arithmetic_constants_functional.cpp
    arithmetic/test_arithmetic_dual_source_functional.cpp
    core/test_npp_core_functional.cpp
    data_exchange/test_transpose_functional.cpp
)

# 使用模块化函数创建功能测试
npp_create_test_suite("functional" "${FUNCTIONAL_TEST_SOURCES}" npp)