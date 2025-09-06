# NPP单元测试CMake配置

# 单元测试源文件
set(UNIT_TEST_SOURCES
    nppcore/test_nppcore.cpp
    nppi/arithmetic/test_add.cpp
    nppi/arithmetic/test_constants.cpp
    nppi/arithmetic/test_dual_source.cpp
    nppi/data_exchange/test_transpose.cpp
)

# 使用模块化函数创建单元测试
npp_create_test_suite("unit" "${UNIT_TEST_SOURCES}" npp)