# NPP单元测试CMake配置
# 基于API头文件结构组织的测试架构

# 递归查找所有测试文件
file(GLOB_RECURSE UNIT_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/nppi/test_*.cpp
    # disable all npps temp
    #${CMAKE_CURRENT_SOURCE_DIR}/npps/test_*.cpp
)

# 显示找到的测试文件 (按模块分组)
message(STATUS "Found unit test files by module:")

# 按模块分类显示
foreach(test_file ${UNIT_TEST_SOURCES})
    file(RELATIVE_PATH rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${test_file})
    get_filename_component(module_dir ${rel_path} DIRECTORY)
    message(STATUS "  [${module_dir}] ${rel_path}")
endforeach()

# 使用模块化函数创建单元测试
# 注意：当USE_NVIDIA_NPP=ON时，library_target参数将被忽略
# 传递一个占位符"NVIDIA"而不是空字符串，避免CMake参数解析问题
if(USE_NVIDIA_NPP)
    npp_create_test_suite("unit" "${UNIT_TEST_SOURCES}" "NVIDIA_NPP_PLACEHOLDER")
else()
    npp_create_test_suite("unit" "${UNIT_TEST_SOURCES}" npp)
endif()

# 为单元测试添加特定的包含路径
# 注意：src目录始终需要包含，因为测试框架依赖于src中的一些头文件(如npp_version_compat.h)
target_include_directories(unit_tests
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/framework      # 测试框架
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src      # 源码目录 (头文件依赖)
    ${CMAKE_CURRENT_SOURCE_DIR}                 # 单元测试根目录
)

# Define golden data directory path
target_compile_definitions(unit_tests
    PRIVATE
    GOLDEN_DATA_DIR="${CMAKE_SOURCE_DIR}/test/golden"
)