# NPP单元测试CMake配置
# 基于API头文件结构组织的测试架构

# 递归查找所有测试文件
file(GLOB_RECURSE UNIT_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/nppi/test_*.cpp

    # npps diff by version. comment temp
    # cuda 12.8
    #           Npp32fc * nppsMalloc_32fc(size_t nSize);
    # cuda 12.2/11.4
    #           Npp32fc * nppsMalloc_32fc(int nSize);
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
npp_create_test_suite("unit" "${UNIT_TEST_SOURCES}" npp)

# 为单元测试添加特定的包含路径
target_include_directories(unit_tests
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/framework   # test framework
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src   # src root
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/include    # src include (npp_version_compat.h, etc.)
    ${CMAKE_CURRENT_SOURCE_DIR}             # ut root
)

# NPP 13 removed the non-Ctx entry points, so the full suite cannot link against
# the reference NPP.  The stream tests only use Ctx entry points; build them as
# a separate target so they can still run against CUDA SDK NPP.
if(USE_NVIDIA_NPP)
    npp_create_test_target(stream_sync_tests
        "${CMAKE_CURRENT_SOURCE_DIR}/nppi/test_npp_stream_sync.cpp"
        npp)
    target_include_directories(stream_sync_tests
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/framework
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    npp_register_test(stream_sync stream_sync_tests)
endif()