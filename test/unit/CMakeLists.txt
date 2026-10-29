# NPP单元测试CMake配置
# 基于API头文件结构组织的测试架构

# NVIDIA NPP 13 removed the non-Ctx entry points, while most existing tests
# call them directly.  The full suite is therefore built only for MPP/MACA;
# the reference build uses the Ctx-only stream and differential targets.
if(NOT USE_NVIDIA_NPP)
    # 递归查找所有测试文件
    # The legacy npps tests are enabled per family by the NPPS work packages
    # (P1.1-P1.3): they currently reference APIs that are not implemented yet,
    # so globbing them here would break the link of the whole suite.
    file(GLOB_RECURSE UNIT_TEST_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/nppi/test_*.cpp
    )
endif()

# 显示找到的测试文件 (按模块分组)
message(STATUS "Found unit test files by module:")

# 按模块分类显示
foreach(test_file ${UNIT_TEST_SOURCES})
    file(RELATIVE_PATH rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${test_file})
    get_filename_component(module_dir ${rel_path} DIRECTORY)
    message(STATUS "  [${module_dir}] ${rel_path}")
endforeach()

if(NOT USE_NVIDIA_NPP)
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
endif()

# Differential tests (record on NVIDIA NPP, verify on MPP).  Only Ctx entry
# points are referenced by generated tests, so the target builds against the
# reference NPP 13 headers as well.
file(GLOB_RECURSE DIFF_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/diff/test_*_generated.cpp
)
if(DIFF_TEST_SOURCES)
    npp_create_test_target(diff_tests "${DIFF_TEST_SOURCES}" npp)
    target_include_directories(diff_tests
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/framework
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    npp_register_test(diff diff_tests)
endif()

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
# Generated reference tests (tools/gen_ref_tests.py).  They call only _Ctx
# entry points, so the same target builds against MPP and against the NVIDIA
# NPP 13 reference.
file(GLOB_RECURSE REF_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ref/test_*_generated.cpp
)
if(REF_TEST_SOURCES)
    npp_create_test_target(ref_tests "${REF_TEST_SOURCES}" npp)
    target_include_directories(ref_tests
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/framework
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/ref
    )
    npp_register_test(ref ref_tests)
endif()
