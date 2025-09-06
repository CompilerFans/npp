# NPP测试配置功能

# 查找并配置GoogleTest
function(npp_setup_gtest)
    find_package(GTest QUIET)
    
    if(NOT GTest_FOUND)
        message(STATUS "Using in-tree GoogleTest")
        add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/googletest googletest EXCLUDE_FROM_ALL)
    else()
        message(STATUS "Using system GoogleTest")
    endif()
endfunction()

# 创建NPP测试目标
function(npp_create_test_target target_name sources library_target)
    add_executable(${target_name} ${sources})
    
    # 链接库
    target_link_libraries(${target_name}
        PRIVATE
        ${library_target}
        gtest
        gtest_main
    )
    
    # 设置通用配置
    npp_setup_target(${target_name})
    
    # 添加包含目录
    target_include_directories(${target_name} 
        PRIVATE
        ${CMAKE_SOURCE_DIR}     # 项目根目录
    )
endfunction()

# 注册CTest测试
function(npp_register_test test_name target_name)
    add_test(NAME ${test_name} COMMAND ${target_name})
endfunction()

# 创建完整的测试套件
function(npp_create_test_suite suite_name test_sources library_target)
    set(target_name "${suite_name}_tests")
    
    npp_create_test_target(${target_name} "${test_sources}" ${library_target})
    npp_register_test(${suite_name} ${target_name})
    
    message(STATUS "Created test suite: ${suite_name}")
endfunction()