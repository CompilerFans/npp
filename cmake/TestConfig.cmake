# NPP测试配置功能

# 查找并配置GoogleTest
function(npp_setup_gtest)
    # 首先检查是否有源码版本的GoogleTest
    if(EXISTS ${CMAKE_SOURCE_DIR}/third_party/googletest/CMakeLists.txt)
        message(STATUS "Using in-tree GoogleTest source")
        # 配置GoogleTest选项
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        set(BUILD_GTEST ON CACHE BOOL "" FORCE)
        
        # 添加GoogleTest源码子目录
        add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/googletest googletest EXCLUDE_FROM_ALL)
        return()
    endif()
    
    # 尝试查找系统安装的GoogleTest
    find_package(GTest QUIET)
    
    if(NOT GTest_FOUND)
        message(STATUS "System GoogleTest not found. Using in-tree GoogleTest.")
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
    
    # 设置输出目录到build根目录
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
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