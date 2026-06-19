# NPP项目公共配置

include(GNUInstallDirs)

# 设置通用编译标准
function(npp_set_common_standards target)
    # MACA and not use cu-bridge, use maca c directly
    if(NOT WITHOUT_CU_BRIDGE)
        set_target_properties(${target} PROPERTIES
	     CUDA_SEPARABLE_COMPILATION OFF
        )
    endif()
endfunction()

# 添加NPP通用包含目录
function(npp_add_common_includes target)

    # NPP 源码库头文件include
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/API>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    # SDK 头文件include
    target_include_directories(${target} PRIVATE ${NPP_SDK_LIB_DIR})
endfunction()

function(npp_link_libs target)
  target_link_directories(${target} PUBLIC ${NPP_SDK_LIB_DIR})
  target_link_libraries(${target} PUBLIC ${NPP_SDK_LINK_LIBS})
endfunction()

# 设置完整的NPP目标配置
function(npp_setup_target target)
    npp_set_common_standards(${target})
    npp_link_libs(${target})
endfunction()
