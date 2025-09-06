# NPP安装配置

include(GNUInstallDirs)

# 设置默认安装前缀到build目录下的install子目录
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Install path prefix" FORCE)
endif()

# 定义安装路径
set(INSTALL_BINDIR ${CMAKE_INSTALL_BINDIR})
set(INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR})
set(INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}/opennpp)
set(INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/OpenNPP)
set(INSTALL_SAMPLEDIR ${CMAKE_INSTALL_DOCDIR}/examples)

# 安装API头文件
function(npp_install_headers)
    # 安装所有API头文件
    file(GLOB_RECURSE API_HEADERS "${CMAKE_SOURCE_DIR}/API/*.h")
    
    foreach(header ${API_HEADERS})
        file(RELATIVE_PATH rel_path ${CMAKE_SOURCE_DIR}/API ${header})
        get_filename_component(header_dir ${rel_path} DIRECTORY)
        
        if(header_dir)
            install(FILES ${header}
                DESTINATION ${INSTALL_INCLUDEDIR}/${header_dir})
        else()
            install(FILES ${header}
                DESTINATION ${INSTALL_INCLUDEDIR})
        endif()
    endforeach()
endfunction()

# 安装库文件
function(npp_install_library target)
    # 安装目标文件
    install(TARGETS ${target}
        EXPORT OpenNPPTargets
        RUNTIME DESTINATION ${INSTALL_BINDIR}
        LIBRARY DESTINATION ${INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${INSTALL_LIBDIR}
        INCLUDES DESTINATION ${INSTALL_INCLUDEDIR}
    )
endfunction()

# 生成并安装CMake配置文件
function(npp_install_cmake_config)
    # 生成版本文件
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/OpenNPPConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
    )
    
    # 生成配置文件
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/OpenNPPConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/OpenNPPConfig.cmake"
        INSTALL_DESTINATION ${INSTALL_CMAKEDIR}
        PATH_VARS INSTALL_INCLUDEDIR INSTALL_LIBDIR
    )
    
    # 安装导出文件
    install(EXPORT OpenNPPTargets
        FILE OpenNPPTargets.cmake
        NAMESPACE OpenNPP::
        DESTINATION ${INSTALL_CMAKEDIR}
    )
    
    # 安装配置文件
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/OpenNPPConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/OpenNPPConfigVersion.cmake"
        DESTINATION ${INSTALL_CMAKEDIR}
    )
endfunction()

# 安装文档和示例
function(npp_install_docs_examples)
    # 安装README等文档
    if(EXISTS "${CMAKE_SOURCE_DIR}/README.md")
        install(FILES "${CMAKE_SOURCE_DIR}/README.md"
            DESTINATION ${CMAKE_INSTALL_DOCDIR}
        )
    endif()
    
    # 安装CLAUDE.md
    if(EXISTS "${CMAKE_SOURCE_DIR}/CLAUDE.md")
        install(FILES "${CMAKE_SOURCE_DIR}/CLAUDE.md"
            DESTINATION ${CMAKE_INSTALL_DOCDIR}
        )
    endif()
    
    # 安装examples源码
    if(EXISTS "${CMAKE_SOURCE_DIR}/examples")
        install(DIRECTORY "${CMAKE_SOURCE_DIR}/examples/"
            DESTINATION ${INSTALL_SAMPLEDIR}
            FILES_MATCHING 
                PATTERN "*.cpp" 
                PATTERN "*.cu" 
                PATTERN "*.h" 
                PATTERN "CMakeLists.txt"
            PATTERN "build*" EXCLUDE
        )
    endif()
endfunction()

# 安装示例可执行文件
function(npp_install_samples)
    # 如果构建了示例，也安装可执行文件到bin目录
    if(BUILD_EXAMPLES AND TARGET basic_add_example)
        install(TARGETS basic_add_example
            RUNTIME DESTINATION ${INSTALL_BINDIR}
            COMPONENT samples
        )
    endif()
endfunction()

# 主安装函数
function(npp_configure_install)
    # 安装头文件
    npp_install_headers()
    
    # 安装库文件 - 这会在src/CMakeLists.txt中调用
    # npp_install_library()
    
    # 安装CMake配置
    npp_install_cmake_config()
    
    # 安装文档和示例
    npp_install_docs_examples()
    
    # 安装示例可执行文件
    npp_install_samples()
    
    # 生成卸载脚本
    if(NOT TARGET uninstall)
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
            "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
            IMMEDIATE @ONLY)
        
        add_custom_target(uninstall
            COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
    endif()
    
    # 添加install目标的打印信息
    message(STATUS "OpenNPP will be installed to: ${CMAKE_INSTALL_PREFIX}")
    message(STATUS "  - Headers: ${CMAKE_INSTALL_PREFIX}/${INSTALL_INCLUDEDIR}")
    message(STATUS "  - Libraries: ${CMAKE_INSTALL_PREFIX}/${INSTALL_LIBDIR}")
    message(STATUS "  - Samples: ${CMAKE_INSTALL_PREFIX}/${INSTALL_SAMPLEDIR}")
    message(STATUS "  - Binaries: ${CMAKE_INSTALL_PREFIX}/${INSTALL_BINDIR}")
endfunction()