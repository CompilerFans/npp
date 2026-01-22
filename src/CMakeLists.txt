# NPP core library CMake configuration

# Find source files
npp_find_all_sources(CPU_SOURCES CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR})
# Set .cu files as cxx files to avoid separate CUDA compilation
if(WITHOUT_CU_BRIDGE)
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CXX)
endif()

# =============================================================================
# Main library: libmpp (contains all implementations)
# Target name "npp" is kept for internal CMake references (tests depend on it)
# Output name is "mpp" -> generates libmpp.so/libmpp.a
# =============================================================================
add_library(npp SHARED
    ${CPU_SOURCES}
    ${CUDA_SOURCES}
)
add_library(npp_STATIC STATIC
    ${CPU_SOURCES}
    ${CUDA_SOURCES}
)

# Set NPP target configuration
npp_setup_target(npp)
npp_setup_target(npp_STATIC)

# Add NPP specific include directories
npp_add_common_includes(npp)
npp_add_common_includes(npp_STATIC)

# Set library output name to mpp (generates libmpp.so/libmpp.a)
set_target_properties(npp PROPERTIES
    OUTPUT_NAME "mpp"
    EXPORT_NAME "npp"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)
set_target_properties(npp_STATIC PROPERTIES
    OUTPUT_NAME "mpp_static"
    EXPORT_NAME "npp_static"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)


# Add dynamic library specific configuration if shared
# Windows DLL export symbols
target_compile_definitions(npp PRIVATE MPP_EXPORTS)
target_compile_definitions(npp INTERFACE MPP_IMPORTS)
target_compile_definitions(npp_STATIC PRIVATE MPP_EXPORTS)
target_compile_definitions(npp_STATIC INTERFACE MPP_IMPORTS)

# Set runtime path for dynamic library
if(NOT WIN32)
    set_target_properties(npp PROPERTIES
        INSTALL_RPATH "$ORIGIN:$ORIGIN/../../lib:$ORIGIN/../../lib64:$ORIGIN/../lib64:/opt/maca-${PACK_VERSION}/lib"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
    set_target_properties(npp_STATIC PROPERTIES
        INSTALL_RPATH "$ORIGIN:$ORIGIN/../../lib:$ORIGIN/../../lib64:$ORIGIN/../lib64:/opt/maca-${PACK_VERSION}/lib"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

# Install main library
npp_install_library(npp)
npp_install_library(npp_STATIC)

# =============================================================================
# Stub Libraries (link to libmpp via npp target)
# =============================================================================
set(MPP_STUB_LIBS
    mppc
    mppial
    mppicc
    mppidei
    mppif
    mppig
    mppim
    mppist
    mppisu
    mppitc
    mpps
)

set(MPP_STUB_LIB_PREFIX lib)
set(MPP_STUB_LIB_STATIC_SUFFIX _static.a)
set(MPP_STUB_LIB_SHARED_SUFFIX .so)

foreach(link_name IN LISTS MPP_STUB_LIBS)
    set(link_name_static_full ${MPP_STUB_LIB_PREFIX}${link_name}${MPP_STUB_LIB_STATIC_SUFFIX})
    set(link_name_shared_full ${MPP_STUB_LIB_PREFIX}${link_name}${MPP_STUB_LIB_SHARED_SUFFIX})
    message("link_name_static_full: ${link_name_static_full}")
    message("link_name_shared_full: ${link_name_shared_full}")

    add_custom_command(
        TARGET npp_STATIC
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            $<TARGET_FILE_NAME:npp_STATIC> ${link_name_static_full}
        WORKING_DIRECTORY $<TARGET_FILE_DIR:npp_STATIC>
        COMMENT "Create symlink ${link_name_static_full} -> $<TARGET_FILE_NAME:npp_STATIC>"
    )
    add_custom_command(
        TARGET npp
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            $<TARGET_FILE_NAME:npp> ${link_name_shared_full}
        WORKING_DIRECTORY $<TARGET_FILE_DIR:npp>
        COMMENT "Create symlink ${link_name_shared_full} -> $<TARGET_FILE_NAME:npp>"
    )

    # 安装目标文件
    install(FILES
      ${CMAKE_CURRENT_BINARY_DIR}/${link_name_static_full}
      ${CMAKE_CURRENT_BINARY_DIR}/${link_name_shared_full}

      DESTINATION ${INSTALL_LIBDIR}
      COMPONENT libs
    )
endforeach()