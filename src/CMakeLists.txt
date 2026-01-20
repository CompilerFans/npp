# NPP core library CMake configuration
#
# =============================================================================
# Library Structure for NPP Compatibility
# =============================================================================
# libmpp.so/a contains all implementations (corresponds to libnppc.so)
# Other libraries are stubs that link to libmpp:
#
# NVIDIA NPP Libraries -> MPP Libraries:
#   libnppc.so    -> libmppc.so     (Core - contains ALL implementations)
#   libnppial.so  -> libmppial.so   (stub -> libmppc)
#   libnppicc.so  -> libmppicc.so   (stub -> libmppc)
#   libnppidei.so -> libmppidei.so  (stub -> libmppc)
#   libnppif.so   -> libmppif.so    (stub -> libmppc)
#   libnppig.so   -> libmppig.so    (stub -> libmppc)
#   libnppim.so   -> libmppim.so    (stub -> libmppc)
#   libnppist.so  -> libmppist.so   (stub -> libmppc)
#   libnppisu.so  -> libmppisu.so   (stub -> libmppc)
#   libnppitc.so  -> libmppitc.so   (stub -> libmppc)
#   libnpps.so    -> libmpps.so     (stub -> libmppc)
# =============================================================================

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
add_library(npp
    ${CPU_SOURCES}
    ${CUDA_SOURCES}
)

# Set NPP target configuration
npp_setup_target(npp)

# Add NPP specific include directories
npp_add_common_includes(npp)

# Set library output name to mpp (generates libmpp.so/libmpp.a)
set_target_properties(npp PROPERTIES
    OUTPUT_NAME "mpp"
    EXPORT_NAME "npp"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Add dynamic library specific configuration if shared
if(BUILD_SHARED_LIBS)
    # Windows DLL export symbols
    target_compile_definitions(npp PRIVATE MPP_EXPORTS)
    target_compile_definitions(npp INTERFACE MPP_IMPORTS)

    # Set runtime path for dynamic library
    if(NOT WIN32)
        set_target_properties(npp PROPERTIES
            INSTALL_RPATH "$ORIGIN:$ORIGIN/../../lib:$ORIGIN/../../lib64:$ORIGIN/../lib64:/opt/maca-${PACK_VERSION}/lib"
            INSTALL_RPATH_USE_LINK_PATH TRUE
        )
    endif()
endif()

# Install main library
npp_install_library(npp)

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

# Create stub source file
set(STUB_SOURCE_FILE "${CMAKE_CURRENT_BINARY_DIR}/mpp_stub.cpp")
file(WRITE ${STUB_SOURCE_FILE} "// MPP stub library - links to libmpp\n")

foreach(stub_name ${MPP_STUB_LIBS})
    # IMPORTANT: stubs should always be SHARED when BUILD_SHARED_LIBS=ON
    # to satisfy -l<stub> at link time with shared objects.
    if(BUILD_SHARED_LIBS)
        add_library(${stub_name} SHARED ${STUB_SOURCE_FILE})
    else()
        add_library(${stub_name} STATIC ${STUB_SOURCE_FILE})
    endif()

    # Link to main library target (npp outputs libmpp)
    target_link_libraries(${stub_name} PRIVATE npp)

    # Output name matches the stub name -> libmppial.so, etc.
    # Do NOT set VERSION/SOVERSION for stubs to keep packaging simple/stable.
    set_target_properties(${stub_name} PROPERTIES
        OUTPUT_NAME "${stub_name}"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )

    # set install_rpath
    if(BUILD_SHARED_LIBS AND NOT WIN32)
        set_target_properties(${stub_name} PROPERTIES
            INSTALL_RPATH "$ORIGIN:$ORIGIN/../../lib:$ORIGIN/../../lib64:$ORIGIN/../lib64:/opt/maca-${PACK_VERSION}/lib"
            INSTALL_RPATH_USE_LINK_PATH TRUE
        )
    endif()

    # Install stub library
    install(TARGETS ${stub_name}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT libs
    )
endforeach()

# Replace shared stub libs with symlinks to libmpp so -l<stub> works without -lmpp
if(BUILD_SHARED_LIBS AND NOT WIN32)
    set(MPP_STUB_BUILD_TEMPLATE "${CMAKE_SOURCE_DIR}/cmake/mpp_stub_build.cmake.in")
    set(MPP_STUB_BUILD_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/mpp_stub_build.cmake.in")
    configure_file(${MPP_STUB_BUILD_TEMPLATE} ${MPP_STUB_BUILD_CONFIG} @ONLY)

    set(MPP_STUB_BUILD_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/mpp_stub_build.cmake")
    file(GENERATE OUTPUT ${MPP_STUB_BUILD_SCRIPT} INPUT ${MPP_STUB_BUILD_CONFIG})

    add_custom_target(mpp_stub_symlinks ALL
        COMMAND ${CMAKE_COMMAND} -P ${MPP_STUB_BUILD_SCRIPT}
        DEPENDS npp ${MPP_STUB_LIBS}
        VERBATIM
    )

    set(MPP_STUB_INSTALL_TEMPLATE "${CMAKE_SOURCE_DIR}/cmake/mpp_stub_install.cmake.in")
    set(MPP_STUB_INSTALL_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/mpp_stub_install.cmake")
    configure_file(${MPP_STUB_INSTALL_TEMPLATE} ${MPP_STUB_INSTALL_SCRIPT} @ONLY)
    install(SCRIPT ${MPP_STUB_INSTALL_SCRIPT})
endif()

message(STATUS "MPP main library: libmpp (target: npp, contains all implementations)")
message(STATUS "MPP stub libraries: ${MPP_STUB_LIBS} (link to libmpp)")