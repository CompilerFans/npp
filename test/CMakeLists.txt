# NPP unit test CMake configuration

# Setup GoogleTest
npp_setup_gtest()

# Unit tests (pure functional tests, no external dependencies)
add_subdirectory(unit)