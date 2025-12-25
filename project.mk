# Minimal one-step build/package for mcMPP under maca SDK.

BUILD_TYPE ?= Release
BUILD_SHARED_LIBS ?= ON
BUILD_EXAMPLES ?= OFF
# default use Ninja, otherwise CMake can choose (usually cmake).
CMAKE_GENERATOR ?=
ifeq ($(CMAKE_GENERATOR),)
  ifneq ($(shell command -v ninja 2>/dev/null),)
    CMAKE_GENERATOR := Ninja
  endif
endif
PACK_ARCH ?= $(ARCH)
PACK_EXT ?=
# Default to DEB; CPack config is DEB-only for this project.
PACKAGE_GENERATOR ?= DEB
# Defaults: build/install under current workspace unless overridden
# BUILD_DIR ?= $(abspath build-local/mcMPP)

.PHONY: convertMpp
convertMpp: PRIVATE_PROJ_DIR ?= $(CURDIR)/library/mcMPP
convertMpp:
	@echo
	@echo "Converting sources to MPP (if needed)..."
	@if [ -d "$(PRIVATE_PROJ_DIR)/src/mppi" ]; then \
	  echo "  Detected converted sources (src/mppi exists), skipping."; \
	else \
	  bash "$(PRIVATE_PROJ_DIR)/mpp_scripts/setup_mpp.sh" --skip-build --skip-deploy; \
	fi

.PHONY: mcMPP
mcMPP: convertMpp
mcMPP: PRIVATE_BUILD_DIR ?= $(BUILDDIR)/library/mcMPP
mcMPP: PRIVATE_MK_FILE := $(abspath $(lastword $(MAKEFILE_LIST)))
mcMPP: PRIVATE_MK_PATH := $(patsubst %/, %, $(dir $(PRIVATE_MK_FILE)))
mcMPP:
	@echo "Configuring + building + installing + packaging mcMPP..."
	cmake_maca -S $(PRIVATE_MK_PATH) \
	  -B $(PRIVATE_BUILD_DIR) \
	  -DBUILD_NUMBER=$(BUILD_NUMBER) \
	  $(if $(CMAKE_GENERATOR),-G "$(CMAKE_GENERATOR)") \
	  -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	  -DBUILD_TESTS=OFF \
	  -DBUILD_EXAMPLES=$(BUILD_EXAMPLES) \
	  -DBUILD_SHARED_LIBS=$(BUILD_SHARED_LIBS) \
	  -DCMAKE_INSTALL_PREFIX=$(BUILDROOT) \
	  -DCMAKE_PACKAGE_DIR=$(PRIVATE_BUILD_DIR) -DPACKAGE_ARCH=$(PACK_ARCH) -DPACKAGE_EXT=$(PACK_EXT) \
	  -DPACK_VERSION=$(PACK_VERSION) -DMACA_VERSION=$(MACA_VERSION) \
	  -DDEB_PACKAGE_ARCH=$(DEB_PACKAGE_ARCH) -DRPM_PACK_ARCH=$(RPM_PACK_ARCH) -DPACKAGE_GENERATOR=$(PACKAGE_GENERATOR)
	cmake_maca --build $(PRIVATE_BUILD_DIR)
	cmake_maca --build $(PRIVATE_BUILD_DIR) --target install
	cmake_maca --build $(PRIVATE_BUILD_DIR) --target package

.PHONY: clean
clean: PRIVATE_BUILD_DIR ?= $(BUILDDIR)/library/mcMPP
clean:
	rm -rf $(PRIVATE_BUILD_DIR)
