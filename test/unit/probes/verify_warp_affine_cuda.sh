#!/usr/bin/env bash
# Build the WarpAffine differential tests against the mcMPP implementation with
# a plain CUDA toolchain (no MACA / cu-bridge build) and verify them against
# the NVIDIA NPP goldens recorded on the 5090.  This is a development aid: it
# only covers the self-contained WarpAffine executor + generated wrappers.
#
#   test/unit/probes/verify_warp_affine_cuda.sh <golden-dir> [out-dir]
#
# GTEST_LIB must contain libgtest.a / libgtest_main.a (the NVIDIA test build's
# build-nvidia/lib works).  NVCC_ARCH defaults to the local device.
set -euo pipefail

root=$(cd "$(dirname "$0")/../../.." && pwd)
golden=${1:?usage: $0 <golden-dir> [out-dir]}
out=${2:-/tmp/warp_affine_diff_cuda}
arch=${NVCC_ARCH:-native}
gtest_lib=${GTEST_LIB:-$root/build-nvidia/lib}
nvcc=${NVCC:-nvcc}
cxx=${CXX:-g++}
cuda_inc=${CUDA_HOME:-/usr/local/cuda}/include

geom=$root/src/nppi/nppi_geometry_transforms
inc=(-I"$cuda_inc" -I"$root/API-unified" -I"$root/src/include" -I"$root/test/unit/framework"
     -I"$root/test/unit/diff" -I"$root/third_party/googletest/googletest/include")
# The generated wrappers call nppGetStreamContext, which the CUDA headers no
# longer declare (NPP 13 removed the managed-stream API from the public
# headers).  npp_stream_compat.h re-declares it for the NVIDIA build; the shim
# below defines it.
compat=(-DUSE_NVIDIA_NPP_TESTS -include "$root/test/unit/framework/npp_stream_compat.h")

mkdir -p "$out"
"$nvcc" -std=c++17 -arch="$arch" -O2 -c "${inc[@]}" "${compat[@]}" -o "$out/warp_executor.o" \
        "$geom/nppi_warp_affine_executor.cu"
for f in "$geom"/nppi_warp_affine_generated.cpp "$geom"/nppi_warp_affine_back_generated.cpp \
         "$geom"/nppi_warp_affine_quad_generated.cpp "$geom"/nppi_warp_affine_batch_generated.cpp \
         "$geom"/nppi_warp_affine_batch_init_generated.cpp "$root/test/unit/probes/npp_stream_shim.cpp"; do
  "$cxx" -std=c++17 -O2 -c "${inc[@]}" "${compat[@]}" -o "$out/$(basename "$f" .cpp).o" "$f"
done
for f in "$root"/test/unit/diff/test_nppi_warp_affine_generated.cpp \
         "$root"/test/unit/diff/test_nppi_warp_affine_back_generated.cpp \
         "$root"/test/unit/diff/test_nppi_warp_affine_quad_generated.cpp \
         "$root"/test/unit/diff/test_nppi_warp_affine_batch_generated.cpp \
         "$root"/test/unit/diff/test_nppi_warp_affine_batch_init_generated.cpp; do
  "$cxx" -std=c++17 -O2 -c "${inc[@]}" "${compat[@]}" -o "$out/$(basename "$f" .cpp).o" "$f"
done

"$nvcc" -std=c++17 -arch="$arch" -o "$out/warp_affine_diff_cuda" "$out"/*.o \
        "$gtest_lib/libgtest_main.a" "$gtest_lib/libgtest.a" -lpthread

NPP_VERIFY_DIR="$golden" "$out/warp_affine_diff_cuda" --gtest_filter='NppiWarpAffine*Diff.*'
