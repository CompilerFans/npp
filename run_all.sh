cmake -S . -B build-nvidia -DUSE_NVIDIA_NPP=ON -G Ninja
cmake --build build-nvidia -j $(nproc)
./build-nvidia/unit_tests

cmake -S . -B build -G Ninja
cmake --build build -j $(nproc)
./build/unit_tests

