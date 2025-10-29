#!/bin/bash

g++ -o test_cpu_vs_npp test_cpu_vs_npp.cpp \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 \
    -lnppig -lnppicc -lnppidei -lcudart \
    -std=c++11 \
    -O2

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Run with: ./test_cpu_vs_npp"
else
    echo "Compilation failed!"
    exit 1
fi
