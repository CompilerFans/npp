#!/bin/bash

g++ -o comprehensive_test comprehensive_test.cpp \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 \
    -lnppig -lnppicc -lnppidei -lcudart \
    -std=c++11 \
    -O2

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Run with: ./comprehensive_test"
else
    echo "Compilation failed!"
    exit 1
fi
