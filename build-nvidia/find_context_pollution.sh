#!/bin/bash

# Script to find which test pollutes CUDA context

echo "Finding CUDA context pollution source..."

# Get all test names
./unit_tests --gtest_list_tests | grep -E "^\s+\w" | grep -v DISABLED | sed 's/^[ \t]*//' > all_tests.txt

# Test that we know fails when context is polluted
TARGET_TEST="SupportFunctionsTest.GetLibVersion_BasicTest"

# Binary search through tests
echo "Running binary search to find pollution source..."

# First, verify target test passes alone
echo "Testing target alone: $TARGET_TEST"
if ./unit_tests --gtest_filter="$TARGET_TEST" 2>&1 | grep -q "PASSED.*1 test"; then
    echo "✓ Target test passes when run alone"
else
    echo "✗ Target test fails even when run alone!"
    exit 1
fi

# Now test with different test groups
echo -e "\nTesting with test groups..."

# Test with each test suite
./unit_tests --gtest_list_tests | grep -E "^[A-Za-z]" | grep -v "^  " | while read -r suite; do
    suite_name="${suite%.*}"
    echo -n "Testing $suite_name + target... "
    
    if ./unit_tests --gtest_filter="$suite_name.*:$TARGET_TEST" 2>&1 | grep -q "illegal memory access"; then
        echo "✗ POLLUTION FOUND!"
        echo "Suite $suite_name causes context pollution"
        
        # Find specific test in suite
        echo "Finding specific test in $suite_name..."
        ./unit_tests --gtest_list_tests | grep -A 100 "^$suite" | grep -E "^\s+\w" | grep -v DISABLED | sed 's/^[ \t]*//' | while read -r test; do
            echo -n "  Testing $suite_name.$test... "
            if ./unit_tests --gtest_filter="$suite_name.$test:$TARGET_TEST" 2>&1 | grep -q "illegal memory access"; then
                echo "✗ FOUND POLLUTING TEST: $suite_name.$test"
                break
            else
                echo "✓ OK"
            fi
        done
        break
    else
        echo "✓ OK"
    fi
done