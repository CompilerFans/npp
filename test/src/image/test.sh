rm *.out
cucc test_nppi_support_functions.cpp  ../../..//src/backend/npp/image/nppi_support_functions.cpp  -I ../../../API/ -o test.out
./test.out
