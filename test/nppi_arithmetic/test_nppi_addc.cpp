#include "test_nppi_addc.h"

namespace NPPTest {

// 实现注册函数
void registerAddCTests() {
    TestRegistry::registerTest(std::make_unique<AddC_8u_C1RSfs_Test>());
    TestRegistry::registerTest(std::make_unique<AddC_16u_C1RSfs_Test>());
    TestRegistry::registerTest(std::make_unique<AddC_16s_C1RSfs_Test>());
    TestRegistry::registerTest(std::make_unique<AddC_32f_C1R_Test>());
}

} // namespace NPPTest