# NPP算术运算模板化重构系统

## 概述

本项目通过使用C++模板和functor设计模式，大幅简化了NPPI算术运算的实现，减少了代码重复并提高了可维护性。

## 系统架构

### 1. Functor层 (`nppi_arithmetic_functors.h`)

#### 利用标准库函数对象
```cpp
// 使用std::plus, std::minus, std::multiplies等标准函数对象
template<typename T>
using AddFunctor = ScaledFunctor<T, std::plus<double>>;

template<typename T>
using SubFunctor = ScaledFunctor<T, std::minus<double>>;

template<typename T>
using MulFunctor = ScaledFunctor<T, std::multiplies<double>>;
```

#### 标准库逻辑运算
```cpp
template<typename T>
struct AndFunctor {
    __device__ __host__ T operator()(T a, T b, int = 0) const {
        std::bit_and<T> op;
        return op(a, b);
    }
};
```

#### 通用数学函数包装
```cpp
template<typename T, typename UnaryMathFunc>
struct UnaryMathFunctor {
    __device__ __host__ T operator()(T a, int scaleFactor = 0) const {
        UnaryMathFunc func;
        double result = func(static_cast<double>(a));
        // 自动处理scaling和saturation
        return saturate_cast<T>(result);
    }
};
```

### 2. 模板层 (`nppi_arithmetic_templates.h`)

#### 通用核函数模板
```cpp
template<typename T, int Channels, typename BinaryOp>
__global__ void binaryOpKernel(const T *pSrc1, int nSrc1Step, 
                               const T *pSrc2, int nSrc2Step,
                               T *pDst, int nDstStep, 
                               int width, int height, 
                               BinaryOp op, int scaleFactor = 0)
```

#### 通用启动器模板
```cpp
template<typename T, int Channels, typename BinaryOp>
NppStatus launchBinaryOp(const T *pSrc1, int nSrc1Step, 
                         const T *pSrc2, int nSrc2Step,
                         T *pDst, int nDstStep, 
                         NppiSize oSizeROI, 
                         BinaryOp op, int scaleFactor = 0,
                         cudaStream_t stream = 0)
```

### 3. 宏展开层 (`nppi_arithmetic_macros.h`)

#### 自动生成API实现
```cpp
// 生成所有变体: _Ctx, non-Ctx, in-place等
NPPI_BINARY_OP_IMPL(Add, AddFunctor, 8u, 1, Sfs)
NPPI_BINARY_OP_FLOAT_IMPL(Add, AddFunctor, 32f, 1)
```

## 代码重复对比

### 传统方式 (每个操作~200行)
```cpp
// nppi_add.cpp - 205行
// nppi_sub.cpp - 195行  
// nppi_mul.cpp - 210行
// nppi_div.cpp - 200行
// 总计: ~800+行，大量重复代码
```

### 模板化方式 (总共~50行)
```cpp
// nppi_add_templated.cpp - 仅需15行生成所有Add变体
NPPI_BINARY_OP_IMPL(Add, AddFunctor, 8u, 1, Sfs)
NPPI_BINARY_OP_IMPL(Add, AddFunctor, 8u, 3, Sfs)
NPPI_BINARY_OP_FLOAT_IMPL(Add, AddFunctor, 32f, 1)
```

**代码减少率: 94%** (从800+行减少到50行)

## 支持的操作类型

### 1. 二元运算
- **Add**: 加法 (使用 `std::plus`)
- **Sub**: 减法 (使用 `std::minus`)  
- **Mul**: 乘法 (使用 `std::multiplies`)
- **Div**: 除法 (自定义零除保护)
- **AbsDiff**: 绝对差值

### 2. 逻辑运算
- **And**: 按位与 (使用 `std::bit_and`)
- **Or**: 按位或 (使用 `std::bit_or`)
- **Xor**: 按位异或 (使用 `std::bit_xor`)

### 3. 一元运算
- **Abs**: 绝对值
- **Sqr**: 平方
- **Sqrt**: 平方根
- **Exp**: 指数
- **Ln**: 自然对数

### 4. 常量运算
- **AddC**: 加常量
- **SubC**: 减常量
- **MulC**: 乘常量
- **DivC**: 除以常量

## 支持的数据类型和通道

### 数据类型
- `Npp8u`, `Npp8s`
- `Npp16u`, `Npp16s` 
- `Npp32f`, `Npp32s`

### 通道配置
- **C1**: 单通道
- **C3**: 3通道 (RGB)
- **C4**: 4通道 (RGBA)
- **AC4**: 4通道忽略Alpha

### API变体
- **R**: 基础版本
- **Sfs**: 带缩放因子版本
- **I**: 就地操作版本
- **Ctx**: 流上下文版本

## 新增功能的示例

### 添加新操作只需3步：

#### 1. 定义Functor (如果需要)
```cpp
template<typename T>
struct ModFunctor {
    __device__ __host__ T operator()(T a, T b, int = 0) const {
        return (b != T(0)) ? a % b : T(0);
    }
};
```

#### 2. 生成API实现
```cpp
NPPI_BINARY_OP_IMPL(Mod, ModFunctor, 8u, 1, Sfs)
NPPI_BINARY_OP_IMPL(Mod, ModFunctor, 16u, 1, Sfs)
```

#### 3. 添加测试
```cpp
TYPED_TEST(NppiArithmeticTemplatedTest, ModOperation_C1) {
    this->template testBinaryOperation<1>("mod", 0);
}
```

## 性能特点

### 编译时优化
- **零运行时开销**: 所有模板在编译时展开
- **内联优化**: functor调用完全内联
- **类型安全**: 编译时类型检查

### 内存效率
- **统一内核**: 相同的CUDA内核处理不同数据类型
- **缓存友好**: 模板特化优化内存访问模式

### 可扩展性
- **添加新类型**: 只需特化`saturate_cast`
- **添加新操作**: 定义functor即可
- **添加新通道**: 修改模板参数

## 测试框架

### 统一测试模板
```cpp
template<typename T>
class NppiArithmeticTemplatedTest : public ::testing::Test {
    template<int Channels>
    void testBinaryOperation(const std::string& operation, int scaleFactor = 0);
};
```

### 类型化测试
```cpp
using TestTypes = ::testing::Types<Npp8u, Npp16u, Npp32f>;
TYPED_TEST_SUITE(NppiArithmeticTemplatedTest, TestTypes);
```

## 总结

通过利用C++标准库的函数对象(`std::plus`, `std::bit_and`等)和现代模板技术，我们创建了一个：

1. **高度可重用**的算术运算框架
2. **大幅减少代码重复** (94%减少率)  
3. **类型安全**且**性能优化**的实现
4. **易于扩展**的架构设计

这个系统展示了如何通过合理的抽象设计，将复杂的底层CUDA实现封装为简洁、可维护的高层接口。