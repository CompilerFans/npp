# NPP 兼容库拆分设计（方案 2）

本文描述把当前「单体 `libmpp.so` + `libmpp*` 别名符号链接」的打包方式，改造为与官方
NVIDIA NPP 一致的「按领域拆分多库」结构的方案与实施计划。

## 1. 背景

### 1.1 官方 NPP 13.x 的打包结构（5090 实测）

```
libnppc.so.13      libnppial.so.13   libnppicc.so.13  libnppidei.so.13
libnppif.so.13     libnppig.so.13    libnppim.so.13   libnppist.so.13
libnppisu.so.13    libnppitc.so.13   libnpps.so.13
```

- 每个域一个共享库，**各自的 SONAME**（如 `libnppial.so.13`），版本三连 `*.so.13.2.0.58`
- 每个域一个静态库 `libnpp<domain>_static.a`
- 各域库依赖 `libnppc.so.13`（stream context / 内存 / 错误码等核心）

### 1.2 当前实现（方案 1）

`src/CMakeLists.txt` 把 `src/` 下全部 CPU/CUDA 源文件编进一个 `libmpp.so.1.0.0`
（SONAME `libmpp.so.1`），再创建 `libmppial.so`、`libmppicc.so` 等**符号链接别名**
（`MPP_STUB_LIBS`，`src/CMakeLists.txt:73-127`）；静态库同样是一个
`libmpp_static.a` 加 `_static.a` 别名。

优点：实现简单、内部测试只需链接一个目标。
不足：无 `libnpp*` 命名、无按域 SONAME/版本、静态库无法按域裁剪、每个别名导出全部符号。

## 2. 目标与非目标

**目标**
1. 按官方结构生成 11 个共享库（`nppc` + 10 个成像/信号域）与对应静态库
2. 官方命名、SONAME、版本三连对齐（兼容版本 13.2）
3. 域库只导出本域符号；域库依赖 `libnppc`
4. 保留现有 `libmpp*` 命名与安装布局，保证向后兼容
5. 内部测试/工具链接方式不受影响（通过聚合 INTERFACE 目标）

**非目标**
- 不改变任何 API 语义、实现算法或性能目标
- 不调整源码目录（仅构建/链接层改造）
- 不删除单体库（过渡期保留，可开关）

## 3. 目标结构

| 域 | 源码目录 | 共享库（SONAME） | 静态库 |
|---|---|---|---|
| core | `src/nppcore/`（stream/logger/core） | `libnppc.so.13` | `libnppc_static.a` |
| arithmetic | `src/nppi/nppi_arithmetic_operations/` | `libnppial.so.13` | `libnppial_static.a` |
| color | `src/nppi/nppi_color_conversion_operations/` | `libnppicc.so.13` | `libnppicc_static.a` |
| data exchange | `src/nppi/nppi_data_exchange_and_initialization/` | `libnppidei.so.13` | `libnppidei_static.a` |
| filtering | `src/nppi/nppi_filtering_functions/` | `libnppif.so.13` | `libnppif_static.a` |
| geometry | `src/nppi/nppi_geometry_transforms/` | `libnppig.so.13` | `libnppig_static.a` |
| morphology | `src/nppi/nppi_morphological_operations/` | `libnppim.so.13` | `libnppim_static.a` |
| statistics | `src/nppi/nppi_statistics_functions/` | `libnppist.so.13` | `libnppist_static.a` |
| support | `src/nppi/nppi_support_functions.cpp` | `libnppisu.so.13` | `libnppisu_static.a` |
| threshold/compare | `src/nppi/nppi_threshold_and_compare_operations/` | `libnppitc.so.13` | `libnppitc_static.a` |
| signal | `src/npps/` | `libnpps.so.13` | `libnpps_static.a` |

聚合与兼容：
- `libmpp.so.1`（单体，过渡期默认构建，`-DMPP_BUILD_MONOLITHIC=ON`）
- `libmpp<domain>.so` / `libmpp<domain>_static.a` 继续作为别名，指向对应域库
- 测试使用 `npp` INTERFACE 目标（`INTERFACE_LINK_LIBRARIES` = 全部域库），
  `npp_create_test_target(... npp)` 无需修改

## 4. CMake 设计

### 4.1 OBJECT 库（复用编译产物）

```cmake
npp_find_domain_sources(nppial_obj nppi/nppi_arithmetic_operations)
add_library(nppial_obj OBJECT ${NPPIAL_SOURCES})
npp_setup_target(nppial_obj)          # 沿用现有编译选项/包含路径
```

每个域先编 OBJECT，再分别生成 SHARED/STATIC，避免单体与拆分重复编译：

```cmake
add_library(nppial SHARED $<TARGET_OBJECTS:nppial_obj>)
add_library(nppial_static STATIC $<TARGET_OBJECTS:nppial_obj>)
set_target_properties(nppial PROPERTIES
    OUTPUT_NAME nppial
    VERSION ${NPP_COMPAT_VERSION}.0.0   # 例如 13.2.0
    SOVERSION ${NPP_COMPAT_SOVERSION})  # 13
target_link_libraries(nppial PRIVATE nppc)
target_compile_definitions(nppial PRIVATE MPP_EXPORTS)
```

### 4.2 符号可见性

- 现有 `MPP_EXPORTS/MPP_IMPORTS` 宏继续使用；建议配合
  `-fvisibility=hidden` + 显式导出标记，确保每库只导出本域 `npp*` 入口
- 第一阶段先保持默认可见性，验证通过后再收紧（防止内部测试/插件依赖隐藏符号）

### 4.3 依赖关系

- 所有域库 `PRIVATE` 链接 `nppc`
- 域间依赖先做审计（见 §7.1）：若存在域间调用，要么改为 `PRIVATE` 链接对方域库
  （与官方不同但可用），要么把公共实现下沉到 `nppc`/`src/include`
- 静态库的依赖通过 `INTERFACE_LINK_LIBRARIES` 传递

### 4.4 CUDA/MACA 设备代码

- 拆分默认**不启用 `-rdc=true`**；前提是每个域的 kernel 定义与 launch 在同一编译单元
  （现状基本满足：executor/kernel 成对位于本域 `.cu`）
- 若审计发现跨域 device 符号引用，两条路：
  1. 把共享 device 工具下沉到公共 device 库/头文件（优先）
  2. 启用 `-rdc=true` + 域库间 device link（会提高编译/链接成本）
- `WITHOUT_CU_BRIDGE` 路径（`.cu` 按 CXX 编译）同样适用上述规则

### 4.5 安装与版本

- 安装 `libnpp<domain>.so.13.2.0` + `libnpp<domain>.so.13` + `libnpp<domain>.so`
- 同时安装 `libmpp*` 别名（现有部署兼容）
- `PROJECT_VERSION` 保持 1.0.0；新增 `NPP_COMPAT_VERSION=13.2`、`NPP_COMPAT_SOVERSION=13`
- 更新 `npp_install_library` 以支持多目标

## 5. 兼容性语义

| 场景 | 方案 1 现状 | 方案 2 目标 |
|---|---|---|
| 应用按 `-lnppial` 链接 | 无该文件 | 正常 |
| 应用 `DT_NEEDED=libnppial.so.13` | 无法按官方 SONAME 解析 | 正常加载 |
| 单域替换/裁剪部署 | 只能整体替换 | 可按域替换 |
| 静态链接 `-lnppial_static` | 拉入全部对象 | 只拉入本域 |
| 符号范围 | 每别名导出全部 | 每库只导出本域 |
| 现有 `libmpp*` 部署 | 正常 | 保留别名，正常 |

## 6. 测试与验证

1. 三套基线全绿：unit 4,803、diff 1,638、ref 6,362（5090）
2. `readelf -d` 检查每个 `.so` 的 SONAME/NEEDED
3. `nm -D --defined-only` 审计：每库只含本域入口 + `nppc` 公共符号
4. 链接矩阵样例：
   - 只链接 `-lnppial` 的样例（仅调用 nppiAdd）→ 链接/运行通过
   - 只链接 `-lnpps` 的样例 → 通过
   - 链接全部域库 → 通过
5. 静态链接样例：`-lnppial_static` 的二进制不包含其他域对象（`nm` 检查）
6. 用官方样例/现有 benchmark 在替换 `libnppial.so.13` 后运行
7. 单体模式与拆分模式的 diff/benchmark 结果一致性（无性能回退）

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 域间符号依赖被忽视导致链接失败 | 实施前做 `nm`/`ldd` 审计；公共代码下沉 `nppc` 或显式域间依赖 |
| CUDA 跨 TU device 符号需要 RDC | 优先下沉公共 device 代码；必要时按域启用 `-rdc=true` |
| 符号可见性收紧破坏内部使用 | 分阶段：先默认可见，验证后再隐藏 |
| 安装/打包脚本依赖 `libmpp*` | 保留全部 `libmpp*` 别名 |
| 构建时间增加 | OBJECT 库复用；单体与拆分共用对象；并行构建 |
| 回退困难 | `-DMPP_BUILD_MONOLITHIC=ON` 一键回到方案 1 |

## 8. 实施步骤（P0.2）

1. **P0.2a 审计**：生成源→域映射；`nm -D` 统计当前符号；扫描跨域引用；
   输出 `docs/library_split_audit.md`
2. **P0.2b OBJECT 化**：按域建立 OBJECT 库；单体库改为链接全部 OBJECT
   （默认行为不变；先验证三套测试）
3. **P0.2c 域库目标**：生成 11 个 SHARED/STATIC，官方命名/SONAME/版本；
   `npp` INTERFACE 聚合；测试切换链接
4. **P0.2d 符号与安装**：导出范围审计；安装规则；`libmpp*` 别名保留
5. **P0.2e 默认切换**：文档更新、CI/打包脚本调整、基准一致性验证

每步合并门槛：unit/diff/ref 全绿 + 符号审计通过 + 无性能回退。

## 9. 验收标准

1. 生成 11 个共享库与 11 个静态库，命名/SONAME/版本与官方一致（兼容版本 13.2）
2. 每库符号范围正确、依赖 `libnppc`；无非本域符号泄漏
3. unit 4,803 / diff 1,638 / ref 6,362 全绿
4. 现有 `libmpp*` 部署方式不受影响
5. 单体与拆分两种构建的 benchmark 无回退
