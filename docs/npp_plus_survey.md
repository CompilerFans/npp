# NPP_Plus（NPP+）调研记录

> 日期：2026-11-07
> 范围：NPP+ 的产品定位、接口形态、分发版本、安装方式与兼容性验证
> 结论：NPP+ 是 NVIDIA 在经典 NPP C API 之外独立发布的新一代 C++ 接口线；
> 本仓库的公共契约仍以 API-13.4 的经典 C 声明为准，NPP+ 不在该契约内。

## 1. 官方资料

- 产品介绍：<https://docs.nvidia.com/cuda/nppplus/introduction.html>
- 发布说明：<https://docs.nvidia.com/cuda/nppplus/releasenotes.html>
- 随 CUDA 版本归档的文档：<https://docs.nvidia.com/cuda/archive/13.4.2/nppplus/index.html>
- 独立分发包清单：<https://developer.download.nvidia.com/compute/nppplus/redist/>

## 2. 接口形态

NPP+ 不是另一套独立的头文件入口，而是通过宏切换同一套 umbrella header：

- 编译时定义 `NPP_PLUS_ENABLE`，`nppPlus/nppPlus.h` 会据此定义 `NPP_PLUS`；
- `nppdefs.h`、`nppcore.h`、`nppi.h`、`npps.h` 在 `NPP_PLUS` 下打开 `namespace nppPlusV`；
- `nppi.h` 改为包含 `nppPlus/nppiPlus_*.h`（10 个图像模块），
  `npps.h` 改为包含 `nppPlus/nppsPlus_*.h`（6 个信号模块）；
- Linux 上通过 `NPP_LNX_EXTERN_C` 保持 C 链接。

核心变化是**折叠参数（collapsed combined parameter）**：数据类型和通道数从函数名后缀
变成运行时枚举参数，避免为每种类型/通道组合重复声明函数：

```c
/* 经典 NPP */
nppiAdd_8u_C3R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep,
                   oSizeROI, nppStreamCtx);

/* NPP+ */
nppiAdd_Ctx(NPP_8U, NPP_CH_3, pSrc1, nSrc1Step, pSrc2, nSrc2Step,
            pDst, nDstStep, oSizeROI, nppStreamCtx);
```

官方约定：

- 无状态、线程安全；
- 必须使用应用管理的 `NppStreamContext`，非 `_Ctx` 入口已弃用；
- 0.10.0 起启用 C++17，支持当前一代 GPU 架构；
- 以 0.x 库版本独立演进，与经典 NPP 的 13.x 版本号彼此独立。

## 3. 分发与版本

NPP+ 不随 CUDA toolkit 安装，也不在 CUDA 的 apt 仓库中，只通过独立 redist 发布：

- 可查版本：0.9.0、0.10.0（0.10.0 为当前最新）；
- 变体：CUDA 11、CUDA 12；没有 CUDA 13 变体，也没有 0.11；
- 归档内容：`include/`（经典 NPP umbrella 头文件 + `nppPlus/` 头文件树）、
  `lib/`（`libnpp_plus_*.so*`、静态库、stubs）、`LICENSE`。

因此，仅安装 CUDA toolkit 时，定义 `NPP_PLUS` 会因缺少 `nppPlus/*.h` 而编译失败；
需要额外安装 NPP+ 分发包。

## 4. 安装方式

推荐独立前缀安装，避免覆盖 toolkit 自带头文件：

```sh
tar -xf libnpp_plus-linux-x86_64-0.10.0.0_cuda12-archive.tar.xz
sudo mkdir -p /usr/local/nppplus-0.10.0
sudo cp -a libnpp_plus-*/include libnpp_plus-*/lib /usr/local/nppplus-0.10.0/
sudo cp -a libnpp_plus-*/include/nppPlus /usr/local/cuda/include/nppPlus
echo /usr/local/nppplus-0.10.0/lib | sudo tee /etc/ld.so.conf.d/nppplus.conf
sudo ldconfig
```

说明：

- `nppPlus/` 复制到 CUDA include 目录后，CUDA 13.4 的 umbrella header 可以直接找到它；
- 也可以完全不复制，改用 `-I/usr/local/nppplus-0.10.0/include` 使用包内自带的
  umbrella header；
- 库目录同时加入 `/usr/local/lib` 的链接路径后，`-lnpp_plus_*` 无需额外 `-L`。

## 5. 验证

最小验证程序同时覆盖版本查询和真实 GPU 调用：

```c
#include <nppPlus/nppPlus.h>
#include <npp.h>

NppStreamContext ctx = ...;            /* hStream + 设备属性 */
nppPlusV::nppiFilterGauss_Ctx(NPP_8U, NPP_CH_1, pSrc, step, pDst, step,
                              roi, NPP_MASK_SIZE_3_X_3, ctx);
```

```sh
nvcc -std=c++17 -DNPP_PLUS_ENABLE smoke.cu -lnpp_plus_c -lnpp_plus_if
```

在 CUDA 13.4 环境下的验证结果：

- `nppGetLibVersion()` 返回 12.4.0（NPP+ 内部的经典 API 版本）；
- 64x48 单通道高斯滤波返回 `NPP_NO_ERROR`，设备同步无错误；
- 库本身不依赖 `libcudart`（`DT_NEEDED` 中无 CUDA runtime），
  运行时通过 `libcuda.so.1` 驱动接口工作；
- fatbin 覆盖当前数据中心与工作站 GPU 架构，包含本机安装的架构。

## 6. 使用注意

- include 顺序：先 `<nppPlus/nppPlus.h>`，再 `<npp.h>`；
- NPP+ 库同时导出经典 NPP 符号名，不要与经典 NPP 库或本仓库的 libmpp
  链接进同一个进程，否则会产生符号冲突；
- 包内自带的经典头文件属于 NPP 12.x 时代；若需要与 API-13.4 的经典声明共存，
  使用 CUDA toolkit 的 umbrella header + 复制后的 `nppPlus/`；
- NPP+ 是独立 ABI，是否纳入本仓库的支持范围应单独立项，不计入经典 NPP
  契约的覆盖率统计。
