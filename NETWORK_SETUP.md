# 🌐 网络受限环境设置指南

如果你的服务器访问 GitHub 受限（如中国大陆服务器），请按以下步骤操作。

---

## 🎯 问题症状

```
fatal: 无法访问 'https://github.com/google/benchmark.git/'
GnuTLS recv error (-110): The TLS connection was non-properly terminated
Failed to connect to github.com port 443
```

---

## ✅ 解决方案：使用预下载的 Submodules

项目已将所有依赖作为 Git submodule 包含，无需编译时下载！

### 方法 1：首次克隆项目（推荐）

```bash
# 一次性下载项目和所有依赖
cd ~
git clone --recursive git@github.com:UniBoy222/npp.git

# 或使用 HTTPS
git clone --recursive https://github.com/UniBoy222/npp.git
```

**`--recursive` 参数会自动下载所有 submodule！**

---

### 方法 2：已有项目，更新 Submodules

```bash
cd ~/npp

# 更新主项目
git pull

# 初始化并更新所有 submodule
git submodule update --init --recursive
```

---

### 方法 3：使用代理或镜像（高级）

#### 选项 A：使用代理

```bash
# 临时使用代理
git config --global http.proxy http://proxy.example.com:8080
git config --global https.proxy https://proxy.example.com:8080

# 取消代理
git config --global --unset http.proxy
git config --global --unset https.proxy
```

#### 选项 B：使用 Gitee 镜像（中国用户）

```bash
# 如果 GitHub 不可用，手动从 Gitee 下载 submodule
cd ~/npp

# GoogleTest
rm -rf third_party/googletest
git clone https://gitee.com/mirrors/googletest.git third_party/googletest

# Google Benchmark
rm -rf third_party/benchmark
git clone https://gitee.com/mirrors/benchmark.git third_party/benchmark
```

---

## 📋 验证依赖是否完整

```bash
cd ~/npp

# 检查 GoogleTest
ls third_party/googletest/CMakeLists.txt

# 检查 Google Benchmark
ls third_party/benchmark/CMakeLists.txt

# 如果两个文件都存在，说明依赖完整
```

---

## 🚀 编译和运行

依赖下载完成后，正常编译：

```bash
cd ~/npp
./quick_benchmark.sh
```

脚本会自动验证依赖：

```
========================================
Step 2: Update Source Code
========================================

→ Pulling latest changes...
→ Updating submodules...
✓ Source code updated
✓ GoogleTest submodule verified
✓ Google Benchmark submodule verified
```

---

## 🔍 常见问题

### Q: 克隆时没加 `--recursive` 怎么办？

```bash
cd ~/npp
git submodule update --init --recursive
```

### Q: submodule 更新失败？

```bash
cd ~/npp

# 清理 submodule
git submodule deinit -f .
rm -rf .git/modules/*

# 重新初始化
git submodule update --init --recursive
```

### Q: 仍然尝试从 GitHub 下载？

检查 submodule 是否完整：

```bash
cd ~/npp
ls -la third_party/

# 应该看到：
# benchmark/
# googletest/
```

如果目录为空，执行：

```bash
git submodule update --init --recursive
```

### Q: 使用 SSH 还是 HTTPS？

**SSH 方式（推荐，如果配置了 SSH 密钥）：**
```bash
git clone --recursive git@github.com:UniBoy222/npp.git
```

**HTTPS 方式：**
```bash
git clone --recursive https://github.com/UniBoy222/npp.git
```

---

## 📊 项目依赖结构

```
npp/
├── third_party/
│   ├── googletest/         ← Git submodule (自动下载)
│   └── benchmark/          ← Git submodule (自动下载)
├── quick_benchmark.sh      ← 一键测试脚本
└── CMakeLists.txt
```

---

## 🎯 总结

1. **首次克隆**：使用 `git clone --recursive`
2. **已有项目**：使用 `git submodule update --init --recursive`
3. **验证依赖**：检查 `third_party/googletest` 和 `third_party/benchmark`
4. **编译运行**：`./quick_benchmark.sh`

---

**现在不再需要在编译时从 GitHub 下载任何东西！** 🎉
