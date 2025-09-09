# NPP API状态报告系统

本目录包含NPP API实现状态报告生成系统，用于跟踪API实现进度和测试覆盖情况。

## 文件结构

```
status/
├── api_features.yaml          # API特性配置文件
├── generate_status_report.py  # 状态报告生成脚本
├── run_status_report.sh       # 便捷运行脚本
├── API_STATUS.md              # 生成的状态报告
└── README.md                  # 本说明文件
```

## 快速使用

### 方法1: 使用便捷脚本（推荐）

```bash
# 进入项目根目录
cd /path/to/npp

# 运行状态报告生成
./status/run_status_report.sh
```

脚本会自动：
1. 检查依赖（Python3、PyYAML）
2. 运行单元测试并生成XML结果（如果需要）
3. 分析源码和测试实现
4. 生成状态报告

### 方法2: 手动步骤

1. **生成测试结果XML**
   ```bash
   ./unit_tests --gtest_output=xml:test_results.xml --gtest_brief
   ```

2. **运行状态报告生成器**
   ```bash
   python3 status/generate_status_report.py \
       --project-root . \
       --features-file status/api_features.yaml \
       --xml-results test_results.xml \
       --output status/API_STATUS.md
   ```

## 配置文件说明

### api_features.yaml

定义了需要跟踪的所有NPP API函数，按模块组织：

```yaml
nppi_arithmetic_operations:
  description: "NPPI图像算术运算"
  functions:
    - name: "nppiAdd_8u_C1RSfs"
      description: "8位无符号单通道加法运算（饱和）"
    - name: "nppiAdd_32f_C1R"
      description: "32位浮点单通道加法运算"
    # ...更多函数
```

**添加新API的步骤**：
1. 在相应模块下添加函数条目
2. 提供函数名和描述
3. 重新运行报告生成器

## 生成的报告内容

生成的`API_STATUS.md`包含：

### 总体统计
- 总API数量
- 已实现API数量和百分比
- 已测试API数量和百分比
- 实现率和测试覆盖率

### 模块详细状态
每个模块显示：
- 模块描述和统计
- 函数实现状态表格
- 测试状态和结果

### 测试结果详情
- 测试通过/失败概况
- 失败测试的详细信息

### 实现文件映射
- 每个已实现函数对应的源文件列表
- 便于代码定位和维护

## 工作原理

系统通过以下步骤生成报告：

1. **解析API特性配置**
   - 读取`api_features.yaml`
   - 构建待跟踪的API函数列表

2. **解析测试结果**
   - 读取gtest生成的XML结果
   - 提取每个测试用例的通过/失败状态

3. **扫描源码实现**
   - 递归搜索`src/`目录下的`.cpp`和`.cu`文件
   - 使用正则表达式识别函数实现

4. **扫描测试实现**
   - 递归搜索`test/`目录下的`test_*.cpp`文件
   - 识别测试中调用的API函数

5. **分析和关联**
   - 将实现状态、测试状态和测试结果关联
   - 生成综合状态信息

6. **生成Markdown报告**
   - 按模块组织输出
   - 生成统计表格和详细信息

## 自定义选项

### 命令行参数

```bash
python3 status/generate_status_report.py \
    --project-root /path/to/project \
    --features-file custom_features.yaml \
    --xml-results custom_results.xml \
    --output custom_status.md
```

### 扩展功能

可以通过修改`generate_status_report.py`来：
- 添加新的数据源（如代码覆盖率）
- 修改报告格式
- 添加更多统计指标
- 集成其他分析工具

## 集成到CI/CD

可以将状态报告生成集成到持续集成流程中：

```yaml
# GitHub Actions示例
- name: Generate API Status Report
  run: |
    ./status/run_status_report.sh
    git add status/API_STATUS.md
    git commit -m "Update API status report" || exit 0
```

## 依赖要求

- Python 3.6+
- PyYAML库（自动安装）
- 已构建的unit_tests可执行文件

## 疑难解答

### 常见问题

1. **找不到unit_tests**
   - 确保已构建项目：`make -j4`
   - 检查可执行文件位置：`build/unit_tests`或当前目录

2. **PyYAML安装失败**
   - 手动安装：`pip3 install pyyaml`
   - 使用系统包管理器：`apt install python3-yaml`

3. **报告显示"部分测试"**
   - 表示找到了相关测试但可能测试用例命名不匹配
   - 检查测试用例名称是否包含API函数名

4. **源码扫描不准确**
   - 源码扫描基于简单的正则表达式
   - 可能需要调整匹配模式以适应特定的代码风格

## 维护建议

1. **定期更新api_features.yaml**
   - 当添加新API时及时更新配置
   - 保持描述信息的准确性

2. **监控实现率变化**
   - 设置目标实现率和测试覆盖率
   - 跟踪进度变化趋势

3. **集成到开发流程**
   - 在每次发布前生成状态报告
   - 作为代码审查的参考材料

4. **自动化报告更新**
   - 配置定时任务或CI触发器
   - 确保报告与代码同步更新