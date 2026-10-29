# 并行 subagent 工作流（编码并行 / 验证串行）

目标：在不破坏共享 build 缓存的前提下，让多个 GLM-5.3 subagent 同时推进。

## 1. 参考测试 agent（可多路并行）

- 每个 agent：独立 worktree `~/worktrees/mcMPP-ref-<id>` + 独立 5090 目录 `~/mcMPP-ref-<id>`。
- runner：`tools/agents/run_ref_agent.sh <id> <modules...>`，使用 `~/.config/mcpp-agent.env`（公司 GLM-5.3）。
- 5090 并发：20 核；建议每路构建 `jobs=2`，record 本身只需数秒。10 路时远程磁盘约 300MB/路。
- 生成器 `tools/gen_ref_tests.py` 是共享文件：**并行 agent 不得同时修改它**。生成器功能冻结后再按模块 fan-out。

## 2. 实现包 agent（编码并行 / 验证串行）

当前工具链限制：全新 worktree 无法用 CMake 4.4 配置 cu-bridge（`Failed to extract nvcc implicit link line`），
因此 worktree 内不能本地 build。采用如下流水线：

1. 启动（每个包一个 worktree，不做本地构建）：
   ```bash
   tools/agents/run_package.sh <pkg> --worktree --no-build
   ```
   agent 只做实现、生成 wrapper、刷新覆盖率/包列表、静态检查、提交；系统提示会明确禁止 `./build.sh`/测试。
2. 串行验证（Lead，主 checkout 的 build 缓存）：
   ```bash
   tools/agents/verify_package.sh <pkg>
   ```
   脚本用 `flock` 串行化；自动 checkout `pkg/<pkg>` → `./build.sh` → `unit_tests` → `diff_tests`
   （`NPP_VERIFY_DIR=artifacts/5090/npp_records`），结束后切回原分支；失败时保留日志
   `/tmp/verify_<pkg>_<stamp>.*.log`。
3. 通过后由 Lead `git merge --no-ff pkg/<pkg>` 并更新账本。

注意：
- 验证脚本会拒绝在 `run_package.sh`/`run_ref_agent.sh` 仍运行时执行（避免抢占 checkout）。
- 编码并行不等于验证并行：同一时间只有一个包在本地构建；包越多，验证队列越长。
- agent 未本地验证，合并前必须跑完整验证；不要跳过。

## 3. 并发上限建议

| 资源 | 建议 |
|---|---|
| GLM-5.3 网关 | 先 4 路，稳定后 6–8，最多 10；观察 429/超时 |
| C500 CPU | 20 核；10 个轻量 agent 可接受，构建仅验证脚本串行进行 |
| 5090 | 参考测试每路 `jobs=2`；避免 10 路同时全量构建 |
| 磁盘 | 每个 worktree/远程目录数百 MB，1.4TB 空闲足够 |

## 4. 待解决

- 若要让实现包真正并行构建，需要修复 cu-bridge/CMake 4.4 的 nvcc implicit-link 解析
  （补丁 CMake 模块、CMake 3.x、或修 nvcc `-dryrun` 输出），之后可在每个 worktree 独立 `--build`。
