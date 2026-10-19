# 恋之瞳 · Third Eye Insight

本地聊天互动证据查看器。新版桌面以匿名成员、可追溯行号、互动次数和结构覆盖展示结果；不再输出亲密度、情感或关系强弱总分。

## 安装与启动

支持 Windows / Python 3.13；先准备 Python 3.13 和 `uv`，然后在项目目录执行：

```powershell
uv sync --locked
```

安装依赖可能联网；正常分析不会下载模型或调用外部 AI。安装完成后双击 **Start Insight.vbs**，主窗口和后台任务均使用 pythonw，不弹命令行窗口。如果系统禁用 VBScript，可为以下命令创建快捷方式：

```powershell
.venv\Scripts\pythonw.exe desktop.py
```

调试入口为 `debug_desktop.ps1`。建议先点“体验合成演示”：界面支持群聊/私聊、日期筛选、匿名成员搜索、每页 100 行、图表与滚动证据详情；900×700 窗口保留结果和分页区。耗时工作在后台进程执行，可取消或关闭，重复点击不会启动重复任务。

## 输入与 v2 指标

选择只读 SQLite 数据库及对应格式：原生 NT 或显式结构导入。群聊填写群号，私聊填写自己的 QQ；日期按 Asia/Shanghai 理解，先 SQL 筛选再检查上限。限制为 50,000 行、300 成员、100,000 条原始定向证据，超限提示缩小范围，不静默截断。SQLCipher 仅在显式勾选时读取本机私有配置；不要提交配置或修改原始库。

v2 展示回应轮次、双向会话、定向引用/@ 次数、发言量、共同活跃和有效样本延迟。群聊明确引用与私聊发言转换使用不同口径；同话题或同时出现不自动算作回应。原生 NT 只支持有限正文与引用缓存标记解码，引用目标/@ 收件人映射尚未验证，因此方向数据可能是“未知”。

**未知不等于零。** 结构不完整时，正数显示“已观测 N”，也不是下限：补全事件可能合并原先分开的轮次或会话。延迟和比例仅针对有效已观察样本。指标不能证明关系好坏或真实情感。完整定义与导入格式见 [INTERACTION_V2.md](INTERACTION_V2.md)。

## 可选本地 AI：默认关闭

辅助选择器有三种模式，AI 候选始终不改变直接统计：

| 模式 | 配置与用途 | 边界 |
|---|---|---|
| 辅助关闭 | 默认；无需模型 | 只计算直接证据 |
| 向量候选 | 仅使用已存在的 multilingual MiniLM 本地快照，位置与固定版本见 [AI_ASSISTANCE.md](AI_ASSISTANCE.md) | 最多 2,000 条；相似度不是回应事实或置信概率 |
| 生成实验 | 点“生成设置”选择仓库外的本地 JSON，显式指定运行时、模型和 CUDA DLL 目录 | 最多 12 条；实验性对象、粗话题、行为类型候选及匿名证据说明 |

生成配置只包含三个字段，值必须是已有资源的绝对本地路径；以下是占位示例，不是可直接运行的配置：

```json
{
  "runtime": "C:/approved-runtime/llama-cli.exe",
  "model": "C:/approved-models/Qwen3-4B-Q4_K_M.gguf",
  "dll_dir": "C:/approved-cuda-dlls"
}
```

仅支持代码中核验指纹的 llama.cpp b11723 Windows CUDA 12.4 运行时与 Qwen3-4B-Q4_K_M；不自动获取资源，缺失或指纹不符会显示不可用。配置路径仅留在当前桌面会话；CLI 可用 `--generate-assist --generation-config <local-config.json>`。运行时子进程使用离线参数和 loopback 主机设置；这是本地运行边界，不是完整网络隔离认证。详细资源指纹、输入/时间限制与命令见 [GENERATION.md](GENERATION.md)。

**实际合成评估并不证明真实聊天有效性。** 独立 16 例中，生成模式提出 12 个对象候选，8 对、4 错；话题正确 6/16，行为类型正确 9/16，并误判一个合成指令注入案例。生成说明是模型所选类别加验证后的行号模板，不展示自由生成正文，也不声称识别真实情感。8 条合成 GUI 演示约 20 秒完成；这不是通用速度承诺或优化倍数。向量模式的独立评测也出现过错误归因。

## 隐私与安全

原始数据库只读，不强行解锁、不关闭用户应用、不写原始库。正常 v2 分析不调用外部 AI、不自动下载模型、不持久化聊天向量；生成运行时不接收工具执行权限。界面中经过校验的模型结果、固定事件日志和导出结果不包含聊天正文或原始账号/消息 ID，只保留匿名成员与选区行号。固定事件日志保存在被忽略的 `output/logs/desktop.log`。

输入控件仍会显示用户填写的路径和账号；截图前切换合成演示并清空输入。时间、次数和匿名标识仍可能被关联识别，匿名化不是绝对保证。数据库/WAL/SHM、聊天导出、私有配置、模型、日志和私人截图不可提交；测试不应把真实聊天发送给外部模型。当前验证使用合成数据，未验证真实加密私聊库。检查范围与剩余风险见 [SECURITY.md](SECURITY.md)。

## CLI、测试与恢复

默认 CLI 与桌面共享同一 v2 核心，输出仅允许新建文件：

```powershell
.venv\Scripts\python.exe train.py --demo --output demo-result.json
.venv\Scripts\python.exe train.py --help
uv sync --locked --group dev
.venv\Scripts\python.exe -m pytest tests -q --ignore=tests/test_desktop.py
```

Windows 的 Tk 工作流按 [.github/workflows/tests.yml](.github/workflows/tests.yml) 将每个 `tests/test_desktop.py` 测试节点放在独立进程运行，全部执行、不跳过。最新版验证含真实事件循环、取消/关闭、中文与空格路径、只读提取、44,850 对合成结果的分页搜索、实际本地模型运行和独立只读评审。详见 [VALIDATION.md](VALIDATION.md)、[BENCHMARK_V2.md](BENCHMARK_V2.md)、[GENERATION.md](GENERATION.md) 和 [CHANGELOG.md](CHANGELOG.md)。

旧 PostgreSQL、外部评分和图片导出不在当前默认 CLI 路径，历史模块保留供溯源。生成实验之前的恢复提交为 `b137c152b4d3727c8a9cbdf31703b83c369ade53`；更早的源码恢复点为 `a942818ad2bdbc46543fbc956f8eca2e4a520a65`。本机已验证的源码 bundle 保存在仓库外。建议新建恢复分支，不用 hard reset 覆盖未提交工作。
