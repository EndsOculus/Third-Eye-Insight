# 第三眼 · Third Eye Insight

本地聊天互动证据查看器。v2 展示可解释的次数、观察时间和结构覆盖，不给亲密度、感情或关系强弱评分。

## 启动

Windows / Python 3.13：`uv sync --locked`。双击 **Start Insight.vbs**，正常入口及后台任务使用 pythonw，不弹命令行窗口。禁用 VBScript 时可为 `.venv\Scripts\pythonw.exe desktop.py` 创建快捷方式。调试入口：`debug_desktop.ps1`。

先点击合成演示即可体验。支持群聊/私聊、日期筛选、搜索、每页 100 行、证据详情、取消和关闭。900×700 窗口保留搜索和分页；耗时分析在独立进程执行。固定事件日志写入忽略的 `output/logs/desktop.log`，不记录正文与账号。

默认分析不需要模型。可选“本地话题候选”只使用已有离线向量模型，默认关闭，不发送外部 API、不下载模型、不写向量缓存。它不能可靠判断回答、安慰、争吵或情感，且不会修改直接互动次数，详见 [AI_ASSISTANCE.md](AI_ASSISTANCE.md)。

## 数据与解释

原生 NT SQLite 只读访问；现支持有限 BLOB 正文提取并排除引用缓存，但尚未验证引用目标/@ 收件人映射，因此群聊方向证据可能显示“未知”，绝不把文字 @ 或共同活跃当成回复。另支持明确选择的结构导入表。日期在 SQL 中先筛选再检查数量上限，按 Asia/Shanghai 日历日理解。上限 50,000 行、300 成员、100,000 个收件人证据；超过上限需缩小范围，不静默截断。

SQLCipher 仅在用户勾选时读取本地私有配置；不要提交配置或原始库。测试未接触真实聊天库。输入路径/账号控件仍包含用户输入，截图前应切回演示并清空输入。结果匿名化是展示保护，不保证统计无法被关联识别。

定义与导入格式见 [INTERACTION_V2.md](INTERACTION_V2.md)。未知不等于零；数量不是关系质量。旧加权/语义亲密分已退出默认桌面和 CLI 路径。

## CLI 与验证

`.venv\Scripts\python.exe train.py --demo --output demo-result.json`

`train.py --help` 列出同一 v2 核心的参数，`run_train.ps1` 转发参数。输出仅新建文件，拒绝覆盖。旧 PostgreSQL、报告与图片导出不属于当前默认 CLI；遗留模块只保留作兼容源码与回归参考。

`uv sync --locked --group dev` 后运行 `.venv\Scripts\python.exe -m pytest tests -q`。Windows 若连续重建 Tk 解释器报初始化错误，请按 CI 工作流将 tests/test_desktop.py 的每个节点置于独立进程运行，保留全部断言。测试、性能口径和限制见 [VALIDATION.md](VALIDATION.md)、[BENCHMARK_V2.md](BENCHMARK_V2.md)、[SECURITY.md](SECURITY.md)。

回滚基线为 `a942818ad2bdbc46543fbc956f8eca2e4a520a65`；源码 Git bundle 留在本机交付工作区，未入仓。建议新建恢复分支，不用 hard reset 覆盖未提交工作。

## Optional generative experiment

The assistance selector now separates off, vector candidates and default-off local generation. Generation uses an explicitly configured local runtime/model; it can classify coarse topics and utterance types and propose uncertain targets with anonymous evidence IDs. The independent synthetic holdout includes four wrong target predictions, so this is not validated emotion/relationship inference. See [GENERATION.md](GENERATION.md) for configuration, actual results, limits and cancellation behavior.
