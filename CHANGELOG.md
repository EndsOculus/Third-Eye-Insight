# Release Notes

[中文](#chinese-version) | [English](#english-version)

---

<a id="chinese-version"></a>

# 中文版本

## v2.5 - 2026-04-10

### ✨ 新增功能

- **进度条与可视化反馈**：
  - 矩阵计算阶段（语义、行为、网络、私聊指标）新增 tqdm 进度条，`desc` 显示计算类型，`unit` 为"条"或"对"
  - 整体分析流程用 8 步进度条（数据提取 → 时间筛选 → 用户过滤 → embedding 编码 → 指标计算 → 融合权重 → 可视化 → 报告生成）展示全局进度，每步前更新 `postfix_str` 说明当前阶段
  - embedding 编码新增进度显示：GPU 路径启用 `SentenceTransformer.show_progress_bar=True`，CPU 多线程路径用 tqdm 显示分块进度

- **缓存版本管理与清理**：
  - Embedding 缓存新增版本戳：缓存字典中写入 `"__version__"` 键（值为模型名），加载时若版本不匹配则自动丢弃旧缓存
  - 新增 `purge_stale_caches(model_name)` 函数：扫描 `output/cache/embeddings/` 目录，删除版本过期的 `.pkl` 文件，防止模型升级后旧缓存混用
  - 每次 `attach_text_embeddings` 调用时自动执行清理策略

- **运行结果自动整理与展示**：
  - 分析完成后，自动将本次结果复制到固定位置 `output/new/`，历史结果归档至 `output/archive/<时间戳>/`
  - 自动用 Windows 文件浏览器打开 `output/new/`，便于快速查看最新结果
  - 打开失败时仅记录警告，不中断程序

### 📊 参数配置化

- 6 个核心分析参数（`L=1000`, `TAU=150`, `WINDOW_SIZE=300`, `BATCH_SIZE=32`, `N_THREADS=4`, `TOP_N=20`）已移至 `config.json` 中 `analysis_params` 字段，`train.py` 启动时读取（失败回退默认值）

### ✅ 测试框架

- 新建 `tests/` 目录，`test_extract.py`、`test_visualization.py`、`test_entrypoint.py`、`test_run_test_import.py` 共 17 个单元测试用例，覆盖 `clean_message`、`_rank_name_candidates`、`_build_local_query`、`filter_for_gbk`、入口调用与脚本导入安全
- 运行方式：`uv run python -m pytest tests/ -v`

### 🔧 重构与精简

- 入口收敛：删除 `main.py`，统一保留 `train.py` 作为唯一启动入口
- 测试脚本精简：删除重复/临时脚本 `module_test.py`、`simple_test.py`、`test_logging.py`
- 修复测试收集副作用：`run_test.py` 改为 `main()` 包装，仅在脚本直跑时退出，避免 `pytest` 收集阶段触发 `SystemExit`
- 文档同步：更新 `HANDOVER.md` 与 `v2.4_COMPLETION.md` 的入口与验证命令说明

### ✅ 验证（补充）

- 模块校验通过：`uv run python verify_modules.py`
- 自动化测试通过：`python -m pytest tests -q`（17 passed）

---

## v2.4 - 2026-04-09

### ✨ 新增功能

- 模块化重构完成：`train.py`/`main.py` 作为薄入口，主流程拆分至 `app_pipeline.py`、`runtime_config.py`、`analysis_types.py`、`embedding_cache.py`、`metrics.py`、`reporting.py`。
- 新增 embedding 增量缓存：基于文本 SHA-256 进行命中，仅对未命中文本编码；缓存文件默认位于 `output/cache/embeddings/paraphrase-multilingual-MiniLM-L12-v2.pkl`。
- 新增聚焦用户统一视角输出：聚焦模式下热力图切换为单行视图，Top pairs/网络图/CSV/AI 报告统一约束为焦点用户相关关系。
- 完成日志系统替换：全链路由 `print()` 迁移至 `logging`，支持 `LOG_LEVEL` 控制输出级别（默认 `INFO`）。

### 🐛 Bug Fixes

- 修复 SQLCipher 群聊查询字段：`"40080"` 更正为 `CAST("40800" AS TEXT)`。
- 修复 SQLCipher 容错提取：由 `LIMIT/OFFSET` 改为按主键 `40001` 分段恢复，损坏页可跳过并继续提取后续数据。
- 修复 `interactive_config()` 中 `exclude_users` 未定义导致排除用户失效的问题。
- 修复 `top_pairs` 口径不一致问题：排序统一按最终 `IntimacyScore`，条形图分段展示加权分量。
- 修复聚焦报告误导问题：聚焦模式 prompt 禁止“绝对中心/绝对活跃度”叙述，改为角色与关系模式分析。
- 修复 SQLCipher 密码优先级回归：优先使用 `config.json.password`，环境变量 `SQLCIPHER_PASSWORD` 作为兜底（避免旧环境变量覆盖配置密码）。

### 📋 其他改进

- 输出编码标准化：`user_mapping.txt` 与 `interaction_scores*.csv` 统一为 `utf-8-sig`，兼容 Windows Excel 与跨平台读取。
- 项目入口统一：`main.py` 改为标准入口，`python main.py` 与 `python train.py` 行为一致。
- 输出目录治理：历史散落产物归档至 `output/archive/`，保持 `output/YYYY/MM/DD/<mode>/<id>/` 结构稳定。
- README/HANDOVER 同步更新：补充聚焦模式与 lite 模式行为边界、输出形态、密码优先级规则。

### ✅ 验证

- 语法校验通过：`python -m py_compile train.py`。
- 核心模块导入与基础调用验证通过：`analysis_types.py`、`embedding_cache.py`、`metrics.py`、`reporting.py`、`runtime_config.py`。
- 群聊回归验证通过：群号 `8*3`，时间范围 `2026/03/01` 至 `2026/03/31`，GPU + AI 报告启用，产物成功生成至 `output/2026/04/09/group/8*3/`。
- 聚焦一致性验证通过：聚焦用户 `2*7` 的 CSV 导出仅包含焦点用户相关用户对（非焦点-非焦点对为 0）。

---

<a id="english-version"></a>

# English Version

## v2.5 - 2026-04-10

### ✨ New Features

- **Progress Bars and Real-Time Feedback**:
  - Matrix computation stages (semantic, behavior, network, private metrics) now display tqdm progress bars with stage type in `desc` and row/pair counts in `unit`.
  - Overall pipeline uses 8-step progress bar (data extraction → time filtering → user exclusion → embedding encoding → metric computation → weight fusion → visualization → report generation), updating `postfix_str` before each stage.
  - Embedding encoding now shows progress: GPU path enables `SentenceTransformer.show_progress_bar=True`; CPU multi-threaded path displays chunk progress via tqdm.

- **Embedding Cache Versioning and Cleanup**:
  - Cache now includes version stamp: `"__version__"` key holds the model name. Mismatched versions trigger auto-discard of stale cache.
  - New `purge_stale_caches(model_name)` function scans `output/cache/embeddings/`, deletes outdated `.pkl` files to prevent model-upgrade cache pollution.
  - Cleanup runs automatically every time `attach_text_embeddings` is called.

- **Auto-Organization and Result Display**:
  - After analysis completes, outputs are copied to fixed location `output/new/`; historical results archive to `output/archive/<timestamp>/`.
  - Windows Explorer automatically opens `output/new/` for quick result access.
  - Open failures log warnings only; execution continues uninterrupted.

### 📊 Parameter Configuration

- 6 core analysis parameters (`L=1000`, `TAU=150`, `WINDOW_SIZE=300`, `BATCH_SIZE=32`, `N_THREADS=4`, `TOP_N=20`) moved to `config.json` under `analysis_params` field; `train.py` reads at startup with fallback defaults.

### ✅ Test Framework

- New `tests/` directory with `test_extract.py`, `test_visualization.py`, `test_entrypoint.py`, and `test_run_test_import.py`: 17 unit tests covering `clean_message`, `_rank_name_candidates`, `_build_local_query`, `filter_for_gbk`, entrypoint invocation, and import safety.
- Run via `uv run python -m pytest tests/ -v`.

### 🔧 Refactor and Simplification

- Entrypoint consolidation: removed `main.py`; `train.py` is now the single startup entrypoint.
- Test-script cleanup: removed redundant/temp scripts `module_test.py`, `simple_test.py`, and `test_logging.py`.
- Fixed pytest collection side effect: `run_test.py` now uses a guarded `main()` and exits only in direct execution, avoiding `SystemExit` during collection.
- Documentation sync: updated `HANDOVER.md` and `v2.4_COMPLETION.md` to reflect entrypoint and validation command changes.

### ✅ Validation (Update)

- Module verification passed: `uv run python verify_modules.py`.
- Automated test suite passed: `python -m pytest tests -q` (17 passed).

---

## v2.4 - 2026-04-09

### ✨ New Features

- Completed modular refactor: `train.py`/`main.py` are thin entrypoints; core workflow split into `app_pipeline.py`, `runtime_config.py`, `analysis_types.py`, `embedding_cache.py`, `metrics.py`, and `reporting.py`.
- Added incremental embedding cache: SHA-256 text-keyed cache with miss-only encoding; default path `output/cache/embeddings/paraphrase-multilingual-MiniLM-L12-v2.pkl`.
- Added focus-user unified output mode: in focus mode, heatmaps become single-row views, and Top pairs/network/CSV/AI report are constrained to focus-related relations.
- Completed logging migration: replaced `print()` with `logging` across the pipeline, with `LOG_LEVEL` runtime control.

### 🐛 Bug Fixes

- Fixed SQLCipher group message field: `"40080"` -> `CAST("40800" AS TEXT)`.
- Fixed SQLCipher resilient extraction path: replaced `LIMIT/OFFSET` with primary-key (`40001`) chunk recovery to skip damaged pages and continue reading.
- Fixed undefined `exclude_users` in `interactive_config()` that broke exclusion filtering.
- Fixed Top pairs consistency: ranking unified by final `IntimacyScore`, stacked bars show weighted component contributions.
- Fixed focus-report bias: focus prompt now forbids “absolute center/absolute activity” narratives and enforces role-pattern analysis.
- Fixed SQLCipher password precedence regression: prefer `config.json.password`, with `SQLCIPHER_PASSWORD` as fallback.

### 📋 Other Improvements

- Output encoding standardized: `user_mapping.txt` and `interaction_scores*.csv` now use `utf-8-sig` for Excel and cross-platform compatibility.
- Entrypoint normalization: `main.py` now behaves identically to `train.py`.
- Output directory hygiene: legacy artifacts moved to `output/archive/`; active outputs remain under `output/YYYY/MM/DD/<mode>/<id>/`.
- Documentation sync: README/HANDOVER updated with focus/lite behavior boundaries, output semantics, and password precedence.

### ✅ Validation

- Syntax check passed: `python -m py_compile train.py`.
- Core module import/basic invocation checks passed for `analysis_types.py`, `embedding_cache.py`, `metrics.py`, `reporting.py`, and `runtime_config.py`.
- Group regression completed: group `8*3`, time range `2026/03/01`-`2026/03/31`, GPU + AI report enabled, outputs generated under `output/2026/04/09/group/8*3/`.
- Focus consistency check passed: for focus user `2*7`, exported CSV contains only focus-related pairs (non-focus/non-focus pairs = 0).
