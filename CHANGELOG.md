# Release Notes

[中文](#chinese-version) | [English](#english-version)

---

<a id="chinese-version"></a>

# 中文版本

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
