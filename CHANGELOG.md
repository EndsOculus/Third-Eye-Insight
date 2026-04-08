# Release Notes

[点击查看英文版本](#english-version)

## v2.2 - 2026-04-08

### 新增功能

**排除用户功能**
- 交互式配置中新增"排除用户 QQ 号"输入项，支持逗号分隔多选。
- 过滤在时间筛选之后、嵌入计算之前执行，不影响其他流程。

**按日期+群号分级输出目录**
- 输出目录结构改为 `output/YYYY/MM/DD/<群号>/`，多次运行结果自动按日期归档，不再相互覆盖。

### 问题修复

**SQL 查询 rc=11 SQLITE_CORRUPT 修复**
- `_build_local_query` 中 WHERE 子句原先引用了 SELECT 别名 `content`，SQLCipher 无法解析导致部分群号报 `database disk image is malformed`。
- 修复为使用实际列名 `"40080"`，并将系统账号过滤移至 Python 层（`_postprocess`）。

**GPU 加速路径修复**
- `parallel_encode` 未将 `device` 参数传入子线程，导致即使选择 GPU 也全走 CPU。
- 修复：GPU 模式下改为单模型批量编码，CPU 模式保留多线程分块。

### 性能优化

**启动速度优化（懒导入）**
- 将 `sentence_transformers` 和 `scipy.optimize` 从文件顶部移除，改为在各自函数内部按需导入。
- 交互式配置界面现在可以立即响应，不再等待十余秒的重库初始化。

### 依赖管理

**PyTorch CUDA 版本通过 pyproject.toml 固定**
- 在 `pyproject.toml` 中通过 `[tool.uv.sources]` 指定本地 cu124 whl 文件。
- `uv sync` / `uv run` 不再自动降级为 CPU 版 torch。
- `requires-python` 限制为 `>=3.13,<3.14`，与 cp313 wheel 标签匹配。

---

## v2.1 - 2026-03-28

### 新增功能

**语义指标改为“真实交互对”计算**
- 在 `train.py` 中改为基于 5 分钟窗口内实际发生交互的消息对计算余弦相似度均值。
- 相比“用户平均向量互相比较”，该策略更贴近对话级回应关系。

**行为指标引入时间衰减**
- 行为互动矩阵改为 5 分钟滑动窗口统计并叠加指数衰减（`TAU=150s`）。
- 更近时间的共现贡献更高，弱化历史偶发共现噪声。

**网络指标改为 Jaccard 共同邻居系数**
- 网络得分由度中心性均值升级为共同邻居重叠比例。
- 更强调“圈层重叠”而非单纯活跃度。

**自动调权参数区间与回退值调整**
- `optimize_weights` 的 `bounds` 从 `[0.2, 0.4]` 调整为 `[0.1, 0.6]`。
- 初始值改为 `[1/3, 1/3, 1/3]`，失败回退为有效默认值 `[0.4, 0.4, 0.2]`。

**可视化新增 Top 互动对拆解图**
- 新增 `top_pairs.png`，按行为/语义/网络分量展示高分用户对。
- 新增综合亲密度热力图 `intimacy_heatmap.png` 并与时间区间命名联动。

### 问题修复

- **报告质量提升**：重构 AI 报告提示词，加入 Top 20 互动对表格、分指标解释与结构化分析要求。
- **模型切换**：DeepSeek 调用模型改为 `deepseek-chat`，增强稳定性与可用性。
- **排序一致性**：
	- `extract_chat_data.py` 的本地 SQLite 查询补充 `ORDER BY "40050"`。
	- PostgreSQL 的群聊/私聊查询补充 `ORDER BY time`。
	- `train.py` 在时间筛选后使用 `sort_values('timestamp').reset_index(drop=True)` 再次保证有序。

### 移除内容

- **旧版英文文档拆分方式**：取消 `readme_en.md` 独立维护，改为单文件双语 `readme.md`。

### 文档更新

- 中文 README 已更新为“亲密度”叙述，并补充滑动窗口语义、时间衰减行为、Jaccard 网络指标与新增可视化产物。
- 英文 README 已与中文内容对齐，补齐 Implementation Details、输出文件清单与指标定义。
- GitHub About 建议文案已同步更新，描述 SQLCipher 与远程 PostgreSQL 支持。

### 其他变更

- 自动调权策略保持默认融合权重为语义 0.4 / 行为 0.4 / 网络 0.2，失败回退策略更明确。
- 输出目录继续沿用时间后缀命名，便于多批次实验结果并存。

---

## English Version

<a id="english-version"></a>

## v2.2 - 2026-04-08

### New Features

**User exclusion**
- Added an interactive prompt for excluding specific QQ numbers (comma-separated, multi-select).
- Filtering runs after time-range selection and before embedding computation.

**Date-based output directory structure**
- Output path changed to `output/YYYY/MM/DD/<identifier>/` so results from multiple runs are automatically archived by date without overwriting each other.

### Bug Fixes

**SQL query rc=11 SQLITE_CORRUPT fix**
- `_build_local_query` referenced the SELECT alias `content` in the WHERE clause; SQLCipher cannot resolve aliases, causing some group IDs to return `database disk image is malformed`.
- Fixed by using the actual column name `"40080"` and moving system-account filtering to the Python layer in `_postprocess`.

**GPU acceleration path fix**
- `parallel_encode` was not forwarding the `device` parameter to worker threads, so GPU mode silently fell back to CPU.
- Fixed: GPU mode now uses single-model batch encoding; CPU mode retains multi-threaded chunked processing.

### Performance

**Lazy imports for faster startup**
- Moved `sentence_transformers` and `scipy.optimize` imports from the module top level into the functions that actually use them.
- The interactive configuration prompt is now immediately responsive instead of waiting 10+ seconds for heavy library initialization.

### Dependency Management

**PyTorch CUDA version pinned via pyproject.toml**
- `[tool.uv.sources]` in `pyproject.toml` now points to a local cu124 wheel file.
- `uv sync` / `uv run` no longer silently downgrades to the CPU-only torch.
- `requires-python` restricted to `>=3.13,<3.14` to match the cp313 wheel tag.

---

## v2.1 - 2026-03-28

### New Features

**Semantic metric switched to real interaction-pair scoring**
- `train.py` now computes semantic similarity from actual interaction message pairs inside a 5-minute sliding window.
- This is more conversation-level than comparing only global user-mean embeddings.

**Behavior metric upgraded with time decay**
- Behavioral interaction now uses sliding-window co-occurrence with exponential decay (`TAU=150s`).
- Recent interactions contribute more than older co-occurrences.

**Network metric upgraded to Jaccard common neighbors**
- Network score changed from degree-centrality average to common-neighbor overlap.
- Better reflects social-circle overlap rather than raw activity.

**Auto-weight tuning range and fallback updated**
- `optimize_weights` bounds changed from `[0.2, 0.4]` to `[0.1, 0.6]`.
- Initial weights changed to `[1/3, 1/3, 1/3]`, and fallback now uses valid defaults `[0.4, 0.4, 0.2]`.

**Visualization enhancements**
- Added `top_pairs.png` with behavior/semantic/network component breakdown.
- Added `intimacy_heatmap.png` and kept time-range-aware output naming.

### Bug Fixes

- **Higher-quality AI report output**: report prompt redesigned with Top-20 pair table, metric explanations, and structured analysis requirements.
- **Model update**: DeepSeek model switched to `deepseek-chat` for better stability.
- **Ordering consistency**: local SQLite query now uses `ORDER BY "40050"`, PostgreSQL group/private queries use `ORDER BY time`, and `train.py` enforces `sort_values('timestamp').reset_index(drop=True)` after time filtering.

### Removed

- **Separate English README workflow**: `readme_en.md` removed from standalone maintenance; bilingual content is now integrated in `readme.md`.

### Documentation

- Chinese README now describes intimacy-oriented scoring and the new semantic/behavior/network definitions.
- English README has been aligned with the Chinese section (implementation details, outputs, and metric definitions).
- GitHub About suggestion updated to reflect SQLCipher and remote PostgreSQL support.

### Other Changes

- Auto-weight defaults remain semantic 0.4 / behavior 0.4 / network 0.2 with clearer fallback behavior.
- Time-range suffix naming for output files is retained for experiment traceability.

### Implementation Notes

- `train.py` explicitly imports `deque` from `collections` for sliding-window semantic/behavior computation.
- Edge filtering in visualization path has been aligned to weighted interactions (`interaction_weights > 0`).
