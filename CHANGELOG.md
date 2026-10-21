## Unreleased — 安全与隔离只读兼容 / Security and isolated read-only compatibility

[中文](#unreleased-zh) | [English](#unreleased-en)

<a id="unreleased-zh"></a>

### 中文

- **发布范围**：本节记录未发布的源码变更，不新增版本号、发布日期或 Release；既有 Beta2.6 标签和发布资源不变。下方旧记录保留其当时的状态与验证范围。
- **依赖安全**：将 PyTorch 限制为 `>=2.10.0,<2.11`，处理 CVE-2025-2999 / GHSA-vgrw-7cvw-pwgx 的空 `PackedSequence` 原生崩溃。Windows CPU 2.10.0 的子进程回归验证了空输入被拒绝及正常输入仍可用；2.9.1 Windows CPU 构建未通过该回归。本更新不宣称修复独立的 CVE-2025-3000，也不认证其他平台或全部依赖安全，详见 [SECURITY.md](SECURITY.md)。
- **群聊读取兼容**：仅对明确识别的 1024 字节 NT 前缀、4096 字节页及保留页布局启用隔离只读进程，在子进程内调整 pending-byte 映射；不修改原库、DLL 或主进程的其他 SQLite 连接，也不执行数据库修复。普通 SQLite / SQLCipher 保持标准读取路径，界面区分兼容读取与读取失败。
- **兼容边界**：仅验证 Windows、SQLite `3.46.1` / SQLCipher `4.6.1 community`；使用的 `sqlite3_test_control` 是不保证跨版本稳定的测试接口，其他引擎版本或不匹配布局明确拒绝。兼容准备最多复制 4 GiB、限时 120 秒；非空 WAL/日志、来源变化、超时及取消均失败关闭。Windows Job、有限 IPC、批次背压及所属子进程清理覆盖退出路径；异常退出后仍可能留下临时文件，不保证安全擦除。
- **审核修正与来源记录**：群聊和私聊分别识别 `40027` / `40030`，消息键 `40001` 可选，无关精简表不阻碍打开，实际选中模式仍校验其必需字段。前缀复制同时计算原始/载荷 SHA-256；普通库不再强制全文件哈希，收据以空哈希及 `not_computed_standard` 明确标记，保留只读与来源变化检查。提供每次调用独立的 `config_file` 和计数演示入口，不记录聊天正文、不自动导入训练集。 详情界面区分合成演示、标准只读未计算哈希与缺失来源信息，不再将普通库误标为演示。
- **既有修复保留**：保留 PR #15 的导入限长、按相关会话限定缺失证据提示和重复别名修正，以及 PR #16 的依赖约束与安全回归；本次读取兼容不改变这些统计语义。
- **验证范围**：最终本地回归为 215 项非 GUI 加 18 项桌面测试全部通过，零失败或跳过；桌面用例含 15 项实际 Tk 测试和 3 项逻辑测试，lint 通过。新增回归覆盖模式字段/可选键、错误布局与引擎拒绝、普通库避免全文件哈希、前缀哈希和临时文件清理；既有用例覆盖取消、并发、重复打开、只读拒写及父进程映射不变。先前 PR 提交的托管 Windows CI 为 198 项通过、7 项因缺少受支持 SQLCipher 运行时跳过，另有 15 项桌面测试通过；该结果不冒充最终提交的托管验证。真实本地只读来源检查保留独立证据，不发布聊天正文或来源指纹；这不是全量消息正文解析、跨平台兼容或视觉验收的证明。

<a id="unreleased-en"></a>

### English

- **Release scope**: This section records unreleased source changes; it adds no version number, release date or Release. Existing Beta2.6 tags and published assets remain unchanged. Older entries below retain their status and validation scope at the time they were written.
- **Dependency security**: Restrict PyTorch to `>=2.10.0,<2.11` for the empty-`PackedSequence` native crash in CVE-2025-2999 / GHSA-vgrw-7cvw-pwgx. Child-process regressions verify rejection of empty input and continued handling of valid input on Windows CPU 2.10.0; the Windows CPU 2.9.1 build failed this regression. This update does not claim to fix the separate CVE-2025-3000 or certify other platforms or all dependencies. See [SECURITY.md](SECURITY.md).
- **Group-chat read compatibility**: Enable an isolated read-only process only for an explicitly recognized 1024-byte NT prefix, 4096-byte pages and reserved-page layout; adjust the pending-byte mapping inside that child. Do not modify the source database, DLL or other SQLite connections in the parent, or perform database repair. Ordinary SQLite / SQLCipher files retain standard reading; the UI distinguishes compatibility reads from read failures.
- **Compatibility limits**: Only Windows with SQLite `3.46.1` / SQLCipher `4.6.1 community` is verified. The `sqlite3_test_control` API is a test interface without cross-version stability guarantees; other engine versions or mismatched layouts are explicitly rejected. Compatibility preparation copies at most 4 GiB within 120 seconds; nonempty WAL/journals, source changes, timeouts and cancellation fail closed. Windows Jobs, bounded IPC, batch backpressure and cleanup of owned children cover exit paths. Abnormal termination may still leave temporary files; secure erasure is not guaranteed.
- **Review corrections and provenance**: Recognize group/private peer fields `40027` / `40030` respectively, make message key `40001` optional, and allow an unrelated reduced table; the selected-mode adapter still validates its required fields. Prefix copying computes source/payload SHA-256 together. Ordinary files no longer require a full-file hash scan: receipts explicitly use null hashes and `not_computed_standard`, while retaining read-only and source-change checks. Add per-call `config_file` and a count-only demo; do not log chat bodies or automatically import records into training data. Details distinguish synthetic demos, uncomputed standard-file hashes and missing provenance, rather than labeling ordinary files as demos.
- **Prior fixes retained**: Preserve PR #15's import limits, conversation-scoped missing-evidence warnings and duplicate-alias corrections, plus PR #16's dependency constraints and security regressions. Read compatibility does not change those analysis semantics.
- **Validation scope**: Final local regression evidence is 215 non-GUI plus 18 desktop tests passed, with zero failures or skips. Desktop coverage comprises 15 actual Tk tests and 3 logic tests; lint passes. New regressions cover mode-specific fields/optional keys, rejected layouts and engines, avoiding full-file hashes for ordinary files, prefix provenance and temporary-file cleanup. Existing cases cover cancellation, concurrency, repeated opens, write rejection and unchanged parent mapping. Hosted Windows CI on the earlier PR commit recorded 198 passes and 7 skips for the unavailable supported SQLCipher runtime, plus 15 desktop passes; this is not presented as hosted verification of the final commit. Real local read-only source checks retain separate evidence without publishing chat bodies or source fingerprints. They do not prove full message-body parsing, cross-platform compatibility or visual acceptance.

---

## Unreleased — 分批统计与私聊对方显示

- 去掉默认全局 50,000 行上限，以游标分批读取、磁盘辅助去重/索引和增量统计处理较大选区；保留可解释的内存、工作库、时间与参与者预算。
- 私聊按输入的本人 QQ 识别，只列对方；可靠备注/昵称优先，无昵称显示 QQ 并提示。新增匿名开关及一致的搜索、排序和详情展示，默认公共输出仍匿名。
- 修复冲突引用的平方级作者查询，联系人字段读取前限长；补充大数据取消/关闭、隐私切换和资源采样并发回归。
- 基于自动解密修复分支，尚未合并或发布；依赖安全升级另在独立 PR，不混入本变更。

<!-- translation: streaming -->
### English translation — Unreleased — Streaming statistics and private-chat peer display

- Remove the default global 50,000-row cap. Read cursor batches, use disk-assisted deduplication/indexing and incremental statistics for larger selections, while retaining explainable memory, working-database, time and participant budgets.
- Identify private chats using the entered self QQ identifier and list only peers. Prefer reliable contact notes/nicknames; otherwise display the QQ identifier with a notice. Add a privacy switch and consistent search, sorting and details, while keeping public output anonymous by default.
- Fix quadratic author queries for conflicting references and bound contact fields before reading. Add regressions for large-data cancel/close, privacy switching and concurrent resource sampling.
- Based on the automatic-decryption fix branch, not yet merged or released at the time of this entry. Dependency security upgrades are in a separate PR and are not part of this change.

<!-- /translation: streaming -->

## Unreleased — 自动识别与只读解密修复

- 修复未勾选 SQLCipher 时错误走普通 SQLite 的根因：按文件内容和配置自动识别，支持 1024 字节 NT 前缀，使用既有本地解密配置。
- 使用只读连接及隔离临时副本；拒绝非空 WAL/回滚日志，加入复制空间/时限、查询时限、变化检测和取消清理，错误采用固定脱敏提示。
- 修复 SQLCipher 含 NUL 文本字段截断，以及临时文件清理失败中断界面轮询的问题。
- 验证合成大数据预算、中文/空格路径、重复点击、取消/关闭、清理失败重试及真实离线库最小流程。真实库验证不发送外部模型，原文件前后哈希一致。无屏幕截图/视觉复核，仅验证真实 Tk 事件循环与控件行为。
- 启动仍为 Start Insight.vbs；调试入口 debug_desktop.ps1。Beta2.6 标签及已发布资源保持原样；此修复尚未发布。

<!-- translation: decryption -->
### English translation — Unreleased — Automatic detection and read-only decryption fixes

- Fix the cause of incorrectly using ordinary SQLite when SQLCipher is not selected: detect from file contents and configuration, support the 1024-byte NT prefix and use the existing local decryption configuration.
- Use read-only connections and isolated temporary copies. Reject nonempty WAL/rollback journals; add copy-space/time budgets, query deadlines, change detection and cancellation cleanup, with fixed redacted error messages.
- Fix truncation of SQLCipher text fields containing NUL and temporary-file cleanup failures interrupting GUI polling.
- Verify synthetic large-data budgets, Chinese/space-containing paths, repeated clicks, cancel/close, cleanup retries and a minimal real offline-database flow. Real-source validation sends nothing to external models and preserves the original file hash. No screenshots or visual review were performed; only the real Tk event loop and widget behavior were verified.
- Keep Start Insight.vbs as the launcher and debug_desktop.ps1 as the debug entry. The Beta2.6 tag and published assets remain unchanged; this fix was not yet released at the time of this entry.

<!-- /translation: decryption -->

## Beta2.6 / 2.6.0b1 — 2026-10-19

Source-only prerelease of the modern Windows desktop and interaction v2. Start with `uv sync --locked`, then `Start Insight.vbs`; no executable, dependencies or models are bundled. Optional local vector/generation assistance remains default-off and experimental. Generation holdout: 8 correct / 4 wrong targets among 12 predictions; not validated real-chat or emotion inference. See README.md and GENERATION.md for installation, upgrade boundaries, limits and privacy. This release supersedes the historical v2.5 notes below; only the default v2 desktop/CLI behavior describes this release.

<!-- translation: beta26 -->
### 中文译文 — Beta2.6 / 2.6.0b1 — 2026-10-19

现代 Windows 桌面与交互 v2 的纯源码预发行版。先运行 `uv sync --locked`，再启动 `Start Insight.vbs`；不捆绑可执行文件、依赖或模型。可选的本地向量/生成辅助仍默认关闭，且属于实验功能。生成留出评估：12 次预测中目标归属 8 次正确、4 次错误；未验证真实聊天或情感推断能力。安装、升级边界、限制和隐私说明见 README.md 与 GENERATION.md。本版本取代下方历史 v2.5 说明；只有默认 v2 桌面/CLI 行为描述本次发布。

<!-- /translation: beta26 -->

## Optional local generation — 2026-10-19

- Add default-off, fingerprinted local model classification and target candidates with validated anonymous evidence, separate from direct counts. No downloads, external API calls or raw model output in results.
- Keep the UI responsive with bounded stdin/output and nested Windows process jobs; fix synchronous stdin writes that could block timeout/cancel and check total limits after the final inference.
- Record the frozen independent synthetic holdout including four wrong attributions and limited topic/type accuracy. See GENERATION.md; no real-chat validity claim.
- Verify actual GPU-backed synthetic GUI completion/cancel/close and independent process cleanup regressions.

<!-- translation: generation -->
### 中文译文 — 可选本地生成 / Optional local generation — 2026-10-19

- 新增默认关闭、带指纹的本地模型分类与目标候选，使用经过校验的匿名证据，并与直接统计分开展示。无下载、外部 API 调用，结果中不含模型原始输出。
- 用有界标准输入/输出和嵌套 Windows 进程 Job 保持界面响应；修复同步标准输入写入可能阻塞超时/取消的问题，并在最后一次推理后检查总预算。
- 记录冻结的独立合成留出评估，其中包含四次错误归属及有限的主题/类型准确率。详见 GENERATION.md；不声称真实聊天有效性。
- 验证实际 GPU 支持下的合成 GUI 完成/取消/关闭，以及独立的进程清理回归。

<!-- /translation: generation -->

## Independent review corrections — 2026-10-19

- Restrict conflicting-message quality warnings to affected participants. An unrelated group's conflicting ID no longer changes a clean pair's evidence status; preserve the global conflict count.
- Regressions cover same-conversation unrelated members and every conflicting version's sender, mention target and quoted-source author.
- Collect explicit-link delay samples independently of turn deduplication; omit private lexical samples without unique source/target bodies.
- Compute simultaneous group/private turns from pre-batch state; sender renumbering cannot change counts. Expose direction-specific response/link/@ counts.
- Display incomplete directional values as unknown or observed counts with incomplete-structure status, and explicitly distinguish group-reference and private-transition latency.
- Parse bounded native 40800/45101 BLOB body elements, excluding 47423 cached quotes. Recipient fields remain unknown; preserve AI candidate-only scope.

<!-- translation: review-corrections -->
### 中文译文 — 独立审核修正 / Independent review corrections — 2026-10-19

- 将冲突消息质量提示限定于受影响的参与者；无关群组的冲突 ID 不再改变正常成员对的证据状态，同时保留全局冲突计数。
- 回归覆盖同一会话中的无关成员，以及每个冲突版本的发送者、提及目标和被引用来源作者。
- 独立于轮次去重收集显式链接的延迟样本；没有唯一来源/目标正文时，不输出私聊词汇样本。
- 根据批次处理前的状态计算同时发生的群聊/私聊轮次，发送者编号变化不再影响计数；显示分方向的响应/链接/@ 计数。
- 将不完整的方向值显示为未知，或显示为带结构不完整状态的观测计数，并明确区分群聊引用与私聊转接延迟。
- 解析有界的原生 40800/45101 BLOB 正文元素，排除 47423 缓存引用；接收者字段仍为未知，保留 AI 仅提供候选的范围。

<!-- /translation: review-corrections -->

## Interaction v2 — 2026-10-19

- Replace default weighted intimacy scores with auditable direct rounds, bidirectional sessions, raw counts, coverage and evidence aliases; remove unsupported certainty.
- Fix cross-session/private-pair leakage, same-timestamp ordering, source conflicts, date-limit ordering and untrusted native IDs. Conservative native metadata handling and explicit separate import metadata prevent body text from inventing replies.
- Share desktop and CLI core; retain hidden launch, bounded background processing, cancellation, search/pagination and compact accessible evidence details.
- Add default-off offline topic candidates with abstention and explicit limitations; actual synthetic holdout exposes a false attribution. No relationship/emotion classifier claim.
- Add counterexample, held-out, privacy-boundary and UI regression tests. See current validation and benchmark documents. Earlier entries below describe historical behavior only.

<!-- translation: interaction-v2 -->
### 中文译文 — 交互 v2 / Interaction v2 — 2026-10-19

- 用可审计的直接轮次、双向会话、原始计数、覆盖率和证据别名替代默认加权亲密度评分，移除缺乏依据的确定性表述。
- 修复跨会话/私聊成员对泄漏、同时间戳排序、来源冲突、日期限制顺序和不可信原生 ID；保守处理原生元数据，并明确区分导入元数据，防止正文凭空产生回复关系。
- 桌面与 CLI 共用核心；保留隐藏启动、有界后台处理、取消、搜索/分页和紧凑可访问的证据详情。
- 新增默认关闭、可弃权且明确限制的离线主题候选；实际合成留出评估暴露一次错误归属，不宣称具备关系/情感分类能力。
- 新增反例、留出、隐私边界及 UI 回归测试。详见当前验证和基准文档；下方更早记录仅描述历史行为。

<!-- /translation: interaction-v2 -->

## Visual acceptance fixes — 2026-10-19

- Real Tk screenshots revealed table/footer clipping; compact the form and chart, reserve result table height and keep search/pagination visible at 900x700.
- Add persistent synthetic-data identification and a visible progress-bar accent.
- Add default/compact geometry regression tests; screenshots remain private Library deliverables.

<!-- translation: visual -->
### 中文译文 — 视觉验收修正 / Visual acceptance fixes — 2026-10-19

- 实际 Tk 截图发现表格/页脚裁切；压缩表单与图表，为结果表保留高度，并在 900x700 下保持搜索/分页可见。
- 新增持续显示的合成数据标识，以及明显的进度条强调色。
- 新增默认/紧凑窗口尺寸回归测试；截图仍是私有 Library 交付物。

<!-- /translation: visual -->

## Independent review fixes — 2026-10-19

- Fix worker-exit race: drain final results/errors again after confirmed process exit, before closing its queue. Cancellation still skips potentially interrupted queue writes.
- Remove random semantic rank jitter. Only observed finite pairs are ranked; missing interactions remain zero and ties share a deterministic midrank. This corrects scores in both desktop and CLI without changing the CLI interface.

- Fix private-chat SQL placeholder binding: the self identifier now precedes peer identifiers, matching CASE then IN in the query.
- Initialize offline embeddings for desktop private analysis as well as group analysis; preserve the existing cosine-plus-lexical metric rather than silently changing its meaning.
- Reproduce failures using actual synthetic SQLite extraction; add single/multiple-peer extraction and desktop encoding regression tests without prefilled embeddings.

<!-- translation: review-fixes -->
### 中文译文 — 独立审核修复 / Independent review fixes — 2026-10-19

- 修复工作进程退出竞争：确认进程退出后、关闭队列前，再次取尽最终结果/错误。取消仍跳过可能被中断的队列写入。
- 移除随机语义排名扰动；仅对已观测的有限成员对排名，缺失交互保持为零，并列使用确定性平均名次。同步修正桌面与 CLI 评分，不改变 CLI 接口。
- 修复私聊 SQL 占位符绑定：本人标识先于对方标识，与查询中 CASE 后 IN 的顺序一致。
- 为桌面私聊分析初始化离线嵌入，与群聊一致；保留既有余弦加词汇指标，不暗中改变其含义。
- 用实际合成 SQLite 提取复现问题；新增单对方/多对方提取和桌面编码回归，不预填嵌入。

<!-- /translation: review-fixes -->

## Desktop modernization — 2026-10-19

- Add Tk desktop, pythonw launch, synthetic demo, validation, stage progress, cancellation, anonymous charts, search and pagination.
- Preserve CLI; desktop does not invoke external AI or export private data.
- Fix writable database handles, startup file deletion offer, pickle cache loading, truncated-result detection and exception cleanup.
- Batch CPU inference in one model; tuple window iteration and vectorized nickname fallback. Synthetic performance details in VALIDATION.md.
- Publish missing core modules/tests; replace local wheel lock and update bounded security dependencies.
- Encrypted private data, GPU, external services and native screenshots not validated. See VALIDATION.md and SECURITY.md.

<!-- translation: desktop -->
### 中文译文 — 桌面现代化 / Desktop modernization — 2026-10-19

- 新增 Tk 桌面、pythonw 启动、合成演示、校验、阶段进度、取消、匿名图表、搜索和分页。
- 保留 CLI；桌面不调用外部 AI，也不导出私有数据。
- 修复可写数据库句柄、启动时提供删文件选项、pickle 缓存加载、截断结果检测和异常清理问题。
- 单一模型批量执行 CPU 推理；采用元组窗口遍历和向量化昵称回退。合成性能细节见 VALIDATION.md。
- 发布缺失的核心模块/测试；替换本地 wheel 锁定并更新带边界的安全依赖。
- 加密私有数据、GPU、外部服务和原生截图未获验证。详见 VALIDATION.md 与 SECURITY.md。

<!-- /translation: desktop -->

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

### Desktop visual refinement

Flat white cards on a neutral backdrop, restrained body text, one purple primary action, clear section and statistics hierarchy, softer inputs, striped tables and explicit disabled states. Completed progress bars disappear. Both 1120x780 and 900x700 layouts were checked using real app-window captures with synthetic data.
