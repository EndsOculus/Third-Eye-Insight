# 恋之瞳 (Third Eye Insight)

[Click here to view the English version of this README](#english-version)

---

## 项目名称与来源

本项目名称"恋之瞳"源自《东方Project》中的角色古明地恋（Komeiji Koishi）。恋主动关闭了自己的第三只眼，将自身隐入他人的无意识之中——正如聊天记录中那些从未被人注意过的互动痕迹。项目的目标，正是从这些无意识留下的数据里，还原出两人之间真实的关系温度。
项目英文名“Third Eye Insight”也体现了通过数据分析来洞察群聊互动、捕捉用户之间隐含关系的理念。

---

## 项目差异与扩展

相比基于数据清洗与传统统计的前代方案，"恋之瞳"在以下几个方面有了显著提升：

- **🧠 深度学习文本嵌入**  
  使用预训练 SentenceTransformer 模型将每条消息转化为语义向量，更精准地捕捉两人之间"说话的对话感"，而不仅仅依赖词频统计。

- **📈 多指标融合**  
  群聊模式融合语义、行为、网络拓扑三项指标；私聊模式独立设计，融合行为特征、语义上下文、时间粘性，更贴切单对单关系。

- **🎉 灵活聚焦与排除分析**  
  支持以某个用户为中心的聚焦分析（只看该用户的互动关系）；还有精简模式，从源头只提取该用户参与的互动记录。支持排除指定 QQ 号，灵活剔除不需要分析的账号。

- **⚡ Embedding 增量缓存**  
  文本嵌入结果基于 SHA-256 缓存，同一条消息无需重复计算，大幅加速重跑速度。

- **🌐 远程数据库支持**  
  直接连接远程 PostgreSQL，满足多种部署和数据源场景。

- **🔐 SQLCipher 加密数据库支持**  
  通过 `config.json` 配置，可直接读取 NTQQ 原始加密数据库，无需手动导出转换。

- **🎨 全面可视化与 AI 报告生成**  
  输出 CSV、多维热力图、互动网络图、Top 互动对可视化，并自动生成 DeepSeek 驱动的结构化分析报告。

---

## 🌟 核心算法

"恋之瞳"用混合多指标模型来感知两个人之间的互动温度。群聊和私聊设计了不同的指标体系：

### 📊 群聊模式的三维视角

**1 💬 语义相似度：在乎彼此说的话**  
每条消息被 SentenceTransformer 转化为语义向量。我们不是简单比较全局平均向量，而是在 5 分钟滑动窗口内找出真正「互相回应」的消息对，计算它们的余弦相似度均值。这里体现的是"两人在认真交流"的程度。用 ECDF 离散化来放大细微差异。

**2 🤝 行为互动：在乎一起出现的频率**  
同样 5 分钟窗口内，统计两人以及其他用户的所有消息交互，施加指数时间衰减（τ=150s）——越久远的共现贡献越小。体现"互动的密集程度"。

**3 🔗 网络拓扑：在乎社交圈的重叠**  
用 Jaccard 共同邻居系数计算：两个人共同聊过的第三方，占他们各自聊过的总人数的比例。这反映"社交圈有多契合"，与两人本身是否是群里活跃分子无关。

### 📤 私聊模式的三维视角

**1. ⚕️ 行为特征**  
互动的对称性（谁更主动）、消息长度的对称性、响应延迟、主动发起的平衡度。

**2. 📣 语义上下文**  
相邻来回发言的语义连贯性，以及高频词/语气词的共现程度（专属小语境）。

**3. ⏱️ 时间粘性**  
活跃聊天天数的密度，以及深夜聊天（23:00-04:00）的占比——反映关系的"粘度"。

### 🖯 指标融合

将三个维度的指标按权重（默认：语义 0.4、行为 0.4、网络 0.2）融合成最终亲密度分数。支持 scipy 自动优化权重，最大化 score 方差来强化区分度。私聊模式使用固定权重：行为特征 0.4 / 语义上下文 0.4 / 时间粘性 0.2。

---

## 🚀 实现流程

1️⃣ **数据提取与清洗**

- 从数据库（SQLite 或远程 PostgreSQL）中提取群聊或私聊记录，自动过滤 QQ 号为 10000 和 2854196310 的系统消息。
- 合并群昵称（字段 40090）与 QQ 名称（字段 40093），默认优先使用群昵称。
- 清洗消息内容，保留中文、英文、数字、常见符号和 Emoji，无法显示的字符替换为 “?”。

2️⃣ **数据库预处理**

- 程序启动时会自动检测当前目录下是否存在 `nt_msg.db`，若存在则提示是否剥离前 1024 字节文件头生成 `nt_msg.clean_e.db`，并可选择删除原文件。

- 也可手动执行：
  
  ```bash
  python -c "open('nt_msg.clean_e.db','wb').write(open('nt_msg.db','rb').read()[1024:])"
  ```

- 加密数据库通过 `config.json` 配置 SQLCipher 参数（密码、页大小、KDF 迭代次数等），程序自动识别并解密读取。

3️⃣ **文本嵌入与特征计算**

- 使用批量处理和多线程加速计算文本嵌入。
- 群聊模式计算交互消息对语义相似度、行为互动加权量、Jaccard 网络得分。
- 私聊模式计算行为特征、语义上下文、时间粘性三类指标。

4️⃣ **指标融合与输出**

- 可选自动调整权重，使最终融合得分更加离散。
- 生成 CSV 文件、多维热力图（语义/行为/网络/综合亲密度）、网络图、Top 互动对条形图，同时生成用户映射文件和结构化 Markdown 分析报告。

---

## ✨ 特性

- **📄 数据提取与清洗**
  
  - 支持群聊（group_msg_table）和私聊（c2c_msg_table）的数据提取
  - 自动过滤系统消息（QQ 号 10000 与 2854196310）
  - 交互式时间范围筛选（YYYY/MM/DD 格式）
  - 支持排除指定 QQ 号（逗号分隔，可多选）

- **🤖 互动指标计算**
  
  - 文本嵌入（SentenceTransformer，多线程批量计算，带增量缓存）
  - 群聊模式：
    - 语义：交互消息对余弦相似度均值（ECDF 离散化）
    - 行为：5 分钟滑动窗口 + 指数时间衰减（τ=150s）
    - 网络：Jaccard 共同邻居系数
    - 自动权重优化（scipy 最大化方差）
  - 私聊模式：
    - 行为特征：互动对称性、响应延迟、主动发起平衡
    - 语义上下文：问答连贯性 + 高频词/语气词共现
    - 时间粘性：活跃天数密度 + 深夜聊天占比
    - 固定权重融合（0.4/0.4/0.2）

- **🎨 可视化与报告生成**
  
  - 热力图（语义/行为/网络/综合亲密度，按活跃度排序，对角遮蔽）
    - 群聊：完整方阵视图  
    - 聚焦模式：单行视图（焦点用户与其他人的互动）
  - Top N 互动对水平堆叠条形图（分量拆解，聚焦模式只显示焦点相关对）
  - 互动网络图（节点大小/颜色编码活跃度，边粗细/透明度编码强度）
  - AI 分析报告（DeepSeek）：结构化 prompt，包含 Top 20 互动对数据  
  - 文件名包含时间区间信息；输出编码统一为 UTF-8-SIG（Excel 兼容）

- **远程数据库支持**
  
  - 可直接使用数据库连接字符串访问远程 PostgreSQL 数据库

- **SQLCipher 加密数据库支持**
  
  - 通过 `config.json` 配置加密参数，自动识别并解密 NTQQ 加密数据库
  - 支持 `nt_msg.db` 文件头剥离（启动时自动检测并提示）

---

## ⚙️ 安装方法

确保使用 Python 3.13（Windows 官方安装包，非 MSYS2 内置 Python）。

**推荐：使用 `uv sync`（项目已含 pyproject.toml）**

```bash
uv sync
```

`pyproject.toml` 中通过 `[tool.uv.sources]` 将 torch 指向项目根目录的本地 `cu124` wheel，`uv sync` 会优先使用该文件安装 GPU 版 PyTorch。

本地 wheel 文件名也记录在 `config.json` 的 `torch_whl` 字段中；若你替换了 wheel 文件名，可执行：

```bash
pwsh ./apply_torch_source_from_config.ps1
```

然后重新运行 `uv lock` / `uv sync`。

> **MSYS2 Python 冲突**：若 PATH 中 MSYS2 Python 排在 Windows Python 之前，`uv` 命令会报 `Unknown operating system: mingw_x86_64_ucrt_gnu`。需显式指定：
> 
> ```bash
> uv sync --python "C:/Users/<你的用户名>/AppData/Local/Programs/Python/Python313/python.exe"
> ```

如需读取 SQLCipher 加密数据库，还需安装 MSYS2 并确保 `libsqlcipher-0.dll` 存在于 `C:/msys64/mingw64/bin/`（MSYS2 中执行 `pacman -S mingw-w64-x86_64-sqlcipher`）。

---

## 💻 使用方法

直接运行程序，按交互式提示操作：

```bash
uv run python train.py
```

如果已存在 `.venv` 且你只想直接复用它，也可以改用：

```bash
pwsh ./run_train.ps1
```

这个入口会直接复用当前 `.venv` 运行 `train.py`，不执行 `uv` 的预同步检查。若需要先用当前目录的本地 wheel 覆盖安装 `torch`，可用：

```bash
pwsh ./run_train.ps1 -EnsureLocalTorch
```

程序会依次询问以下配置项（括号内为默认值，直接回车接受）：

| 提示            | 说明                                                                     |
| ------------- | ---------------------------------------------------------------------- |
| 数据库文件路径       | 本地 SQLite 路径或 PostgreSQL 连接字符串，默认读取 `config.json` 的 `db_file`          |
| 分析模式          | 1 = 群聊，2 = 私聊                                                          |
| 群号 / 好友 QQ 号  | 根据模式填写对应 ID                                                            |
| 聚焦用户 QQ 号     | 可选，仅分析该用户相关互动；留空则分析所有用户                                                |
| 精简模式（Lite 模式） | 聚焦用户存在时可选。启用时：仅从源头保留焦点用户参与的互动记录（**前置过滤**）；禁用时：提取全量数据再以聚焦视角展示（**输出过滤**） |
| 排除用户 QQ 号     | 可选，逗号分隔多个 QQ 号，这些用户将从分析中移除                                             |
| 使用远程数据库       | 连接字符串模式（PostgreSQL）                                                    |
| 启用 GPU 加速     | 需要 CUDA 环境（torch cu124）                                                |
| 自动调整融合权重      | 使用 scipy 优化语义/行为/网络三权重                                                 |
| 生成 AI 分析报告    | 需设置环境变量 `DEEPSEEK_API_KEY`                                             |
| 图表中文字体        | 默认 Microsoft YaHei                                                     |

**关于聚焦模式与精简模式的区别：**

- **聚焦模式启用，精简模式关闭**：提取全量数据，分析完整的互动图谱，但输出（热力图、Top对、CSV、报告）只展示焦点用户相关的内容。适合想看焦点用户在全群的互动地位。
- **聚焦模式启用，精简模式启用**：从源头只提取焦点用户参与的会话，大幅缩小数据体量，加速分析。适合只关心焦点用户的独立互动数据。

数据提取完成后还会提示输入时间范围（格式 YYYY/MM/DD），直接回车表示不限制。

---

## 🧭 项目结构（含运行后目录）

```text
恋之瞳/
├── train.py                            # 主入口（交互式分析流程）
├── app_pipeline.py                     # 主流程编排（提取→计算→可视化→报告）
├── extract_chat_data.py                # SQLite/SQLCipher/PostgreSQL 数据提取
├── metrics.py                          # 群聊/私聊指标计算与融合
├── visualization.py                    # 热力图、网络图、Top 对图
├── reporting.py                        # CSV 导出与 AI 报告
├── runtime_config.py                   # 交互配置、时间筛选、输出路径
├── embedding_cache.py                  # embedding 增量缓存
├── config.json                         # SQLCipher 与分析参数配置
├── output/                             # [运行后自动生成]
│   ├── YYYY/MM/DD/
│   │   ├── group/<群号>/
│   │   │   ├── interaction_scores*.csv
│   │   │   ├── semantic_heatmap*.png
│   │   │   ├── behavior_heatmap*.png
│   │   │   ├── network_heatmap*.png
│   │   │   ├── intimacy_heatmap*.png
│   │   │   ├── top_pairs*.png
│   │   │   ├── interaction_network*.png
│   │   │   ├── user_mapping.txt
│   │   │   └── analysis_report.md      # 可选
│   │   └── private/<QQ号>/             # 文件集合与群聊目录同构
│   ├── new/                            # 最新一次运行结果快照
│   ├── archive/<YYYYMMDD-HHMMSS>/      # 历史 new 目录自动归档
│   └── cache/embeddings/
│       └── paraphrase-multilingual-MiniLM-L12-v2.pkl
├── nt_msg.clean_e.db                   # [可选] 由 nt_msg.db 剥离 1024B 文件头生成
└── README.md
```

说明：

- 每次运行会输出到 `output/YYYY/MM/DD/<mode>/<id>/`。
- 程序结束后会把本次结果复制到 `output/new/`，并把旧的 `output/new/` 自动移动到 `output/archive/<时间戳>/`。
- embedding 缓存常驻在 `output/cache/embeddings/`，用于后续增量复用。

---

## 📊 数据库预处理说明

- **SQLite 加密数据库（NTQQ nt_msg.db）**：
  程序启动时自动检测当前目录下的 `nt_msg.db`，提示是否剥离前 1024 字节文件头生成 `nt_msg.clean_e.db`，并可选择删除原文件。
  也可手动执行：
  
  ```bash
  python -c "open('nt_msg.clean_e.db','wb').write(open('nt_msg.db','rb').read()[1024:])"
  ```

- **config.json 加密配置**：
  在项目目录创建 `config.json`，配置 SQLCipher 解密参数：
  
  ```json
  {
    "db_file": "nt_msg.clean_e.db",
    "encrypted": true,
    "sqlcipher_dll": "C:/msys64/mingw64/bin/libsqlcipher-0.dll",
    "cipher_page_size": 4096,
    "kdf_iter": 4000,
    "cipher_hmac_algorithm": "HMAC_SHA1",
    "cipher_kdf_algorithm": "PBKDF2_HMAC_SHA512",
    "password": "your_password_here"
  }
  ```
  
  `encrypted` 设为 `false` 或省略时，直接使用明文 SQLite。

- **PostgreSQL 数据库**：
  确保连接字符串正确，格式：
  
  ```
  postgresql://username:password@host:port/dbname
  ```

---

## 📈 输出结果

程序输出的文件均存储在按模式区分的目录下：

- 群聊：`output/YYYY/MM/DD/group/<群号>/`
- 私聊：`output/YYYY/MM/DD/private/<QQ号>/`

| 文件                        | 说明                                                                           |
| ------------------------- | ---------------------------------------------------------------------------- |
| `interaction_scores.csv`  | 用户对得分表。包含 QQ、昵称、各维度得分。群聊：行为/语义/网络/亲密度；私聊：行为特征/语义上下文/时间粘性/亲密度。聚焦模式下仅包含焦点用户相关对 |
| `semantic_heatmap.png`    | 语义相似度热力图。群聊为完整方阵；聚焦模式为单行（焦点用户与他人）；私聊为单行                                      |
| `behavior_heatmap.png`    | 行为互动热力图。群聊为完整方阵；聚焦模式为单行；私聊为单行                                                |
| `network_heatmap.png`     | 网络/拓扑热力图。群聊为完整方阵；聚焦模式为单行；私聊为时间粘性热力图                                          |
| `intimacy_heatmap.png`    | 综合亲密度热力图。群聊为完整方阵；聚焦模式为单行；私聊为单行                                               |
| `top_pairs.png`           | Top 20 互动对水平条形图。拆解各维度的加权贡献。聚焦模式下仅显示焦点用户相关对                                   |
| `interaction_network.png` | 互动网络图。节点大小=活跃度，边粗细/透明度=亲密度。聚焦模式下只显示焦点用户的连接                                   |
| `user_mapping.txt`        | 用户索引、QQ 号、昵称对照表（UTF-8-SIG 编码）                                                |
| `analysis_report.md`      | （可选）DeepSeek AI 生成的结构化人际关系分析报告。聚焦模式下视角聚焦于焦点用户                                |

文件名中包含时间区间信息（如 `_2025-01-01-end`）。

所有 CSV 和文本文件均采用 **UTF-8-SIG 编码**，兼容 Windows Excel 和跨平台读取。

私聊模式下，联系人昵称会优先从私聊历史记录中回查可读名称；若数据库中长期缺失昵称，仍会回退为 QQ 号显示。为避免昵称混乱，补全逻辑仅使用该联系人本人发言时留下的昵称，不会用"对方会话对象字段"反推。

---

## 📜 许可证

本项目采用 MIT 许可证。

---

## 🙏 鸣谢

- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [NetworkX](https://networkx.org/)
- [PyTorch](https://pytorch.org/)
- [SentenceTransformers](https://www.sbert.net/)
- [Scipy](https://www.scipy.org/)
- [DeepSeek API](https://api.deepseek.com/)

---

<a id="english-version"></a>

# Third Eye Insight (恋之瞳)

## Project Name and Origin

The Chinese name “恋之瞳” is inspired by Komeiji Koishi from Touhou Project. Koishi closes her third eye and slips into others' unconscious perception, much like those subtle interaction traces in chat history that people rarely notice. This project aims to recover the real “relationship temperature” between users from those latent traces.

The English name “Third Eye Insight” reflects the same idea: using data analysis to understand hidden interaction patterns in chats.

---

## 🚀 Differences and Extensions

Compared with earlier data-cleaning + traditional-statistics approaches, this project adds:

- **🧠 Deep-learning text embedding**
  Uses a pretrained SentenceTransformer model to map every message into a semantic vector, capturing dialogue-level resonance instead of relying on token frequency only.

- **📈 Multi-metric fusion**
  Group-chat mode fuses semantic, behavioral, and topology metrics. Private-chat mode uses a dedicated design with behavioral traits, semantic context, and time stickiness.

- **🎉 Flexible focus and exclusion**
  Supports focus-user analysis (show only relations around one user), optional Lite mode (source-level pre-filtering), and exclusion of comma-separated QQ IDs.

- **⚡ Incremental embedding cache**
  SHA-256 based embedding cache avoids recomputation of repeated messages and significantly speeds up reruns.

- **🌐 Remote database support**
  Connects directly to remote PostgreSQL for flexible deployment.

- **🔐 SQLCipher encrypted database support**
  Reads NTQQ encrypted databases directly with `config.json`, without manual conversion.

- **🎨 Full visualization and AI reporting**
  Exports CSV, heatmaps, interaction network, Top-pairs chart, and an optional structured DeepSeek report.

---

## 🌟 Core Algorithm

Third Eye Insight models interaction intimacy with hybrid metrics. Group and private chat use different metric systems.

### 📊 Group Chat: Three Dimensions

1. **💬 Semantic Similarity**
   Messages are encoded with SentenceTransformer. Instead of only comparing global user centroids, the program detects real interaction pairs in a 5-minute sliding window and averages cosine similarity. ECDF mapping amplifies subtle differences.

2. **🤝 Behavioral Interaction**
   Within the same 5-minute window, cross-user interactions are counted with exponential time decay ($\tau=150s$). Older co-occurrences contribute less.

3. **🔗 Network Topology**
   Uses Jaccard common-neighbor coefficient to measure overlap in social circles, independent of absolute activity volume.

### 📤 Private Chat: Three Dimensions

1. **⚕️ Behavioral Traits**
   Symmetry of interaction, message-length symmetry, response delay, and initiative balance.

2. **📣 Semantic Context**
   Coherence of adjacent turns plus high-frequency word/particle co-occurrence.

3. **⏱️ Time Stickiness**
   Active-day density and late-night chat ratio (23:00-04:00).

### 🧮 Fusion

Group-chat default fusion weights: semantic 0.4 / behavior 0.4 / topology 0.2, with optional scipy optimization to maximize score variance.

Private-chat fixed fusion weights: behavioral traits 0.4 / semantic context 0.4 / time stickiness 0.2.

---

## ⚙️ Installation

Ensure Python 3.13 is used (official Windows installer, not MSYS2 built-in Python).

**Recommended: use `uv sync` (this repo includes `pyproject.toml`)**

```bash
uv sync
```

`pyproject.toml` maps torch to the local `cu124` wheel via `[tool.uv.sources]`, so `uv sync` installs GPU PyTorch from that file.

The local wheel name is also stored in `config.json` as `torch_whl`. If you replace the wheel filename, run:

```bash
pwsh ./apply_torch_source_from_config.ps1
```

Then run `uv lock` / `uv sync` again.

> **MSYS2 Python conflict**: if MSYS2 Python appears before Windows Python in PATH, `uv` can fail with `Unknown operating system: mingw_x86_64_ucrt_gnu`. Use:
>
> ```bash
> uv sync --python "C:/Users/<your-username>/AppData/Local/Programs/Python/Python313/python.exe"
> ```

For SQLCipher support, install MSYS2 and ensure `libsqlcipher-0.dll` exists at `C:/msys64/mingw64/bin/`.

---

## 💻 Usage

Run interactively:

```bash
uv run python train.py
```

If `.venv` already exists and you want to reuse it directly:

```bash
pwsh ./run_train.ps1
```

If you need to force reinstall local wheel torch before launch:

```bash
pwsh ./run_train.ps1 -EnsureLocalTorch
```

Prompts shown during startup:

| Prompt                     | Description                                                                               |
| -------------------------- | ----------------------------------------------------------------------------------------- |
| Database path              | Local SQLite path or PostgreSQL connection string; defaults to `db_file` in `config.json` |
| Analysis mode              | 1 = group chat, 2 = private chat                                                          |
| Group / QQ number          | Group ID or friend QQ number depending on mode                                            |
| Focus user QQ number       | Optional; analyze only relations around this user                                          |
| Lite mode                  | Optional when focus user is set; source-level filtering                                    |
| Exclude user QQ numbers    | Optional; comma-separated QQ IDs to remove                                                |
| Remote database            | PostgreSQL connection-string mode                                                         |
| GPU acceleration           | Requires CUDA environment (torch cu124)                                                   |
| Auto-adjust fusion weights | Uses scipy optimization                                                                   |
| Generate AI report         | Requires `DEEPSEEK_API_KEY` environment variable                                          |
| Chart font                 | Default: Microsoft YaHei                                                                  |

You will also be prompted for an optional time range (format: YYYY/MM/DD).

**Focus vs Lite mode:**

- **Focus ON + Lite OFF**: extract full data, compute the full interaction graph, but output only focus-related rows/edges.
- **Focus ON + Lite ON**: keep only focus-related records at extraction stage for faster runs and smaller data volume.

---

## 🧭 Project Structure (Including Runtime-Generated Folders)

```text
Third-Eye-Insight/
├── train.py                            # Main entrypoint (interactive pipeline)
├── app_pipeline.py                     # Pipeline orchestration
├── extract_chat_data.py                # SQLite/SQLCipher/PostgreSQL extraction
├── metrics.py                          # Metric computation and fusion
├── visualization.py                    # Heatmaps, network graph, top-pairs chart
├── reporting.py                        # CSV export and AI report generation
├── runtime_config.py                   # Interactive config, filters, output path
├── embedding_cache.py                  # Incremental embedding cache
├── config.json                         # SQLCipher + analysis params config
├── output/                             # [generated at runtime]
│   ├── YYYY/MM/DD/
│   │   ├── group/<group-id>/
│   │   │   ├── interaction_scores*.csv
│   │   │   ├── semantic_heatmap*.png
│   │   │   ├── behavior_heatmap*.png
│   │   │   ├── network_heatmap*.png
│   │   │   ├── intimacy_heatmap*.png
│   │   │   ├── top_pairs*.png
│   │   │   ├── interaction_network*.png
│   │   │   ├── user_mapping.txt
│   │   │   └── analysis_report.md      # optional
│   │   └── private/<qq-id>/            # same artifact set
│   ├── new/                            # latest run snapshot
│   ├── archive/<YYYYMMDD-HHMMSS>/      # auto-archived previous `new/`
│   └── cache/embeddings/
│       └── paraphrase-multilingual-MiniLM-L12-v2.pkl
├── nt_msg.clean_e.db                   # [optional] generated from nt_msg.db
└── README.md
```

Notes:

- Each run writes artifacts to `output/YYYY/MM/DD/<mode>/<id>/`.
- On completion, current artifacts are copied to `output/new/`.
- Previous `output/new/` is automatically moved to `output/archive/<timestamp>/`.
- Embedding cache persists in `output/cache/embeddings/` for incremental reuse.

---

## 📊 Database Preprocessing

- **SQLite encrypted database (NTQQ `nt_msg.db`)**:
  On startup, the program can detect `nt_msg.db` and prompt to strip the first 1024 bytes into `nt_msg.clean_e.db`. You can also do this manually:

  ```bash
  python -c "open('nt_msg.clean_e.db','wb').write(open('nt_msg.db','rb').read()[1024:])"
  ```

- **`config.json` SQLCipher settings**:

  ```json
  {
    "db_file": "nt_msg.clean_e.db",
    "encrypted": true,
    "sqlcipher_dll": "C:/msys64/mingw64/bin/libsqlcipher-0.dll",
    "cipher_page_size": 4096,
    "kdf_iter": 4000,
    "cipher_hmac_algorithm": "HMAC_SHA1",
    "cipher_kdf_algorithm": "PBKDF2_HMAC_SHA512",
    "password": "your_password_here"
  }
  ```

  Set `encrypted` to `false` (or omit it) for plaintext SQLite.

- **PostgreSQL**:
  Make sure connection string is valid:

  ```
  postgresql://username:password@host:port/dbname
  ```

---

## 📈 Outputs

Primary output directories:

- Group chat: `output/YYYY/MM/DD/group/<group-id>/`
- Private chat: `output/YYYY/MM/DD/private/<qq-id>/`

Additional runtime directories:

- Latest snapshot: `output/new/`
- Historical snapshot archive: `output/archive/<timestamp>/`
- Embedding cache: `output/cache/embeddings/`

| File                      | Description                                                                                                   |
| ------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `interaction_scores*.csv` | Pairwise score table (QQ IDs, nicknames, metric components, final intimacy). Focus mode includes focus-only pairs |
| `semantic_heatmap*.png`   | Semantic heatmap: full matrix in normal mode; focus-row in focus mode                                        |
| `behavior_heatmap*.png`   | Behavioral heatmap                                                                                             |
| `network_heatmap*.png`    | Group mode: topology heatmap; private mode: stickiness heatmap                                                |
| `intimacy_heatmap*.png`   | Final intimacy heatmap                                                                                         |
| `top_pairs*.png`          | Top interaction pairs stacked bar chart (weighted component decomposition)                                     |
| `interaction_network*.png`| Interaction network graph (node size/color = activity, edge width/alpha = strength)                           |
| `user_mapping.txt`        | User index / QQ / nickname mapping                                                                             |
| `analysis_report.md`      | Optional structured AI report (DeepSeek)                                                                       |

Filenames include selected time range suffixes (for example, `_2025-01-01-end`).

All CSV and text files are encoded in **UTF-8-SIG**, compatible with Windows Excel and cross-platform readers.

In private-chat mode, contact nicknames are preferably backfilled from that contact's own historical messages. If missing, QQ ID fallback is used. To avoid nickname pollution, no reverse inference is made from counterpart-session fields.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [NetworkX](https://networkx.org/)
- [PyTorch](https://pytorch.org/)
- [SentenceTransformers](https://www.sbert.net/)
- [Scipy](https://www.scipy.org/)
- [DeepSeek API](https://api.deepseek.com/)
