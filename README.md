# 觉之瞳 (Third Eye Insight)

[Click here to view the English version of this README](#english-version)

---

## 项目名称与来源

本项目名称“觉之瞳”源自《东方Project》系列中的角色古明地觉（Komeiji Satori）。
古明地觉拥有读取他人内心的能力，其“第三只眼”象征着洞察与直觉。
项目英文名“Third Eye Insight”也体现了通过数据分析来洞察群聊互动、捕捉用户之间隐含关系的理念。

---

## 项目差异与扩展

与之前仅基于数据清洗与传统统计方法的互动分析项目相比，“觉之瞳”在以下方面进行了改进和扩展：

- **深度学习文本嵌入**  
  利用预训练 SentenceTransformer 模型获取文本嵌入，捕捉消息语义信息，使语义相似度计算更精准。

- **多指标融合**  
  除了文本语义相似度，还结合行为互动（连续消息互动次数）和网络拓扑指标（基于度中心性），通过离散化映射拉大微小差异，并支持自动调整融合权重。

- **灵活聚焦与排除分析**
  可以只提取指定用户与其他用户之间的互动数据；同时支持精简模式，仅基于该用户的相关数据进行分析。支持排除指定 QQ 号（多选），从数据集中移除不需要纳入分析的用户。

- **远程数据库支持**
  允许通过数据库连接字符串直接连接远程 PostgreSQL 数据库，满足多种部署场景的需求。

- **SQLCipher 加密数据库支持**
  通过 `config.json` 配置加密参数，可直接读取 NTQQ 原始加密数据库，无需额外转换步骤。

- **全面可视化与报告生成**  
  输出 CSV 文件、生成语义、行为、网络等多个指标的热力图和网络图，并自动生成包含用户映射表的详细分析报告。

---

## 原理解释

本项目采用混合多指标模型衡量用户互动亲密度，核心原理包括：

1. **文本嵌入与语义相似度**
   - 使用预训练 SentenceTransformer 模型将每条消息转换为向量。
   - 通过 5 分钟滑动窗口识别实际交互的消息对，计算这些消息对之间的余弦相似度均值——衡量的是"两人在互相回应对方说的话"，而非全局平均向量的相似性。
   - 采用经验累计分布（ECDF）离散化映射，拉大细微差异。

2. **行为互动**
   - 以 5 分钟为窗口，统计窗口内所有不同用户间的消息交互（非仅相邻两条），并施以指数时间衰减权重（τ=150s），越早的共现贡献越低。
   - 对 log1p(加权互动量) 归一化，反映互动密集程度。

3. **网络拓扑：Jaccard 共同邻居系数**
   - 基于行为互动网络，计算两用户共同互动过的第三方占各自互动对象的比例。
   - 衡量"社交圈重叠程度"，共同朋友越多得分越高，与两人自身是否活跃无关。

4. **指标融合与自动权重**
   - 将语义、行为与网络三个指标按权重（默认：语义 0.4，行为 0.4，网络 0.2）融合，得到最终亲密度得分。
   - 支持 scipy 自动优化权重，最大化得分方差以增强区分度。

---

## 实现方法

1. **数据提取与清洗**  
   - 从数据库（SQLite 或远程 PostgreSQL）中提取群聊或私聊记录，自动过滤 QQ 号为 10000 和 2854196310 的系统消息。  
   - 合并群昵称（字段 40090）与 QQ 名称（字段 40093），默认优先使用群昵称。  
   - 清洗消息内容，保留中文、英文、数字、常见符号和 Emoji，无法显示的字符替换为 “?”。

2. **数据库预处理**
   - 程序启动时会自动检测当前目录下是否存在 `nt_msg.db`，若存在则提示是否剥离前 1024 字节文件头生成 `nt_msg.clean_e.db`，并可选择删除原文件。
   - 也可手动执行：
     ```bash
     python -c "open('nt_msg.clean_e.db','wb').write(open('nt_msg.db','rb').read()[1024:])"
     ```
   - 加密数据库通过 `config.json` 配置 SQLCipher 参数（密码、页大小、KDF 迭代次数等），程序自动识别并解密读取。
     
3. **文本嵌入与特征计算**
   - 使用批量处理和多线程加速计算文本嵌入。
   - 通过滑动窗口计算交互消息对语义相似度、行为互动加权量、Jaccard 网络得分。

4. **指标融合与输出**
   - 可选自动调整权重，使最终融合得分更加离散。
   - 生成 CSV 文件、多维热力图（语义/行为/网络/综合亲密度）、网络图、Top 互动对条形图，同时生成用户映射文件和结构化 Markdown 分析报告。

---

## 特性

- **数据提取与清洗**  
  - 支持群聊（group_msg_table）和私聊（c2c_msg_table）的数据提取  
  - 自动过滤系统消息（QQ 号 10000 与 2854196310）  
  - 交互式时间范围筛选（YYYY/MM/DD 格式）
  - 支持排除指定 QQ 号（逗号分隔，可多选）

- **互动指标计算**
  - 文本嵌入（SentenceTransformer，多线程批量计算）
  - 语义：交互消息对余弦相似度均值（ECDF 离散化）
  - 行为：5 分钟滑动窗口 + 指数时间衰减
  - 网络：Jaccard 共同邻居系数
  - 支持自动调整指标融合权重

- **可视化与报告生成**
  - 热力图（语义/行为/网络/综合亲密度，按活跃度排序，对角遮蔽）
  - Top N 互动对水平堆叠条形图（分量拆解一目了然）
  - 互动网络图（节点大小/颜色编码活跃度，边粗细/透明度编码强度）
  - AI 分析报告：结构化 prompt，包含 Top 20 互动对数据和具体分析要求
  - 输出文件名包含时间区间信息

- **远程数据库支持**
  - 可直接使用数据库连接字符串访问远程 PostgreSQL 数据库

- **SQLCipher 加密数据库支持**
  - 通过 `config.json` 配置加密参数，自动识别并解密 NTQQ 加密数据库
  - 支持 `nt_msg.db` 文件头剥离（启动时自动检测并提示）

---

## 安装方法

确保使用 Python 3.13（Windows 官方安装包，非 MSYS2 内置 Python）。

**推荐：使用 `uv sync`（项目已含 pyproject.toml）**

```bash
uv sync
```

`pyproject.toml` 中已通过 `[tool.uv.sources]` 将 torch 指向本地 cu124 whl 文件，`uv sync` 会自动安装所有依赖（含 GPU 版 PyTorch）。

> **注意**：`torch` 的本地 whl 路径在 `pyproject.toml` 中硬编码，首次运行前请确认路径正确或替换为你的实际路径：
> ```toml
> [tool.uv.sources]
> torch = { path = "D:/下载/torch-2.6.0+cu124-cp313-cp313-win_amd64.whl" }
> ```

> **MSYS2 Python 冲突**：若 PATH 中 MSYS2 Python 排在 Windows Python 之前，`uv` 命令会报 `Unknown operating system: mingw_x86_64_ucrt_gnu`。需显式指定：
> ```bash
> uv sync --python "C:/Users/<你的用户名>/AppData/Local/Programs/Python/Python313/python.exe"
> ```

如需读取 SQLCipher 加密数据库，还需安装 MSYS2 并确保 `libsqlcipher-0.dll` 存在于 `C:/msys64/mingw64/bin/`（MSYS2 中执行 `pacman -S mingw-w64-x86_64-sqlcipher`）。

---

## 使用方法

直接运行程序，按交互式提示操作：

```bash
uv run python train.py
```

程序将依次询问以下配置项（括号内为默认值，直接回车接受）：

| 提示 | 说明 |
|------|------|
| 数据库文件路径 | 本地 SQLite 路径或 PostgreSQL 连接字符串，默认读取 `config.json` 中的 `db_file` |
| 分析模式 | 1 = 群聊，2 = 私聊 |
| 群号 / 好友 QQ 号 | 根据模式填写对应 ID |
| 聚焦用户 QQ 号 | 可选，仅分析该用户相关互动；留空则分析所有用户 |
| 精简模式 | 聚焦用户存在时可选，仅保留该用户参与的互动记录 |
| 排除用户 QQ 号 | 可选，逗号分隔多个 QQ 号，这些用户将从数据集中移除 |
| 使用远程数据库 | 连接字符串模式（PostgreSQL） |
| 启用 GPU 加速 | 需要 CUDA 环境（torch cu124） |
| 自动调整融合权重 | 使用 scipy 优化语义/行为/网络三权重 |
| 生成 AI 分析报告 | 需设置环境变量 `DEEPSEEK_API_KEY` |
| 图表中文字体 | 默认 Microsoft YaHei |

数据提取完成后还会提示输入时间范围（格式 YYYY/MM/DD），直接回车表示不限制。

---

## 数据库预处理说明

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

## 输出结果

程序输出的文件均存储在 `output/YYYY/MM/DD/<群号或QQ号>/` 目录下，按执行日期和分析对象自动归档：

| 文件 | 说明 |
|------|------|
| `interaction_scores.csv` | 用户对得分表（QQ 号、昵称、行为/语义/网络/亲密度，GBK 编码） |
| `semantic_heatmap.png` | 语义相似度热力图（按活跃度排序，对角遮蔽） |
| `behavior_heatmap.png` | 行为互动热力图 |
| `network_heatmap.png` | 网络拓扑热力图 |
| `intimacy_heatmap.png` | 综合亲密度热力图 |
| `top_pairs.png` | Top 20 互动对水平条形图（行为/语义/网络分量拆解） |
| `interaction_network.png` | 互动网络图（节点大小=活跃度，边粗细=亲密度） |
| `user_mapping.txt` | 用户索引、QQ 号、昵称对照表 |
| `analysis_report.md` | （可选）DeepSeek AI 生成的结构化人际关系分析报告 |

文件名中包含时间区间信息（如 `_2025-01-01-end`）。

---

## 许可证

本项目采用 MIT 许可证。

---

## 鸣谢

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

# Chat Interaction Affinity Analysis Tool (Third Eye Insight)

This project is a deep learning and natural language processing based tool for automated analysis of user interaction intimacy in group and private chats. It extracts chat records from SQLite (plaintext or SQLCipher-encrypted) or remote PostgreSQL databases, fuses semantic, behavioral, and network-topology signals, and produces detailed visualizations plus an optional AI analysis report.

## Features

- **Data Extraction & Cleaning**
  - Extracts chat data from SQLite databases (group_msg_table for groups, c2c_msg_table for private chats).
  - Automatically filters out system messages (QQ numbers 10000 and 2854196310).
  - Supports interactive time-range filtering (format: YYYY/MM/DD).
  - Supports excluding specific QQ numbers (comma-separated, multi-select).

- **SQLCipher Encrypted Database Support**
  - Reads NTQQ's native encrypted database directly via `config.json` — no manual conversion needed.
  - Automatic detection of `nt_msg.db` at startup with a prompt to strip the 1024-byte header.

- **Interaction Metrics Calculation**
  - Text embeddings via SentenceTransformer (multi-threaded batch processing).
  - Semantic score: mean cosine similarity of actual interaction message pairs in a 5-minute sliding window.
  - Behavioral score: 5-minute sliding-window co-occurrence with exponential time decay.
  - Network score: Jaccard common-neighbor coefficient on the interaction graph.
  - Supports automatic fusion-weight optimization with scipy.

- **Visualization & Report Generation**
  - Heatmaps for semantic, behavior, network, and final intimacy scores (activity-sorted with masked diagonal).
  - Top-N interaction pairs stacked bar chart with component decomposition.
  - Interaction network graph with node/edge visual encoding.
  - Structured AI report prompt with Top-20 pair data and targeted analysis requirements.
  - Output file names include the selected time range.

- **Interactive Configuration**
  - No command-line arguments — all options are presented as interactive prompts at startup.

## Principle

The core principle combines three metrics:

1. **Text Embedding & Semantic Similarity**
  Each message is encoded with SentenceTransformer. Instead of comparing global per-user mean vectors only, the program identifies real interaction message pairs within a 5-minute sliding window and computes their mean cosine similarity. This better captures "reply-level" semantic resonance. ECDF-based mapping is used to amplify subtle differences.

2. **Behavioral Statistics**
  Within each 5-minute window, all cross-user interactions are counted (not just adjacent messages), with exponential time-decay weighting (tau=150s). Older co-occurrences contribute less. The weighted count is log-transformed and normalized.

3. **Network Topology (Jaccard Common Neighbors)**
  Based on the behavioral interaction graph, the network score is the Jaccard coefficient of common neighbors between two users. It measures social-circle overlap (shared contacts ratio), independent of absolute activity level.

4. **Fusion**
  The semantic, behavioral, and network metrics are fused with configurable weights (default: semantic 0.4, behavior 0.4, network 0.2). Automatic weight optimization via scipy can maximize score variance for stronger separability.

## Project Structure

```
Third-Eye-Insight/
├── extract_chat_data.py       # Data extraction and cleaning (SQLite / SQLCipher / PostgreSQL)
├── train.py                   # Main program: metric calculation, fusion, output
├── visualization.py           # Heatmaps, network graph, top-pairs chart
├── config.json                # SQLCipher encryption configuration
├── output/                    # Output directory: CSV, charts, analysis report
└── README.md                  # This document
```

## Installation

Ensure you are using Python 3.13 (the official Windows installer, **not** the MSYS2 built-in Python).

**Using `uv sync` (recommended — project includes `pyproject.toml`):**

```bash
uv sync
```

`pyproject.toml` already pins torch to a local cu124 wheel via `[tool.uv.sources]`, so `uv sync` installs all dependencies including the GPU-enabled PyTorch.

> **Note**: The local wheel path is hardcoded in `pyproject.toml`. Verify or update the path before running:
> ```toml
> [tool.uv.sources]
> torch = { path = "D:/downloads/torch-2.6.0+cu124-cp313-cp313-win_amd64.whl" }
> ```

> **MSYS2 Python conflict**: If MSYS2's Python appears before the Windows Python in PATH, `uv` will fail with `Unknown operating system: mingw_x86_64_ucrt_gnu`. Fix by specifying the interpreter explicitly:
> ```bash
> uv sync --python "C:/Users/<your-username>/AppData/Local/Programs/Python/Python313/python.exe"
> ```

For SQLCipher encrypted database support, install MSYS2 and run `pacman -S mingw-w64-x86_64-sqlcipher` so that `C:/msys64/mingw64/bin/libsqlcipher-0.dll` is available.

## Usage

Run the program and follow the interactive prompts:

```bash
uv run python train.py
```

The program will ask you step by step for:

| Prompt | Description |
|--------|-------------|
| Database path | Local SQLite path or PostgreSQL connection string; defaults to `db_file` in `config.json` |
| Analysis mode | 1 = group chat, 2 = private chat |
| Group / QQ number | Group ID or friend QQ number depending on mode |
| Focus user QQ number | Optional; leave blank to analyze all users |
| Lite mode | If a focus user is set, keep only that user's related records |
| Exclude user QQ numbers | Optional; comma-separated list of QQ numbers to remove from the dataset |
| Remote database | Use PostgreSQL connection string mode |
| GPU acceleration | Requires CUDA environment (torch cu124) |
| Auto-adjust fusion weights | Uses scipy optimization |
| Generate AI report | Requires `DEEPSEEK_API_KEY` environment variable |
| Chart font | Default: Microsoft YaHei |

After data extraction, you will be prompted for an optional time range filter (format: YYYY/MM/DD).

## Implementation Details

The implementation includes the following steps:

1. **Data Extraction and Cleaning**
  Extract chat records from SQLite or remote PostgreSQL, filter system accounts (10000 and 2854196310), merge group nickname / QQ name fields, and normalize message text.

2. **Database Preprocessing**
  At startup, the program can detect `nt_msg.db` and guide 1024-byte header stripping to produce `nt_msg.clean_e.db`. SQLCipher parameters are loaded from `config.json` for transparent decryption.

3. **Embedding and Feature Computation**
  Compute sentence embeddings in batches with multi-threading, then derive semantic pair scores, time-decayed behavioral interaction scores, and Jaccard network scores with sliding-window logic.

4. **Fusion and Output**
  Optionally auto-tune fusion weights, then export CSV, multi-metric heatmaps, interaction network graph, Top-N interaction pairs chart, user mapping, and an optional structured AI analysis report.

## Database Preprocessing

**Encrypted NTQQ database (`nt_msg.db`):**
The program automatically detects `nt_msg.db` at startup and prompts you to strip the 1024-byte header to produce `nt_msg.clean_e.db`. You can also do this manually:

```bash
python -c "open('nt_msg.clean_e.db','wb').write(open('nt_msg.db','rb').read()[1024:])"
```

**`config.json` for SQLCipher decryption:**

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

Set `encrypted` to `false` (or omit it) for plaintext SQLite databases.

**PostgreSQL:** Ensure your connection string is correct and the service is running.

## Output

All output files are saved under `output/YYYY/MM/DD/<identifier>/`, automatically archived by execution date:

| File | Description |
|------|-------------|
| `interaction_scores.csv` | Pairwise score table (QQ IDs, nicknames, behavior/semantic/network/final intimacy; GBK encoding) |
| `semantic_heatmap.png` | Semantic heatmap (activity-sorted, masked diagonal) |
| `behavior_heatmap.png` | Behavioral heatmap |
| `network_heatmap.png` | Network-topology heatmap |
| `intimacy_heatmap.png` | Final intimacy heatmap |
| `top_pairs.png` | Top 20 interaction pairs horizontal stacked bar chart (component breakdown) |
| `interaction_network.png` | Interaction network graph (node size = activity, edge width = intimacy) |
| `user_mapping.txt` | User index / QQ / nickname mapping |
| `analysis_report.md` | (Optional) Structured relationship analysis report generated by DeepSeek AI |

File names include the selected time range (for example, `_2025-01-01-end`).

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [NetworkX](https://networkx.org/)
- [PyTorch](https://pytorch.org/)
- [SentenceTransformers](https://www.sbert.net/)
- [Scipy](https://www.scipy.org/)
- [DeepSeek API](https://api.deepseek.com/)
