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

- **灵活聚焦分析**
  可以只提取指定用户与其他用户之间的互动数据；同时支持精简模式，仅基于该用户的相关数据进行分析。

- **远程数据库支持**
  允许通过数据库连接字符串直接连接远程 PostgreSQL 数据库，满足多种部署场景的需求。

- **SQLCipher 加密数据库支持**
  通过 `config.json` 配置加密参数，可直接读取 NTQQ 原始加密数据库，无需额外转换步骤。

- **全面可视化与报告生成**  
  输出 CSV 文件、生成语义、行为、网络等多个指标的热力图和网络图，并自动生成包含用户映射表的详细分析报告。

---

## 原理解释

本项目采用混合多指标模型衡量用户互动活跃度，核心原理包括：

1. **文本嵌入与语义相似度**  
   - 使用预训练 SentenceTransformer 模型将每条消息转换为向量，计算每个用户的平均嵌入，并利用余弦相似度构造语义相似度矩阵。  
   - 采用经验累计分布（ECDF）离散化映射，将连续得分离散化到指定等级，使中间值较多、极端值较少。

2. **行为统计**  
   - 统计用户在 5 分钟内连续互动的次数，对 np.log1p(互动次数) 进行归一化与离散化映射，反映用户之间的行为互动频率。

3. **网络拓扑指标**  
   - 基于行为数据构建用户互动网络，计算各用户的度中心性，并定义用户对之间的网络得分为二者中心性的平均值。

4. **指标融合与自动权重**  
   - 将语义、行为与网络三个指标按照设定或自动调整的权重（示例默认：语义 0.4，行为 0.4，网络 0.2）融合，得到最终互动活跃度得分。

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
   - 使用批量处理和多线程加速计算文本嵌入；对每个用户的消息向量取均值，并归一化后构造语义相似度矩阵。  
   - 统计行为互动和构建网络拓扑指标。

4. **指标融合与输出**  
   - 可选自动调整权重，使最终融合得分更加离散。  
   - 生成 CSV 文件、热力图（语义、行为、网络各指标）和网络图，同时生成用户映射文件和详细的 Markdown 分析报告。

---

## 特性

- **数据提取与清洗**  
  - 支持群聊（group_msg_table）和私聊（c2c_msg_table）的数据提取  
  - 自动过滤系统消息（QQ 号 10000 与 2854196310）  
  - 交互式时间范围筛选（YYYY/MM/DD 格式）  

- **互动指标计算**  
  - 文本嵌入（使用 SentenceTransformer，支持多线程批量计算）  
  - 语义相似度、行为互动与网络拓扑指标计算  
  - 离散化映射（ECDF 映射）拉大细微差异  
  - 支持自动调整指标融合权重  

- **可视化与报告生成**
  - 输出 CSV 文件、热力图、网络图等
  - 自动生成包含用户映射表的详细分析报告
  - 支持聚焦分析特定用户
  - 输出文件名中包含时间区间信息

- **远程数据库支持**
  - 可直接使用数据库连接字符串访问远程 PostgreSQL 数据库

- **SQLCipher 加密数据库支持**
  - 通过 `config.json` 配置加密参数，自动识别并解密 NTQQ 加密数据库
  - 支持 `nt_msg.db` 文件头剥离（启动时自动检测并提示）

---

## 安装方法

确保使用 Python 3.13.2（Windows 官方安装包，非 MSYS2 内置 Python）。

**推荐：使用 `uv`**

> **注意**：若系统 PATH 中 MSYS2 的 Python（如 `C:/msys64/ucrt64/bin/python.exe`）排在 Windows Python 之前，`uv pip install` 会报错 `Unknown operating system: mingw_x86_64_ucrt_gnu`。
> 解决方法：显式指定 Python 解释器路径：
> ```bash
> uv pip install --python "C:/Users/<你的用户名>/AppData/Local/Programs/Python/Python313/python.exe" \
>   pandas numpy matplotlib seaborn networkx torch sentence-transformers scipy requests scikit-learn openai
> ```

> **注意**：`uv add` 需要项目存在 `pyproject.toml`，直接在脚本目录运行会报错 `No pyproject.toml found`。
> 此项目为单目录脚本，请使用 `uv pip install` 而非 `uv add`。

```bash
uv pip install --python "C:/Users/<你的用户名>/AppData/Local/Programs/Python/Python313/python.exe" \
  pandas numpy matplotlib seaborn networkx torch sentence-transformers scipy requests scikit-learn openai
```

**或者直接使用 pip（更简单）：**

```bash
"C:/Users/<你的用户名>/AppData/Local/Programs/Python/Python313/python.exe" -m pip install \
  pandas numpy matplotlib seaborn networkx torch sentence-transformers scipy requests scikit-learn openai
```

其他版本不确定兼容性，若遇到问题请切换至 Python 3.13.2。

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
| 使用远程数据库 | 连接字符串模式（PostgreSQL） |
| 启用 GPU 加速 | 需要 CUDA 环境 |
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

程序输出的文件均存储在 `output/` 目录下：
- **interaction_scores.csv**：包含用户对（唯一组合）的 QQ 号、昵称、行为得分、语义得分、网络得分和最终互动活跃度得分（GBK 编码）。
- **语义相似度热力图**、**行为得分热力图**、**网络拓扑得分热力图**：PNG 格式图表，文件名中包含时间区间（如有）。
- **interaction_network.png**：用户互动网络图。
- **user_mapping.txt**：用户映射文件，列出所有用户的索引、QQ 号和昵称。
- **analysis_report.md**（可选）：调用 DeepSeek API 自动生成的详细分析报告，报告中包含用户映射表及其它关键信息。

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

This project is a deep learning and natural language processing based tool for automated analysis of user interaction affinity in group and private chats. It extracts chat records from an SQLite (plaintext or SQLCipher-encrypted) or remote PostgreSQL database, calculates interaction scores between users by combining text embeddings and behavioral statistics, and generates detailed visualizations and an automated analysis report.

## Features

- **Data Extraction & Cleaning**
  - Extracts chat data from SQLite databases (group_msg_table for groups, c2c_msg_table for private chats).
  - Automatically filters out system messages (QQ numbers 10000 and 2854196310).
  - Supports interactive time-range filtering (format: YYYY/MM/DD).

- **SQLCipher Encrypted Database Support**
  - Reads NTQQ's native encrypted database directly via `config.json` — no manual conversion needed.
  - Automatic detection of `nt_msg.db` at startup with a prompt to strip the 1024-byte header.

- **Interaction Metrics Calculation**
  - Uses SentenceTransformer to obtain text embeddings and computes each user's average embedding (mean pooling and normalization).
  - Constructs a behavior matrix by counting consecutive interactions within a 5-minute window.
  - Calculates a semantic similarity matrix (cosine similarity between average embeddings).
  - Applies ECDF-based discrete mapping to amplify subtle differences.
  - Fuses semantic, behavior, and network topology scores with configurable or auto-optimized weights.

- **Visualization & Report Generation**
  - Generates heatmaps (semantic, behavior, network), network graphs, and CSV output.
  - Optionally generates a detailed analysis report using the DeepSeek API.
  - Supports focusing on a specific user.

- **Interactive Configuration**
  - No command-line arguments — all options are presented as interactive prompts at startup.

## Principle

The core principle combines three metrics:

1. **Text Embedding & Semantic Similarity**
   Each message is converted to a vector using SentenceTransformer. Per-user average embeddings are computed and normalized; cosine similarity forms the semantic matrix. ECDF discrete mapping spreads scores across a discrete scale.

2. **Behavioral Statistics**
   Counts consecutive interactions within a 5-minute window. Log-transforms the counts and normalizes via ECDF mapping.

3. **Network Topology**
   Builds an interaction graph with NetworkX and computes degree centrality per user. The network score for a pair is the average of their centralities.

4. **Fusion**
   Three metrics are fused with configurable weights (default: semantic 0.4, behavior 0.4, network 0.2). Automatic weight optimization via scipy is also available.

## Project Structure

```
QQ-Interaction-Analysis-Tool/
├── extract_chat_data.py       # Data extraction and cleaning (SQLite / SQLCipher / PostgreSQL)
├── train.py                   # Main program: metrics calculation, fusion, output
├── visualization.py           # Heatmaps and network graph generation
├── config.json                # SQLCipher encryption configuration
├── output/                    # Output directory: CSV, charts, analysis report
└── README.md                  # This document
```

## Installation

Ensure you are using Python 3.13.2 (the official Windows installer, **not** the MSYS2 built-in Python).

**Using `uv` (recommended):**

> **Known issue — MSYS2 Python in PATH**: If MSYS2's Python (e.g. `C:/msys64/ucrt64/bin/python.exe`) appears before the Windows Python in your PATH, `uv pip install` will fail with `Unknown operating system: mingw_x86_64_ucrt_gnu`. Fix: specify the interpreter path explicitly.

> **Known issue — no `pyproject.toml`**: `uv add` requires a `pyproject.toml` and will fail with `No pyproject.toml found` in a plain script directory. Use `uv pip install` instead.

```bash
uv pip install --python "C:/Users/<your-username>/AppData/Local/Programs/Python/Python313/python.exe" \
  pandas numpy matplotlib seaborn networkx torch sentence-transformers scipy requests scikit-learn openai
```

**Or using pip directly:**

```bash
"C:/Users/<your-username>/AppData/Local/Programs/Python/Python313/python.exe" -m pip install \
  pandas numpy matplotlib seaborn networkx torch sentence-transformers scipy requests scikit-learn openai
```

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
| Remote database | Use PostgreSQL connection string mode |
| GPU acceleration | Requires CUDA environment |
| Auto-adjust fusion weights | Uses scipy optimization |
| Generate AI report | Requires `DEEPSEEK_API_KEY` environment variable |
| Chart font | Default: Microsoft YaHei |

After data extraction, you will be prompted for an optional time range filter (format: YYYY/MM/DD).

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

All output files are saved in the `output/` directory:

- **interaction_scores.csv**: CSV (GBK encoding) with unique user pair data: UserID, Nickname, BehaviorScore, SemanticScore, NetworkScore, Final Interaction Activity Score.
- **Heatmaps**: PNG files for semantic similarity, behavior scores, and network topology scores. File names include the time range if specified.
- **interaction_network.png**: Network graph of user interactions.
- **user_mapping.txt**: Maps user index, QQ number, and nickname.
- **analysis_report.md** (optional): Detailed analysis report generated via DeepSeek API.

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
