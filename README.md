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
  通过 `--focus-user` 参数，可以只提取指定用户与其他用户之间的互动数据；同时支持 `--lite` 模式，仅基于该用户的相关数据进行分析。

- **远程数据库支持**  
  增加 `--remote` 参数，允许通过数据库连接字符串直接连接远程 PostgreSQL 数据库，满足多种部署场景的需求。

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
   - 对于 SQLite 数据库，若原始数据库文件包含前1024字节的无用数据，可使用如下命令进行预处理：
     ```bash
     python -c "open('nt_msg.clean.db','wb').write(open('nt_msg.db','rb').read()[1024:])"
     ```
     
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
  - 支持聚焦分析特定用户（--focus-user 参数）  
  - 输出文件名中包含时间区间信息

- **远程数据库支持**  
  - 通过 `--remote` 参数可直接使用数据库连接字符串访问远程 PostgreSQL 数据库

---

## 安装方法

确保使用 Python 3.13.2，并使用 pip 安装以下依赖：

```bash
pip install pandas numpy matplotlib seaborn networkx torch sentence-transformers scipy requests scikit-learn
```

其他版本不确定兼容性，若遇到问题请切换至 Python 3.13.2。

---

## 使用方法

### 命令行参数说明

- **--group**：群聊号码（群聊模式下有效）。
- **--db**：数据库连接字符串或本地数据库文件路径。例如：
  - SQLite 本地数据库：`nt_msg.clean.db`
  - PostgreSQL 连接字符串：`postgresql://username:password@host:port/dbname`
- **--mode**：模式，`group` 表示群聊，`c2c` 表示私聊。
- **--id**：在群聊模式下与 --group 相同；在私聊模式下为好友 QQ 号（可选）。
- **--focus-user**：指定聚焦分析某个用户（仅提取该用户与其他用户的互动数据）。
- **--lite**：启用精简模式，仅保留与 focus-user 相关的记录（需同时指定 --focus-user）。
- **--font**：图表使用的中文字体（默认 "Microsoft YaHei"）。
- **--boost**：启用 GPU 加速（默认关闭）。
- **--report**：生成详细分析报告（调用 API）。
- **--auto-weight**：自动调整指标融合权重。
- **--remote**：使用远程数据库连接（否则默认使用本地数据库文件）。

### 示例

**群聊模式（分析群聊 114514，聚焦用户 1919810）：**

```bash
python train.py --group 114514 --db nt_msg.clean.db --mode group --id 114514 --focus-user 1919810 --font "Microsoft YaHei"
```

**私聊模式（分析与 QQ 号 1919810 的私聊数据）：**

```bash
python train.py --group 0 --db nt_msg.clean.db --mode c2c --id 1919810 --font "Microsoft YaHei"
```

**Lite 模式（仅分析 focus-user 的相关互动）：**

```bash
python train.py --group 114514 --db nt_msg.clean.db --mode group --id 114514 --focus-user 1919810 --lite --font "Microsoft YaHei"
```

**远程数据库连接示例：**

```bash
python train.py --group 114514 --db "postgresql://sr:your_password@127.0.0.1:5432/botmsg" --mode group --id 114514 --font "Microsoft YaHei" --remote
```

程序启动后会提示输入起始和结束日期（格式 YYYY/MM/DD）：
- 同时输入起始和结束日期：仅分析该时间段数据。
- 直接回车：默认分析所有数据。
- 只输入起始日期：从该日期开始分析。
- 只输入结束日期：截止至该日期。

---

## 数据库预处理说明

- **SQLite 数据库预处理**：  
  如果原始数据库（nt_msg.db）包含前 1024 字节的无用数据，请运行以下命令生成清洗后的数据库文件：
  ```bash
  python -c "open('nt_msg.clean.db','wb').write(open('nt_msg.db','rb').read()[1024:])"
  ```

- **PostgreSQL 数据库**：  
  请确保你的数据库连接字符串正确，并且 PostgreSQL 服务已启动。连接字符串格式如：  
  ```bash
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

## English Version

# Third Eye Insight

## Project Name Origin

The project name "Third Eye Insight" is derived from the character Komeiji Satori in the Touhou Project series. 
Satori possesses the ability to read minds, and her "third eye" symbolizes insight and perception. 
The name reflects our goal to analyze chat records deeply and uncover the underlying interaction patterns among users.

## Differences and Enhancements

Compared to our previous project based on data cleaning and traditional statistics, this project has been enhanced in several ways:
- **Deep Learning Text Embeddings**: Utilizes pre-trained SentenceTransformer models to capture semantic features, leading to more accurate similarity calculations.
- **Multi-Metric Fusion**: Combines semantic similarity, behavioral statistics, and network topology metrics. It applies ECDF-based discrete mapping to amplify subtle differences and supports auto-weight optimization.
- **Focused Analysis**: With the `--focus-user` and `--lite` parameters, you can target the interactions of a specific user.
- **Remote Database Support**: Offers the option to connect to remote PostgreSQL databases via a connection string.
- **Comprehensive Visualization and Reporting**: Outputs CSV files, heatmaps, network graphs, and can automatically generate a detailed analysis report.

## Implementation Details

The implementation includes the following steps:
1. **Data Extraction and Cleaning**:  
   Extract chat records from an unencrypted SQLite or remote PostgreSQL database, filtering out system messages (e.g., QQ numbers 10000 and 2854196310) and merging group nicknames with QQ names.
2. **Text Embedding Extraction**:  
   Compute text embeddings using SentenceTransformer in batch mode with multi-threading.
3. **Semantic Similarity Calculation**:  
   Compute the cosine similarity between users’ average text embeddings and apply ECDF-based discrete mapping.
4. **Behavior Score Calculation**:  
   Count consecutive interactions within 5 minutes, normalize using log transformation, and apply discrete mapping.
5. **Network Topology Metrics**:  
   Construct an interaction network using NetworkX, compute degree centrality, and derive network scores.
6. **Fusion and Output**:  
   Fuse the semantic, behavior, and network scores (with auto-weight optimization if enabled) to derive the final Interaction Activity Score. Output results in CSV format, along with various heatmaps, network graphs, a user mapping file, and an optional detailed report.
7. **Parameter Options**:  
   Command-line parameters allow you to specify the group/chat mode, focus on a specific user, enable remote database access, set the analysis time range, and more.

## Installation

Ensure you are using Python 3.13.2 and install the following dependencies:

```bash
pip install pandas numpy matplotlib seaborn networkx torch sentence-transformers scipy requests scikit-learn
```

## Usage

### Command-Line Parameters

- **--group**: Group chat number (effective in group mode).
- **--db**: Database connection string or local database file path.
- **--mode**: Mode: `group` for group chat, `c2c` for private chat.
- **--id**: In group mode, should be the same as --group; in c2c mode, friend QQ number.
- **--focus-user**: Focus analysis on a specific user (QQ number).
- **--lite**: If set along with --focus-user, only interactions involving that user are analyzed.
- **--font**: Font used in charts (default "Microsoft YaHei").
- **--boost**: Enable GPU acceleration (default off).
- **--report**: Generate a detailed analysis report via API.
- **--auto-weight**: Auto-adjust fusion weights.
- **--remote**: Use remote database connection; otherwise, use local database.

### Examples

**Group Chat Analysis (with focus):**

```bash
python train.py --group 114514 --db nt_msg.clean.db --mode group --id 114514 --focus-user 1919810 --font "Microsoft YaHei"
```

**Private Chat Analysis:**

```bash
python train.py --group 0 --db nt_msg.clean.db --mode c2c --id 1919810 --font "Microsoft YaHei"
```

**Lite Mode (focused only on specified user's interactions):**

```bash
python train.py --group 114514 --db nt_msg.clean.db --mode group --id 114514 --focus-user 1919810 --lite --font "Microsoft YaHei"
```

**Remote Database Connection:**

```bash
python train.py --group 114514 --db "postgresql://sr:your_password@127.0.0.1:5432/botmsg" --mode group --id 114514 --font "Microsoft YaHei" --remote
```

After starting the program, you will be prompted to input a start date and an end date (format: YYYY/MM/DD). Press Enter to leave unrestricted, or specify one or both dates to limit the analysis period.

## Database Preprocessing

For SQLite databases, if the original database (nt_msg.db) contains unnecessary data in the first 1024 bytes, run:

```bash
python -c "open('nt_msg.clean.db','wb').write(open('nt_msg.db','rb').read()[1024:])"
```

For PostgreSQL, ensure your connection string is correct and the PostgreSQL service is running.

## Output

All output files are saved in the `output/` directory:
- **interaction_scores.csv**: CSV file (GBK encoding) with unique user pair data (UserID, Nickname, BehaviorScore, SemanticScore, NetworkScore, Final Interaction Activity Score).
- **Heatmaps**: PNG files for semantic similarity, behavior scores, and network topology scores (file names include the time range, if specified).
- **interaction_network.png**: Network graph of user interactions.
- **user_mapping.txt**: A file mapping user index, QQ number, and nickname.
- **analysis_report.md**: (Optional) A detailed analysis report generated via API.

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

---
