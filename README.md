# 觉之瞳

[Click here to view the English version of the README](https://github.com/EndsOculus/QQ-Interaction-Analysis-Tool/blob/main/README_en.md)

本项目基于深度学习与自然语言处理技术，用于自动化分析群聊和私聊中用户之间的互动亲密度。工具从未加密的 SQLite 数据库中提取聊天记录，通过文本嵌入与行为统计计算用户对之间的互动得分，并生成详细的可视化图表与自动化分析报告。

## 项目差异与扩展
与之前的项目相比，本工具在以下几个方面做出了改进和扩展：

- **深度学习文本嵌入**：  
  之前的项目主要基于数据清洗和传统统计方法计算互动指标，本项目引入 SentenceTransformer 获取文本嵌入，从语义角度衡量用户间的相似度。

- **离散化映射方法**：  
  为了拉大微小差异，本项目采用 ECDF 离散化映射方法，将原始的连续得分离散化，使得中等相似度更常见，而极高或极低得分更稀少，从而更好地反映用户之间的真实差异。

- **融合多源指标**：  
  本项目不仅计算文本相似度，还统计行为互动数据，并将两者融合，提供更全面的互动亲密度评价。

- **灵活的用户聚焦**：  
  通过 --focus-user 参数，可以专注分析特定用户与其他用户的互动情况，方便针对性运营与研究。

- **全面的可视化与报告**：  
  除了 CSV 数据输出外，还生成热力图、网络图等多种可视化图表，并支持自动生成详细分析报告，帮助用户更直观地理解数据。
  
## 特性

- **数据提取与清洗**  
  - 从 SQLite 数据库中提取群聊（group_msg_table）或私聊（c2c_msg_table）数据  
  - 自动过滤 QQ 号为 10000 和 2854196310 的系统消息  
  - 支持交互式时间范围筛选（格式：YYYY/MM/DD）

- **互动指标计算**  
  - 使用 SentenceTransformer 提取文本嵌入，计算每个用户的平均嵌入（归一化均值）  
  - 构造行为矩阵（统计 5 分钟内连续消息互动次数），并使用经验累计分布（ECDF）离散化映射  
  - 计算语义相似度矩阵（基于余弦相似度），并采用 ECDF 离散映射以拉大微小差异  
  - 融合行为得分与语义得分，各占 50% 得到最终互动亲密度得分

- **可视化与报告生成**  
  - 生成热力图、网络图等可视化图表  
  - 可选调用 DeepSeek API 自动生成详细分析报告  
  - 支持聚焦分析指定用户（--focus-user 参数）

- **灵活的命令行参数**  
  - 支持群聊/私聊模式、指定群号/好友 QQ 号、时间筛选、GPU 加速（可选）及报告生成

## 原理解释

本项目的核心原理分为两部分：

1. **文本嵌入与语义相似度**  
   - 利用预训练的 SentenceTransformer 模型将每条消息转换为向量表示，并对每个用户的消息向量求均值，得到用户整体的文本特征。  
   - 基于余弦相似度计算用户之间的语义相似度，采用经验累计分布函数（ECDF）的离散化映射，使得相似度得分在 0～1 内分布得更加离散，中间值更多，极端值较少。

2. **行为统计与融合计算**  
   - 统计在 5 分钟内连续互动的消息次数构造行为矩阵，对数值进行对数变换（np.log1p）后使用 ECDF 离散化映射。  
   - 将行为得分与语义得分各占 50% 融合，形成最终的互动亲密度得分，从而反映用户之间的互动强度与内容相似性。

## 实现方法

项目的实现方法主要包括以下步骤：

1. **数据提取与清洗**  
   - 从数据库中提取聊天记录，并根据输入的时间范围筛选数据。  
   - 合并群昵称和 QQ 名称，确保每条记录包含用户标识与清晰的时间戳。

2. **文本特征提取**  
   - 使用 SentenceTransformer 模型计算每条消息的文本嵌入，然后求各用户的平均向量并归一化。  

3. **语义相似度计算**  
   - 基于用户平均嵌入向量计算余弦相似度矩阵，再通过 ECDF 离散化映射，将连续得分离散化到指定等级数，拉大细微差异。

4. **行为得分计算**  
   - 统计用户之间在 5 分钟内的连续互动次数，采用 np.log1p 归一化后，通过 ECDF 映射离散化行为得分。

5. **得分融合与输出**  
   - 将离散化后的语义得分与行为得分按 50% 权重融合，生成最终的互动亲密度得分。  
   - 生成 CSV 文件（GBK编码），仅输出唯一用户对数据，并支持聚焦某个用户的分析。

6. **可视化与报告生成**  
   - 通过调用可视化模块生成热力图、网络图等图表，并可选调用 DeepSeek API 自动生成详细的 Markdown 格式分析报告。

## 示例

假设我们的群聊号码为 **114514**，群名为 **幻想乡**，且有用户 **1919810**。示例中两个用户分别为：  
- 用户1：**魔理沙**  
- 用户2：**灵梦**

**群聊模式示例：**

```bash
python train.py --group 114514 --db nt_msg.clean.db --mode group --id 114514 --focus-user 1919810 --font "Microsoft YaHei"
```

**私聊模式示例：**

```bash
python train.py --group 0 --db nt_msg.clean.db --mode c2c --id 1919810 --font "Microsoft YaHei"
```

**Lite 模式 (`--lite`)：**  
  当同时指定 `--lite` 参数和 `--focus-user` 参数时，程序将仅保留与指定用户相关的互动记录（即只分析该用户与其他用户之间在 5 分钟内连续互动的聊天数据），从而实现聚焦分析。  
  **使用示例：**  
  ```bash
  python train.py --group 114514 --db nt_msg.clean.db --mode group --id 114514 --focus-user 1919810 --lite --font "Microsoft YaHei"
  ```

## 项目结构

```
QQ-Interaction-Analysis-Tool/
├── extract_chat_data.py       # 数据提取与清洗模块
├── train.py                   # 主程序：互动指标计算、融合、CSV 输出及可视化/报告生成
├── visualization.py           # 可视化图表生成模块（热力图、网络图等）
├── output/                    # 输出目录：保存 CSV、图表和报告
└── README.md                  # 本文档
```

## 安装

确保安装 Python 3.13.2，并使用 pip 安装以下依赖：

```bash
pip install pandas numpy matplotlib seaborn networkx torch sentence-transformers scipy requests scikit-learn
```

## 使用方法

运行 `train.py` 时支持以下命令行参数：

- `--group`：群聊号码（群聊模式下有效）。
- `--db`：数据库文件路径（例如 `nt_msg.clean.db`）。
- `--mode`：模式，`group` 表示群聊，`c2c` 表示私聊。
- `--id`：在群聊模式下与 `--group` 相同；在私聊模式下为好友 QQ 号。
- `--focus-user`：聚焦分析某个用户（QQ号）。
- `--font`：图表显示使用的中文字体（默认 “Microsoft YaHei”）。
- `--boost`：是否启用 GPU 加速（默认关闭）。
- `--report`：是否生成详细分析报告（调用 DeepSeek API）。
- `--lite`：是否仅保留与指定用户相关的互动记录,实现高度聚焦分析。

程序启动后会提示输入起始日期和结束日期（格式：YYYY/MM/DD）：  
- 同时输入起始和结束日期：仅分析该时间段数据。  
- 直接回车：默认分析所有数据。  
- 只输入起始日期：从该日期开始分析。  
- 只输入结束日期：截止至该日期。

## 许可证

本项目采用 MIT 许可证。

## 鸣谢

- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [NetworkX](https://networkx.org/)
- [PyTorch](https://pytorch.org/)
- [SentenceTransformers](https://www.sbert.net/)
- [DeepSeek API](https://api.deepseek.com/)
